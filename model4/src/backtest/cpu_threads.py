"""User-authorized CPU thread amendment for only the last frozen evaluation origin.

The original source/config/checkpoints stay byte-for-byte unchanged.  This runner
validates five original origins and executes the last origin under a separately
pinned manifest whose sole protocol difference is catboost.thread_count=4.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

from src.backtest.execution import Attempt, LockBusy, RunLock, atomic_json, read_json, replace_file, run_child, utc_now
from src.backtest.protocol import ROOT, digest_json

EXTENSION = Path(__file__).resolve()


def owned(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / 'artifacts/backtest').resolve()):
        raise ValueError('run must be inside workspace artifacts/backtest')
    return path


def amended_protocol(base, threads=4):
    if threads != 4 or base['catboost']['thread_count'] != 1:
        raise ValueError('only the authorized CPU 1-to-4 thread change is allowed')
    config = copy.deepcopy(base)
    config['catboost']['thread_count'] = threads
    return config


def require_only_thread_change(base, actual):
    if actual != amended_protocol(base):
        raise ValueError('amendment changed something other than CPU thread_count')


def freeze(base_root, base_hash, root):
    from src.backtest.data import sha256
    from src.backtest.frozen import read_manifest, verify_runtime
    from src.backtest.selection import verify_selection
    base_root, root = owned(base_root), owned(root)
    if root.exists():
        raise ValueError('amendment freeze requires a new run directory')
    base = read_manifest(base_root, base_hash)
    verify_runtime(base)
    origins = base['protocol']['evaluation_origins']
    if len(origins) != 6 or origins[-1] != '2024-12':
        raise ValueError('expected the predeclared six-origin experiment')
    if (base_root / 'evaluation' / origins[-1]).exists():
        raise ValueError('last CPU-1 origin already started; refusing to discard or mix its work')
    selected = read_json(base_root / 'selected_trend.json')
    verify_selection(base_root, base, selected['selection_sha256'])
    extension = {'relative_path': str(EXTENSION.relative_to(ROOT)).replace('\\', '/'),
                 'sha256': sha256(EXTENSION)}
    config = amended_protocol(base['protocol'])
    environment = copy.deepcopy(base['environment'])
    environment['execution_extension'] = extension
    manifest = copy.deepcopy(base)
    manifest.update(created_utc=utc_now(), protocol=config, protocol_sha256=digest_json(config), environment=environment)
    manifest['execution_amendment'] = {
        'base_run_dir': str(base_root), 'base_manifest_sha256': base_hash,
        'selection_sha256': selected['selection_sha256'], 'target_origin': origins[-1],
        'original_thread_count': 1, 'amended_thread_count': 4,
        'changed_protocol_key': 'catboost.thread_count',
        'authorization': 'User explicitly requested the sixth window use CPU multithreading; all models and backtest rules unchanged.',
        'extension': extension,
    }
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = digest_json(manifest)
    root.mkdir(parents=True)
    atomic_json(root / 'frozen_protocol.json', manifest)
    verify_amendment(root, manifest['manifest_sha256'])
    return manifest


def verify_amendment(root, expected_hash):
    from src.backtest.data import sha256
    from src.backtest.frozen import read_manifest, verify_runtime
    manifest = read_manifest(root, expected_hash)
    amendment = manifest['execution_amendment']
    base_root = owned(amendment['base_run_dir'])
    base = read_manifest(base_root, amendment['base_manifest_sha256'])
    # This is the original strict source/data/dependency/config verifier.
    verify_runtime(base)
    require_only_thread_change(base['protocol'], manifest['protocol'])
    target = base['protocol']['evaluation_origins'][-1]
    if (len(base['protocol']['evaluation_origins']) != 6 or target != '2024-12'
            or amendment['target_origin'] != target or amendment['changed_protocol_key'] != 'catboost.thread_count'
            or amendment['original_thread_count'] != 1 or amendment['amended_thread_count'] != 4):
        raise ValueError('execution amendment scope differs from user authorization')
    extension = {'relative_path': str(EXTENSION.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256(EXTENSION)}
    expected_environment = copy.deepcopy(base['environment'])
    expected_environment['execution_extension'] = extension
    if amendment['extension'] != extension or manifest['environment'] != expected_environment:
        raise ValueError('execution extension/source/dependency snapshot changed')
    permitted_changes = {'created_utc', 'protocol', 'protocol_sha256', 'environment', 'manifest_sha256', 'execution_amendment'}
    if {k: v for k, v in manifest.items() if k not in permitted_changes} != {k: v for k, v in base.items() if k not in permitted_changes}:
        raise ValueError('amendment changed frozen row plan, selection rule, or dataset')
    return manifest, base_root, base


def worker(root, pin, dispatch_id):
    from src.backtest import frozen, selection
    from src.backtest.worker import run_origin
    manifest, base_root, base = verify_amendment(root, pin)
    amendment = manifest['execution_amendment']
    original_runtime, original_selection = frozen.verify_runtime, selection.verify_selection

    def runtime_check(candidate):
        if candidate == manifest:
            # Avoid recursion: verify_amendment must call the original verifier on base.
            current, _, _ = verify_amendment(root, pin)
            if current != candidate:
                raise ValueError('amended manifest changed during fitting')
            return
        return original_runtime(candidate)

    def selection_check(candidate_root, candidate_manifest, selection_pin):
        if Path(candidate_root).resolve() == root:
            runtime_check(candidate_manifest)
            if selection_pin != amendment['selection_sha256']:
                raise ValueError('trend selection pin changed')
            return original_selection(base_root, base, selection_pin)
        return original_selection(candidate_root, candidate_manifest, selection_pin)

    # The original worker is reused with explicit amendment verifiers, never with
    # disabled verification. Its real config/audit/contracts record thread_count=4.
    frozen.verify_runtime, selection.verify_selection = runtime_check, selection_check
    try:
        with RunLock(root / 'worker.lock'):
            attempt = Attempt(root / 'attempts/worker', root / 'worker_status.json', 'worker',
                              origin=amendment['target_origin'], phase='evaluation', dispatch_id=dispatch_id,
                              command=sys.argv, execution_thread_count=4)
            try:
                args = argparse.Namespace(run_dir=str(root), manifest_hash=pin, phase='evaluation',
                                          origin=amendment['target_origin'], selection_hash=amendment['selection_sha256'])
                run_origin(args, attempt)
            except BaseException as error:
                attempt.fail(error)
                raise
    finally:
        frozen.verify_runtime, selection.verify_selection = original_runtime, original_selection


def aggregate(root, manifest, base_root, base, selection, attempt):
    import numpy as np
    import pandas as pd
    from src.backtest.data import sha256
    from src.backtest.frozen import DIRECT_MODELS, DEVELOPMENT_MODELS
    from src.backtest.scoring import metrics
    from src.backtest.selection import validate_origin
    target = manifest['execution_amendment']['target_origin']
    summary = root / 'summary'
    summary.mkdir(exist_ok=True)
    artifacts, evidence, phase_tables = [], {}, {}
    for phase in ['development', 'evaluation']:
        names = list(DEVELOPMENT_MODELS) if phase == 'development' else [*DIRECT_MODELS, f"decomposed_{selection['selected_trend']}"]
        frames = []
        for origin in base['protocol'][f'{phase}_origins']:
            revised = phase == 'evaluation' and origin == target
            source_root, source_manifest = (root, manifest) if revised else (base_root, base)
            frame, receipt = validate_origin(source_root, source_manifest, phase, origin, names)
            frame = frame.copy()
            frame['execution_thread_count'] = 4 if revised else 1
            frames.append(frame)
            evidence[f'{phase}/{origin}'] = {'run_dir': str(source_root), 'manifest_sha256': source_manifest['manifest_sha256'],
                                            'receipt_sha256': digest_json(receipt), 'thread_count': 4 if revised else 1}
        combined = pd.concat(frames, ignore_index=True)
        table = metrics(combined, names, base['protocol'])
        phase_tables[phase] = table
        for suffix in ['parquet', 'csv.gz']:
            path = summary / f'{phase}_predictions.{suffix}'
            temporary = path.with_name(path.name + '.partial')
            if suffix == 'parquet':
                combined.to_parquet(temporary, index=False)
            else:
                combined.to_csv(temporary, index=False, compression='gzip')
            replace_file(temporary, path)
            artifacts.append(path)
        path = summary / f'{phase}_metrics.csv'
        table.to_csv(path, index=False)
        artifacts.append(path)
        deltas = []
        for (origin, bucket), group in table.groupby(['origin', 'horizon'], sort=False):
            for candidate_name in [n for n in names if n.startswith('decomposed_')]:
                candidate = group.loc[group.model == candidate_name].iloc[0]
                for baseline_name in ['direct_all93', 'direct_legacy40']:
                    baseline = group.loc[group.model == baseline_name].iloc[0]
                    diff = float(baseline.rmse - candidate.rmse)
                    deltas.append(dict(origin=origin, horizon=bucket, candidate=candidate_name, baseline=baseline_name,
                                       n=int(candidate.n), baseline_rmse=float(baseline.rmse), candidate_rmse=float(candidate.rmse),
                                       improvement_sgd=diff, improvement_percent=100*diff/float(baseline.rmse),
                                       candidate_bias=float(candidate.bias)))
        path = summary / f'{phase}_deltas.csv'
        pd.DataFrame(deltas).to_csv(path, index=False)
        artifacts.append(path)
    pooled = phase_tables['evaluation'].query("origin == 'pooled_instances' and horizon == 'all'")
    lines = ['# 严格固定截点本地回测结果', '',
             '用户授权执行变更：两个开发窗口及前五个正式窗口为CPU 1线程，最后2024-12正式窗口为CPU 4线程。',
             '仅改变线程数；深度、学习率、种子、轮数上限、早停、特征、预测行、标签边界及开发期已冻结趋势均不变。',
             '原始manifest和已完成产物保留；执行变更有独立manifest、代码hash、模型合同及逐窗口来源记录。', '',
             f"开发期选定趋势：{selection['selected_trend']}。", '',
             '| 方法 | 正式预测实例加权RMSE（SGD） | bias（预测-真值） | n |', '|---|---:|---:|---:|']
    for row in pooled.itertuples():
        lines.append(f'| {row.model} | {row.rmse:.4f} | {row.bias:.4f} | {row.n} |')
    lines += ['', '逐origin、逐月、1–3/4–6/7–12/13–15/16月分桶和两套对照的货币/百分比差值见CSV。负改善值表示退步，全部保留。',
              '', '窗口重叠，汇总按预测实例加权，不能视为独立交易或独立重复实验。40列对照有历史特征筛选信息；修正对照不等于旧程序精确复现。',
              '', '含外生变量的方法条件于提供的未来COE/股票协变量。分解方法的静态输入集合也不同；这里只比较完整预测方案。构成调整不能消除未观测混淆。本地结果不证明外部泛化改善。',
              '', '首次训练成本保留在各来源目录observations及attempt中。线程数变化属于执行变更，不作为模型改进。']
    path = summary / 'results_summary.zh.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    artifacts.append(path)
    path = summary / 'origin_sources.json'
    atomic_json(path, evidence)
    artifacts.append(path)
    atomic_json(root / 'complete.json', dict(manifest_sha256=manifest['manifest_sha256'],
                base_manifest_sha256=base['manifest_sha256'], selection_sha256=selection['selection_sha256'],
                completed_utc=utc_now(), completion_attempt=attempt.attempt_id,
                artifact_sha256={str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in artifacts}))


def run(root, pin):
    from src.backtest.selection import validate_origin, verify_selection
    from src.backtest.frozen import DIRECT_MODELS
    manifest, base_root, base = verify_amendment(root, pin)
    with RunLock(root / 'supervisor.lock'):
        attempt = Attempt(root / 'attempts/supervisor', root / 'status.json', 'supervisor',
                          command=sys.argv, manifest_sha256=pin, execution_thread_count=4)
        try:
            # The old supervisor is handed off independently. Never terminate the
            # live worker; it owns a non-inherited OS lock and finishes unchanged.
            attempt.update('running', stage='waiting_for_original_supervisor_handoff')
            while True:
                try:
                    base_lock = RunLock(base_root / 'supervisor.lock')
                    base_lock.__enter__()
                    break
                except LockBusy:
                    if (base_root / 'evaluation' / manifest['execution_amendment']['target_origin']).exists():
                        raise ValueError('original final origin started before handoff; refusing duplicate execution')
                    time.sleep(2)
            try:
                attempt.update('running', stage='waiting_for_original_fifth_window', origin=base['protocol']['evaluation_origins'][-2])
                while True:
                    try:
                        worker_lock = RunLock(base_root / 'worker.lock')
                        worker_lock.__enter__()
                        break
                    except LockBusy:
                        observed = read_json(base_root / 'worker_status.json')
                        if observed.get('origin') != base['protocol']['evaluation_origins'][-2]:
                            raise ValueError('unexpected original worker; refusing duplicate execution')
                        time.sleep(2)
                try:
                    manifest, _, base = verify_amendment(root, pin)
                    selection = verify_selection(base_root, base, manifest['execution_amendment']['selection_sha256'])
                    names = [*DIRECT_MODELS, f"decomposed_{selection['selected_trend']}"]
                    for origin in base['protocol']['evaluation_origins'][:-1]:
                        validate_origin(base_root, base, 'evaluation', origin, names)
                    if not (root / 'evaluation' / manifest['execution_amendment']['target_origin'] / 'origin_complete.json').exists():
                        dispatch = uuid.uuid4().hex
                        command = [sys.executable, '-u', str(EXTENSION), 'worker', '--run-dir', str(root), '--manifest-hash', pin, '--dispatch-id', dispatch]
                        attempt.update('running', stage='dispatch_worker', phase='evaluation', origin=manifest['execution_amendment']['target_origin'])
                        code = run_child(command, ROOT, attempt, 'evaluation_2024-12_cpu4', root / 'worker_status.json', dispatch)
                        if code:
                            return code
                    attempt.update('running', stage='aggregate_verified_results')
                    manifest, _, base = verify_amendment(root, pin)
                    aggregate(root, manifest, base_root, base, selection, attempt)
                    attempt.finish(0, stage='complete')
                finally:
                    worker_lock.__exit__(None,None,None)
            finally:
                base_lock.__exit__(None,None,None)
        except BaseException as error:
            attempt.fail(error)
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','probe','run','worker','status'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--manifest-hash')
    parser.add_argument('--base-run-dir')
    parser.add_argument('--base-manifest-hash')
    parser.add_argument('--dispatch-id')
    args = parser.parse_args()
    root = owned(args.run_dir)
    try:
        if args.action == 'freeze':
            value = freeze(args.base_run_dir,args.base_manifest_hash,root)
            print(json.dumps({'run_dir':str(root),'manifest_sha256':value['manifest_sha256']}),flush=True)
        elif args.action == 'status':
            manifest, _, _ = verify_amendment(root,args.manifest_hash)
            print(json.dumps({'manifest_sha256':manifest['manifest_sha256'],'complete_marker':(root/'complete.json').exists(),
                              'supervisor':read_json(root/'status.json') if (root/'status.json').exists() else None,
                              'worker':read_json(root/'worker_status.json') if (root/'worker_status.json').exists() else None},indent=2),flush=True)
        elif args.action == 'probe':
            verify_amendment(root,args.manifest_hash)
            with RunLock(root/'supervisor.lock'), RunLock(root/'worker.lock'):
                pass
            print('amendment frozen and successor locks free',flush=True)
        elif args.action == 'worker':
            worker(root,args.manifest_hash,args.dispatch_id)
        else:
            return run(root,args.manifest_hash)
        return 0
    except LockBusy as error:
        print(str(error),file=sys.stderr,flush=True)
        return 3
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
