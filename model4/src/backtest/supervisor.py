"""Frozen sequential full experiments, intended for an independent background process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
import uuid

from src.backtest.execution import Attempt, LockBusy, RunLock, atomic_json, ensure_no_worker, read_json, replace_file, run_child, utc_now
from src.backtest.protocol import ROOT


def owned_run(path: str) -> Path:
    root = Path(path).resolve()
    if not root.is_relative_to((ROOT / "artifacts/backtest").resolve()):
        raise ValueError("run directory must be within the approved artifacts/backtest workspace")
    return root


def aggregate_run(root, manifest, selection, attempt):
    import pandas as pd
    from src.backtest.data import sha256
    from src.backtest.frozen import DIRECT_MODELS, DEVELOPMENT_MODELS
    from src.backtest.scoring import metrics
    from src.backtest.selection import validate_origin
    import os

    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    artifacts, phase_metrics = [], {}
    for phase in ["development", "evaluation"]:
        names = list(DEVELOPMENT_MODELS) if phase == "development" else [*DIRECT_MODELS, f"decomposed_{selection['selected_trend']}"]
        frames = [validate_origin(root, manifest, phase, origin, names)[0] for origin in manifest["protocol"][f"{phase}_origins"]]
        combined = pd.concat(frames, ignore_index=True)
        table = metrics(combined, names, manifest["protocol"])
        phase_metrics[phase] = table
        paths = [summary / f"{phase}_predictions.parquet", summary / f"{phase}_predictions.csv.gz", summary / f"{phase}_metrics.csv"]
        temporary = paths[0].with_suffix(".parquet.partial")
        combined.to_parquet(temporary, index=False)
        replace_file(temporary, paths[0])
        temporary = paths[1].with_suffix(".gz.partial")
        combined.to_csv(temporary, index=False, compression="gzip")
        replace_file(temporary, paths[1])
        temporary = paths[2].with_suffix(".csv.partial")
        table.to_csv(temporary, index=False)
        replace_file(temporary, paths[2])
        deltas = []
        for (origin, bucket), group in table.groupby(["origin", "horizon"], sort=False):
            for name in [n for n in names if n.startswith("decomposed_")]:
                candidate = group.loc[group.model == name].iloc[0]
                for baseline in ["direct_all93", "direct_legacy40"]:
                    base = group.loc[group.model == baseline].iloc[0]
                    difference = float(base.rmse - candidate.rmse)
                    deltas.append({"origin": origin, "horizon": bucket, "candidate": name, "baseline": baseline,
                                   "n": int(candidate.n), "baseline_rmse": float(base.rmse), "candidate_rmse": float(candidate.rmse),
                                   "improvement_sgd": difference, "improvement_percent": 100 * difference / float(base.rmse),
                                   "candidate_bias": float(candidate.bias)})
        delta_path = summary / f"{phase}_deltas.csv"
        pd.DataFrame(deltas).to_csv(delta_path, index=False)
        artifacts += [*paths, delta_path]
    evaluation = phase_metrics["evaluation"]
    pooled = evaluation.loc[(evaluation.origin == "pooled_instances") & (evaluation.horizon == "all")]
    text = ["# 严格固定截点本地回测结果", "", f"开发期选定趋势：{selection['selected_trend']}。开发与正式指标分别保存。", "",
            "| 方法 | 正式预测实例加权RMSE（SGD） | bias（预测-真值） | n |", "|---|---:|---:|---:|"]
    for row in pooled.itertuples():
        text.append(f"| {row.model} | {row.rmse:.4f} | {row.bias:.4f} | {row.n} |")
    text += ["", "按origin、逐月和1–3/4–6/7–12/13–15/16月分桶的完整RMSE、bias、n及相对两套CatBoost的货币/百分比差值见同目录CSV。负改善值表示退步，全部保留。",
             "", "这些是重叠预测实例汇总，不能视为独立交易或独立重复实验。93列为本次预声明对照；40列有历史选择信息。两套对照经过OOF和固定截点修正，与旧程序并非精确逐字复现。",
             "", "包含外生变量的方法条件于提供的未来COE/股票协变量；分解方案使用不同静态属性集合，差异属于完整预测方案比较。月份构成调整不能消除未观测混淆。这里只报告本地回测，不证明外部泛化改善。",
             "", "首次训练资源记录保留在各模型observations和各attempt目录；恢复耗时不替代训练耗时。"]
    report = summary / "results_summary.zh.md"
    report.write_text("\n".join(text) + "\n", encoding="utf-8")
    artifacts.append(report)
    receipt = {"manifest_sha256": manifest["manifest_sha256"], "selection_sha256": selection["selection_sha256"],
               "completed_utc": utc_now(), "completion_attempt": attempt.attempt_id,
               "artifact_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in artifacts}}
    atomic_json(root / "complete.json", receipt)


def run_supervisor(args):
    from src.backtest.data import sha256
    from src.backtest.frozen import read_manifest, verify_runtime
    from src.backtest.selection import select_and_freeze, verify_selection

    root = owned_run(args.run_dir)
    if not (root / "frozen_protocol.json").is_file():
        raise ValueError("run directory is not an owned frozen experiment")
    with RunLock(root / "supervisor.lock"):
        # No attempt/status writes before both lifetime locks have been checked.
        ensure_no_worker(root)
        attempt_parent = root / "attempts/supervisor"
        previous = list(attempt_parent.glob("*")) if attempt_parent.exists() else []
        if previous and not args.resume:
            raise ValueError("previous attempts exist; use explicit --resume")
        attempt = Attempt(attempt_parent, root / "status.json", "supervisor", command=sys.argv,
                          resumed=bool(args.resume), previous_attempts=[p.name for p in previous], manifest_sha256=args.manifest_hash)
        try:
            manifest = read_manifest(root, args.manifest_hash)
            verify_runtime(manifest)
            if (root / "complete.json").exists():
                receipt = json.loads((root / "complete.json").read_text(encoding="utf-8"))
                selection = select_and_freeze(root, manifest)
                if receipt["manifest_sha256"] != manifest["manifest_sha256"] or receipt["selection_sha256"] != selection["selection_sha256"]:
                    raise ValueError("completed run identity changed")
                for file, expected in receipt["artifact_sha256"].items():
                    if sha256(root / file) != expected:
                        raise ValueError("completed summary checksum mismatch")
                # Completed evaluation receipts/history are verified before reuse too.
                from src.backtest.frozen import DIRECT_MODELS
                from src.backtest.selection import validate_origin
                for origin in manifest["protocol"]["evaluation_origins"]:
                    validate_origin(root, manifest, "evaluation", origin, [*DIRECT_MODELS, f"decomposed_{selection['selected_trend']}"])
                attempt.finish(0, stage="restored_complete_run")
                return 0
            for phase in ["development", "evaluation"]:
                if phase == "evaluation":
                    attempt.update("running", stage="freeze_development_selection")
                    selection = select_and_freeze(root, manifest)
                    verify_selection(root, manifest, selection["selection_sha256"])
                for origin in manifest["protocol"][f"{phase}_origins"]:
                    verify_runtime(manifest)
                    ensure_no_worker(root)
                    command = [sys.executable, "-u", "-m", "src.backtest.worker", "--run-dir", str(root),
                               "--manifest-hash", manifest["manifest_sha256"], "--phase", phase, "--origin", origin]
                    dispatch_id = uuid.uuid4().hex
                    command += ["--dispatch-id", dispatch_id]
                    if phase == "evaluation":
                        command += ["--selection-hash", selection["selection_sha256"]]
                    attempt.update("running", phase=phase, origin=origin, stage="dispatch_worker")
                    code = run_child(command, ROOT, attempt, f"{phase}_{origin}", root / "worker_status.json", dispatch_id)
                    if code != 0:
                        return code
            verify_runtime(manifest)
            verify_selection(root, manifest, selection["selection_sha256"])
            attempt.update("running", stage="aggregate_verified_results")
            aggregate_run(root, manifest, selection, attempt)
            attempt.finish(0, stage="complete")
            return 0
        except BaseException as error:
            attempt.fail(error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ["freeze", "probe", "status", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True)
        if name != "freeze":
            command.add_argument("--manifest-hash", required=True)
        if name == "run":
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        root = owned_run(args.run_dir)
        if args.action == "freeze":
            from src.backtest.frozen import freeze_run
            manifest = freeze_run(root)
            print(json.dumps({"run_dir": str(root), "manifest_sha256": manifest["manifest_sha256"]}), flush=True)
            return 0
        if args.action == "probe":
            from src.backtest.frozen import read_manifest, verify_runtime
            with RunLock(root / "supervisor.lock"):
                ensure_no_worker(root)
                verify_runtime(read_manifest(root, args.manifest_hash))
            print("ready; supervisor and worker locks are free", flush=True)
            return 0
        if args.action == "status":
            from src.backtest.frozen import read_manifest
            manifest = read_manifest(root, args.manifest_hash)
            value = {"manifest_sha256": manifest["manifest_sha256"], "complete_marker": (root / "complete.json").exists(),
                     "supervisor": read_json(root / "status.json") if (root / "status.json").exists() else {"state": "not_started"},
                     "worker": read_json(root / "worker_status.json") if (root / "worker_status.json").exists() else None}
            print(json.dumps(value, indent=2), flush=True)
            return 0
        return run_supervisor(args)
    except LockBusy as error:
        print(str(error), file=sys.stderr, flush=True)
        return 3
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
