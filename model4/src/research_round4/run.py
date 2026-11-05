"""Four new B0/C3 backtest origins and verified historical references."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from src.backtest.protocol import ROOT,digest_json
from src.backtest.data import sha256,frame_hash
from src.backtest.execution import read_json,atomic_json,RunLock,LockBusy,Attempt,run_child,utc_now
from src.backtest.resources import ResourceMeter
from src.research_round2 import run as old
from src.research_round2 import models as b0_models
from src.research_round3 import models as c3_models
from src.research_round4.common import CONFIG,recipe
from src.research_round4.summary import collect_references,copy_references,summarize


def own(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'artifacts/backtest') or not path.name.startswith('round4_'):
        raise ValueError('requires independent artifacts/backtest/round4_ directory')
    return path


def job_plan(config):
    if config['new_origins'] != ['2023-12','2024-06','2024-09','2024-12'] or config['reused_origins'] != ['2023-09','2024-03'] or config['models'] != ['B0','C3']:
        raise ValueError('outside the frozen four-window backtest extension')
    if config['normal_new_tree_fit_budget'] != 16 or config['backtest_normal_fit_budget'] != 16:
        raise ValueError('outside the frozen backtest-only 16-fit budget')
    return [{'task':'backtest','origin':o,'model':m} for o in config['new_origins'] for m in config['models']]


def freeze(root):
    root = own(root)
    if root.exists(): raise ValueError('freeze requires a fresh round4 output directory')
    config = read_json(CONFIG); jobs = job_plan(config); base = old.load_base()
    if config['origins'] != ['2023-09','2023-12','2024-03','2024-06','2024-09','2024-12']:
        raise ValueError('six-origin aggregate plan changed')
    refs = collect_references(base, config)
    sources = set(old.source_paths()) | set((ROOT/'src/research_round3').glob('*.py')) | set((ROOT/'src/research_round4').glob('*.py'))
    sources |= {ROOT/'configs/research_round3/protocol.json', CONFIG}
    first = ROOT/config['reuse_sources']['B0']['root']
    inputs = {ROOT/p for p in read_json(first/'manifest.json')['inputs']}
    for origin, models in refs.items():
        for ref in models.values():
            inputs |= {ROOT/ref['source_manifest'], ROOT/ref['source_root_complete'], ROOT/ref['source_checkpoint'], ROOT/ref['source_predictions']}
    plan = {}
    for origin in config['origins']:
        history, query = old.data_at(base, origin, recipe(config,'C3'))
        from src.backtest.protocol import inner_split
        train, valid, _, cutoff = inner_split(history, origin, recipe(config,'C3'))
        plan[origin] = {'history_rows':len(history), 'query_rows':len(query), 'query_identity_sha256':frame_hash(query[['row_id','RENT_APPROVAL_DATE']]),
                        'inner_cutoff':cutoff, 'inner_train_rows':len(train), 'inner_validation_rows':len(valid), 'source':'new' if origin in config['new_origins'] else 'reused'}
    value = {'created_utc':utc_now(), 'config':config, 'jobs':jobs, 'environment':old.environment(),
             'sources':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(sources)},
             'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(inputs)},
             'references':refs, 'row_plan':plan, 'scope':config['scope']}
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    value['hardware_at_freeze'] = {'nvidia_smi':gpu.stdout.strip(), 'exit_code':gpu.returncode}
    value['manifest_sha256'] = digest_json(value)
    root.mkdir(parents=True); copy_references(root, refs); atomic_json(root/'manifest.json', value)
    return value


def verify(root,pin):
    value=read_json(root/'manifest.json')
    if value['manifest_sha256']!=pin or digest_json({k:v for k,v in value.items() if k!='manifest_sha256'})!=pin: raise ValueError('round4 manifest mismatch')
    if value['environment']!=old.environment() or value['config']!=read_json(CONFIG): raise ValueError('round4 environment/config drift')
    if value['jobs']!=job_plan(value['config']): raise ValueError('frozen dispatch plan changed')
    for kind in ('sources','inputs'):
        if any(sha256(ROOT/p)!=v for p,v in value[kind].items()): raise ValueError(f'round4 frozen {kind} changed')
    for models in value['references'].values():
        for ref in models.values():
            if sha256(root/ref['local_copy'])!=ref['source_predictions_sha256']: raise ValueError('reused prediction copy changed')
    return value


def probe(root,pin):
    verify(root,pin)
    with RunLock(root/'supervisor.lock'),RunLock(root/'worker.lock'):
        return {'dispatch_allowed':not (root/'complete.json').exists(),'manifest_sha256':pin}


def worker(root,pin,task,origin,name,dispatch):
    manifest = verify(root,pin); config = manifest['config']; job = {'task':task,'origin':origin,'model':name}
    if job not in manifest['jobs']: raise ValueError('unapproved backtest model/origin/task')
    with RunLock(root/'worker.lock'):
        attempt = Attempt(root/'attempts/worker', root/'worker_status.json', 'worker', manifest_sha256=pin, task=task, origin=origin, model=name, dispatch_id=dispatch)
        def progress(stage,**extra):
            attempt.update('running',stage=stage,**extra); print(json.dumps({'time':utc_now(),'stage':stage,**extra}),flush=True)
        try:
            with ResourceMeter() as meter:
                rec = recipe(config,name); history,query = old.data_at(old.load_base(),origin,rec)
                if name == 'B0': b0_models.fit_candidate(history,query,origin,name,rec,root/'models'/origin/name,root/'unused_snapshots',root/'unused_features',pin,progress)
                else: c3_models.fit_candidate(history,query,origin,name,rec,root/'models'/origin/name,pin,progress)
            atomic_json(attempt.path/'resources.json',meter.result)
            attempt.finish(0,stage='model_complete',resources=meter.result)
        except BaseException as error: attempt.fail(error); raise


def run(root,pin):
    manifest = verify(root,pin)
    with RunLock(root/'supervisor.lock'):
        with RunLock(root/'worker.lock'): pass
        if (root/'complete.json').exists(): return 0
        attempt = Attempt(root/'attempts/supervisor',root/'status.json','supervisor',manifest_sha256=pin)
        try:
            for job in manifest['jobs']:
                verify(root,pin); dispatch = uuid.uuid4().hex
                command = [sys.executable,'-u','-m','src.research_round4.run','worker','--run-dir',str(root),'--manifest-hash',pin,
                           '--task',job['task'],'--origin',job['origin'],'--model',job['model'],'--dispatch-id',dispatch]
                attempt.update('running',stage=job['task'],origin=job['origin'],model=job['model'])
                code = run_child(command,ROOT,attempt,f"{job['task']}_{job['origin']}_{job['model']}",root/'worker_status.json',dispatch)
                if code: return code
            verify(root,pin); summary = summarize(root,manifest)
            atomic_json(root/'complete.json', {'manifest_sha256':pin,'new_origins':manifest['config']['new_origins'],
                        'reused_origins':manifest['config']['reused_origins'],'completed_origins':manifest['config']['origins'],
                        'models':['B0','C3'],'normal_new_fit_budget':16,'reused_models_retrained':False,
                        'summary':{str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in summary.iterdir() if p.is_file()}})
            attempt.finish(0,stage='six_window_backtest_complete')
        except BaseException as error: attempt.fail(error); raise
    return 0


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','probe','run','worker','status']); p.add_argument('--run-dir',type=Path,required=True)
    for key in ('manifest-hash','task','origin','model','dispatch-id'): p.add_argument('--'+key)
    args=p.parse_args(); root=own(args.run_dir)
    if args.action=='freeze': print(json.dumps(freeze(root)))
    elif args.action=='probe': print(json.dumps(probe(root,args.manifest_hash)))
    elif args.action=='run': return run(root,args.manifest_hash)
    elif args.action=='worker': worker(root,args.manifest_hash,args.task,args.origin,args.model,args.dispatch_id)
    else:
        verify(root,args.manifest_hash); print(json.dumps({k:read_json(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('status','worker_status','complete')}))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except LockBusy as error: print(str(error),file=sys.stderr); raise SystemExit(3)
