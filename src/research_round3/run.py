"""Two-origin direct-property batch, reusing frozen data/process primitives."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
from src.backtest.data import sha256
from src.backtest.execution import RunLock,LockBusy,Attempt,run_child,read_json,atomic_json,utc_now
from src.backtest.protocol import ROOT,TARGET,digest_json,months
from src.backtest.scoring import attach_truth,metrics
from src.backtest.resources import ResourceMeter
from src.research_round2 import run as previous
from src.research_round2 import models as old_models
from src.research_round3.models import fit_candidate,restore,expected_contract

from configs import config_path
CONFIG = config_path('research_round3/protocol.json')


def own(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'artifacts/backtest') or not path.name.startswith('round3_'):
        raise ValueError('round3 requires a new dedicated output directory')
    return path


def reference(base,config):
    ref=config['baseline_reference']; root=ROOT/ref['root']; pin=ref['manifest_sha256']
    manifest=previous.verify(root,pin); result={}
    for origin in config['origins']:
        history,query=previous.data_at(base,origin,config)
        contract=old_models.expected_contract(history,query,origin,'B0',manifest['config'],pin); contract['smoke_iterations']=None
        directory=root/'models'/origin/'B0'; frame=old_models.restore(directory,contract)
        if frame is None or not np.array_equal(frame.row_id,query.row_id): raise ValueError('baseline reference query mismatch')
        receipt=read_json(directory/'complete.json'); prediction=directory/receipt['attempt']/'predictions.parquet'
        result[origin]={'source_checkpoint':str((directory/'complete.json').relative_to(ROOT)).replace('\\','/'),
                        'source_checkpoint_sha256':sha256(directory/'complete.json'),
                        'source_predictions':str(prediction.relative_to(ROOT)).replace('\\','/'),'source_predictions_sha256':sha256(prediction),
                        'rows':len(frame),'local_copy':f'references/B0_{origin}.parquet'}
    return result


def freeze(root):
    root=own(root)
    if root.exists(): raise ValueError('freeze requires a fresh round3 directory')
    config=read_json(CONFIG)
    if config['models']!=['C1','C2','C3'] or config['origins']!=['2023-09','2024-03'] or config['normal_tree_fit_budget']!=12:
        raise ValueError('unapproved direct-property batch')
    base=previous.load_base(); refs=reference(base,config)
    sources=set(previous.source_paths())|set((ROOT/'src/research_round3').glob('*.py'))|{CONFIG}
    old=ROOT/config['baseline_reference']['root']; inputs=[ROOT/n for n in read_json(old/'manifest.json')['inputs']]
    inputs += [old/'manifest.json',old/'complete.json']
    for ref in refs.values(): inputs += [ROOT/ref['source_checkpoint'],ROOT/ref['source_predictions']]
    value={'created_utc':utc_now(),'config':config,'environment':previous.environment(),
           'sources':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(sources)},
           'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in inputs},'baseline_references':refs,
           'baseline_retrained':False,'row_plan':{},'scope':config['scope']}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,
                       timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    value['hardware_at_freeze']={'nvidia_smi':gpu.stdout.strip(),'query_exit_code':gpu.returncode}
    for origin in config['origins']:
        p=read_json(old/'manifest.json')['row_plan'][origin]
        value['row_plan'][origin]={k:p[k] for k in ('history_rows','query_rows','query_identity_sha256','inner_cutoff','inner_validation_rows')}
    value['manifest_sha256']=digest_json(value); root.mkdir(parents=True); (root/'references').mkdir()
    for ref in refs.values():
        shutil.copyfile(ROOT/ref['source_predictions'],root/ref['local_copy'])
        if sha256(root/ref['local_copy'])!=ref['source_predictions_sha256']: raise ValueError('baseline copy checksum mismatch')
    atomic_json(root/'manifest.json',value)
    return value


def verify(root,pin):
    value=read_json(root/'manifest.json')
    if value['manifest_sha256']!=pin or digest_json({k:v for k,v in value.items() if k!='manifest_sha256'})!=pin:
        raise ValueError('round3 manifest mismatch')
    if value['environment']!=previous.environment() or value['config']!=read_json(CONFIG): raise ValueError('round3 environment/config drift')
    for kind in ('sources','inputs'):
        if any(sha256(ROOT/n)!=v for n,v in value[kind].items()): raise ValueError(f'round3 frozen {kind} changed')
    for ref in value['baseline_references'].values():
        if sha256(root/ref['local_copy'])!=ref['source_predictions_sha256']: raise ValueError('local baseline copy changed')
    return value


def probe(root,pin):
    verify(root,pin)
    with RunLock(root/'supervisor.lock'),RunLock(root/'worker.lock'):
        return {'dispatch_allowed':not (root/'complete.json').exists(),'manifest_sha256':pin}


def worker(root,pin,origin,name,dispatch):
    manifest=verify(root,pin); config=manifest['config']
    if origin not in config['origins'] or name not in config['models']: raise ValueError('unapproved worker')
    with RunLock(root/'worker.lock'):
        attempt=Attempt(root/'attempts/worker',root/'worker_status.json','worker',manifest_sha256=pin,origin=origin,model=name,dispatch_id=dispatch)
        def progress(stage,**extra):
            attempt.update('running',stage=stage,**extra); print(json.dumps({'time':utc_now(),'stage':stage,**extra}),flush=True)
        try:
            with ResourceMeter() as meter:
                history,query=previous.data_at(previous.load_base(),origin,config)
                fit_candidate(history,query,origin,name,config,root/'models'/origin/name,pin,progress)
            atomic_json(attempt.path/'resources.json',meter.result); attempt.finish(0,stage='model_complete',resources=meter.result)
        except BaseException as error: attempt.fail(error); raise


def summarize(root,manifest):
    config=manifest['config']; base=previous.load_base(); parts=[]
    for origin in config['origins']:
        history,query=previous.data_at(base,origin,config)
        baseline=pd.read_parquet(root/manifest['baseline_references'][origin]['local_copy'])
        if not np.array_equal(baseline.row_id,query.row_id): raise ValueError('B0 row identity differs')
        joined=baseline
        for name in config['models']:
            contract=expected_contract(history,query,origin,name,config,manifest['manifest_sha256']); contract['smoke_iterations']=None
            frame=restore(root/'models'/origin/name,contract)
            if frame is None or not np.array_equal(frame.row_id,query.row_id): raise ValueError('incomplete/misaligned new model')
            joined=joined.merge(frame,on=['origin','row_id','RENT_APPROVAL_DATE','horizon'],how='inner',validate='one_to_one')
        if len(joined)!=len(query): raise ValueError('candidate row sets differ')
        parts.append(joined)
    predictions=attach_truth(pd.concat(parts,ignore_index=True)); names=['B0',*config['models']]
    scores=metrics(predictions,names,config); directory=root/'summary'; directory.mkdir(exist_ok=True)
    predictions.to_parquet(directory/'predictions.parquet',index=False); scores.to_csv(directory/'metrics.csv',index=False)
    b0=scores.loc[scores.model=='B0'].set_index(['origin','horizon']).rmse
    delta=scores.loc[scores.model!='B0'].copy(); delta['B0_rmse']=[b0.loc[(o,h)] for o,h in zip(delta.origin,delta.horizon)]
    delta['improvement_rmse']=delta.B0_rmse-delta.rmse; delta['improvement_pct']=100*delta.improvement_rmse/delta.B0_rmse
    delta.to_csv(directory/'deltas.csv',index=False)
    test=pd.read_csv(ROOT/'data/raw/query.csv',usecols=['RENT_APPROVAL_DATE']); hs=months(test).asi8-pd.Period(config['last_label_month'],'M').ordinal
    weights=pd.Series(hs).value_counts().sort_index(); diag=[]
    for name in names:
        mse=scores.loc[(scores.origin=='pooled_instances')&scores.horizon.str.startswith('month_')&(scores.model==name)].copy()
        mse['h']=mse.horizon.str.removeprefix('month_').astype(int); mse=mse.set_index('h'); positive=weights.index[weights.index>=1]
        value=float(np.sqrt(np.average(mse.loc[positive,'sse']/mse.loc[positive,'n'],weights=weights.loc[positive])))
        diag.append({'model':name,'query_horizon_reweighted_rmse_diagnostic':value,'excluded_h0':int(weights.get(0,0))})
    atomic_json(directory/'test_horizon_diagnostic.json',{'scores':diag,'exact_horizon_counts':{str(k):int(v) for k,v in weights.items()}})
    atomic_json(directory/'baseline_reference.json',{'manifest':config['baseline_reference'],'checkpoints':manifest['baseline_references'],'baseline_retrained':False})
    return directory


def run(root,pin):
    manifest=verify(root,pin)
    with RunLock(root/'supervisor.lock'):
        with RunLock(root/'worker.lock'): pass
        if (root/'complete.json').exists(): return 0
        attempt=Attempt(root/'attempts/supervisor',root/'status.json','supervisor',manifest_sha256=pin)
        try:
            for origin in manifest['config']['origins']:
                for name in manifest['config']['models']:
                    verify(root,pin); dispatch=uuid.uuid4().hex
                    command=[sys.executable,'-u','-m','src.research_round3.run','worker','--run-dir',str(root),'--manifest-hash',pin,'--origin',origin,'--model',name,'--dispatch-id',dispatch]
                    attempt.update('running',stage='candidate',origin=origin,model=name)
                    code=run_child(command,ROOT,attempt,f'{origin}_{name}',root/'worker_status.json',dispatch)
                    if code: return code
            verify(root,pin); summary=summarize(root,manifest)
            atomic_json(root/'complete.json',{'manifest_sha256':pin,'models':manifest['config']['models'],'completed_origins':manifest['config']['origins'],
                        'baseline_retrained':False,'additional_origins_started':False,'summary':{str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in summary.iterdir() if p.is_file()}})
            attempt.finish(0,stage='batch_complete')
        except BaseException as error: attempt.fail(error); raise
    return 0


def smoke(root):
    root=own(root)
    if root.exists(): raise ValueError('smoke requires new output')
    root.mkdir(parents=True); config=read_json(CONFIG)
    history,query=previous.data_at(previous.load_base(),'2024-03',config)
    def progress(stage,**extra):
        atomic_json(root/'status.json',{'updated_utc':utc_now(),'pid':os.getpid(),'stage':stage,**extra}); print(json.dumps({'stage':stage,**extra}),flush=True)
    with ResourceMeter() as meter:
        for name in config['models']:
            fit_candidate(history,query,'2024-03',name,config,root/'models'/name,'round3_full_input_smoke16',progress,smoke_iterations=16)
    atomic_json(root/'resources.json',meter.result)
    atomic_json(root/'complete.json',{'scope':'three actual-input 16-tree GPU correctness/resource smokes; no truth scoring or selection','models':config['models'],'baseline_retrained':False})


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','probe','run','worker','smoke','status']); p.add_argument('--run-dir',type=Path,required=True)
    for key in ('manifest-hash','origin','model','dispatch-id'): p.add_argument('--'+key)
    args=p.parse_args(); root=own(args.run_dir)
    if args.action=='freeze': print(json.dumps(freeze(root)))
    elif args.action=='probe': print(json.dumps(probe(root,args.manifest_hash)))
    elif args.action=='run': return run(root,args.manifest_hash)
    elif args.action=='worker': worker(root,args.manifest_hash,args.origin,args.model,args.dispatch_id)
    elif args.action=='smoke': smoke(root)
    else:
        verify(root,args.manifest_hash); print(json.dumps({k:read_json(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('status','worker_status','complete')}))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except LockBusy as error: print(str(error),file=sys.stderr); raise SystemExit(3)
