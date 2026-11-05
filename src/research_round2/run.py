"""Small serial runner for the approved two-origin, three-candidate research batch."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import pandas as pd
import numpy as np
from src.backtest.data import sha256, frame_hash, _read_selected_labels
from src.backtest.protocol import ROOT, TARGET, digest_json, months
from src.backtest.execution import atomic_json, read_json, RunLock, LockBusy, Attempt, run_child, utc_now
from src.backtest.scoring import attach_truth, metrics
from src.backtest.resources import ResourceMeter
from src.research_round2.models import fit_candidate, expected_contract, restore
from src.research_round2.snapshots import SnapshotStore, cached_training

from configs import config_path
CONFIG = config_path('research_round2/protocol.json')
PACKAGES=('numpy','pandas','pyarrow','scipy','scikit-learn','xgboost','catboost','joblib','psutil','threadpoolctl')


def own(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'artifacts/backtest') or not path.name.startswith('round2_'):
        raise ValueError('runner requires a dedicated artifacts/backtest/round2_ directory')
    return path


def source_paths():
    paths=list((ROOT/'src/research_round2').glob('*.py'))+list((ROOT/'src/backtest').glob('*.py'))+list((ROOT/'src/features').glob('*.py'))
    paths += [ROOT/'src/pipeline/preprocess.py',ROOT/'src/pipeline/rules.py',ROOT/'src/paths.py',ROOT/'configs/__init__.py',CONFIG,
              ROOT/'pyproject.toml']
    return sorted(paths)


def environment():
    return {'python':platform.python_version(),'interpreter':sys.executable,
            'packages':{p:importlib.metadata.version(p) for p in PACKAGES}}


def row_plan(base,config):
    periods=months(base)
    start=pd.Period(config['snapshot_start'],freq='M').ordinal
    output={}
    for origin in config['origins']:
        cutoff=pd.Period(origin,freq='M')
        hs=periods.asi8-cutoff.ordinal
        q=base.loc[(hs>=1)&(hs<=config['max_horizon'])]
        inner=cutoff-config['inner_validation_months']
        example_counts={}
        for label,end in [('inner',inner),('outer',cutoff)]:
            counts={str(h):int(np.count_nonzero((periods<=end)&(periods.asi8-h>=start))) for h in config['snapshot_horizons']}
            if any(n==0 for n in counts.values()):
                raise ValueError('first research batch lacks a predeclared training horizon')
            example_counts[label]={'fit_cutoff':str(end),'by_h':counts,'instances':sum(counts.values()),'unique_target_rows':counts['1']}
        output[origin]={'history_rows':int(np.count_nonzero(periods<=cutoff)), 'query_rows':len(q),
                        'query_identity_sha256':frame_hash(q[['row_id','RENT_APPROVAL_DATE']]),
                        'inner_cutoff':str(inner),'inner_validation_rows':int(np.count_nonzero((periods>inner)&(periods<=cutoff))),
                        'snapshot_training_examples':example_counts}
    return output


def freeze(root):
    root=own(root)
    if root.exists():
        raise ValueError('freeze needs a fresh run directory')
    config=read_json(CONFIG)
    if config['models']!=['B0','B1','B2'] or config['origins']!=['2023-09','2024-03'] or config['cpu_threads']!=4:
        raise ValueError('outside the approved first research batch')
    inputs=[ROOT/'data/raw/train.csv',ROOT/'data/raw/query.csv']
    inputs += [ROOT/'artifacts/backtest/cache'/f'{split}_{suffix}' for split in ('train','test') for suffix in ('base.parquet','manifest.json')]
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,
                       timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    import xgboost as xgb
    value={'created_utc':utc_now(),'config':config,'environment':environment(),
           'hardware_at_freeze':{'nvidia_smi':gpu.stdout.strip(),'query_exit_code':gpu.returncode,
                                 'xgboost_build':xgb.build_info()},
           'sources':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in source_paths()},
           'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in inputs},
           'scope':'two prespecified exploratory origins; no candidate selection or additional four origins authorized',
           'devices':{'training_priority':'cuda:0','CPU_fallback_threads':4,'fallback_requires_separate_model_attempt':True}}
    value['row_plan']=row_plan(load_base(),config)
    value['manifest_sha256']=digest_json(value)
    root.mkdir(parents=True)
    atomic_json(root/'manifest.json',value)
    return value


def verify(root,pin):
    value=read_json(root/'manifest.json')
    claimed=value['manifest_sha256']
    if claimed!=pin or digest_json({k:v for k,v in value.items() if k!='manifest_sha256'})!=pin:
        raise ValueError('manifest pin mismatch')
    if value['environment']!=environment():
        raise ValueError('interpreter/dependency environment changed')
    for kind in ('sources','inputs'):
        if any(sha256(ROOT/name)!=digest for name,digest in value[kind].items()):
            raise ValueError(f'frozen {kind} changed')
    if value['config']!=read_json(CONFIG):
        raise ValueError('resolved config drift')
    return value


def load_base():
    # Reuse only immutable target-free first-round predictor cache, checking its
    # byte provenance under the original manifest rather than pandas-version hashes.
    directory=ROOT/'artifacts/backtest/cache'
    info=read_json(directory/'train_manifest.json')
    if sha256(directory/'train_base.parquet')!=info['parquet_sha256'] or sha256(ROOT/'data/raw/train.csv')!=info['raw_file_sha256_at_build_for_audit_only']:
        raise ValueError('target-free cache/raw source byte provenance changed')
    for name,pin in info['source_sha256'].items():
        if '#' not in name and sha256(ROOT/name)!=pin:
            raise ValueError('cache feature source changed')
    base=pd.read_parquet(directory/'train_base.parquet')
    if TARGET in base or any(c.startswith('mean_rent_') for c in base) or not base.row_id.is_unique:
        raise ValueError('unsafe predictor cache')
    return base


def data_at(base,origin,config):
    history=base.loc[months(base)<=pd.Period(origin,freq='M')].copy()
    history[TARGET]=_read_selected_labels(ROOT/'data/raw/train.csv',history.row_id.to_numpy())
    hs=months(base).asi8-pd.Period(origin,freq='M').ordinal
    query=base.loc[(hs>=1)&(hs<=config['max_horizon'])].copy()
    return history,query


def probe(root,pin):
    verify(root,pin)
    with RunLock(root/'supervisor.lock'),RunLock(root/'worker.lock'):
        return {'dispatch_allowed':not (root/'complete.json').exists(),'manifest_sha256':pin}


def worker(root,pin,origin,name,dispatch):
    manifest=verify(root,pin)
    config=manifest['config']
    if origin not in config['origins'] or name not in config['models']:
        raise ValueError('unapproved model/origin')
    with RunLock(root/'worker.lock'):
        attempt=Attempt(root/'attempts/worker',root/'worker_status.json','worker',origin=origin,model=name,dispatch_id=dispatch,manifest_sha256=pin)
        def progress(stage,**extra):
            attempt.update('running',stage=stage,**extra)
            print(json.dumps({'time':utc_now(),'origin':origin,'model':name,'stage':stage,**extra}),flush=True)
        try:
            with ResourceMeter() as meter:
                base=load_base()
                history,query=data_at(base,origin,config)
                fit_candidate(history,query,origin,name,config,root/'models'/origin/name,root/'snapshots',root/'features',pin,progress)
            atomic_json(attempt.path/'resources.json',meter.result)
            attempt.finish(0,stage='model_complete',resources=meter.result)
        except BaseException as error:
            attempt.fail(error)
            raise


def summarize(root,manifest):
    config=manifest['config']
    base=load_base()
    all_frames=[]
    for origin in config['origins']:
        history,query=data_at(base,origin,config)
        joined=None
        for name in config['models']:
            contract=expected_contract(history,query,origin,name,config,manifest['manifest_sha256'])
            contract['smoke_iterations']=None
            frame=restore(root/'models'/origin/name,contract)
            if frame is None or not np.array_equal(frame.row_id.to_numpy(),query.row_id.to_numpy()):
                raise ValueError('incomplete/misaligned candidate at summary gate')
            if joined is None:
                joined=frame
            else:
                joined=joined.merge(frame,on=['origin','row_id','RENT_APPROVAL_DATE','horizon'],validate='one_to_one',how='inner')
        if len(joined)!=len(query):
            raise ValueError('candidate row sets differ')
        all_frames.append(joined)
    predictions=attach_truth(pd.concat(all_frames,ignore_index=True))
    scores=metrics(predictions,config['models'],config)
    directory=root/'summary'
    directory.mkdir(exist_ok=True)
    predictions.to_parquet(directory/'predictions.parquet',index=False)
    scores.to_csv(directory/'metrics.csv',index=False)
    b0=scores.loc[scores.model=='B0'].set_index(['origin','horizon']).rmse
    delta=scores.loc[scores.model!='B0'].copy()
    delta['B0_rmse']=[b0.loc[(o,h)] for o,h in zip(delta.origin,delta.horizon)]
    delta['improvement_rmse']=delta.B0_rmse-delta.rmse
    delta['improvement_pct']=100*delta.improvement_rmse/delta.B0_rmse
    delta.to_csv(directory/'deltas.csv',index=False)
    test=pd.read_csv(ROOT/'data/raw/query.csv',usecols=['RENT_APPROVAL_DATE'])
    htest=months(test).asi8-pd.Period(config['last_label_month'],freq='M').ordinal
    weights=pd.Series(htest).value_counts().sort_index()
    diag=[]
    for name in config['models']:
        mse=scores.loc[(scores.origin=='pooled_instances')&scores.horizon.str.startswith('month_')&(scores.model==name)].copy()
        mse['h']=mse.horizon.str.removeprefix('month_').astype(int)
        mse=mse.set_index('h')
        hs=weights.index[weights.index>=1]
        if not set(hs).issubset(mse.index):
            raise ValueError('reweighting missing horizon support')
        score=float(np.sqrt(np.average(mse.loc[hs,'sse']/mse.loc[hs,'n'],weights=weights.loc[hs])))
        diag.append({'model':name,'query_horizon_reweighted_rmse_diagnostic':score,'excluded_h0':int(weights.get(0,0))})
    atomic_json(directory/'test_horizon_diagnostic.json',{'scores':diag,'exact_horizon_counts':{str(k):int(v) for k,v in weights.items()},'scope':config['scope']})
    return directory


def run(root,pin):
    manifest=verify(root,pin)
    with RunLock(root/'supervisor.lock'):
        with RunLock(root/'worker.lock'): pass
        if (root/'complete.json').exists():
            return
        attempt=Attempt(root/'attempts/supervisor',root/'status.json','supervisor',manifest_sha256=pin)
        try:
            for origin in manifest['config']['origins']:
                for name in manifest['config']['models']:
                    verify(root,pin)
                    dispatch=uuid.uuid4().hex
                    command=[sys.executable,'-u','-m','src.research_round2.run','worker','--run-dir',str(root),'--manifest-hash',pin,'--origin',origin,'--model',name,'--dispatch-id',dispatch]
                    attempt.update('running',stage='candidate',origin=origin,model=name)
                    code=run_child(command,ROOT,attempt,f'{origin}_{name}',root/'worker_status.json',dispatch)
                    if code: return code
            verify(root,pin)
            directory=summarize(root,manifest)
            atomic_json(root/'complete.json',{'manifest_sha256':pin,'scope':manifest['config']['scope'],'completed_origins':manifest['config']['origins'],
                        'models':manifest['config']['models'],'additional_origins_started':False,
                        'summary':{str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in directory.iterdir() if p.is_file()}})
            attempt.finish(0,stage='first_batch_complete')
        except BaseException as error:
            attempt.fail(error)
            raise
    return 0


def smoke(root):
    """One full-size feature build and two 16-tree fits, no scoring/selection."""
    root=own(root)
    if root.exists(): raise ValueError('smoke requires a fresh output directory')
    root.mkdir(parents=True)
    config=read_json(CONFIG)
    base=load_base()
    history,query=data_at(base,'2024-03',config)
    namespace='phaseB_full_size_smoke_16_trees_'+digest_json(config)
    def progress(stage,**extra):
        atomic_json(root/'status.json',{'stage':stage,'updated_utc':utc_now(),'pid':os.getpid(),**extra})
        print(json.dumps({'stage':stage,**extra}),flush=True)
    with ResourceMeter() as meter:
        started=time.perf_counter()
        store=SnapshotStore(history,config,root/'snapshots',namespace,progress)
        directory,audit=cached_training(history,'2024-03',config,store,root/'features',namespace)
        build_s=time.perf_counter()-started
        if audit['supported_horizons']!=config['snapshot_horizons']:
            raise ValueError('first research origin lacks far-horizon training support')
        atomic_json(root/'full_feature_build.json',{'elapsed_s':build_s,'audit':audit,'cache':str(directory)})
        del store
        gc.collect()
        for name in ('B1','B2'):
            fit_candidate(history,query,'2024-03',name,config,root/'models'/name,root/'snapshots',root/'features',namespace,progress,smoke_iterations=16)
    atomic_json(root/'resources.json',meter.result)
    atomic_json(root/'complete.json',{'scope':'full-size feature construction plus B1/B2 16-tree correctness/resource smoke; no truth scoring or candidate selection',
                                    'feature_build':read_json(root/'full_feature_build.json'),'resources':meter.result})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','probe','run','worker','status','smoke'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--manifest-hash')
    parser.add_argument('--origin')
    parser.add_argument('--model')
    parser.add_argument('--dispatch-id')
    args=parser.parse_args()
    root=own(args.run_dir)
    if args.action=='freeze': print(json.dumps(freeze(root)))
    elif args.action=='probe': print(json.dumps(probe(root,args.manifest_hash)))
    elif args.action=='run': return run(root,args.manifest_hash)
    elif args.action=='worker': worker(root,args.manifest_hash,args.origin,args.model,args.dispatch_id)
    elif args.action=='smoke': smoke(root)
    else:
        verify(root,args.manifest_hash)
        print(json.dumps({k:read_json(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('status','worker_status','complete')}))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except LockBusy as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(3)
