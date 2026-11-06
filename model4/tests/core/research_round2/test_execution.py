"""Execution risks: residual stopping, honest fallback, durable reload and bounds."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.execution import atomic_json, read_json, RunLock, LockBusy
from src.backtest.data import sha256
from src.research_round2 import models
from src.research_round2 import run as runner


ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture
def config():
    value=json.loads((ROOT/'configs/research_round2/protocol.json').read_text())
    # Explicit fixture budget; never used by the formal CLI/config.
    value['xgboost'].update(max_depth=2,n_estimators=8,early_stopping_rounds=2)
    return value


@pytest.fixture
def source():
    rows=[]
    for month in pd.period_range('2022-01','2023-09',freq='M'):
        for building in range(3):
            rows.append({'row_id':len(rows),'RENT_APPROVAL_DATE':str(month),'TOWN':'town',
                         'BLOCK':str(building),'STREET':'street','FLAT_TYPE':'3-room',
                         'FLAT_MODEL':'improved','FLOOR_AREA_SQM':65+building,'FURNISHED':'yes',
                         'LEASE_COMMENCE_DATE':1990+building,'FEE':0,
                         'MONTHLY_RENT':1500+20*(month.ordinal-pd.Period('2022-01','M').ordinal)+30*building})
    history=pd.DataFrame(rows)
    query=history.iloc[:2].drop(columns='MONTHLY_RENT').copy()
    query['row_id']=[1000,1001]
    query['RENT_APPROVAL_DATE']=['2023-10','2025-01']
    return history,query


def test_category_dictionary_is_training_only_and_unknown_is_missing(config):
    train=pd.DataFrame({'TOWN':['a','b'],'FLOOR_AREA_SQM':[60.0,70.0]})
    query=pd.DataFrame({'TOWN':['not_in_training'],'FLOOR_AREA_SQM':[80.0]})
    _,q,prep,unknown=models.encode(train,query,config)
    assert prep.levels_['TOWN']==['a','b']
    assert q.TOWN.isna().all() and unknown=={'TOWN':1}


def test_backend_verifier_rejects_silent_cpu_and_single_thread():
    def fake(device,threads,updater):
        config={'learner':{'generic_param':{'device':device,'nthread':str(threads)},
                           'gradient_booster':{'gbtree_train_param':{'updater':updater}}}}
        return SimpleNamespace(get_booster=lambda:SimpleNamespace(save_config=lambda:json.dumps(config)))
    with pytest.raises(models.BackendFailure,match='silent CPU'):
        models.assert_backend(fake('cpu',4,'grow_quantile_histmaker'),'cuda:0')
    with pytest.raises(models.BackendFailure,match='four'):
        models.assert_backend(fake('cpu',1,'grow_quantile_histmaker'),'cpu')
    models.assert_backend(fake('cuda:0',4,'grow_gpu_hist'),'cuda:0')


def test_progress_refuses_silent_cpu_at_the_first_tree():
    config={'learner':{'generic_param':{'device':'cpu','nthread':'4'},
                       'gradient_booster':{'gbtree_train_param':{'updater':'grow_quantile_histmaker'}}}}
    booster=SimpleNamespace(save_config=lambda:json.dumps(config))
    with pytest.raises(models.BackendFailure,match='first-iteration'):
        models.Progress(None,'inner_fit','cuda:0').after_iteration(booster,0,{})


def test_residual_validation_base_and_total_only_floor(monkeypatch,tmp_path,config,source):
    history,query=source
    calls=[]
    def inputs(hist,q,cutoff,name,conf,store,cache,namespace):
        calls.append(('inputs',cutoff,str(pd.PeriodIndex(hist.RENT_APPROVAL_DATE,freq='M').max())))
        x=pd.DataFrame({'x':np.arange(len(hist),dtype=float)})
        v=pd.DataFrame({'x':np.arange(len(q),dtype=float)})
        base=np.full(len(q),100.0 if cutoff!='2023-09' else 150.0)
        return x,v,np.full(len(hist),-10.0),base,np.ones(len(hist)),None,None,{},{}
    def fit(x,y,weight,conf,device,iters,progress,phase,valid=None):
        calls.append((phase,np.asarray(y).copy(),None if valid is None else np.asarray(valid[0][1]).copy(),iters))
        return SimpleNamespace(best_iteration=2),{'actual_config':{'learner':{'generic_param':{'device':device,'nthread':'4'}}}}
    def save(bundle,path):
        (path/'model.ubj').write_bytes(b'fixture-native-model')
        (path/'bundle.joblib').write_bytes(b'fixture-bundle')
    monkeypatch.setattr(models,'candidate_inputs',inputs)
    monkeypatch.setattr(models,'fit_tree',fit)
    monkeypatch.setattr(models,'predict_native',lambda *_:np.array([-200.0,20.0]))
    monkeypatch.setattr(models.Bundle,'save',save)
    monkeypatch.setattr(models,'load_bundle',lambda _:SimpleNamespace(predict=lambda q:np.array([0.0,170.0])))
    result=models.fit_candidate(history,query,'2023-09','B1',config,tmp_path/'model',tmp_path/'snapshots',tmp_path/'features','fixture')
    inner=[item for item in calls if item[0]=='inner_fit'][0]
    expected=history.loc[history.RENT_APPROVAL_DATE>'2023-06','MONTHLY_RENT'].to_numpy()-100.0
    np.testing.assert_array_equal(inner[1],-10.0)
    np.testing.assert_array_equal(inner[2],expected)
    outer=[item for item in calls if item[0]=='outer_fit'][0]
    assert outer[2] is None and outer[3]==3
    assert ('inputs','2023-06','2023-06') in calls
    np.testing.assert_array_equal(result.B1,[0.0,170.0])


@pytest.fixture
def cpu_checkpoint(monkeypatch,tmp_path,config,source):
    """One real small CPU4 fit pair; simulated unavailable GPU, no GPU work."""
    history,query=source
    actual=models.fit_tree
    calls=[]
    def force_explicit_fallback(*args,**kwargs):
        device=args[4]
        calls.append((device,args[7]))
        if device.startswith('cuda'):
            raise models.BackendFailure('fixture: GPU unavailable')
        return actual(*args,**kwargs)
    monkeypatch.setattr(models,'fit_tree',force_explicit_fallback)
    directory=tmp_path/'model'
    frame=models.fit_candidate(history,query,'2023-09','B0',config,directory,tmp_path/'snapshots',tmp_path/'features','fixture')
    return directory,frame,calls,history,query,config


@pytest.mark.training
def test_explicit_cpu4_fallback_native_io_and_resume_without_training(monkeypatch,cpu_checkpoint):
    directory,frame,calls,history,query,config=cpu_checkpoint
    assert calls==[('cuda:0','inner_fit'),('cpu','inner_fit'),('cpu','outer_fit')]
    receipt=read_json(directory/'complete.json')
    assert receipt['actual_contract']['backend']=='cpu'
    assert receipt['actual_contract']['cpu_threads']==4
    assert receipt['actual_contract']['fallback_evidence'][0]['backend']=='cuda:0'
    attempt=directory/receipt['attempt']
    obs=read_json(attempt/'observations.json')
    assert obs['outer']['actual_config']['learner']['generic_param']['device']=='cpu'
    assert obs['outer']['actual_config']['learner']['generic_param']['nthread']=='4'
    np.testing.assert_array_equal(models.load_bundle(attempt).predict(query),frame.B0)
    monkeypatch.setattr(models,'fit_tree',lambda *a,**kw:pytest.fail('completed checkpoint refit'))
    again=models.fit_candidate(history,query,'2023-09','B0',config,directory,directory/'unused_snapshots',directory/'unused_cache','fixture')
    pd.testing.assert_frame_equal(frame,again)


@pytest.mark.parametrize('damage',['backend','missing_fallback'])
@pytest.mark.training
def test_backend_receipt_cannot_masquerade_as_saved_model(cpu_checkpoint,damage):
    directory,frame,calls,history,query,config=cpu_checkpoint
    receipt=read_json(directory/'complete.json')
    if damage=='backend':
        receipt['actual_contract']['backend']='cuda:0'
    else:
        receipt['actual_contract']['fallback_evidence']=[]
        actual_path=directory/receipt['attempt']/'actual_contract.json'
        atomic_json(actual_path,receipt['actual_contract'])
        receipt['files']['actual_contract.json']=sha256(actual_path)
    atomic_json(directory/'complete.json',receipt)
    with pytest.raises(ValueError):
        models.restore(directory,receipt['expected_contract'])


def test_first_batch_dispatches_only_two_origins_three_candidates(monkeypatch,tmp_path):
    config={'origins':['2023-09','2024-03'],'models':['B0','B1','B2'],'scope':'exploratory fixture'}
    manifest={'config':config,'manifest_sha256':'fixture'}
    calls=[]
    monkeypatch.setattr(runner,'verify',lambda *args:copy.deepcopy(manifest))
    def child(command,cwd,attempt,label,*args):
        calls.append((command[command.index('--origin')+1],command[command.index('--model')+1],label))
        return 0
    monkeypatch.setattr(runner,'run_child',child)
    def summary(root,manifest):
        path=root/'summary'
        path.mkdir(parents=True)
        (path/'metrics.csv').write_text('fixture')
        return path
    monkeypatch.setattr(runner,'summarize',summary)
    assert runner.run(tmp_path,'fixture')==0
    assert [(o,m) for o,m,_ in calls]==[(o,m) for o in config['origins'] for m in config['models']]
    complete=read_json(tmp_path/'complete.json')
    assert complete['additional_origins_started'] is False
    assert complete['completed_origins']==config['origins']
    assert read_json(tmp_path/'status.json')['stage']=='first_batch_complete'


def test_worker_binds_actual_pid_and_lock_refusal_preserves_status(monkeypatch,tmp_path,config,source):
    history,query=source
    config['origins']=['2023-09']
    manifest={'config':config,'manifest_sha256':'fixture'}
    monkeypatch.setattr(runner,'verify',lambda *args:copy.deepcopy(manifest))
    monkeypatch.setattr(runner,'load_base',lambda:None)
    monkeypatch.setattr(runner,'data_at',lambda *args:(history,query))
    monkeypatch.setattr(runner,'fit_candidate',lambda *args:None)
    class Meter:
        result={'elapsed_seconds':0.0}
        def __enter__(self): return self
        def __exit__(self,*args): return False
    monkeypatch.setattr(runner,'ResourceMeter',Meter)
    runner.worker(tmp_path,'fixture','2023-09','B0','unique-dispatch')
    status=read_json(tmp_path/'worker_status.json')
    assert status['pid']==os.getpid()
    assert status['dispatch_id']=='unique-dispatch'
    assert status['exit_status']==0 and status['stage']=='model_complete'
    saved=(tmp_path/'worker_status.json').read_bytes()
    with RunLock(tmp_path/'worker.lock'):
        with pytest.raises(LockBusy):
            runner.worker(tmp_path,'fixture','2023-09','B0','rejected-dispatch')
    assert (tmp_path/'worker_status.json').read_bytes()==saved
