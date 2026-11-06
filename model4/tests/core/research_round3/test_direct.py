"""Round-3 deltas only: predictor identity, native categories and device contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostError
import xgboost as xgb

from src.backtest.execution import read_json
from src.backtest.data import sha256
from src.research_round3 import models
from src.research_round3 import run as runner

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture
def config():
    cfg=json.loads((ROOT/'configs/research_round3/protocol.json').read_text())
    # Test-only tiny CPU4 budget; no formal recipe or existing artifacts change.
    cfg['xgboost'].update(max_depth=2,n_estimators=8,early_stopping_rounds=2)
    cfg['catboost'].update(depth=2,iterations=8,early_stopping_rounds=2)
    return cfg


@pytest.fixture
def source(config):
    columns=set(config['features']['C2'])-{'building_key','town_flat_type'}
    text={'TOWN','STREET','FLAT_TYPE','FLAT_MODEL','FURNISHED','postal_code','subzone','planning_area','region'}
    text.update(c for c in columns if c.endswith(('_code','_name','_line')))
    template={c:('known' if c in text else 1.0) for c in columns}
    rows=[]
    for month in pd.period_range('2022-01','2023-09',freq='M'):
        for town in ('North','South','North'):
            row=dict(template)
            row.update(row_id=len(rows),RENT_APPROVAL_DATE=str(month),TOWN=town,BLOCK='1',STREET='Main Street',
                       FLAT_TYPE='3-room',FLAT_MODEL='Improved',FURNISHED='yes',FLOOR_AREA_SQM=67.0,
                       LEASE_COMMENCE_DATE=1990.0,FEE=0.0,MONTHLY_RENT=1500+20*(month.ordinal-pd.Period('2022-01','M').ordinal)+10*(town=='South'))
            rows.append(row)
    history=pd.DataFrame(rows)
    query=history.iloc[:2].drop(columns='MONTHLY_RENT').copy()
    query['row_id']=[1000,1001]
    query['RENT_APPROVAL_DATE']=['2023-10','2025-01']
    query.loc[query.index[1],'TOWN']='Never Seen Town'
    return history,query


def test_declared_columns_are_exact_original_static_union(config):
    first=json.loads((ROOT/'configs/backtest/protocol.json').read_text())
    expected=set(first['decomposed']['property_features'])|{'RENT_APPROVAL_DATE','town_flat_type'}
    assert len(expected)==70
    assert config['features']['C2']==config['features']['C3']
    assert len(config['features']['C2'])==70 and set(config['features']['C2'])==expected
    assert 'building_key' in config['features']['C1'] and 'BLOCK' not in config['features']['C1']
    for columns in config['features'].values():
        assert not models.FORBIDDEN.intersection(columns)
        assert not any(c.startswith(('mean_rent_','coe_','bank_basket_','property_basket_')) for c in columns)


def test_native_building_identity_distinguishes_towns_without_rent_mean(config,source):
    history,query=source
    projected=models.projection(history,config['features']['C1'])
    assert projected.building_key.iloc[0]!=projected.building_key.iloc[1]
    x,_,prep,_,audit=models.inputs(history,query,'2023-09','C1',config)
    assert 'building_key' in x and isinstance(x.building_key.dtype,pd.CategoricalDtype)
    assert 'BLOCK__building' not in x
    assert not prep.include_building_ and prep.building_values_=={} and prep.target_values_=={}
    assert audit['manual_target_statistics'] is False
    for bad in ('BLOCK','MONTHLY_RENT','mean_rent_postal_sector','building_age_years','flat_age_from_lease','building_age_x_central','coe_e_premium_lag1','bank_basket_adj_close_lag1'):
        with pytest.raises(ValueError,match='forbidden'):
            models.projection(history,[bad])


def test_future_and_inner_validation_labels_never_change_direct_features(config,source):
    history,_=source
    cutoff='2023-06'
    valid=history.loc[history.RENT_APPROVAL_DATE>cutoff].drop(columns='MONTHLY_RENT')
    changed=history.copy()
    changed.loc[changed.RENT_APPROVAL_DATE>cutoff,'MONTHLY_RENT']=1e9
    for name in ('C1','C2','C3'):
        a=models.inputs(history,valid,cutoff,name,config)
        b=models.inputs(changed,valid,cutoff,name,config)
        pd.testing.assert_frame_equal(a[0],b[0]); pd.testing.assert_frame_equal(a[1],b[1])
        assert a[4]['training_rows']==int((history.RENT_APPROVAL_DATE<=cutoff).sum())
    # Manual target-statistics are absent even if allowed historical y changes.
    changed['MONTHLY_RENT']=-99999.0
    pd.testing.assert_frame_equal(models.inputs(history,valid,cutoff,'C2',config)[0],models.inputs(changed,valid,cutoff,'C2',config)[0])
    truth=valid.copy(); truth['MONTHLY_RENT']=1.0
    with pytest.raises(ValueError,match='target'):
        models.inputs(history,truth,cutoff,'C2',config)


def test_xgb_unknown_is_missing_but_catboost_retains_normalized_identity(config,source):
    history,query=source
    _,xgb_query,xgb_prep,_,xgb_audit=models.inputs(history,query,'2023-09','C2',config)
    _,cb_query,cb_prep,_,cb_audit=models.inputs(history,query,'2023-09','C3',config)
    assert xgb_query.TOWN.isna().iloc[1] and xgb_query.building_key.isna().iloc[1]
    assert cb_query.TOWN.iloc[1]=='never seen town'
    assert 'never seen town' in cb_query.building_key.iloc[1]
    assert not cb_query.TOWN.isna().any() and cb_query.building_key.iloc[1]!=cb_query.building_key.iloc[0]
    assert 'never seen town' not in xgb_prep.levels_['TOWN'] and 'never seen town' not in cb_prep.levels_['TOWN']
    assert xgb_audit['unknown_query_counts']['TOWN']==cb_audit['unknown_query_counts']['TOWN']==1


@pytest.mark.parametrize('name',['C1','C3'])
@pytest.mark.training
def test_cpu4_native_io_complete_checkpoint_does_not_refit(monkeypatch,tmp_path,config,source,name):
    history,query=source
    calls=[]
    actual_xgb=models.fit_tree; actual_cb=models.cb_fit
    def xgb_fit(*args,**kwargs):
        calls.append((args[4],args[7]))
        if args[4].startswith('cuda'): raise models.BackendFailure('fixture GPU unavailable')
        return actual_xgb(*args,**kwargs)
    def cat_fit(*args,**kwargs):
        calls.append((args[3],args[7]))
        if args[3].startswith('cuda'): raise CatBoostError('fixture GPU unavailable')
        return actual_cb(*args,**kwargs)
    monkeypatch.setattr(models,'fit_tree',xgb_fit); monkeypatch.setattr(models,'cb_fit',cat_fit)
    directory=tmp_path/name
    result=models.fit_candidate(history,query,'2023-09',name,config,directory,'fixture')
    assert calls==[('cuda:0','inner_fit'),('cpu','inner_fit'),('cpu','outer_fit')]
    saved=read_json(directory/'complete.json')
    assert saved['actual_contract']['device']=='cpu' and saved['actual_contract']['cpu_threads']==4
    assert saved['actual_contract']['fallback_evidence']
    loaded,_=models.load_bundle(directory/saved['attempt']).predict(query)
    np.testing.assert_array_equal(loaded,result[name])
    monkeypatch.setattr(models,'fit_tree',lambda *a,**kw:pytest.fail('XGBoost checkpoint refit'))
    monkeypatch.setattr(models,'cb_fit',lambda *a,**kw:pytest.fail('CatBoost checkpoint refit'))
    pd.testing.assert_frame_equal(models.fit_candidate(history,query,'2023-09',name,config,directory,'fixture'),result)


class FakeCatModel:
    def get_best_iteration(self): return 1
    def save_model(self,path): Path(path).write_bytes(b'fixture native model')
    def predict(self,matrix,task_type,thread_count):
        assert thread_count==4
        if task_type=='GPU': raise CatBoostError('Model contains categorical features, GPU evaluation impossible')
        return np.linspace(1600.0,1700.0,len(matrix))


def test_c3_gpu_training_cpu4_prediction_are_distinct_contracts(monkeypatch,tmp_path,config,source):
    history,query=source
    calls=[]
    def fit(*args,**kwargs):
        calls.append((args[3],args[7]))
        return FakeCatModel(),{'actual_params':{'task_type':'GPU'},'constructor_params':{'thread_count':4},'executed_iterations':2}
    def load(path):
        bundle=joblib.load(path/'bundle.joblib'); bundle.model=FakeCatModel(); return bundle
    monkeypatch.setattr(models,'cb_fit',fit); monkeypatch.setattr(models,'load_bundle',load)
    directory=tmp_path/'C3'
    models.fit_candidate(history,query,'2023-09','C3',config,directory,'fixture')
    assert calls==[('cuda:0','inner_fit'),('cuda:0','outer_fit')]
    saved=read_json(directory/'complete.json')
    actual=saved['actual_contract']
    assert actual['device']=='cuda:0' and actual['library']=='catboost'
    assert actual['prediction']['prediction_device']=='CPU4'
    assert actual['prediction']['training_backend_unchanged'] is True
    assert 'categorical features' in actual['prediction']['reason']
    assert actual['fallback_evidence']==[]
    models.restore(directory,saved['expected_contract'])


def test_gpu_prediction_failure_does_not_trigger_cpu_training(monkeypatch,tmp_path,config,source):
    history,query=source
    calls=[]
    def fit(*args,**kwargs):
        calls.append((args[4],args[7]))
        return SimpleNamespace(best_iteration=1),{'actual_config':{'learner':{'generic_param':{'device':'cuda:0','nthread':'4'}}}}
    def fail_prediction(*args): raise xgb.core.XGBoostError('fixture CUDA prediction failure')
    monkeypatch.setattr(models,'fit_tree',fit); monkeypatch.setattr(models,'predict_native',fail_prediction)
    with pytest.raises(xgb.core.XGBoostError,match='prediction failure'):
        models.fit_candidate(history,query,'2023-09','C1',config,tmp_path/'C1','fixture')
    assert calls==[('cuda:0','inner_fit'),('cuda:0','outer_fit')]
    assert len(list((tmp_path/'C1').glob('*_cpu_*')))==0


def test_baseline_reference_is_verified_clone_without_training(monkeypatch,tmp_path,config,source):
    history,template=source
    # The frozen paths are relative to the experiment workspace, not to the
    # developer's checkout or the operating system's temporary-directory root.
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    protocol=tmp_path/'configs/research_round3/protocol.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps(config),encoding='utf-8')
    monkeypatch.setattr(runner,'CONFIG',protocol)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='fixture GPU',returncode=0))
    prior=tmp_path/'prior'; prior.mkdir(parents=True)
    config['baseline_reference'].update(root=str(prior.relative_to(tmp_path)),manifest_sha256='prior-fixture')
    frames={}
    queries={}
    row_plan={}
    for origin in config['origins']:
        q=template.copy()
        q['RENT_APPROVAL_DATE']=[str(pd.Period(origin,'M')+1),str(pd.Period(origin,'M')+16)]
        queries[origin]=q
        frame=q[['row_id','RENT_APPROVAL_DATE']].reset_index(drop=True).copy()
        frame.insert(0,'origin',origin); frame['horizon']=[1,16]; frame['B0']=[1700.0,1800.0]
        frames[origin]=frame
        directory=prior/'models'/origin/'B0'; (directory/'saved').mkdir(parents=True)
        frame.to_parquet(directory/'saved/predictions.parquet',index=False)
        (directory/'complete.json').write_text(json.dumps({'attempt':'saved'}))
        row_plan[origin]={'history_rows':len(history),'query_rows':len(q),'query_identity_sha256':'fixture',
                          'inner_cutoff':str(pd.Period(origin,'M')-3),'inner_validation_rows':3}
    (prior/'manifest.json').write_text(json.dumps({'inputs':{},'row_plan':row_plan}))
    (prior/'complete.json').write_text(json.dumps({'complete':True}))
    read=runner.read_json
    monkeypatch.setattr(runner,'read_json',lambda p:config if Path(p)==runner.CONFIG else read(p))
    monkeypatch.setattr(runner.previous,'verify',lambda root,pin:{'config':config})
    monkeypatch.setattr(runner.previous,'source_paths',lambda:[])
    monkeypatch.setattr(runner.previous,'environment',lambda:{'fixture':True})
    monkeypatch.setattr(runner.previous,'load_base',lambda:None)
    monkeypatch.setattr(runner.previous,'data_at',lambda base,origin,conf:(history,queries[origin]))
    monkeypatch.setattr(runner.old_models,'restore',lambda path,contract:frames[contract['origin']])
    monkeypatch.setattr(runner.old_models,'fit_candidate',lambda *a,**kw:pytest.fail('B0 retrained'))
    monkeypatch.setattr(runner,'fit_candidate',lambda *a,**kw:pytest.fail('freeze started new training'))
    root=tmp_path/'artifacts/backtest/round3_reference'
    manifest=runner.freeze(root)
    assert manifest['baseline_retrained'] is False
    for origin,ref in manifest['baseline_references'].items():
        assert ref['rows']==2
        assert sha256(root/ref['local_copy'])==ref['source_predictions_sha256']==sha256(tmp_path/ref['source_predictions'])
        pd.testing.assert_frame_equal(pd.read_parquet(root/ref['local_copy']),frames[origin])


def test_runner_dispatches_only_six_new_candidates(monkeypatch,tmp_path,config):
    manifest={'config':config,'manifest_sha256':'fixture'}
    calls=[]
    monkeypatch.setattr(runner,'verify',lambda *args:manifest)
    def child(command,cwd,attempt,label,*args):
        calls.append((command[command.index('--origin')+1],command[command.index('--model')+1]))
        return 0
    monkeypatch.setattr(runner,'run_child',child)
    def summary(root,manifest):
        path=root/'summary'; path.mkdir(parents=True); (path/'metrics.csv').write_text('fixture'); return path
    monkeypatch.setattr(runner,'summarize',summary)
    assert runner.run(tmp_path,'fixture')==0
    assert calls==[(o,m) for o in config['origins'] for m in ('C1','C2','C3')]
    done=read_json(tmp_path/'complete.json')
    assert done['baseline_retrained'] is False and done['additional_origins_started'] is False
    assert read_json(tmp_path/'status.json')['stage']=='batch_complete'
