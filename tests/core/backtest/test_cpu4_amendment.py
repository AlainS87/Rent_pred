import copy
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.protocol import ROOT, TARGET, months

from src.backtest import cpu_threads as extension


def test_only_authorized_thread_count_may_change(tiny_config):
    changed = extension.amended_protocol(tiny_config)
    assert changed['catboost']['thread_count'] == 4
    assert tiny_config['catboost']['thread_count'] == 1
    extension.require_only_thread_change(tiny_config, changed)
    poisoned = copy.deepcopy(changed)
    poisoned['catboost']['iterations'] += 1
    with pytest.raises(ValueError, match='something other'):
        extension.require_only_thread_change(tiny_config, poisoned)
    with pytest.raises(ValueError, match='1-to-4'):
        extension.amended_protocol(tiny_config, 8)


@pytest.mark.parametrize('name',['direct_legacy40','direct_all93','decomposed_damped'])
@pytest.mark.training
def test_actual_fitters_keep_iterations_and_predictions_across_threads(labeled_fixture,tiny_config,name):
    from src.backtest.models import fit_direct
    from src.backtest.decomposed import fit_decomposed
    origin='2022-12'
    history=labeled_fixture.loc[months(labeled_fixture)<=pd.Period(origin,freq='M')].copy()
    query=labeled_fixture.loc[months(labeled_fixture)>pd.Period(origin,freq='M')].drop(columns=TARGET).copy()
    config4=extension.amended_protocol(tiny_config)
    if name.startswith('direct_'):
        bundle1, values1=fit_direct(history,query,origin,name,tiny_config)
        bundle4, values4=fit_direct(history,query,origin,name,config4)
    else:
        bundle1, frame1=fit_decomposed(history,query,origin,tiny_config,['damped'])['damped']
        bundle4, frame4=fit_decomposed(history,query,origin,config4,['damped'])['damped']
        values1,values4=frame1.prediction.to_numpy(),frame4.prediction.to_numpy()
    assert bundle1.audit['selected_iterations']==bundle4.audit['selected_iterations']
    assert bundle1.audit['inner']['executed_iterations']==bundle4.audit['inner']['executed_iterations']
    np.testing.assert_array_equal(values1,values4)
    assert bundle4.model.get_params()['thread_count']==4


@pytest.mark.training
def test_amended_worker_writes_honest_contracts_and_restores_without_fitting(labeled_fixture,tiny_config,checkpoint_path,monkeypatch):
    from src.backtest import data, frozen, selection, models, decomposed
    from src.backtest.execution import atomic_json
    from src.backtest.frozen import make_manifest, DIRECT_MODELS
    from src.backtest.protocol import digest_json
    root=checkpoint_path/'amended'
    root.mkdir(parents=True)
    parent=checkpoint_path/'original'
    parent.mkdir(parents=True)
    dataset=checkpoint_path/'dataset'
    (dataset/'data/raw').mkdir(parents=True)
    pieces=[labeled_fixture.reset_index(drop=True)]
    for month in pd.period_range('2024-05','2025-03',freq='M'):
        extra=labeled_fixture.iloc[-12:].copy().reset_index(drop=True)
        extra['RENT_APPROVAL_DATE']=str(month)
        extra[TARGET]+=int(month.ordinal-pd.Period('2024-04',freq='M').ordinal)*25
        pieces.append(extra)
    raw=pd.concat(pieces,ignore_index=True)
    raw['row_id']=np.arange(len(raw),dtype=np.int64)
    raw.to_csv(dataset/'data/raw/train.csv',index=False)
    base=raw.drop(columns=TARGET)
    config1=copy.deepcopy(tiny_config)
    config1['evaluation_origins']=['2024-12']
    config4=extension.amended_protocol(config1)
    original_manifest=make_manifest(config1,base,{'fixture':True},dataset)
    manifest=make_manifest(config4,base,{'fixture':True,'execution_extension':{'fixture':True}},dataset)
    manifest['execution_amendment']={'target_origin':'2024-12','selection_sha256':'frozen-choice'}
    manifest['manifest_sha256']=digest_json({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    atomic_json(root/'frozen_protocol.json',manifest)
    monkeypatch.setattr(extension,'verify_amendment',lambda actual_root,pin:(manifest,parent,original_manifest))
    monkeypatch.setattr(frozen,'verify_runtime',lambda candidate:None)
    selection_calls=[]
    def base_selection(actual_root,candidate,pin):
        assert actual_root==parent and candidate==original_manifest and pin=='frozen-choice'
        selection_calls.append(pin)
        return {'selected_trend':'damped'}
    monkeypatch.setattr(selection,'verify_selection',base_selection)
    original_history_at=data.history_at
    monkeypatch.setattr(data,'load_cache',lambda split:base.copy())
    monkeypatch.setattr(data,'history_at',lambda frame,origin,source=dataset:original_history_at(frame,origin,source))
    extension.worker(root,manifest['manifest_sha256'],'fixture-dispatch')
    folder=root/'evaluation/2024-12'
    originals={model:(folder/model/'observations.json').read_bytes() for model in [*DIRECT_MODELS,'decomposed_damped']}
    for model in originals:
        contract=json.loads((folder/model/'contract.json').read_text())
        assert contract['config']['catboost']['thread_count']==4
        assert contract['environment']==manifest['environment']
    receipt=(folder/'origin_complete.json').read_bytes()
    assert selection_calls
    def forbidden(*args,**kwargs):
        raise AssertionError('a complete amended origin must never refit')
    monkeypatch.setattr(models,'fit_direct',forbidden)
    monkeypatch.setattr(models,'fit_simple',forbidden)
    monkeypatch.setattr(decomposed,'fit_decomposed',forbidden)
    extension.worker(root,manifest['manifest_sha256'],'fixture-resume')
    assert receipt==(folder/'origin_complete.json').read_bytes()
    for model,content in originals.items():
        assert content==(folder/model/'observations.json').read_bytes()
