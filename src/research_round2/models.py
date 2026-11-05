"""Three approved candidates; explicit GPU/CPU4 contracts and native model IO."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import time
import traceback
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits

from src.backtest.data import frame_hash, sha256
from src.backtest.features import OOFBuildingEncoder
from src.backtest.execution import atomic_json, read_json
from src.backtest.protocol import TARGET, history_hash, digest_json, months, horizon, inner_split, require_query
from src.features.build_features import ORIGINAL_COLUMNS
from src.pipeline.preprocess import TabularPreprocessor
from src.research_round2.snapshots import RAW, SnapshotStore, cached_training, bounded


class BackendFailure(RuntimeError):
    pass


def assert_backend(model,device,check_updater=True):
    config = __import__('json').loads(model.get_booster().save_config())
    generic=config['learner']['generic_param']
    updater=config['learner']['gradient_booster']['gbtree_train_param']['updater']
    if int(generic['nthread']) != 4:
        raise BackendFailure('actual XGBoost thread limit differs from four')
    if device.startswith('cuda') and (not generic['device'].startswith('cuda') or (check_updater and 'grow_gpu_hist' not in updater)):
        raise BackendFailure(f'silent CPU fallback rejected: {generic}, {updater}')
    if device=='cpu' and generic['device']!='cpu':
        raise BackendFailure('CPU4 contract is using a different device')
    return config


def is_gpu_error(error):
    if isinstance(error,BackendFailure):
        return True
    message=str(error).lower()
    return isinstance(error,xgb.core.XGBoostError) and any(s in message for s in ('cuda','gpu','device','bad_alloc','out of memory','thrust'))


class Progress(xgb.callback.TrainingCallback):
    def __init__(self,callback,phase,device):
        self.callback,self.phase,self.device=callback,phase,device

    def after_iteration(self,model,epoch,evals_log):
        if epoch==0:
            config=__import__('json').loads(model.save_config())['learner']
            actual=config['generic_param']
            updater=config['gradient_booster']['gbtree_train_param']['updater']
            if int(actual['nthread'])!=4 or (self.device.startswith('cuda') and
                (not actual['device'].startswith('cuda') or 'grow_gpu_hist' not in updater)):
                raise BackendFailure('first-iteration actual backend differs from explicit GPU/CPU4 request')
        if self.callback and (epoch==0 or (epoch+1)%25==0):
            metric={name:{k:float(v[-1]) for k,v in values.items()} for name,values in evals_log.items()}
            self.callback(self.phase,iteration=epoch+1,metrics=metric)
        return False


def encode(train,query,config):
    # No BLOCK and no target categorical mode: this preprocessor only learns
    # training numeric fills and native category vocabularies.
    prep=TabularPreprocessor(config['preprocess'])
    prep.fit(train,np.zeros(len(train)),list(train.columns))
    x,y=prep.transform(train),prep.transform(query)
    unknown={c:int(y[c].isna().sum()) for c in prep.categorical_columns_}
    return x,y,prep,unknown


def b0_inputs(history,query,config):
    columns=[c for c in ORIGINAL_COLUMNS if c!='BLOCK']
    x,q,prep,unknown=encode(history[columns].reset_index(drop=True),query[columns].reset_index(drop=True),config)
    encoder=OOFBuildingEncoder(strength=20,folds=5,seed=config['seed'])
    x['BLOCK__building']=encoder.fit_transform(history)
    q['BLOCK__building']=encoder.transform(query)
    return x,q,history[TARGET].to_numpy(dtype=float),np.zeros(len(query)),np.ones(len(history)),prep,encoder,unknown


def candidate_inputs(history,query,cutoff,name,config,store,cache,namespace):
    if name=='B0':
        result=b0_inputs(history,query,config)
        return (*result,{'instances':len(history),'unique_target_rows':len(history),'snapshot_count':0})
    variant={'B1':'raw','B2':'adjusted'}[name]
    directory,audit=cached_training(history,cutoff,config,store,cache,namespace)
    frame=pd.read_parquet(directory/f'{variant}.parquet')
    meta=pd.read_parquet(directory/'meta.parquet')
    snap=store.get(cutoff)
    frames,bases=snap.transform(query,horizon(query,cutoff))
    x,q,prep,unknown=encode(frame,frames[variant],config)
    residual=meta.target.to_numpy(dtype=float)-meta[f'{variant}_base'].to_numpy(dtype=float)
    return x,q,residual,bases[variant],meta.weight.to_numpy(dtype=float),prep,snap,unknown,audit


def fit_tree(x,y,weights,config,device,iterations,progress,phase,valid=None):
    params=dict(config['xgboost'])
    params.pop('early_stopping_rounds')
    params.update(device=device,n_estimators=iterations,callbacks=[Progress(progress,phase,device)])
    if valid is not None:
        params['early_stopping_rounds']=config['xgboost']['early_stopping_rounds']
        params['eval_metric']='rmse'
    model=xgb.XGBRegressor(**params)
    started=time.perf_counter()
    with threadpool_limits(limits=config['cpu_threads']),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x,y,sample_weight=weights,eval_set=valid,verbose=False)
    actual=assert_backend(model,device)
    return model,{'elapsed_s':time.perf_counter()-started,'actual_config':actual,'warnings':[str(w.message) for w in caught],
                  'executed_iterations':model.get_booster().num_boosted_rounds()}


def predict_native(model,matrix,device):
    model.get_booster().set_param({'device':device,'nthread':4})
    assert_backend(model,device,check_updater=False)
    result=model.get_booster().predict(xgb.DMatrix(matrix,enable_categorical=True,nthread=4))
    if result.shape!=(len(matrix),) or not np.isfinite(result).all():
        raise ValueError('nonfinite or misaligned model output')
    return result.astype(float)


@dataclass
class Bundle:
    origin: str
    name: str
    config: dict
    prep: TabularPreprocessor
    feature_state: object
    device: str
    model: object = None

    def predict(self,query):
        require_query(query,self.origin,self.config['max_horizon'])
        if self.name=='B0':
            matrix=self.prep.transform(query[RAW].reset_index(drop=True))
            matrix['BLOCK__building']=self.feature_state.transform(query)
            base=np.zeros(len(query))
        else:
            frames,bases=self.feature_state.transform(query,horizon(query,self.origin))
            variant={'B1':'raw','B2':'adjusted'}[self.name]
            matrix=self.prep.transform(frames[variant])
            base=bases[variant]
        residual=predict_native(self.model,matrix,self.device)
        return np.maximum(self.config['prediction_floor'],base+residual)

    def save(self,directory):
        self.model.save_model(directory/'model.ubj')
        model=self.model
        try:
            self.model=None
            joblib.dump(self,directory/'bundle.joblib')
        finally:
            self.model=model


def load_bundle(directory):
    bundle=joblib.load(directory/'bundle.joblib')
    bundle.model=xgb.XGBRegressor(device=bundle.device,n_jobs=4)
    bundle.model.load_model(directory/'model.ubj')
    bundle.model.get_booster().set_param({'device':bundle.device,'nthread':4})
    return bundle


def expected_contract(history,query,origin,name,config,namespace):
    require_query(query,origin,config['max_horizon'])
    return {'origin':origin,'model':name,'config':config,'namespace':namespace,'labels':history_hash(history),
            'history_predictors':frame_hash(history.drop(columns=TARGET)),'query_predictors':frame_hash(query)}


def restore(directory,contract):
    marker=directory/'complete.json'
    if not marker.exists():
        return None
    receipt=read_json(marker)
    if receipt['expected_contract']!=contract:
        raise ValueError('completed model contract mismatch')
    actual=receipt['actual_contract']
    if actual['expected_contract_sha256']!=digest_json(contract) or actual['backend'] not in ('cuda:0','cpu'):
        raise ValueError('invalid actual backend contract')
    attempt=directory/receipt['attempt']
    if any(sha256(attempt/name)!=value for name,value in receipt['files'].items()):
        raise ValueError('completed model checksum mismatch')
    if actual!=read_json(attempt/'actual_contract.json'):
        raise ValueError('checkpoint backend contract differs from saved fit contract')
    actual_params=read_json(attempt/'observations.json')['outer']['actual_config']['learner']['generic_param']
    if actual_params['device']!=actual['backend'] or int(actual_params['nthread'])!=4:
        raise ValueError('checkpoint backend differs from recorded actual model configuration')
    if actual['backend']=='cpu' and not actual['fallback_evidence']:
        raise ValueError('CPU fallback is missing its explicit GPU failure evidence')
    frame=pd.read_parquet(attempt/'predictions.parquet')
    if frame.duplicated('row_id').any() or not np.isfinite(frame[contract['model']]).all():
        raise ValueError('invalid saved model prediction')
    return frame


def fit_candidate(history,query,origin,name,config,directory,snapshots,cache,namespace,progress=None,smoke_iterations=None):
    history=bounded(history,origin)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    contract=expected_contract(history,query,origin,name,config,namespace)
    contract['smoke_iterations']=smoke_iterations
    restored=restore(directory,contract)
    if restored is not None:
        if not np.array_equal(restored.row_id.to_numpy(),query.row_id.to_numpy()):
            raise ValueError('checkpoint row order differs from expected query')
        return restored
    store=SnapshotStore(history,config,snapshots,namespace,progress)
    failures=[]
    for device in ('cuda:0','cpu'):
        attempt_name=time.strftime('%Y%m%dT%H%M%S')+'_'+device.split(':')[0]+'_'+uuid.uuid4().hex[:8]
        attempt=directory/attempt_name
        attempt.mkdir()
        try:
            if smoke_iterations is None:
                train,valid,y_valid,cutoff=inner_split(history,origin,config)
                if progress: progress('inner_features',cutoff=cutoff,backend=device)
                x,v,y,vbase,weight,_,_,unknown,inner_audit=candidate_inputs(train,valid,cutoff,name,config,store,cache,namespace)
                # Validation residual matches this same validation forecast base.
                model,inner_obs=fit_tree(x,y,weight,config,device,config['xgboost']['n_estimators'],progress,'inner_fit',[(v,y_valid-vbase)])
                selected=int(model.best_iteration)+1
                atomic_json(attempt/'inner.json',{'selected_iterations':selected,'audit':inner_audit,'unknown_categories':unknown,**inner_obs})
                del x,v,y,model
                gc.collect()
            else:
                selected=smoke_iterations
            if progress: progress('outer_features',cutoff=origin,backend=device)
            x,q,y,qbase,weight,prep,state,unknown,outer_audit=candidate_inputs(history,query,origin,name,config,store,cache,namespace)
            model,outer_obs=fit_tree(x,y,weight,config,device,selected,progress,'outer_fit')
            # Residuals can be negative; floor only after reconstructing total rent.
            residual=predict_native(model,q,device)
            prediction=np.maximum(config['prediction_floor'],qbase+residual)
            bundle=Bundle(origin,name,config,prep,state,device,model)
            bundle.save(attempt)
            after=load_bundle(attempt).predict(query)
            if not np.array_equal(prediction,after):
                raise ValueError('native save/load changed reconstructed prediction')
            frame=query[['row_id','RENT_APPROVAL_DATE']].reset_index(drop=True).copy()
            frame.insert(0,'origin',origin)
            frame['horizon']=horizon(query,origin)
            frame[name]=prediction
            frame.to_parquet(attempt/'predictions.parquet',index=False)
            actual={'expected_contract_sha256':digest_json(contract),'backend':device,'cpu_threads':4,'selected_iterations':selected,
                    'fallback_evidence':failures,'smoke_iterations':smoke_iterations}
            observations={'outer':outer_obs,'training_audit':outer_audit,'unknown_categories':unknown,'save_load_max_abs_difference':float(np.max(np.abs(prediction-after))),
                          'residual_min':float(residual.min()),'base_min':float(qbase.min()),'total_floor_count':int(np.count_nonzero(qbase+residual<0))}
            atomic_json(attempt/'observations.json',observations)
            atomic_json(attempt/'actual_contract.json',actual)
            names=['model.ubj','bundle.joblib','predictions.parquet','observations.json','actual_contract.json']+(['inner.json'] if smoke_iterations is None else [])
            atomic_json(directory/'complete.json',{'expected_contract':contract,'actual_contract':actual,'attempt':attempt_name,
                         'files':{file:sha256(attempt/file) for file in names}})
            return frame
        except Exception as error:
            failure={'backend':device,'error':repr(error),'traceback':traceback.format_exc(),'attempt':attempt_name}
            atomic_json(attempt/'failed.json',failure)
            failures.append(failure)
            if device!='cuda:0' or not is_gpu_error(error):
                raise
            if progress: progress('explicit_cpu4_fallback',reason=repr(error),failed_backend=device)
            gc.collect()
    raise RuntimeError('both execution backends failed')
