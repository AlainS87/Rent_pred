"""Native building identity and static property inputs; no manual target means."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import time
import traceback
import uuid

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,CatBoostError
from threadpoolctl import threadpool_limits

from src.backtest.data import sha256
from src.backtest.execution import atomic_json,read_json
from src.backtest.protocol import TARGET,digest_json,horizon,inner_split,require_query
from src.backtest.decomposed import static_frame
from src.pipeline.preprocess import _category_text
from src.research_round2.models import encode,fit_tree,predict_native,expected_contract,is_gpu_error,BackendFailure
from src.research_round2.snapshots import bounded

FORBIDDEN={'BLOCK',TARGET,'building_age_years','flat_age_from_lease','building_age_x_central'}


def projection(frame,columns):
    if len(columns)!=len(set(columns)) or any(c in FORBIDDEN or c.startswith(('mean_rent_','coe_','bank_basket_','property_basket_')) for c in columns):
        raise ValueError('forbidden direct-property feature')
    # static_frame performs deterministic predictor-only key/area derivations.
    prepared=static_frame(frame.drop(columns=TARGET,errors='ignore'))
    return prepared[columns].reset_index(drop=True).copy()


def inputs(history,query,origin,name,config):
    require_query(query,origin,config['max_horizon'])
    history=bounded(history,origin)
    columns=config['features'][name]
    train=projection(history,columns); future=projection(query,columns)
    x,q,prep,unknown=encode(train,future,config)
    if prep.include_building_ or prep.building_values_ or prep.target_values_:
        raise ValueError('special target-statistic preprocessor was activated')
    cats=list(prep.categorical_columns_)
    if name=='C3':
        # CatBoost hashes normalized strings. Retain genuinely unseen values;
        # XGBoost instead gets missing codes outside its training vocabulary.
        for c in cats:
            x[c]=_category_text(train[c],c).to_numpy(dtype=str)
            q[c]=_category_text(future[c],c).to_numpy(dtype=str)
    audit={'source_columns':columns,'matrix_columns':list(x),'categorical_columns':cats,'unknown_query_counts':unknown,
           'manual_target_statistics':False,'training_rows':len(history),'query_rows':len(query),
           'unseen_policy':'CatBoost normalized unseen string hashing/CTR prior' if name=='C3' else 'XGBoost training-vocabulary categorical missing code',
           'categorical_vocabulary_source':'training rows only; rare_min_count does not prune native levels'}
    return x,q,prep,cats,audit


def cb_fit(x,y,config,device,iterations,cats,progress,phase,valid=None):
    params=dict(config['catboost']); params.pop('early_stopping_rounds')
    params.update(iterations=iterations,task_type='GPU' if device.startswith('cuda') else 'CPU')
    if params['task_type']=='CPU':
        params.pop('devices'); params.pop('gpu_ram_part')
    if valid is not None: params['early_stopping_rounds']=config['catboost']['early_stopping_rounds']
    if progress: progress(phase,requested_training_device=device,params=params)
    model=CatBoostRegressor(**params); start=time.perf_counter()
    with threadpool_limits(limits=config['cpu_threads']):
        # GPU CatBoost does not support Python training callbacks. Durable
        # stdout progress every 25 rounds is captured by the serial worker.
        model.fit(x,y,cat_features=cats,eval_set=valid,use_best_model=valid is not None,verbose=25)
    actual=model.get_all_params()
    if actual['task_type']!=params['task_type'] or model.get_params()['thread_count']!=4:
        raise BackendFailure('CatBoost actual training backend/thread contract mismatch')
    executed=len(model.get_evals_result()['validation']['RMSE']) if valid is not None else model.tree_count_
    return model,{'elapsed_s':time.perf_counter()-start,'actual_params':actual,'constructor_params':params,'executed_iterations':executed}


def cb_predict(model,matrix,device):
    if device=='cpu':
        return np.asarray(model.predict(matrix,task_type='CPU',thread_count=4),dtype=float),{'prediction_device':'CPU4','reason':'explicit CPU4 training backend'}
    try:
        return np.asarray(model.predict(matrix,task_type='GPU',thread_count=4),dtype=float),{'prediction_device':'GPU','reason':None}
    except CatBoostError as error:
        text=str(error).lower()
        if 'categorical features' not in text or 'gpu evaluation impossible' not in text:
            raise
        result=np.asarray(model.predict(matrix,task_type='CPU',thread_count=4),dtype=float)
        return result,{'prediction_device':'CPU4','reason':str(error),'training_backend_unchanged':True}


@dataclass
class DirectBundle:
    origin: str
    name: str
    config: dict
    prep: object
    cats: list
    device: str
    model: object=None

    def predict(self,query):
        require_query(query,self.origin,self.config['max_horizon'])
        raw=projection(query,self.config['features'][self.name]); matrix=self.prep.transform(raw)
        if self.name=='C3':
            for c in self.cats: matrix[c]=_category_text(raw[c],c).to_numpy(dtype=str)
            prediction,metadata=cb_predict(self.model,matrix,self.device)
        else:
            prediction=predict_native(self.model,matrix,self.device); metadata={'prediction_device':self.device}
        if prediction.shape!=(len(query),) or not np.isfinite(prediction).all():
            raise ValueError('invalid direct prediction')
        return np.maximum(self.config['prediction_floor'],prediction),metadata

    def save(self,path):
        filename='model.cbm' if self.name=='C3' else 'model.ubj'
        self.model.save_model(path/filename)
        model=self.model
        try:
            self.model=None; joblib.dump(self,path/'bundle.joblib')
        finally: self.model=model


def load_bundle(path):
    bundle=joblib.load(path/'bundle.joblib')
    if bundle.name=='C3':
        bundle.model=CatBoostRegressor(); bundle.model.load_model(path/'model.cbm')
    else:
        import xgboost as xgb
        bundle.model=xgb.XGBRegressor(device=bundle.device,n_jobs=4); bundle.model.load_model(path/'model.ubj')
    return bundle


def restore(directory,contract):
    if not (directory/'complete.json').exists(): return None
    receipt=read_json(directory/'complete.json'); actual=receipt['actual_contract']; path=directory/receipt['attempt']
    if receipt['expected_contract']!=contract or actual['expected_contract_sha256']!=digest_json(contract):
        raise ValueError('round3 checkpoint expected contract mismatch')
    if any(sha256(path/n)!=v for n,v in receipt['files'].items()): raise ValueError('round3 checkpoint checksum mismatch')
    if actual!=read_json(path/'actual_contract.json'): raise ValueError('round3 actual backend receipt mismatch')
    if actual['device'] not in ('cuda:0','cpu') or actual['cpu_threads']!=4: raise ValueError('invalid device contract')
    if actual['library']!=('catboost' if contract['model']=='C3' else 'xgboost'): raise ValueError('library contract mismatch')
    if actual['device']=='cpu' and not actual['fallback_evidence']: raise ValueError('CPU training lacks explicit GPU failure')
    observation=read_json(path/'observations.json')['outer']
    recorded=observation['actual_params']['task_type'] if contract['model']=='C3' else observation['actual_config']['learner']['generic_param']['device']
    expected=('GPU' if actual['device']=='cuda:0' else 'CPU') if contract['model']=='C3' else actual['device']
    if recorded!=expected: raise ValueError('saved training backend differs from receipt')
    actual_threads=observation['constructor_params']['thread_count'] if contract['model']=='C3' else int(observation['actual_config']['learner']['generic_param']['nthread'])
    if actual_threads!=4 or observation['executed_iterations']!=actual['selected_iterations']:
        raise ValueError('saved thread/iteration contract mismatch')
    frame=pd.read_parquet(path/'predictions.parquet')
    if frame.row_id.duplicated().any() or not np.isfinite(frame[contract['model']]).all(): raise ValueError('invalid saved direct predictions')
    return frame


def fit_candidate(history,query,origin,name,config,directory,namespace,progress=None,smoke_iterations=None):
    history=bounded(history,origin); require_query(query,origin,config['max_horizon'])
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    contract=expected_contract(history,query,origin,name,config,namespace); contract['smoke_iterations']=smoke_iterations
    saved=restore(directory,contract)
    if saved is not None:
        if not np.array_equal(saved.row_id.to_numpy(),query.row_id.to_numpy()): raise ValueError('restored row order changed')
        return saved
    failures=[]
    for device in ('cuda:0','cpu'):
        attempt_name=time.strftime('%Y%m%dT%H%M%S')+'_'+device.split(':')[0]+'_'+uuid.uuid4().hex[:8]
        path=directory/attempt_name; path.mkdir()
        fit_in_progress=False
        try:
            if smoke_iterations is None:
                train,valid,y_valid,cutoff=inner_split(history,origin,config)
                if progress: progress('inner_features',cutoff=cutoff,training_device=device)
                x,v,_,cats,audit=inputs(train,valid,cutoff,name,config); y=train[TARGET].to_numpy(dtype=float)
                fit_in_progress=True
                if name=='C3':
                    model,obs=cb_fit(x,y,config,device,config['catboost']['iterations'],cats,progress,'inner_fit',(v,y_valid)); selected=max(1,int(model.get_best_iteration())+1)
                else:
                    model,obs=fit_tree(x,y,np.ones(len(y)),config,device,config['xgboost']['n_estimators'],progress,'inner_fit',[(v,y_valid)]); selected=int(model.best_iteration)+1
                fit_in_progress=False
                atomic_json(path/'inner.json',{'selected_iterations':selected,'audit':audit,**obs})
                del x,v,y,model; gc.collect()
            else: selected=smoke_iterations
            if progress: progress('outer_features',cutoff=origin,training_device=device)
            x,q,prep,cats,audit=inputs(history,query,origin,name,config); y=history[TARGET].to_numpy(dtype=float)
            fit_in_progress=True
            if name=='C3': model,obs=cb_fit(x,y,config,device,selected,cats,progress,'outer_fit')
            else: model,obs=fit_tree(x,y,np.ones(len(y)),config,device,selected,progress,'outer_fit')
            fit_in_progress=False
            bundle=DirectBundle(origin,name,config,prep,cats,device,model)
            prediction,pred_device=bundle.predict(query); bundle.save(path)
            restored,_=load_bundle(path).predict(query)
            if not np.array_equal(prediction,restored): raise ValueError('direct native save/load changed predictions')
            frame=query[['row_id','RENT_APPROVAL_DATE']].reset_index(drop=True).copy(); frame.insert(0,'origin',origin); frame['horizon']=horizon(query,origin); frame[name]=prediction
            frame.to_parquet(path/'predictions.parquet',index=False)
            actual={'expected_contract_sha256':digest_json(contract),'library':'catboost' if name=='C3' else 'xgboost',
                    'device':device,'cpu_threads':4,'selected_iterations':selected,'fallback_evidence':failures,'prediction':pred_device,'smoke_iterations':smoke_iterations}
            atomic_json(path/'actual_contract.json',actual)
            atomic_json(path/'observations.json',{'outer':obs,'feature_audit':audit,'save_load_max_abs_difference':float(np.max(abs(prediction-restored)))})
            names=['model.cbm' if name=='C3' else 'model.ubj','bundle.joblib','predictions.parquet','actual_contract.json','observations.json']+(['inner.json'] if smoke_iterations is None else [])
            atomic_json(directory/'complete.json',{'expected_contract':contract,'actual_contract':actual,'attempt':attempt_name,'files':{n:sha256(path/n) for n in names}})
            return frame
        except Exception as error:
            failure={'device':device,'attempt':attempt_name,'error':repr(error),'traceback':traceback.format_exc()}; atomic_json(path/'failed.json',failure); failures.append(failure)
            gpu_failure=is_gpu_error(error) or (isinstance(error,CatBoostError) and any(t in str(error).lower() for t in ('cuda','gpu','driver','out of memory','bad_alloc')))
            if device!='cuda:0' or not gpu_failure or not fit_in_progress: raise
            if progress: progress('explicit_CPU4_training_fallback',reason=repr(error))
            gc.collect()
