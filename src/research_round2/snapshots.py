"""Past-only hierarchical price snapshots and matched raw/detrended matrices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import gc
import copy

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from src.backtest.data import frame_hash, sha256
from src.backtest.execution import atomic_json, read_json
from src.backtest.protocol import TARGET, ROW_ID, months, history_hash, digest_json, require_history, require_identity
from src.backtest.decomposed import static_frame, MarketTrend
from src.pipeline.preprocess import _category_text
from src.features.build_features import ORIGINAL_COLUMNS

LEVELS = ('global', 'flat', 'town', 'street', 'building')
COMPONENTS = ((), ('FLAT_TYPE',), ('TOWN','FLAT_TYPE'), ('TOWN','STREET','FLAT_TYPE'), ('TOWN','BLOCK','STREET','FLAT_TYPE'))
RAW = [c for c in ORIGINAL_COLUMNS if c != 'BLOCK']


def keys(frame):
    text = {c: _category_text(frame[c], c).astype(str).reset_index(drop=True) for c in ('TOWN','BLOCK','STREET','FLAT_TYPE')}
    result = {}
    for level, components in zip(LEVELS, COMPONENTS):
        key = pd.Series('*', index=np.arange(len(frame)))
        valid = pd.Series(True, index=key.index)
        for c in components:
            if text[c].str.contains('\x1f', regex=False).any():
                raise ValueError('reserved key separator in source')
            key = key + '\x1f' + text[c]
            valid &= ~text[c].str.lower().isin(['', '__missing__', 'nan'])
        result[level] = (key, valid)
    return result


def bounded(history, cutoff):
    require_identity(history)
    result = history.loc[months(history) <= pd.Period(cutoff, freq='M')].reset_index(drop=True).copy()
    require_history(result, cutoff)
    return result


def hedonic_index(history, cutoff, config):
    """Same joint Ridge/month specification, with a new explicit CPU4 fit path."""
    require_history(history, cutoff)
    opts = config['hedonic']
    prepared = static_frame(history)
    numeric = []
    for column in opts['hedonic_numeric']:
        values = pd.to_numeric(prepared[column], errors='coerce').to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        values = np.where(np.isfinite(values), values, fill)
        numeric.append((values-values.mean())/(values.std() or 1.0))
    categories = pd.DataFrame({c: _category_text(prepared[c], c).to_numpy() for c in opts['hedonic_categorical']})
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
    prop = sparse.hstack([enc.fit_transform(categories), sparse.csr_matrix(np.column_stack(numeric))], format='csr')
    month = months(history).astype(str)
    menc = OneHotEncoder(handle_unknown='error', sparse_output=True, dtype=np.float64)
    temporal = menc.fit_transform(np.asarray(month).reshape(-1,1))
    design = sparse.hstack([prop,temporal], format='csr')
    model = Ridge(alpha=opts['ridge_alpha'], solver=opts['ridge_solver'], tol=opts['ridge_tol'], max_iter=opts['ridge_max_iter'], fit_intercept=True)
    with threadpool_limits(limits=config['cpu_threads']):
        model.fit(design, history[TARGET].to_numpy(dtype=float))
    labels = menc.categories_[0]
    counts = pd.Series(month).value_counts().reindex(labels).to_numpy(dtype=float)
    raw = model.coef_[prop.shape[1]:]
    center = float(np.average(raw, weights=counts))
    index = pd.DataFrame({'month':labels,'count':counts,'market_effect':raw-center})
    recon = np.asarray(prop @ model.coef_[:prop.shape[1]]).reshape(-1)+float(model.intercept_)+center+(raw-center)[temporal.indices]
    gap = float(np.max(np.abs(model.predict(design)-recon)))
    if gap > 1e-7:
        raise ValueError('hedonic centering changed reconstruction')
    return index, {'cpu_threads':config['cpu_threads'],'reconstruction_max_abs_error':gap,
                   'ridge_iterations':np.asarray(model.n_iter_).tolist(),'design_shape':list(design.shape)}


@dataclass
class Snapshot:
    cutoff: str
    config: dict

    def fit(self, history):
        require_history(history, self.cutoff)
        self.allowed_history_sha256 = history_hash(history)
        self.index, self.hedonic_audit = hedonic_index(history,self.cutoff,self.config)
        self.trend = MarketTrend('damped',self.cutoff,self.config['trend']).fit(self.index)
        rent = history[TARGET].to_numpy(dtype=float)
        market = self.index.set_index('month').market_effect.reindex(months(history).astype(str)).to_numpy(dtype=float)
        values = {'raw':rent, 'adjusted':rent-market}
        self.fallback = {k:float(v.mean()) for k,v in values.items()}
        self.tables = {}
        all_keys = keys(history)
        ordinal = months(history).asi8
        now = pd.Period(self.cutoff,freq='M').ordinal
        for window in self.config['windows']:
            use = np.ones(len(history),dtype=bool) if window == 'all' else ordinal > now-int(window)
            for level,(key,valid) in all_keys.items():
                mask = use & valid.to_numpy()
                table = pd.DataFrame({'key':key.to_numpy()[mask], 'raw':rent[mask], 'adjusted':values['adjusted'][mask], 'month':ordinal[mask]})
                self.tables[(str(window),level)] = table.groupby('key',sort=True).agg(raw=('raw','sum'),adjusted=('adjusted','sum'),n=('raw','size'),last=('month','max'))
        return self

    def transform(self, query, offsets):
        if TARGET in query:
            raise ValueError('snapshot query must not contain truth')
        require_identity(query)
        offsets = np.asarray(offsets)
        actual = months(query).asi8-pd.Period(self.cutoff,freq='M').ordinal
        if not np.issubdtype(offsets.dtype,np.integer) or not np.array_equal(offsets,actual) or ((offsets<1)|(offsets>16)).any():
            raise ValueError('query date/horizon disagrees with snapshot cutoff')
        keyset = keys(query)
        frames = {name:query[RAW].reset_index(drop=True).copy() for name in ('raw','adjusted')}
        for frame in frames.values():
            frame['horizon'] = offsets.astype(np.float32)
        anchors = {}
        now = pd.Period(self.cutoff,freq='M').ordinal
        for window in self.config['windows']:
            parents = {name:np.full(len(query),self.fallback[name]) for name in frames}
            for level,(key,valid) in keyset.items():
                table = self.tables[(str(window),level)].reindex(key)
                n = table.n.fillna(0).to_numpy(dtype=float)
                n = np.where(valid.to_numpy(),n,0)
                known = n>0
                age = np.where(known,now-table['last'].fillna(now+1).to_numpy(dtype=float),-1)
                for name,frame in frames.items():
                    sums = np.where(known,table[name].fillna(0).to_numpy(dtype=float),0)
                    if level == 'global':
                        price = np.divide(sums,n,out=parents[name].copy(),where=known)
                    else:
                        price = (sums+self.config['strength']*parents[name])/(n+self.config['strength'])
                    prefix = f'{window}_{level}'
                    frame[f'price_{prefix}'] = price.astype(np.float32)
                    frame[f'count_{prefix}'] = n.astype(np.float32)
                    frame[f'recency_{prefix}'] = age.astype(np.float32)
                    frame[f'known_{prefix}'] = known.astype(np.float32)
                    parents[name] = price
                    if window == 12 and level == 'building':
                        anchors[name] = price
        bases = {'raw':anchors['raw'], 'adjusted':anchors['adjusted']+self.trend.predict_horizon(offsets)}
        if list(frames['raw']) != list(frames['adjusted']):
            raise ValueError('B1/B2 schemas differ')
        return frames,bases


class SnapshotStore:
    def __init__(self, history, config, path=None, namespace='fixture', progress=None):
        # Own an immutable-by-convention source snapshot: later caller edits
        # cannot invalidate a previously verified cutoff in this store.
        self._history,self._config=history.copy(deep=True),copy.deepcopy(config)
        self.path,self.namespace,self.progress=Path(path) if path else None,namespace,progress
        self.memory = {}
        self._cutoffs = {}

    def get(self, cutoff):
        if cutoff in self._cutoffs:
            return self.memory[self._cutoffs[cutoff]]
        hist = bounded(self._history,cutoff)
        contract = {'cutoff':cutoff,'labels':history_hash(hist),'predictors':frame_hash(hist.drop(columns=TARGET)),
                    'config':self._config,'namespace':self.namespace}
        pin = digest_json(contract)
        if pin in self.memory:
            return self.memory[pin]
        directory = self.path/pin if self.path else None
        if directory and (directory/'complete.json').exists():
            receipt=read_json(directory/'complete.json')
            if receipt['contract'] != contract or receipt['sha256'] != sha256(directory/'snapshot.joblib'):
                raise ValueError('snapshot cache contract/checksum mismatch')
            snapshot = joblib.load(directory/'snapshot.joblib')
        else:
            if self.progress:
                self.progress('snapshot_fit',snapshot=cutoff,history_rows=len(hist))
            snapshot = Snapshot(cutoff,copy.deepcopy(self._config)).fit(hist)
            if directory:
                directory.mkdir(parents=True,exist_ok=True)
                joblib.dump(snapshot,directory/'snapshot.joblib')
                atomic_json(directory/'complete.json',{'contract':contract,'sha256':sha256(directory/'snapshot.joblib'),
                            'hedonic_audit':snapshot.hedonic_audit,'trend':snapshot.trend.state()})
        self.memory[pin]=snapshot
        self._cutoffs[cutoff]=pin
        return snapshot


def training_rows(history,cutoff,config,store):
    hist=bounded(history,cutoff)
    periods=months(hist)
    pieces={k:[] for k in ('raw','adjusted')}
    metadata=[]
    start=pd.Period(config['snapshot_start'],freq='M').ordinal
    # Target row t is always after its snapshot s; same-month labels cannot enter.
    for h in config['snapshot_horizons']:
        if store.progress:
            store.progress('snapshot_training_rows',horizon=h,fit_cutoff=cutoff)
        for target_month in sorted(periods.unique()):
            s=target_month-h
            if s.ordinal<start:
                continue
            subset=hist.loc[periods==target_month].copy()
            query=subset.drop(columns=TARGET)
            snapshot=store.get(str(s))
            matrices,bases=snapshot.transform(query,np.full(len(query),h,dtype=int))
            for name in pieces:
                pieces[name].append(matrices[name])
            metadata.append(pd.DataFrame({'row_id':subset.row_id.to_numpy(),'target_month':str(target_month),'snapshot':str(s),
                                          'horizon':h,'target':subset[TARGET].to_numpy(dtype=float),
                                          'raw_base':bases['raw'],'adjusted_base':bases['adjusted']}))
    if not metadata:
        raise ValueError('no eligible past-only snapshot examples')
    meta=pd.concat(metadata,ignore_index=True)
    if meta.duplicated(['snapshot','horizon','row_id']).any():
        raise ValueError('duplicate snapshot training identity')
    counts=meta.horizon.value_counts()
    meta['weight']=len(meta)/(len(counts)*meta.horizon.map(counts).to_numpy(dtype=float))
    matrices={k:pd.concat(v,ignore_index=True) for k,v in pieces.items()}
    audit={'fit_cutoff':cutoff,'instances':len(meta),'unique_target_rows':int(meta.row_id.nunique()),
           'snapshot_count':int(meta.snapshot.nunique()),'per_horizon':{str(h):{'n':int(n),'unique_target_rows':int(meta.loc[meta.horizon==h,'row_id'].nunique()),
                       'snapshot_count':int(meta.loc[meta.horizon==h,'snapshot'].nunique()),'weight_sum':float(meta.loc[meta.horizon==h,'weight'].sum())} for h,n in counts.sort_index().items()},
           'weight_rule':'equal total weight per supported h; equal rows within h; mean weight one',
           'expected_horizons':config['snapshot_horizons'],'supported_horizons':sorted(counts.index.tolist()),
           'column_count_before_constant_omission':len(matrices['raw'].columns)}
    return matrices,meta,audit


def cached_training(history,cutoff,config,store,path,namespace):
    hist=bounded(history,cutoff)
    contract={'cutoff':cutoff,'labels':history_hash(hist),'predictors':frame_hash(hist.drop(columns=TARGET)), 'config':config,'namespace':namespace}
    directory=Path(path)/digest_json(contract)
    receipt=directory/'complete.json'
    if receipt.exists():
        saved=read_json(receipt)
        if saved['contract'] != contract or any(sha256(directory/k)!=v for k,v in saved['files'].items()):
            raise ValueError('training feature cache changed')
        return directory,saved['audit']
    matrices,meta,audit=training_rows(hist,cutoff,config,store)
    directory.mkdir(parents=True,exist_ok=True)
    for name,frame in matrices.items():
        frame.to_parquet(directory/f'{name}.parquet',index=False)
    meta.to_parquet(directory/'meta.parquet',index=False)
    atomic_json(receipt,{'contract':contract,'audit':audit,'files':{name:sha256(directory/name) for name in ('raw.parquet','adjusted.parquet','meta.parquet')}})
    del matrices,meta
    gc.collect()
    return directory,audit
