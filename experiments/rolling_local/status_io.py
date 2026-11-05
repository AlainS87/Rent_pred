"""Recoverable status records; immutable fallbacks and strict critical writes."""
import json,os,time,uuid
from pathlib import Path

def _encoded(payload,default=None):
    return json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False,default=default)+'\n'

def append_record(path,payload,default=None):
    # Serialization and append failures are intentionally never suppressed.
    line=json.dumps(payload,ensure_ascii=False,allow_nan=False,default=default)+'\n'
    with Path(path).open('a',encoding='utf-8') as handle:
        handle.write(line);handle.flush()

def write_critical(path,payload,default=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=_encoded(payload,default)
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with temp.open('x',encoding='utf-8') as handle:
        handle.write(text);handle.flush()
    os.replace(temp,path)

def write_status(path,payload,*,deadline_epoch=None,default=None,warning_path=None,run_id='initial'):
    """Only Windows sharing/access failures replacing a status file may degrade."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=_encoded(payload,default)
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with temp.open('x',encoding='utf-8') as handle:
        handle.write(text);handle.flush()
    error=None
    for delay in (0.0,0.01,0.03,0.06,0.10):
        if delay and (deadline_epoch is None or time.time()+delay<deadline_epoch): time.sleep(delay)
        try:
            os.replace(temp,path)
            return {'status_write':'primary','path':str(path)}
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33): raise
            error=exc
        if deadline_epoch is not None and time.time()>=deadline_epoch: break
    fallback=path.with_name(path.stem+f'.fallback.{run_id}.{time.time_ns()}.{os.getpid()}.json')
    # A new, immutable fallback has no pre-existing target handle to conflict.
    # Failure even here remains fatal: this is not permission/disk-error hiding.
    os.replace(temp,fallback)
    record={'stage':'status_fallback','epoch':time.time(),'run_id':run_id,
            'primary':str(path),'fallback':str(fallback),'error_type':type(error).__name__,
            'error':str(error),'training_retry':False}
    if warning_path is not None: append_record(warning_path,record)
    return {'status_write':'fallback','path':str(fallback),'warning':record}

def read_json_if_ready(path):
    try:
        with Path(path).open('r',encoding='utf-8') as handle: return json.load(handle)
    except (FileNotFoundError,PermissionError,json.JSONDecodeError): return None
