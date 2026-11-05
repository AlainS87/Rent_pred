"""Bounded JSON replacement retries; optional snapshots cannot abort training."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import json
import os
from pathlib import Path
import time

def write_json(path, value, *, required=True, attempts=5, pause=0.04, replace=None):
    path = Path(path)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    do_replace = os.replace if replace is None else replace
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        for attempt in range(attempts):
            try:
                do_replace(tmp, path)
                return True
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(pause)
    except OSError:
        if required:
            raise
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')

def self_check():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='spatial_status_io_', dir=Path(__file__).resolve().parent) as folder:
        target = Path(folder) / 'state.json'
        target.write_text('{"value": 0}', encoding='utf-8')
        count = 0

        def transient(src, dst):
            nonlocal count
            count += 1
            if count < 3:
                raise PermissionError('simulated transient sharing conflict')
            os.replace(src, dst)
        assert write_json(target, {'value': 1}, replace=transient, pause=0)
        assert count == 3 and json.loads(target.read_text())['value'] == 1

        def persistent(src, dst):
            raise PermissionError('simulated persistent sharing conflict')
        assert write_json(target, {'value': 2}, required=False, replace=persistent, pause=0) is False
        assert json.loads(target.read_text())['value'] == 1
        try:
            write_json(target, {'value': 3}, replace=persistent, pause=0)
        except PermissionError:
            pass
        else:
            raise AssertionError('Mandatory artifacts must still report failure')
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            api = ctypes.WinDLL('kernel32', use_last_error=True)
            api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            api.CreateFileW.restype = wintypes.HANDLE
            api.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = api.CreateFileW(str(target), 2147483648, 1 | 2, None, 3, 128, None)
            assert handle != wintypes.HANDLE(-1).value
            try:
                assert write_json(target, {'value': 4}, required=False, pause=0) is False
                assert json.loads(target.read_text())['value'] == 1
            finally:
                assert api.CloseHandle(handle)
            assert write_json(target, {'value': 5})
            assert json.loads(target.read_text())['value'] == 5
        assert not list(Path(folder).glob('*.tmp'))
    return {'transient_retry': True, 'locked_snapshot_nonfatal': True, 'required_artifact_failure_visible': True, 'original_file_preserved_when_locked': True, 'actual_windows_file_lock_test': os.name == 'nt', 'production_training_performed': False}
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(self_check()))
