"""Retrieve already-discovered public sources. Does not train or load models."""
from runtime import CODE_DIR, ARTIFACT_DIR
import concurrent.futures as cf
import hashlib
import json
import subprocess
from pathlib import Path
import requests
OUT = ARTIFACT_DIR

def fetch(kind, item):
    name = kind + '_' + item['quarter'].replace('-', '_')
    path = OUT / (name + '.pdf')
    result = {**item, 'kind': kind}
    try:
        r = requests.get(item['url'], timeout=25)
        r.raise_for_status()
        if not r.content.startswith(b'%PDF'):
            raise ValueError('Response was not a PDF; status=' + str(r.status_code))
        path.write_bytes(r.content)
        text_path = path.with_suffix('.txt')
        subprocess.run(['pdftotext', '-layout', str(path), str(text_path)], check=True, capture_output=True)
        result.update(downloaded=True, sha256=hashlib.sha256(r.content).hexdigest(), path=str(path), text_path=str(text_path))
    except Exception as e:
        result.update(downloaded=False, error=repr(e))
    return result

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES = json.loads((CODE_DIR / 'sources.json').read_text(encoding='utf-8'))
    jobs = [('hdb', s) for s in SOURCES['hdb_dated_releases']] + [('ura', SOURCES['ura_dated_release'])]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda args: fetch(*args), jobs))
    (OUT / 'release_download_ledger.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k in ['kind', 'quarter', 'downloaded', 'error']} for r in results]))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
