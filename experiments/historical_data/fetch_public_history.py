"""Public data retrieval only. No training, model loading, or prediction."""
from runtime import CODE_DIR, ARTIFACT_DIR
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
from pathlib import Path
import requests
OUT = ARTIFACT_DIR
DATASETS = {'hdb_rent_medians': 'd_23000a00c52996c55106084ed0339566', 'ura_stock_vacancy': 'd_01e3556fb916ca19a7e29fc39520fa78', 'ura_private_rent_index': 'd_647dabde09e726b9eeb75e4d9cd96699'}
PAGES = {'nas_search_home': 'https://www.nas.gov.sg/archivesonline/speeches/', 'ura_methodology': 'https://eservice.ura.gov.sg/reis/coverageandMethodology'}

def fetch_one(name, url):
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        extension = '.json' if 'json' in r.headers.get('Content-Type', '') else '.html'
        path = OUT / (name + extension)
        path.write_bytes(r.content)
        result = {'name': name, 'url': url, 'status': r.status_code, 'path': str(path), 'sha256': hashlib.sha256(r.content).hexdigest(), 'downloaded_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        if extension == '.json':
            data = r.json()
            table = data.get('result', {})
            result.update(success=data.get('success'), total=table.get('total'), rows=len(table.get('records', [])), fields=table.get('fields'))
        return result
    except Exception as e:
        return {'name': name, 'url': url, 'error': repr(e)}

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(name, 'https://data.gov.sg/api/action/datastore_search?resource_id=' + ident + '&limit=20000') for name, ident in DATASETS.items()] + list(PAGES.items())
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda args: fetch_one(*args), jobs))
    (OUT / 'download_ledger.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
