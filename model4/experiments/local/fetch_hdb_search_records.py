"""Read public HDB records for the existing Search months only, for provenance."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request
ROOT = Path(__file__).resolve().parent
OUT = asset_path(Path('feature_ablation_01/diagnosis'))
DATASET = 'd_c9f57187485a850908655db0e8cfe651'

def fetch(month):
    params = {'resource_id': DATASET, 'limit': 10000, 'filters': json.dumps({'rent_approval_date': month})}
    url = 'https://data.gov.sg/api/action/datastore_search?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'Rental-data-provenance-audit/1.0'})
    with urllib.request.urlopen(req, timeout=25) as response:
        payload = json.load(response)
    assert payload['success'] is True
    result = payload['result']
    assert len(result['records']) == result['total'], 'Incomplete response: no automatic pagination or retry'
    assert all((row['rent_approval_date'] == month for row in result['records']))
    saved = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'url': url, 'purpose': 'Provenance audit only; never training, prediction, label replacement or Confirm evaluation', 'response': payload}
    (OUT / f'hdb_public_{month}.json').write_text(json.dumps(saved, indent=2) + '\n', encoding='utf-8')
    return {'month': month, 'rows': result['total'], 'path': str(OUT / f'hdb_public_{month}.json')}
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        print(json.dumps(list(pool.map(fetch, ['2024-10', '2024-11', '2024-12'])), indent=2))
