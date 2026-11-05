"""Data availability, publication and integrity audit; no fitting or prediction."""
from runtime import CODE_DIR, ARTIFACT_DIR

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import hashlib
    import json
    import re
    import pandas as pd
    OUT = ARTIFACT_DIR
    ROOT = OUT.parent
    sources = json.loads((CODE_DIR / 'sources.json').read_text(encoding='utf-8'))
    raw = pd.DataFrame(json.loads((OUT / 'hdb_rent_medians.json').read_text())['result']['records'])
    h = raw[raw.quarter <= '2024-Q2'].copy()
    h['rent'] = pd.to_numeric(h.median_rent.str.replace(',', ''), errors='coerce')
    key = ['quarter', 'town', 'flat_type']
    dups = h[h.duplicated(key, keep=False)].copy()
    dups.to_csv(OUT / 'hdb_duplicate_records.csv', index=False)
    expected = [str(q).replace('Q', '-Q') for q in pd.period_range('2005Q2', '2020Q4', freq='Q')]
    missing = sorted(set(expected) - set(h.quarter))
    numeric_conflicts = h.groupby(key).rent.nunique()
    conflicts = numeric_conflicts[numeric_conflicts > 1]
    reports = []
    rows = []
    room_types = ['1-RM', '2-RM', '3-RM', '4-RM', '5-RM', 'EXEC']
    for src in sources['hdb_dated_releases']:
        path = OUT / ('hdb_' + src['quarter'].replace('-', '_') + '.txt')
        pages = path.read_text(encoding='utf-8').split('\x0c')
        matches = [p for p in pages if re.search('Median (?:Subletting )?Rents by Town', p, re.I)]
        assert len(matches) == 1, (src['quarter'], len(matches))
        page = matches[0]
        release_date = pd.Timestamp(src['release_date']).strftime('%d %B %Y').lstrip('0')
        assert release_date in ' '.join(pages[0].split()), (src['quarter'], release_date)
        extracted = []
        for line in page.splitlines():
            m = re.match('^\\s*([A-Z][A-Z /]+?)\\s{2,}(.+)$', line)
            if not m or m.group(1).strip() in ['TOWN', 'TOWNS', 'OVERALL']:
                continue
            vals = re.findall('\\$[\\d,]+\\*?|(?<!\\S)[*-](?!\\S)', m.group(2))
            if len(vals) != 6:
                continue
            town = re.sub('\\s*/\\s*', '/', m.group(1).strip())
            for room, value in zip(room_types, vals):
                number = float(value.replace('$', '').replace(',', '').replace('*', '')) if '$' in value else None
                extracted.append({'quarter': src['quarter'], 'release_date': src['release_date'], 'town': town, 'flat_type': room, 'reported_cell': value, 'rent': number, 'small_sample_flag': '*' in value, 'source_url': src['url']})
        assert len(extracted) == 156, (src['quarter'], len(extracted))
        rows.extend(extracted)
        comparisons = []
        for r in extracted:
            if r['rent'] is None or r['small_sample_flag']:
                continue
            x = h[(h.quarter == r['quarter']) & (h.town.str.strip().str.replace(' / ', '/') == r['town']) & (h.flat_type == r['flat_type'])]
            values = x.rent.dropna().tolist()
            comparisons.append({'town': r['town'], 'flat_type': r['flat_type'], 'release_value': r['rent'], 'current_table_values': values, 'match': values == [r['rent']]})
        reports.append({'quarter': src['quarter'], 'release_date': src['release_date'], 'date_verified_in_document': True, 'rental_cells': len(extracted), 'eligible_numeric_comparisons': len(comparisons), 'mismatches': [c for c in comparisons if not c['match']]})
    pd.DataFrame(rows).to_csv(OUT / 'hdb_release_vintage_cells.csv', index=False)
    quarter_vintage = [r['quarter'] for r in reports]
    longest = []
    run = []
    for q in expected:
        if q in quarter_vintage:
            run.append(q)
            if len(run) > len(longest):
                longest = run.copy()
        else:
            run = []
    result = {'task': 'Earlier rental and vacancy history with verifiable publication dates; feasibility only', 'training_performed': False, 'models_loaded': False, 'new_predictions': False, 'search_evaluated': False, 'confirm_evaluated': False, 'baseline_or_split_or_labels_changed': False, 'hdb_latest_vintage': {'source': 'https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view', 'download_total_rows': len(raw), 'api_reported_rows': json.loads((OUT / 'hdb_rent_medians.json').read_text())['result']['total'], 'analyzed_through': '2024-Q2', 'pre2021_expected_quarters': len(expected), 'pre2021_present_quarters': h[h.quarter < '2021-Q1'].quarter.nunique(), 'pre2021_missing_quarter_labels': missing, 'duplicate_extra_records_through_cutoff': int(h.duplicated(key).sum()), 'numeric_conflicting_keys': len(conflicts), 'conflict_quarters': sorted(set((q for q, t, r in conflicts.index))), 'duplicate_rows_by_quarter': dups.groupby('quarter').size().to_dict(), 'automatic_averaging_or_relabeling_performed': False, 'all_historical_release_dates_verified': False, 'note': 'Current table is a latest historical vintage; creation date and current update date do not certify original quarterly availability.'}, 'hdb_dated_release_audit': reports, 'hdb_verified_release_quarters': len(reports), 'hdb_original_release_numeric_values_compared': sum((r['eligible_numeric_comparisons'] for r in reports)), 'hdb_current_table_numeric_disagreements': sum((bool(m['current_table_values']) for r in reports for m in r['mismatches'])), 'hdb_original_numeric_cells_currently_missing_or_suppressed': sum((not m['current_table_values'] for r in reports for m in r['mismatches'])), 'town_name_alignment': 'Trimmed surrounding whitespace for matching only, including QUEENSTOWN; raw downloads preserved.', 'hdb_longest_contiguous_verified_release_run': longest, 'ura_history': {'source': 'https://data.gov.sg/datasets/d_01e3556fb916ca19a7e29fc39520fa78/view', 'series_start': '1988-Q1', 'scope': 'completed private residential units excluding HDB and EC', 'new_pre2021_current_quarter_release_verified': ['2016-Q1'], 'original_release_pair_verified': '2016-Q1, both HDB and URA released 2016-04-22', 'continuous_pre2021_vintages_verified': False, 'vacancy_rate_definition_break': '2016-Q4: sample to all completed private units', 'private_rent_index_method_break': '2015-Q1: stratified hedonic method', 'latest_private_rent_table_start': '1990-Q1'}, 'other_measurement_issues': ['2009-Q1 HDB original PDF published small-sample values marked <10 transactions; 2016 PDFs suppress values for <20. Do not treat missing values as zero or fill them with neighbouring rents.', 'HDB town/type medians do not control exact property attributes and are not the existing property-adjusted monthly index.', 'Current public individual-rental dataset starts Jan 2021; no compatible earlier row-level history was verified.', 'SRX dated monthly releases exist at least from Jan 2015 reporting Dec 2014, but earlier months are revised in later reports.', 'SRX Dec 2022 changed both underlying data source and definition of month, in addition to rebasing; not a seamless extension of the existing target.'], 'decision': {'earlier_values_retrievable': True, 'some_original_publication_dates_and_values_verified': True, 'complete_continuous_release_certified_history_ready': False, 'sufficient_to_restart_vacancy_candidate': False, 'predictive_improvement_established': False, 'recommendation': 'Do not restart the current vacancy-change correction candidate. Retire it from the next experiment queue on current evidence; retain downloaded historical data for possible separately approved research.', 'reason': 'Longer aggregate/proxy histories do not repair the lack of early compatible property-index calibration targets. Publication-vintage gaps, actual source-table anomalies and definition breaks remain. No evidence yet demonstrates stable incremental predictive value across cycles.', 'not_claimed': 'This audit does not establish that vacancy has no economic relationship with HDB rent, nor that missing official releases do not exist.', 'reopen_only_if': 'A continuous publication-dated vintage panel is reconstructed, definition breaks handled explicitly, and a bridge to the existing HDB target plus distinct-cycle predictive checks is predeclared and passes. No training or further experiment is automatically authorized.'}}
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'verified_hdb_quarters': len(reports), 'longest_run': longest, 'numeric_comparisons': sum((r['eligible_numeric_comparisons'] for r in reports)), 'mismatches': {r['quarter']: len(r['mismatches']) for r in reports}, 'missing_quarters': missing, 'numeric_conflicts': len(conflicts)}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
