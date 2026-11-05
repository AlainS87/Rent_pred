#!/usr/bin/env python3
"""Diagnostic comparison of Kaggle HDB rent predictions with public HDB records.

Public records do not identify individual flats within a block. This script
explicitly separates single-candidate matches from ambiguous group matches.
Results are NOT an official Kaggle leaderboard evaluation.
"""

import argparse
import json
import re
import time
from pathlib import Path
from hdb_rent.data import DEFAULT_DATA_DIR, OUTPUT_DIR

import numpy as np
import pandas as pd

DATASET_ID = "d_c9f57187485a850908655db0e8cfe651"
API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
KEYS = ["date", "town", "block", "street", "flat_type"]

# Unify the expanded street words in Kaggle with short forms in HDB's CSV.
STREET_WORDS = {
    "avenue": "ave", "ave": "ave", "road": "rd", "rd": "rd",
    "street": "st", "st": "st", "drive": "dr", "dr": "dr",
    "crescent": "cres", "cres": "cres", "close": "cl", "cl": "cl",
    "north": "nth", "nth": "nth", "south": "sth", "sth": "sth",
    "central": "ctrl", "ctrl": "ctrl", "lorong": "lor", "lor": "lor",
    "jalan": "jln", "jln": "jln", "upper": "upp", "upp": "upp",
    "place": "pl", "pl": "pl", "terrace": "ter", "ter": "ter",
    "boulevard": "blvd", "blvd": "blvd", "mount": "mt", "mt": "mt",
    "heights": "hts", "hts": "hts", "circuit": "cct", "cct": "cct",
    "garden": "gdn", "gardens": "gdns", "gdn": "gdn", "gdns": "gdns",
    "junction": "jn", "jn": "jn", "square": "sq", "sq": "sq",
}


def normal_text(s):
    return (s.fillna("").astype(str).str.lower().str.strip()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def normal_street(s):
    v = normal_text(s)
    # Vectorised replacement: replace whole street-name tokens only.
    pat = r"\b(" + "|".join(re.escape(x) for x in STREET_WORDS) + r")\b"
    return v.str.replace(pat, lambda m: STREET_WORDS[m.group()], regex=True)


def normal_date(s):
    raw = s.fillna("").astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%Y-%m", errors="coerce")
    if dt.isna().any():
        examples = raw[dt.isna()].head(3).tolist()
        raise ValueError("Invalid YYYY-MM date value(s): " + repr(examples))
    return dt.dt.strftime("%Y-%m")


def normal_flat_type(s):
    v = normal_text(s)
    v = v.str.replace(r"^(\d+)\s*rooms?$", r"\1 room", regex=True)
    v = v.str.replace(r"^multi\s*generation$", "multi generation", regex=True)
    return v


def standardise_table(df, official=False):
    columns = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c
               for c in df.columns}
    want = {
        "date": ["rent_approval_date", "rent_approval_date_month_yyyy_mm"],
        "town": ["town"], "block": ["block"],
        "street": ["street_name", "street"] if official else ["street", "street_name"],
        "flat_type": ["flat_type"],
    }
    if official:
        want["rent"] = ["monthly_rent"]
    result = pd.DataFrame(index=df.index)
    for dst, aliases in want.items():
        found = next((columns[a] for a in aliases if a in columns), None)
        if found is None:
            raise ValueError(f"Missing '{dst}' column. Found: {list(df.columns)}")
        result[dst] = df[found]
    result["date"] = normal_date(result["date"])
    for k in ("town", "block"):
        result[k] = normal_text(result[k])
    result["street"] = normal_street(result["street"])
    result["flat_type"] = normal_flat_type(result["flat_type"])
    if official:
        result["rent"] = pd.to_numeric(result["rent"], errors="coerce")
        result = result.dropna(subset=["rent"]).copy()
        if (result["rent"] <= 0).any():
            raise ValueError("Non-positive rent found in official data")
    return result


def download_official(path):
    """Get the *complete* official CSV via the documented two-step API."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Download mode requires: pip install requests") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    base = f"{API_BASE}/{DATASET_ID}"
    session = requests.Session()

    def get_json(url):
        last_error = None
        for attempt in range(5):
            try:
                res = session.get(url, timeout=60)
                if res.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {res.status_code}"
                    time.sleep(min(3 * 2 ** attempt, 30))
                    continue
                res.raise_for_status()
                obj = res.json()
                if obj.get("code") not in (None, 0):
                    raise RuntimeError(f"Official API error: {obj}")
                return obj
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = str(exc)
                time.sleep(min(3 * 2 ** attempt, 30))
        raise RuntimeError(f"Official API unreachable ({last_error}). Download CSV manually instead.")

    print("Initiating official HDB CSV download ...", flush=True)
    get_json(f"{base}/initiate-download")
    download_url = None
    for _ in range(30):
        obj = get_json(f"{base}/poll-download")
        data = obj.get("data") or {}
        download_url = data.get("url") or data.get("downloadUrl") or obj.get("url")
        if download_url:
            break
        time.sleep(2)
    if not download_url:
        raise RuntimeError("Official download link not ready. Download CSV manually from data.gov.sg.")

    print("Downloading official CSV ...", flush=True)
    with session.get(download_url, timeout=(30, 180), stream=True) as resp:
        resp.raise_for_status()
        temp = path.with_name(path.name + ".part")
        with temp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        if temp.stat().st_size < 100:
            temp.unlink(missing_ok=True)
            raise RuntimeError("Download returned an empty/unexpectedly short CSV")
        temp.replace(path)
    print(f"Downloaded {path} ({path.stat().st_size:,} bytes)", flush=True)


def load_predictions(path, expected_rows):
    d = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in d.columns}
    prediction_col = next((cols[c] for c in ("monthly_rent", "prediction", "predicted", "rent") if c in cols), None)
    if prediction_col is None and len(d.columns) == 1:
        prediction_col = d.columns[0]
    if prediction_col is None:
        raise ValueError(f"Cannot identify prediction column in {path}. Columns: {list(d.columns)}")
    if len(d) != expected_rows:
        raise ValueError(f"Prediction rows={len(d):,}, test rows={expected_rows:,}; order/length must match")
    pred = pd.to_numeric(d[prediction_col], errors="coerce")
    if not np.isfinite(pred.to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain NaN or infinite values")
    return pred.reset_index(drop=True)


def remaining_official(official, train=None):
    """Remove known training transactions, preserving multiplicities per key + rent.

    This can disambiguate overlapping months (e.g. 2025-03). It cannot assign
    distinct rents to indistinguishable test rows.
    """
    price_keys = KEYS + ["rent"]
    official_counts = (official.groupby(price_keys, dropna=False).size()
                       .rename("official_n").reset_index())
    if train is None:
        official_counts["remaining_n"] = official_counts["official_n"]
        return official_counts, {"train_rows_subtracted": 0, "train_rows_not_found_in_official": 0}
    train_counts = (train.groupby(price_keys, dropna=False).size()
                    .rename("train_n").reset_index())
    counts = official_counts.merge(train_counts, on=price_keys, how="outer")
    counts[["official_n", "train_n"]] = counts[["official_n", "train_n"]].fillna(0).astype(int)
    counts["remaining_n"] = (counts["official_n"] - counts["train_n"]).clip(lower=0)
    used = np.minimum(counts["official_n"], counts["train_n"])
    stats = {
        "train_rows_subtracted": int(used.sum()),
        "train_rows_not_found_in_official": int((counts["train_n"] - used).sum()),
    }
    return counts[counts["remaining_n"] > 0].copy(), stats


def make_reference(test, remaining):
    active = remaining[remaining["remaining_n"] > 0].copy()
    # Weighted summaries (no expansion of the source's duplicate records).
    active["weighted_rent"] = active["rent"] * active["remaining_n"]
    grouped = active.groupby(KEYS, sort=False, dropna=False)
    refs = grouped.agg(
        official_records=("remaining_n", "sum"),
        different_rents=("rent", "size"),
        official_mean_sum=("weighted_rent", "sum"),
        official_min=("rent", "min"),
        official_max=("rent", "max"),
    ).reset_index()
    refs["official_mean"] = refs["official_mean_sum"] / refs["official_records"]
    refs = refs.drop(columns="official_mean_sum")

    test_counts = (test.groupby(KEYS, dropna=False).size()
                   .rename("test_records_in_group").reset_index())
    refs = refs.merge(test_counts, how="left", on=KEYS, validate="1:1")
    out = test.merge(refs, how="left", on=KEYS, sort=False, validate="m:1")
    assert len(out) == len(test)
    out["match_type"] = "unmatched"
    matched = out["official_records"].notna()
    out.loc[matched & out["different_rents"].gt(1), "match_type"] = "ambiguous_multiple_rents"
    out.loc[matched & out["different_rents"].eq(1), "match_type"] = "one_possible_rent"
    strong = (matched & out["official_records"].eq(1)
              & out["test_records_in_group"].eq(1))
    out.loc[strong, "match_type"] = "single_candidate_pair"
    # A candidate price is not a verified Kaggle label, even in strong matches.
    return out, active


def compute_metric(y, pred):
    if len(pred) == 0:
        return {"rows": 0, "rmse": None, "mae": None, "bias_pred_minus_reference": None}
    err = pred.astype(float).to_numpy() - y.astype(float).to_numpy()
    return {"rows": int(len(err)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias_pred_minus_reference": float(np.mean(err))}


def score_and_save(test_original, test, pred, official, train, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining, subtraction = remaining_official(official, train)
    matched, active = make_reference(test, remaining)
    matched["prediction"] = pred.to_numpy()
    matched["squared_error_vs_mean"] = (matched["prediction"] - matched["official_mean"]) ** 2

    high = matched["match_type"] == "single_candidate_pair"
    one_rent = matched["match_type"].isin(["single_candidate_pair", "one_possible_rent"])
    any_matched = matched["match_type"] != "unmatched"
    ambiguous = matched["match_type"] == "ambiguous_multiple_rents"

    def metric(mask):
        return compute_metric(matched.loc[mask, "official_mean"], matched.loc[mask, "prediction"])

    summary = {
        "notice": "Diagnostic HDB public-record match; not the Kaggle leaderboard RMSE. Public records omit flat identifiers.",
        "test_rows": int(len(matched)), "official_input_rows": int(len(official)),
        "official_rows_after_training_subtraction": int(active["remaining_n"].sum()),
        "test_date_start": str(test["date"].min()), "test_date_end": str(test["date"].max()),
        "official_date_start": str(official["date"].min()), "official_date_end": str(official["date"].max()),
        "match_counts": {k: int(v) for k, v in matched["match_type"].value_counts().to_dict().items()},
        "coverage_any_official_group": float(any_matched.mean()),
        "coverage_single_candidate_pair": float(high.mean()),
        "training_subtraction": subtraction,
        "single_candidate_pair_vs_official": metric(high),
        "one_possible_rent_vs_official": metric(one_rent),
        "all_matched_vs_group_mean_PROXY_NOT_TRUE_RMSE": metric(any_matched),
        "ambiguous_groups_vs_group_mean_PROXY_NOT_TRUE_RMSE": metric(ambiguous),
    }

    # All rows, in the original Kaggle test order. No public label is used to change predictions.
    detail = test_original.copy().reset_index(drop=True)
    detail.insert(0, "test_row_id", np.arange(len(detail)))
    info_cols = ["prediction", "match_type", "official_records", "different_rents",
                 "test_records_in_group", "official_mean", "official_min", "official_max",
                 "squared_error_vs_mean"]
    detail = pd.concat([detail, matched[info_cols].reset_index(drop=True)], axis=1)
    detail.to_csv(out_dir / "official_match_details.csv", index=False)

    month_rows = []
    for month, group in matched.groupby("date", sort=True):
        ok = group["match_type"] == "single_candidate_pair"
        result = compute_metric(group.loc[ok, "official_mean"], group.loc[ok, "prediction"])
        month_rows.append({"month": month, "test_rows": len(group),
                           "matched_rows": int((group["match_type"] != "unmatched").sum()),
                           "single_candidate_rows": int(ok.sum()),
                           "single_candidate_rmse": result["rmse"],
                           "single_candidate_mae": result["mae"]})
    pd.DataFrame(month_rows).to_csv(out_dir / "official_by_month.csv", index=False)
    with (out_dir / "official_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--test-file", type=Path, default=DEFAULT_DATA_DIR / "test.csv")
    p.add_argument("--pred-file", type=Path, default=OUTPUT_DIR / "f1/submission_F1_1600.csv")
    p.add_argument("--official-csv", type=Path, default=OUTPUT_DIR / "official_data.csv")
    p.add_argument("--train-file", type=Path, default=None,
                   help="Optional: subtract matched known training rents in overlapping months")
    p.add_argument("--download", action="store_true", help="Fetch the official CSV from data.gov.sg")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "official_check")
    return p.parse_args()


def main():
    a = parse_args()
    if a.download:
        download_official(a.official_csv)
    if not a.official_csv.is_file():
        raise FileNotFoundError(f"Missing official CSV: {a.official_csv}. Use --download or download it manually.")
    for item in (a.test_file, a.pred_file):
        if not item.is_file():
            raise FileNotFoundError(f"Missing required CSV: {item}")
    test_original = pd.read_csv(a.test_file, dtype={"BLOCK": str})
    test = standardise_table(test_original)
    pred = load_predictions(a.pred_file, len(test))
    official = standardise_table(pd.read_csv(a.official_csv, dtype={"block": str}), official=True)
    train = None
    if a.train_file is not None:
        if not a.train_file.is_file():
            raise FileNotFoundError(f"Missing training CSV: {a.train_file}")
        train_df = pd.read_csv(a.train_file, dtype={"BLOCK": str})
        train = standardise_table(train_df)
        train["rent"] = pd.to_numeric(train_df["MONTHLY_RENT"], errors="raise")
    summary = score_and_save(test_original, test, pred, official, train, a.output_dir)
    print(f"Test rows: {summary['test_rows']:,}")
    print("Match types:")
    for label, count in summary["match_counts"].items():
        print(f"  {label}: {count:,}")
    score = summary["single_candidate_pair_vs_official"]
    if score["rmse"] is None:
        print("Single-candidate RMSE: N/A (no single-candidate pairs)")
    else:
        print(f"Single-candidate-vs-official RMSE: {score['rmse']:.4f} on {score['rows']:,} rows")
    proxy = summary["all_matched_vs_group_mean_PROXY_NOT_TRUE_RMSE"]
    if proxy["rmse"] is not None:
        print(f"Group-mean proxy RMSE (NOT test RMSE): {proxy['rmse']:.4f} on {proxy['rows']:,} rows")
    print(f"Wrote {a.output_dir / 'official_metrics.json'}")
    print(f"Wrote {a.output_dir / 'official_match_details.csv'}")
    print(f"Wrote {a.output_dir / 'official_by_month.csv'}")
    print("WARNING: These are public-record proxy metrics, NOT the private Kaggle test score.")


if __name__ == "__main__":
    main()
