"""Block-table attributes and ages derived from the rental approval date."""

import numpy as np
import pandas as pd

from src.features.keys import block_key, norm_text


def _nullable_int(values: pd.Series) -> pd.Series:
    return values.round().astype("Int64")


def add_block_columns(rentals: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Left-join block attributes onto rental rows and add age columns.

    ``MONTHLY_RENT`` is not read. The rental row count is unchanged.
    """
    lookup = blocks.copy()
    lookup["block_key"] = block_key(lookup["BLOCK"], lookup["STREET"])
    if not lookup["block_key"].is_unique:
        raise ValueError("block_key is not unique on the block table")

    year = pd.to_numeric(lookup["YEAR_COMPLETED"], errors="coerce")
    floor = pd.to_numeric(lookup["MAX_FLOOR"], errors="coerce")
    postal = norm_text(lookup["POSTAL_CODE"])
    lookup = pd.DataFrame(
        {
            "block_key": lookup["block_key"],
            "latitude": pd.to_numeric(lookup["LATITUDE"], errors="coerce"),
            "longitude": pd.to_numeric(lookup["LONGITUDE"], errors="coerce"),
            "postal_code": postal.mask(postal.eq("nil")),
            "max_floor": floor.mask(floor.eq(-1)),
            "year_completed": year.mask(year.eq(-1)),
            "subzone": norm_text(lookup["SUBZONE"]),
            "planning_area": norm_text(lookup["PLANNING_AREA"]),
            "region": norm_text(lookup["REGION"]),
        }
    )
    lookup["max_floor"] = _nullable_int(lookup["max_floor"])
    lookup["year_completed"] = _nullable_int(lookup["year_completed"])

    keyed = rentals.copy()
    keyed["block_key"] = block_key(keyed["BLOCK"], keyed["STREET"])
    merged = keyed.merge(lookup, on="block_key", how="left", validate="many_to_one")
    if len(merged) != len(rentals):
        raise ValueError("block join changed the rental row count")

    approval_year = merged["RENT_APPROVAL_DATE"].astype("string").str.slice(0, 4)
    approval_year = pd.to_numeric(approval_year, errors="coerce").astype("Int64")
    lease_year = pd.to_numeric(merged["LEASE_COMMENCE_DATE"], errors="coerce").astype("Int64")
    completed = merged["year_completed"]

    merged["building_age_years"] = approval_year - completed
    merged["lease_vs_completion_gap"] = lease_year - completed
    merged["flat_age_from_lease"] = approval_year - lease_year

    central = merged["region"].eq("central region")
    merged["is_central_region"] = central.astype("Int64")
    merged.loc[merged["region"].isna(), "is_central_region"] = pd.NA
    age = merged["building_age_years"]
    product = age.astype("Float64") * merged["is_central_region"].astype("Float64")
    merged["building_age_x_central"] = product.astype("Float64")
    merged.loc[age.isna() | merged["is_central_region"].isna(), "building_age_x_central"] = np.nan
    return merged
