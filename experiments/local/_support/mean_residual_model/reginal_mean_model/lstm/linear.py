"""Internal numerical support selected by the local experiments."""
import pandas as pd

def _as_unit_rent(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work['MONTHLY_RENT'] = work['MONTHLY_RENT'] / work['floor_area_sqm']
    return work
