"""Lagged equal-weight stock baskets. A month is used only when all three symbols exist."""

import numpy as np
import pandas as pd

BANK_SYMBOLS = ("D05.SI", "O39.SI", "U11.SI")
PROPERTY_SYMBOLS = ("C09.SI", "U14.SI", "9CI.SI")


def _month_end_closes(stocks: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    frame = stocks.loc[stocks["SYMBOL"].isin(symbols), ["SYMBOL", "DATE", "ADJUSTED_CLOSE"]].copy()
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    frame = frame.dropna(subset=["ADJUSTED_CLOSE"])
    frame["period"] = frame["DATE"].dt.to_period("M")
    frame = frame.sort_values("DATE")
    month_end = frame.groupby(["SYMBOL", "period"], as_index=False).tail(1)
    wide = month_end.pivot(index="period", columns="SYMBOL", values="ADJUSTED_CLOSE")
    wide = wide.reindex(columns=list(symbols))
    full_index = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    return wide.reindex(full_index)


def _basket_series(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    complete = wide.notna().all(axis=1)
    close = wide.mean(axis=1).where(complete)
    previous = wide.shift(1)
    previous_complete = previous.notna().all(axis=1)
    returns = (wide / previous - 1.0).mean(axis=1)
    returns = returns.where(complete & previous_complete)
    close.name = "close"
    returns.name = "return"
    return close, returns


def _map_lag(approval: pd.PeriodIndex, series: pd.Series) -> np.ndarray:
    lagged = series.reindex(approval - 1)
    return lagged.to_numpy(dtype=np.float64)


def add_stock_columns(rentals: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """Append one-month lagged basket closes and returns."""
    approval = pd.PeriodIndex(rentals["RENT_APPROVAL_DATE"].astype("string"), freq="M")
    out = rentals.copy()
    baskets = {
        "bank": BANK_SYMBOLS,
        "property": PROPERTY_SYMBOLS,
    }
    for name, symbols in baskets.items():
        close, returns = _basket_series(_month_end_closes(stocks, symbols))
        out[f"{name}_basket_adj_close_lag1"] = _map_lag(approval, close)
        out[f"{name}_basket_return_lag1"] = _map_lag(approval, returns)
    return out
