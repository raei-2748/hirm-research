"""Helper utilities for the Streamlit hedging dashboard."""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from hirm.diagnostics.efficiency import compute_er, compute_tr
from hirm.diagnostics.robustness import compute_vr, compute_wg
from hirm.envs.regime_labelling import price_series_volatility
from hirm.envs.regimes import REGIME_NAMES, map_vol_to_regime

PRICE_COLUMN_CANDIDATES = [
    "Adj Close",
    "Adj_Close",
    "AdjClose",
    "Close",
    "close",
    "Price",
    "price",
    "value",
]
DATE_COLUMN_CANDIDATES = ["Date", "date", "Timestamp", "timestamp"]
REGIME_OPTIONS = ["All"] + [name.title() for name in REGIME_NAMES.values()]


@dataclass
class HedgingResult:
    """Container returned by :func:`run_hedging_simulation`."""

    pnl_series: pd.Series
    cumulative_pnl: pd.Series
    metrics: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


class PriceSeriesError(ValueError):
    """Raised when uploaded files cannot be parsed into a price series."""


def _detect_price_column(columns: Iterable[str]) -> str | None:
    for candidate in PRICE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _detect_date_column(columns: Iterable[str]) -> str | None:
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _series_from_dataframe(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise PriceSeriesError("Uploaded data frame is empty")
    price_col = _detect_price_column(df.columns)
    if price_col is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise PriceSeriesError("Could not find a numeric column to use as prices")
        price_col = numeric_cols[0]
    date_col = _detect_date_column(df.columns)
    values = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if values.empty:
        raise PriceSeriesError("Price column does not contain numeric values")
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = dates.notna() & df[price_col].notna()
        values = pd.Series(pd.to_numeric(df.loc[mask, price_col], errors="coerce").values, index=dates[mask])
        values = values.sort_index()
        return values.astype(float)
    return pd.Series(values.astype(float).to_numpy())


def _parse_csv_bytes(payload: bytes) -> pd.Series:
    try:
        df = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # pragma: no cover - pandas errors include context
        raise PriceSeriesError(f"Failed to parse CSV: {exc}") from exc
    return _series_from_dataframe(df)


def _parse_json_bytes(payload: bytes) -> pd.Series:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - helpful error
        raise PriceSeriesError(f"Invalid JSON file: {exc}") from exc
    data: Any = parsed
    if isinstance(parsed, dict):
        for key in ("prices", "values", "series", "data"):
            if key in parsed:
                data = parsed[key]
                break
        else:
            data = list(parsed.values())
    if isinstance(data, dict):
        rows = sorted(data.items())
        dates = pd.to_datetime([item[0] for item in rows], errors="coerce")
        values = pd.Series([float(item[1]) for item in rows], index=dates)
        values = values[values.index.notna()]
        if values.empty:
            values = pd.Series([float(item[1]) for item in rows])
        return values
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            return _series_from_dataframe(df)
        return pd.Series([float(x) for x in data])
    if isinstance(data, (int, float)):
        return pd.Series([float(data)])
    raise PriceSeriesError("Unsupported JSON structure; expected list or dict")


def parse_price_series(payload: bytes, filename: str | None = None) -> pd.Series:
    """Convert uploaded bytes into a clean price series."""

    suffix = (Path(filename).suffix if filename else "").lower()
    if suffix == ".json":
        series = _parse_json_bytes(payload)
    else:
        series = _parse_csv_bytes(payload)
    series = series.dropna()
    if series.empty:
        raise PriceSeriesError("Price series is empty after cleaning")
    return series.astype(float)


def _max_drawdown(pnl: np.ndarray) -> float:
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return float(abs(drawdowns.min()))


def _tail_mean(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return float("nan")
    sorted_vals = np.sort(values)
    tail_count = max(1, int(math.ceil(alpha * sorted_vals.size)))
    return float(sorted_vals[:tail_count].mean())


def run_hedging_simulation(
    price_series: pd.Series,
    *,
    option_maturity_days: int,
    strike_relative: float,
    checkpoint: str,
    seed: int = 0,
    regime_filter: str = "All",
    hedge_ratio: float = 1.0,
    noise_scale: float = 0.0,
    cvar_alpha: float = 0.05,
    wg_alpha: float = 0.2,
) -> HedgingResult:
    """Simulate a lightweight hedging policy using uploaded prices."""

    if option_maturity_days < 1:
        raise ValueError("Option maturity must be at least 1 day")
    if strike_relative <= 0:
        raise ValueError("Strike / spot ratio must be positive")
    series = price_series.dropna().astype(float)
    if series.size <= option_maturity_days:
        raise ValueError("Price series must be longer than the maturity window")
    series = series.sort_index()
    forward = series.shift(-option_maturity_days)
    valid_mask = forward.notna()
    entry_prices = series[valid_mask]
    exit_prices = forward[valid_mask]
    log_returns = np.log(exit_prices.values / entry_prices.values)
    option_payoff = np.maximum(exit_prices.values - strike_relative * entry_prices.values, 0.0)
    pnl = log_returns - hedge_ratio * (option_payoff / entry_prices.values)
    if noise_scale > 0:
        rng = np.random.default_rng(int(seed))
        pnl = pnl + rng.normal(loc=0.0, scale=noise_scale, size=pnl.shape)
    _, price_vol = price_series_volatility(series.values)
    vol_aligned = price_vol[: series.size]
    regimes = np.asarray([map_vol_to_regime(float(v)) for v in vol_aligned], dtype=int)
    regimes = regimes[valid_mask.values]
    pnl_series = pd.Series(pnl, index=entry_prices.index, name="pnl")
    log_return_series = pd.Series(log_returns, index=entry_prices.index, name="log_returns")
    if regime_filter.lower() != "all":
        target = None
        for regime_id, name in REGIME_NAMES.items():
            if name.lower() == regime_filter.lower():
                target = regime_id
                break
        if target is None:
            raise ValueError(f"Unknown regime filter '{regime_filter}'")
        keep_mask = regimes == target
        if not np.any(keep_mask):
            raise ValueError(
                "Selected regime filter removed all samples. "
                "Choose a different regime or upload a longer series."
            )
        mask_series = pd.Series(keep_mask, index=entry_prices.index)
        pnl_series = pnl_series[mask_series]
        entry_prices = entry_prices[mask_series]
        log_return_series = log_return_series[mask_series]
        regimes = regimes[keep_mask]
    if pnl_series.empty:
        raise ValueError("No PnL observations available after processing inputs")
    cumulative = pnl_series.cumsum()
    pnl = pnl_series.values
    actions = np.sign(log_return_series.values) * hedge_ratio
    actions_seq: List[List[List[float]]] = [[[float(val)] for val in actions]]
    hit_rate = float((pnl > 0).mean())
    metrics: Dict[str, float] = {
        "Mean PnL": float(pnl.mean()),
        "CVaR": _tail_mean(pnl, cvar_alpha),
        "Max Drawdown": _max_drawdown(pnl),
        "Hit Rate": hit_rate,
    }
    metrics.update(compute_vr(pnl.tolist(), eps=1e-6))
    metrics.update(compute_er(pnl.tolist(), cvar_alpha=cvar_alpha, eps=1e-6, mode="returns"))
    metrics.update(compute_tr(actions_seq, eps=1e-6))
    env_risks: Dict[str, float] = {}
    for regime_id in sorted(set(regimes.tolist())):
        mask = regimes == regime_id
        if not np.any(mask):
            continue
        name = REGIME_NAMES.get(int(regime_id), f"env_{regime_id}")
        env_risks[name] = float(pnl[mask].mean())
    if env_risks:
        metrics.update(compute_wg(env_risks=env_risks, alpha=float(wg_alpha)))
    regime_summary = [
        {
            "regime": REGIME_NAMES.get(int(regime_id), str(regime_id)).title(),
            "count": int((regimes == regime_id).sum()),
            "avg_pnl": float(pnl[regimes == regime_id].mean()),
        }
        for regime_id in sorted(set(regimes.tolist()))
    ]
    extra: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "regime_summary": regime_summary,
        "hedge_ratio": hedge_ratio,
    }
    return HedgingResult(pnl_series=pnl_series, cumulative_pnl=cumulative, metrics=metrics, extra=extra)


__all__ = [
    "HedgingResult",
    "PriceSeriesError",
    "REGIME_OPTIONS",
    "parse_price_series",
    "run_hedging_simulation",
]
