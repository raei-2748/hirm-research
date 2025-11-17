"""Utilities for the Streamlit hedging dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from hirm.diagnostics import compute_crisis_cvar
from hirm.envs.regimes import REGIME_NAMES, map_vol_to_regime
from hirm.envs.regime_labelling import price_series_volatility
from hirm.models import build_model
from hirm.objectives.common import compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import ConfigNode, load_config


class HedgingUIError(RuntimeError):
    """Exception raised for validation errors surfaced to the UI."""


@dataclass
class PriceSeries:
    """Clean representation of user supplied price data."""

    dates: List[pd.Timestamp]
    prices: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame that can be displayed in Streamlit."""

        return pd.DataFrame({"date": self.dates, "price": self.prices})


@dataclass
class HedgingResult:
    """Container returned by :func:`run_main_model` for the Web UI."""

    pnl_series: np.ndarray
    actions: np.ndarray
    timestamps: List[pd.Timestamp]
    metrics: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


def _infer_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_price_file(file_name: str, file_bytes: bytes) -> PriceSeries:
    """Parse an uploaded CSV or JSON payload into a :class:`PriceSeries`."""

    suffix = Path(file_name).suffix.lower()
    buffer = BytesIO(file_bytes)
    try:
        if suffix == ".json":
            data = pd.read_json(buffer)
        else:
            data = pd.read_csv(buffer)
    except Exception as exc:  # pragma: no cover - defensive parsing branch
        raise HedgingUIError(f"Unable to parse file '{file_name}': {exc}") from exc

    if data.empty:
        raise HedgingUIError("Uploaded file is empty")
    data = data.dropna(how="all")
    if data.empty:
        raise HedgingUIError("Uploaded file does not contain valid rows")

    date_col = _infer_column(data.columns, ("date", "timestamp", "time", "datetime"))
    price_col = _infer_column(
        data.columns,
        ("price", "close", "adj_close", "adj close", "value"),
    )
    if price_col is None:
        numeric_cols = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]
        if not numeric_cols:
            raise HedgingUIError("Could not find a numeric column representing prices")
        price_col = numeric_cols[0]

    if date_col is not None:
        dates = pd.to_datetime(data[date_col], errors="coerce")
        mask = dates.notna() & data[price_col].notna()
        frame = pd.DataFrame({"date": dates[mask], "price": data.loc[mask, price_col].astype(float)})
        frame = frame.sort_values("date")
    else:
        frame = pd.DataFrame({"date": pd.RangeIndex(start=0, stop=len(data)), "price": data[price_col].astype(float)})

    frame = frame.dropna()
    if frame.empty or frame.shape[0] < 2:
        raise HedgingUIError("Price series must contain at least two observations")

    return PriceSeries(dates=list(pd.to_datetime(frame["date"])), prices=frame["price"].to_numpy(dtype=float))


def _one_hot_regime(regime: int) -> List[float]:
    return [1.0 if regime == rid else 0.0 for rid in sorted(REGIME_NAMES.keys())]


def _build_env_batch(
    prices: np.ndarray,
    dates: Sequence[pd.Timestamp],
    *,
    horizon: int,
    strike_relative: float,
    action_dim: int,
    allowed_regimes: Sequence[int],
) -> tuple[Dict[str, Tensor], List[pd.Timestamp]]:
    if prices.size < 2:
        raise HedgingUIError("Need at least two prices to compute returns")

    returns = np.diff(np.log(prices))
    _, vol_prices = price_series_volatility(prices)
    feature_rows: list[list[float]] = []
    hedge_returns: list[list[float]] = []
    base_pnl: list[float] = []
    env_ids: list[int] = []
    timestamp_rows: list[pd.Timestamp] = []

    time_scale = max(1, int(horizon))
    for idx, ret in enumerate(returns):
        price_idx = idx if idx < vol_prices.shape[0] else -1
        vol = float(vol_prices[price_idx]) if vol_prices.size else 0.0
        regime = map_vol_to_regime(vol)
        if regime not in allowed_regimes:
            continue
        frac = float((idx % time_scale) / time_scale)
        feature_rows.append([vol, strike_relative, frac, *_one_hot_regime(regime)])
        hedge_vec = np.zeros(action_dim, dtype=float)
        if action_dim >= 1:
            hedge_vec[0] = ret
        if action_dim >= 2:
            hedge_vec[1] = -ret
        hedge_returns.append(hedge_vec.tolist())
        base_pnl.append(float(ret))
        env_ids.append(0)
        ts_idx = min(idx + 1, len(dates) - 1)
        timestamp_rows.append(pd.to_datetime(dates[ts_idx]))

    if not feature_rows:
        raise HedgingUIError("Selected regime filter removed all samples")

    batch = {
        "features": torch.tensor(feature_rows, dtype=torch.float32),
        "hedge_returns": torch.tensor(hedge_returns, dtype=torch.float32),
        "base_pnl": torch.tensor(base_pnl, dtype=torch.float32),
        "env_ids": torch.tensor(env_ids, dtype=torch.long),
    }
    return batch, timestamp_rows


def _max_drawdown(pnl: np.ndarray) -> float:
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return float(drawdowns.min()) if drawdowns.size else 0.0


def _turnover(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 0.0
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.abs(diffs))) if diffs.size else 0.0


def _compute_metrics(pnl: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    metrics = {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl)),
        "cvar_95": float(np.percentile(pnl, 5)),
        "max_drawdown": _max_drawdown(pnl),
        "turnover": _turnover(actions),
    }
    crisis = compute_crisis_cvar(pnl_time_series=pnl.tolist(), alpha=0.95)
    metrics.update({f"crisis_{k}": float(v) for k, v in crisis.items()})
    return metrics


def run_main_model(
    price_series: PriceSeries,
    *,
    option_maturity_days: int,
    strike_relative: float,
    config_path: str,
    checkpoint: str | None,
    seed: int,
    action_dim: int,
    regime_filter: Sequence[str] | None = None,
) -> HedgingResult:
    """Run the core hedging model on the uploaded price series."""

    if option_maturity_days <= 0:
        raise HedgingUIError("Option maturity must be positive")
    if strike_relative <= 0:
        raise HedgingUIError("Strike ratio must be positive")
    if action_dim <= 0:
        raise HedgingUIError("Action dimension must be positive")

    cfg = load_config(config_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    allowed_regimes = list(REGIME_NAMES.keys())
    if regime_filter:
        normalized = {name.lower() for name in regime_filter}
        allowed_regimes = [rid for rid, name in REGIME_NAMES.items() if name.lower() in normalized]
        if not allowed_regimes:
            raise HedgingUIError("Regime filter did not match any known regimes")
    batch, timestamps = _build_env_batch(
        price_series.prices,
        price_series.dates,
        horizon=option_maturity_days,
        strike_relative=strike_relative,
        action_dim=action_dim,
        allowed_regimes=allowed_regimes,
    )
    device = torch.device("cpu")
    batch = {k: v.to(device) for k, v in batch.items()}
    model = build_model(cfg.model, input_dim=batch["features"].shape[1], action_dim=action_dim).to(device)
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise HedgingUIError(f"Checkpoint '{checkpoint}' does not exist")
        state = torch.load(path, map_location=device)
        if isinstance(state, Mapping):
            model.load_state_dict(state)
        else:
            raise HedgingUIError("Checkpoint file is not a valid state_dict")
    model.eval()

    risk_fn = build_risk_function(cfg.objective)
    env_risks, pnl_tensor, actions_tensor, env_ids = compute_env_risks(
        model,
        batch,
        batch["env_ids"],
        risk_fn,
    )
    pnl = pnl_tensor.detach().cpu().numpy()
    actions = actions_tensor.detach().cpu().numpy()
    metrics = _compute_metrics(pnl, actions)
    env_risk_map = {f"env_{env}": float(val.detach().cpu()) for env, val in env_risks.items()}
    experiment_name = "experiment"
    if isinstance(cfg, ConfigNode) and hasattr(cfg, "experiment"):
        experiment_name = getattr(cfg.experiment, "name", experiment_name)
    extra = {
        "env_risks": env_risk_map,
        "num_samples": int(pnl.shape[0]),
        "config": experiment_name,
        "env_ids": env_ids.detach().cpu().tolist(),
    }
    return HedgingResult(
        pnl_series=pnl,
        actions=actions,
        timestamps=timestamps,
        metrics=metrics,
        extra=extra,
    )


__all__ = [
    "HedgingResult",
    "HedgingUIError",
    "PriceSeries",
    "parse_price_file",
    "run_main_model",
]
