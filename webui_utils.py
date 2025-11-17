"""Helper utilities for the Streamlit web UI."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch

from hirm.envs.regime_labelling import label_series_with_regimes, price_series_volatility
from hirm.envs.regimes import REGIME_NAMES
from hirm.models import build_model
from hirm.objectives.common import compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import ConfigNode, load_config

_REGIME_NAME_TO_ID = {name.lower(): idx for idx, name in REGIME_NAMES.items()}


@dataclass
class HedgingResult:
    """Structured container returned by ``run_hedging_model``."""

    pnl_series: np.ndarray
    actions: np.ndarray
    metrics: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``frame`` into ``[date, price]`` columns."""

    if frame.empty:
        raise ValueError("Uploaded file does not contain any rows")
    df = frame.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lower_map = {col.lower(): col for col in df.columns}
    date_col = None
    for candidate in ("date", "timestamp", "time", "day"):
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break
    if date_col is None:
        df["date"] = pd.RangeIndex(start=0, stop=len(df), step=1)
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    price_col = None
    for candidate in ("price", "close", "adj_close", "adj close", "value"):
        if candidate in lower_map:
            price_col = lower_map[candidate]
            break
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            price_col = numeric_cols[0]
    if price_col is None:
        raise ValueError("Unable to infer the price column; please provide a 'price' field")
    df = df[["date", price_col]].rename(columns={price_col: "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).dropna(subset=["date"])
    if df.empty:
        raise ValueError("No valid price observations found in the upload")
    df = df.sort_values("date")
    return df.reset_index(drop=True)


def _load_json_frame(raw: bytes) -> pd.DataFrame:
    payload = json.loads(raw.decode("utf-8"))
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return pd.DataFrame(payload)
        return pd.DataFrame({"price": payload})
    if isinstance(payload, Mapping):
        if "price" in payload and isinstance(payload["price"], list):
            data = {"price": payload["price"]}
            if "date" in payload and isinstance(payload["date"], list):
                data["date"] = payload["date"]
            return pd.DataFrame(data)
        return pd.DataFrame([payload])
    raise ValueError("Unsupported JSON payload; expected list or dict")


def parse_time_series_upload(upload) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Parse an uploaded CSV/JSON file into a normalized price dataframe."""

    raw = upload.getvalue()
    name = upload.name or "uploaded"
    if name.lower().endswith(".json"):
        frame = _load_json_frame(raw)
    else:
        frame = pd.read_csv(io.BytesIO(raw))
    return _normalize_price_frame(frame)


def load_sample_time_series(path: str = "data/processed/spy_prices.csv") -> pd.DataFrame:
    """Load a repository bundled price series used as a demo."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Sample price file not found at {file_path}")
    frame = pd.read_csv(file_path)
    return _normalize_price_frame(frame)


def _build_feature_matrices(
    prices: np.ndarray,
    *,
    maturity_days: int,
    strike_relative: float,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (features, hedge_returns, base_pnl, env_ids, regime_labels)."""

    if prices.size < 2:
        raise ValueError("Need at least two price points to compute returns")
    returns = np.diff(np.log(prices))
    vol_returns, _ = price_series_volatility(prices)
    regimes = label_series_with_regimes(prices)
    horizon = max(1, int(maturity_days))
    features: list[list[float]] = []
    hedge_rows: list[list[float]] = []
    base = []
    env_ids: list[int] = []
    regime_labels: list[str] = []
    sorted_regimes = [rid for rid in sorted(REGIME_NAMES)]
    for idx, ret in enumerate(returns):
        vol = float(vol_returns[idx]) if idx < vol_returns.size else 0.0
        regime = int(regimes[idx]) if idx < regimes.size else 0
        frac = float(idx % horizon) / float(max(1, horizon - 1))
        one_hot = [1.0 if regime == rid else 0.0 for rid in sorted_regimes]
        features.append([vol, *one_hot, frac])
        row = [float(ret)]
        if action_dim > 1:
            row.append(-float(ret))
            if action_dim > 2:
                row.extend([0.0] * (action_dim - 2))
        while len(row) < action_dim:
            row.append(0.0)
        hedge_rows.append(row[:action_dim])
        base.append(float(ret) * float(strike_relative))
        env_ids.append(regime)
        regime_labels.append(REGIME_NAMES.get(regime, f"env_{regime}"))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(hedge_rows, dtype=np.float32),
        np.asarray(base, dtype=np.float32),
        np.asarray(env_ids, dtype=np.int64),
        regime_labels,
    )


def _ensure_config_node(cfg_or_path: str | ConfigNode) -> ConfigNode:
    if isinstance(cfg_or_path, ConfigNode):
        return cfg_or_path
    return load_config(str(cfg_or_path))


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _maybe_load_checkpoint(model: torch.nn.Module, checkpoint: str | None, device: torch.device) -> None:
    if not checkpoint:
        return
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def _per_regime_metrics(
    pnl: np.ndarray,
    env_ids: np.ndarray,
    env_risks: Mapping[int, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for env, risk in env_risks.items():
        mask = env_ids == env
        if not np.any(mask):
            continue
        env_pnl = pnl[mask]
        metrics[REGIME_NAMES.get(env, f"env_{env}")] = {
            "mean_pnl": float(np.mean(env_pnl)),
            "cvar95": float(np.quantile(env_pnl, 0.05)) if env_pnl.size else 0.0,
            "risk": float(risk.detach().cpu()),
            "count": int(env_pnl.size),
        }
    return metrics


def _compute_scalar_metrics(pnl: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    if pnl.size == 0:
        return {"mean_pnl": 0.0, "cvar95": 0.0, "max_drawdown": 0.0, "turnover": 0.0}
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    turnover = 0.0
    if actions.size > 0:
        diffs = np.diff(actions, axis=0)
        if diffs.size > 0:
            turnover = float(np.mean(np.abs(diffs)))
        else:
            turnover = float(np.mean(np.abs(actions)))
    return {
        "mean_pnl": float(np.mean(pnl)),
        "cvar95": float(np.quantile(pnl, 0.05)),
        "max_drawdown": float(drawdowns.min()) if drawdowns.size else 0.0,
        "turnover": turnover,
    }


def run_hedging_model(
    df: pd.DataFrame,
    *,
    maturity_days: int,
    strike_relative: float,
    config: str | ConfigNode,
    checkpoint: str | None,
    seed: int = 0,
    device: str = "cpu",
    regime_focus: str | None = None,
) -> HedgingResult:
    """Run the HIRM policy on ``df`` and return metrics for the UI."""

    cfg = _ensure_config_node(config)
    env_cfg = getattr(cfg, "env", ConfigNode({}))
    action_dim = int(getattr(env_cfg, "action_dim", 2) or 2)
    device_obj = _resolve_device(device)
    torch.manual_seed(int(seed))
    prices = df["price"].to_numpy(dtype=float)
    (
        features,
        hedge_returns,
        base_pnl,
        env_ids,
        regime_labels,
    ) = _build_feature_matrices(
        prices,
        maturity_days=maturity_days,
        strike_relative=strike_relative,
        action_dim=action_dim,
    )
    feature_dim = features.shape[1]
    model = build_model(cfg.model, input_dim=feature_dim, action_dim=action_dim).to(device_obj)
    _maybe_load_checkpoint(model, checkpoint, device_obj)
    model.eval()
    batch = {
        "features": torch.tensor(features, dtype=torch.float32, device=device_obj),
        "hedge_returns": torch.tensor(hedge_returns, dtype=torch.float32, device=device_obj),
        "base_pnl": torch.tensor(base_pnl, dtype=torch.float32, device=device_obj),
    }
    env_tensor = torch.tensor(env_ids, dtype=torch.long, device=device_obj)
    risk_fn = build_risk_function(cfg.objective)
    with torch.no_grad():
        env_risks, pnl_tensor, actions_tensor, env_tensor = compute_env_risks(
            model, batch, env_tensor, risk_fn
        )
    pnl = pnl_tensor.detach().cpu().numpy()
    actions = actions_tensor.detach().cpu().numpy()
    env_array = env_tensor.detach().cpu().numpy()
    focus_id = None
    if regime_focus and regime_focus.lower() != "all":
        focus_id = _REGIME_NAME_TO_ID.get(regime_focus.lower())
    pnl_for_metrics = pnl
    actions_for_metrics = actions
    if focus_id is not None:
        mask = env_array == focus_id
        if not np.any(mask):
            raise ValueError(
                f"No samples available for regime '{regime_focus}'. Try another selection."
            )
        pnl_for_metrics = pnl[mask]
        actions_for_metrics = actions[mask]
    metrics = _compute_scalar_metrics(pnl_for_metrics, actions_for_metrics)
    env_metrics = _per_regime_metrics(pnl, env_array, env_risks)
    dates = df["date"].astype(str).tolist()[1:]
    result = HedgingResult(
        pnl_series=pnl,
        actions=actions,
        metrics=metrics,
        extra={
            "env_metrics": env_metrics,
            "env_ids": env_array,
            "dates": dates,
            "regime_labels": regime_labels,
            "config": cfg,
            "regime_focus": regime_focus or "All",
        },
    )
    return result


def hedging_result_frame(result: HedgingResult) -> pd.DataFrame:
    """Convert ``HedgingResult`` into a plotting-friendly dataframe."""

    pnl = result.pnl_series
    cumulative = np.cumsum(pnl)
    dates = result.extra.get("dates")
    regimes = result.extra.get("regime_labels")
    if dates is not None and len(dates) == len(pnl):
        index = pd.to_datetime(dates, errors="coerce")
    else:
        index = pd.RangeIndex(start=0, stop=len(pnl), step=1)
    frame = pd.DataFrame(
        {
            "pnl": pnl,
            "cumulative_pnl": cumulative,
            "regime": regimes if regimes is not None else [""] * len(pnl),
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


__all__ = [
    "HedgingResult",
    "parse_time_series_upload",
    "load_sample_time_series",
    "run_hedging_model",
    "hedging_result_frame",
]
