"""Streamlit dashboard for interactive HIRM hedging experiments."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from webui_utils import (
    PriceSeriesError,
    REGIME_OPTIONS,
    parse_price_series,
    run_hedging_simulation,
)

st.set_page_config(page_title="HIRM Hedging Web UI", layout="wide")
st.title("HIRM Hedging Web UI")
st.markdown(
    """
    Upload a spot price history to prototype hedging configurations using the
    lightweight diagnostics implemented in this repository.  Configure the
    option maturity, strike ratio, and advanced hyper-parameters from the left
    sidebar, then launch a run to visualize PnL distributions and metrics.
    """
)

with st.sidebar:
    st.header("Scenario inputs")
    with st.form("hedging-form"):
        uploaded = st.file_uploader(
            "Price history (CSV or JSON)",
            type=["csv", "json"],
            accept_multiple_files=False,
            help="Upload SPY or any asset history with a price column.",
        )
        maturity = st.number_input(
            "Option maturity (days)",
            min_value=5,
            max_value=365,
            value=60,
            step=5,
            help="Window used to measure the terminal payoff.",
        )
        strike_ratio = st.slider(
            "Strike / Spot ratio",
            min_value=0.7,
            max_value=1.3,
            value=1.0,
            step=0.01,
        )
        checkpoint = st.text_input(
            "Model checkpoint / config",
            value="phase7_tiny",
            help="Free-form tag recorded with the run metadata.",
        )
        with st.expander("Advanced parameters", expanded=False):
            seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0)
            regime_choice = st.selectbox("Regime focus", options=REGIME_OPTIONS, index=0)
            hedge_ratio = st.slider(
                "Hedge aggressiveness", min_value=0.0, max_value=2.0, value=1.0, step=0.05
            )
            noise_scale = st.slider(
                "Shock noise scale", min_value=0.0, max_value=0.05, value=0.005, step=0.001
            )
            cvar_alpha = st.slider(
                "CVaR alpha", min_value=0.01, max_value=0.2, value=0.05, step=0.01
            )
            wg_alpha = st.slider(
                "Worst-case alpha", min_value=0.05, max_value=0.5, value=0.2, step=0.05
            )
        submitted = st.form_submit_button("Run simulation", use_container_width=True)

if submitted:
    if uploaded is None:
        st.error("Please upload a CSV or JSON file with price data before running.")
        st.stop()
    try:
        price_series = parse_price_series(uploaded.getvalue(), uploaded.name)
    except PriceSeriesError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # pragma: no cover - surfacing unexpected parsing errors
        st.error(f"Unexpected error while reading file: {exc}")
        st.stop()
    st.success(
        f"Loaded {len(price_series):,} price points spanning "
        f"{price_series.index.min()} to {price_series.index.max() if isinstance(price_series.index, pd.DatetimeIndex) else len(price_series)}."
    )
    try:
        result = run_hedging_simulation(
            price_series,
            option_maturity_days=int(maturity),
            strike_relative=float(strike_ratio),
            checkpoint=checkpoint,
            seed=int(seed),
            regime_filter=regime_choice,
            hedge_ratio=float(hedge_ratio),
            noise_scale=float(noise_scale),
            cvar_alpha=float(cvar_alpha),
            wg_alpha=float(wg_alpha),
        )
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        st.stop()

    st.subheader("Key metrics")
    metric_keys = ["Mean PnL", "CVaR", "Max Drawdown", "VR", "ER", "TR", "WG", "Hit Rate"]
    metric_values: Dict[str, float] = {key: result.metrics.get(key) for key in metric_keys if key in result.metrics}
    cols = st.columns(len(metric_values) or 1)
    for idx, (name, value) in enumerate(metric_values.items()):
        cols[idx].metric(label=name, value=f"{value:.4f}")

    st.subheader("PnL diagnostics")
    chart_df = pd.DataFrame(
        {
            "PnL": result.pnl_series,
            "Cumulative PnL": result.cumulative_pnl,
        }
    )
    st.line_chart(chart_df)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(result.pnl_series, bins=40, color="#2a7de1", alpha=0.85, edgecolor="#0f3057")
    ax.set_title("PnL distribution")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Regime summary")
    regime_summary = pd.DataFrame(result.extra.get("regime_summary", []))
    if not regime_summary.empty:
        st.dataframe(regime_summary.style.format({"avg_pnl": "{:.4f}"}), use_container_width=True)
    else:
        st.info("Regime-level breakdown unavailable for the current selection.")

    st.caption(
        f"Run metadata — checkpoint: {result.extra.get('checkpoint', checkpoint)}, "
        f"hedge ratio: {result.extra.get('hedge_ratio', hedge_ratio):.2f}, "
        f"maturity: {maturity}d, strike ratio: {strike_ratio:.2f}."
    )
else:
    st.info("Configure the scenario on the left and click **Run simulation** to get started.")
