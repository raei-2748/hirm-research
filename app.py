"""Streamlit dashboard for running HIRM hedging diagnostics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from hirm.envs.regimes import REGIME_NAMES
from webui_utils import (
    HedgingResult,
    hedging_result_frame,
    load_sample_time_series,
    parse_time_series_upload,
    run_hedging_model,
)

st.set_page_config(page_title="HIRM Hedging Web UI", layout="wide")
st.title("HIRM Hedging Web UI")
st.markdown(
    """
    Upload a historical price series, choose an option contract specification, and
    run the HIRM hedging policy to inspect PnL distributions across volatility
    regimes. The dashboard wraps the existing training/evaluation pipeline
    without altering the underlying research code.
    """
)

REGIME_OPTIONS = ["All", *REGIME_NAMES.values()]


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@st.cache_data(show_spinner=False)
def _load_sample_series() -> pd.DataFrame:
    return load_sample_time_series()


def _render_metrics(result: HedgingResult) -> None:
    metrics = result.metrics
    st.subheader("Key metrics")
    cols = st.columns(max(1, len(metrics)))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.4f}")
    st.caption(f"Regime focus: {result.extra.get('regime_focus', 'All')}")


with st.sidebar:
    st.header("Scenario setup")
    with st.form("scenario-form"):
        uploaded_file = st.file_uploader(
            "Upload CSV or JSON time series", type=["csv", "json"], accept_multiple_files=False
        )
        use_sample = st.checkbox("Use bundled SPY demo data", value=True)
        maturity_days = st.number_input(
            "Option maturity (days)",
            min_value=5,
            max_value=365,
            value=60,
            step=5,
            help="Controls the rolling window used to engineer state features.",
        )
        strike_relative = st.slider(
            "Strike relative to spot",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.01,
            help="Scales the base PnL contribution to mimic moneyness adjustments.",
        )
        config_path = st.text_input(
            "Config name or path",
            value="configs/experiments/tiny_test.yaml",
            help="Any config accepted by hirm.utils.config.load_config.",
        )
        checkpoint_path = st.text_input(
            "Model checkpoint (optional)",
            value="",
            placeholder="outputs/experiments/checkpoints/model_final.pt",
        )
        with st.expander("Advanced parameters"):
            seed = st.number_input("Random seed", value=0, step=1)
            device = st.selectbox("Compute device", options=_available_devices())
            regime_focus = st.selectbox(
                "Focus regime for metrics", options=REGIME_OPTIONS, help="Metrics table filters to a specific regime."
            )
        submitted = st.form_submit_button("Run hedging model", use_container_width=True)

series_df: pd.DataFrame | None = None
source_label = ""
if submitted:
    try:
        if uploaded_file is not None:
            series_df = parse_time_series_upload(uploaded_file)
            source_label = f"Uploaded: {uploaded_file.name}"
        elif use_sample:
            series_df = _load_sample_series()
            source_label = "Sample SPY (data/processed/spy_prices.csv)"
        else:
            st.error("Please upload a file or enable the sample dataset option.")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to load time series: {exc}")

result: HedgingResult | None = None
if submitted and series_df is not None:
    try:
        with st.spinner("Running hedging model..."):
            result = run_hedging_model(
                series_df,
                maturity_days=int(maturity_days),
                strike_relative=float(strike_relative),
                config=config_path,
                checkpoint=checkpoint_path or None,
                seed=int(seed),
                device=device,
                regime_focus=None if regime_focus == "All" else regime_focus,
            )
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Model execution failed: {exc}")

if result is not None and series_df is not None:
    st.success(f"Finished evaluation using {source_label}")
    _render_metrics(result)
    frame = hedging_result_frame(result)
    st.subheader("Cumulative PnL trajectory")
    st.line_chart(frame[["cumulative_pnl"]])
    st.subheader("PnL distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frame["pnl"], bins=40, color="#2b7de9", alpha=0.85)
    ax.set_xlabel("PnL per step")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    env_metrics = result.extra.get("env_metrics", {})
    if env_metrics:
        st.subheader("Per-regime diagnostics")
        env_df = pd.DataFrame(env_metrics).T
        st.dataframe(env_df)
    st.subheader("Price series preview")
    st.caption("Rows shown are the last 10 entries of the parsed upload.")
    st.dataframe(series_df.tail(10))
else:
    st.info("Submit the form on the left to generate hedging diagnostics.")
