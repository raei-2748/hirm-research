"""Streamlit dashboard for running the HIRM hedging pipeline interactively."""
from __future__ import annotations

import textwrap
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from hirm.envs.regimes import REGIME_NAMES
from webui_utils import HedgingResult, HedgingUIError, parse_price_file, run_main_model


st.set_page_config(page_title="HIRM Hedging Web UI", layout="wide")


def _render_metrics(result: HedgingResult) -> None:
    st.subheader("Key metrics")
    metric_pairs = [
        ("mean_pnl", "Mean PnL"),
        ("cvar_95", "CVaR-95"),
        ("max_drawdown", "Max drawdown"),
        ("turnover", "Turnover"),
    ]
    cols = st.columns(len(metric_pairs))
    for col, (key, label) in zip(cols, metric_pairs):
        value = result.metrics.get(key)
        if value is None:
            continue
        col.metric(label, f"{value:.4f}")

    crisis_cols = st.columns(2)
    crisis_pairs = [
        ("crisis_cvar", "Crisis CVaR"),
        ("crisis_mean", "Crisis mean"),
    ]
    for col, (key, label) in zip(crisis_cols, crisis_pairs):
        if key in result.metrics:
            col.metric(label, f"{result.metrics[key]:.4f}")


def _render_plots(result: HedgingResult) -> None:
    pnl_frame = pd.DataFrame({"date": result.timestamps, "PnL": result.pnl_series})
    pnl_frame = pnl_frame.set_index("date")
    st.subheader("PnL time series")
    st.line_chart(pnl_frame)

    st.subheader("PnL distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(result.pnl_series, bins=30, color="#2563eb", alpha=0.8, edgecolor="white")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)


def _sidebar_form() -> Optional[dict]:
    st.sidebar.header("Scenario inputs")
    with st.sidebar.form("hedging_form"):
        uploaded = st.file_uploader(
            "Upload spot time-series (CSV or JSON)",
            type=["csv", "json"],
            help="Provide at least a 'date' and 'price' column.",
        )
        maturity = st.number_input(
            "Option maturity (days)",
            min_value=5,
            max_value=365,
            value=60,
            step=5,
        )
        strike = st.slider(
            "Strike / Spot ratio",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.01,
        )
        config_path = st.text_input(
            "Experiment config",
            value="configs/experiments/tiny_test.yaml",
            help="Any YAML supported by hirm.utils.config.load_config.",
        )
        checkpoint = st.text_input(
            "Model checkpoint path",
            value="",
            help="Optional PyTorch state_dict to load before evaluation.",
        )
        regime_names = list(REGIME_NAMES.values())
        selected_regimes = st.multiselect(
            "Regimes to include",
            options=regime_names,
            default=regime_names,
            help="Filter episodes by realized volatility regime.",
        )
        with st.expander("Advanced parameters", expanded=False):
            seed = st.number_input("Random seed", value=0, step=1)
            action_dim = st.number_input("Action dimension", min_value=1, max_value=4, value=2, step=1)
        submitted = st.form_submit_button("Run hedging model")

    if not submitted:
        return None
    if uploaded is None:
        st.sidebar.error("Please upload a CSV or JSON file before running the model.")
        return None
    return {
        "uploaded": uploaded,
        "maturity": int(maturity),
        "strike": float(strike),
        "config_path": config_path.strip(),
        "checkpoint": checkpoint.strip() or None,
        "seed": int(seed),
        "action_dim": int(action_dim),
        "regimes": selected_regimes,
    }


def main() -> None:
    st.title("HIRM Hedging Web UI")
    st.markdown(
        textwrap.dedent(
            """
            Upload a daily spot price series, choose the option profile, and run the
            HIRM invariant hedging policy to inspect the resulting performance.
            The dashboard wraps the PyTorch implementation shipped with this repo
            so that you can experiment with different checkpoints and configs
            without leaving the browser.
            """
        )
    )

    form_values = _sidebar_form()
    result: Optional[HedgingResult] = None
    price_frame: Optional[pd.DataFrame] = None
    if form_values:
        uploaded = form_values["uploaded"]
        bytes_data = uploaded.getvalue()
        try:
            price_series = parse_price_file(uploaded.name, bytes_data)
            price_frame = price_series.to_frame().head(1000)
            with st.spinner("Running hedging model..."):
                result = run_main_model(
                    price_series,
                    option_maturity_days=form_values["maturity"],
                    strike_relative=form_values["strike"],
                    config_path=form_values["config_path"],
                    checkpoint=form_values["checkpoint"],
                    seed=form_values["seed"],
                    action_dim=form_values["action_dim"],
                    regime_filter=form_values["regimes"],
                )
        except HedgingUIError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.exception(exc)

    if price_frame is not None:
        st.subheader("Uploaded price sample")
        st.dataframe(price_frame, use_container_width=True)

    if result is None:
        st.info("Fill out the form in the sidebar and click **Run hedging model** to see results.")
        return

    _render_metrics(result)
    _render_plots(result)

    st.subheader("Environment diagnostics")
    st.json(result.extra.get("env_risks", {}))

    st.subheader("Raw PnL data")
    pnl_frame = pd.DataFrame({"date": result.timestamps, "pnl": result.pnl_series})
    st.dataframe(pnl_frame.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
