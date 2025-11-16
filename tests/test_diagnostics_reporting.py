import pytest

from hirm.diagnostics.reporting import (
    compute_diagnostics_correlations,
    summarize_diagnostics_by_method,
)


@pytest.fixture()
def sample_dataframe():
    pd = pytest.importorskip("pandas")
    data = {
        "model_name": ["erm", "hirm"],
        "method": ["ERM", "HIRM"],
        "metrics.ISI": [0.8, 0.95],
        "metrics.IG": [0.3, 0.1],
        "metrics.crisis_cvar95": [0.6, 0.2],
    }
    return pd.DataFrame(data)


def test_correlations_include_crisis(sample_dataframe):
    correlations = compute_diagnostics_correlations(sample_dataframe)
    assert "corr(ISI, crisis_cvar95)" in correlations
    assert "corr(IG, crisis_cvar95)" in correlations


def test_summary_group_by_override(sample_dataframe):
    summary = summarize_diagnostics_by_method(sample_dataframe, group_by="model_name")
    assert "model_name" in summary.columns
    assert any(col.startswith("metrics.ISI") for col in summary.columns)
