import pytest

torch = pytest.importorskip("torch")

from hirm.objectives.risk import make_cvar


def test_cvar_uses_worst_tail():
    cvar = make_cvar(alpha=0.75)
    pnl = torch.tensor([-3.0, -2.0, -1.0, 0.0])
    value = cvar(pnl)
    assert value > 0.0
    assert torch.isclose(value, torch.tensor(3.0))


def test_cvar_worsens_when_tail_gets_worse():
    cvar = make_cvar(alpha=0.5)
    pnl1 = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    pnl2 = torch.tensor([-2.0, 0.0, 1.0, 2.0])
    risk1 = cvar(pnl1)
    risk2 = cvar(pnl2)
    assert risk2 > risk1


def test_cvar_decreases_with_higher_alpha():
    pnl = torch.tensor([-3.0, -2.0, -1.0, 0.0])
    cvar_90 = make_cvar(alpha=0.90)
    cvar_99 = make_cvar(alpha=0.99)
    risk_90 = cvar_90(pnl)
    risk_99 = cvar_99(pnl)
    assert risk_99 <= risk_90
