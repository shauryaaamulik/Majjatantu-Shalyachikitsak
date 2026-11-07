import torch
from torch import nn

from majjatantu.types import ParameterBudget
from majjatantu.weight_surgery import ablate_direction_in_weight, apply_direction_ablation


def test_ablate_direction_removes_component_from_columns():
    torch.manual_seed(0)
    w = torch.randn(8, 4)
    d = torch.randn(8)
    d = d / torch.linalg.vector_norm(d)

    edited, touched = ablate_direction_in_weight(w, d, columns=[0, 1], strength=1.0)
    assert touched == 8 * 2
    # After unit-strength ablation, columns should be orthogonal to d.
    for j in (0, 1):
        assert abs(float(torch.dot(d, edited[:, j]))) < 1e-5
    # Untouched columns unchanged.
    assert torch.allclose(edited[:, 2:], w[:, 2:])


def test_apply_direction_ablation_respects_budget():
    model = nn.Linear(4, 8, bias=False)
    d = torch.ones(8)
    budget = ParameterBudget(max_fraction=1e-9)  # impossibly tight
    try:
        apply_direction_ablation(model, "weight", d, budget=budget)
        assert False, "expected RuntimeError for budget violation"
    except RuntimeError as e:
        assert "max_fraction" in str(e)


def test_apply_direction_ablation_updates_parameter():
    model = nn.Linear(4, 8, bias=False)
    before = model.weight.detach().clone()
    d = torch.randn(8)
    result = apply_direction_ablation(model, "weight", d, columns=slice(0, 2))
    assert result.params_touched == 16
    assert not torch.allclose(model.weight, before)
    assert result.edit.operator == "direction_ablation"
