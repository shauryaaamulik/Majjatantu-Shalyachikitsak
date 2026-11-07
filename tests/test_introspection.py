import torch
from torch import nn

from majjatantu.introspection import summarize_model, summarize_tensor
from majjatantu.introspection.activations import capture_activations, pool_token_mean


def test_summarize_tensor_2d():
    t = torch.randn(16, 8)
    s = summarize_tensor("w", t, top_k_singular=3)
    assert s.name == "w"
    assert s.shape == [16, 8]
    assert s.numel == 128
    assert len(s.spectral_top_k) == 3
    assert s.std >= 0


def test_summarize_model_filter():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    summaries = summarize_model(model, name_contains="1")  # second Linear is index 2... 
    # named_parameters: "0.weight", "0.bias", "2.weight", "2.bias"
    summaries = summarize_model(model, name_contains="2.weight")
    assert len(summaries) == 1
    assert summaries[0].name == "2.weight"


def test_capture_activations_stats_only():
    model = nn.Linear(3, 5)
    x = torch.randn(2, 3)
    with capture_activations(model, module_names=[""], keep_tensors=False) as cap:
        # root module name is ""
        model(x)
    assert len(cap.stats) == 1
    assert cap.stats[0].shape == (2, 5)
    assert cap.last_tensors == {}


def test_pool_token_mean():
    t = torch.ones(2, 4, 8)
    out = pool_token_mean(t)
    assert out.shape == (2, 8)
