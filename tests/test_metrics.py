import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import numpy as np
import math
import pytest

from src.utils.metrics import (
    compute_nldd,
    compute_rsa,
    compute_tas,
    compute_all_metrics,
)


def _make_acts(n_layers: int = 4, seq_len: int = 5, d_model: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    return {l: torch.randn(seq_len, d_model) for l in range(n_layers)}


def test_compute_nldd_positive_faithful():
    assert compute_nldd(-1.0, -3.0) == pytest.approx(2.0)


def test_compute_nldd_zero_identical():
    assert compute_nldd(-2.5, -2.5) == pytest.approx(0.0)


def test_compute_nldd_negative_unfaithful():
    assert compute_nldd(-3.0, -1.0) == pytest.approx(-2.0)


def test_compute_rsa_returns_per_layer():
    clean = _make_acts(seed=0)
    corrupted = _make_acts(seed=99)
    result = compute_rsa(clean, corrupted)
    assert set(result.keys()) == set(clean.keys())
    for v in result.values():
        assert -1.0 - 1e-5 <= v <= 1.0 + 1e-5


def test_compute_rsa_identical_returns_one():
    acts = _make_acts()
    result = compute_rsa(acts, acts)
    for v in result.values():
        assert abs(v - 1.0) < 1e-5


def test_compute_tas_returns_float():
    clean = _make_acts(seed=0)
    corrupted = _make_acts(seed=99)
    tas = compute_tas(clean, corrupted)
    assert isinstance(tas, float)
    assert not math.isnan(tas)


def test_compute_tas_single_layer_returns_nan():
    single = {0: torch.randn(5, 8)}
    tas = compute_tas(single, single)
    assert math.isnan(tas)


def test_compute_all_metrics_has_all_keys():
    clean = _make_acts(seed=0)
    corrupted = _make_acts(seed=99)
    result = compute_all_metrics(clean, corrupted, clean_log_p=-1.0, corrupted_log_p=-2.0)
    assert "nldd" in result
    assert "rsa" in result
    assert "tas" in result
    assert result["nldd"] == pytest.approx(1.0)
