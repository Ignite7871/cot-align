import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import numpy as np
import pytest
import dataclasses

from src.pipeline.closed_loop import PipelineTrace, find_reasoning_horizon


def test_pipeline_trace_dataclass_fields():
    trace = PipelineTrace(prompt="test")
    fields = {f.name for f in dataclasses.fields(trace)}
    expected = {
        "prompt", "generated_tokens", "generated_text",
        "labels", "confidences", "nldd_scores", "steering_event_indices",
    }
    assert expected.issubset(fields)


def test_find_reasoning_horizon_threshold():
    # NLDD drops sharply at step 3
    scores = [0.8, 0.7, 0.9, 0.05, 0.1, 0.2]
    k = find_reasoning_horizon(scores, method="threshold", threshold_factor=0.5)
    assert k == 3


def test_find_reasoning_horizon_gradient():
    scores = [1.0, 0.9, 0.8, 0.2, 0.15, 0.1]
    k = find_reasoning_horizon(scores, method="gradient")
    assert isinstance(k, int)
    assert 0 <= k < len(scores)


def test_find_reasoning_horizon_flat_returns_last():
    scores = [0.5, 0.5, 0.5, 0.5]
    k = find_reasoning_horizon(scores, method="threshold", threshold_factor=0.5)
    # No value drops below 0.5 * mean(0.5) = 0.25 → returns last index
    assert k == len(scores) - 1


def test_find_reasoning_horizon_empty_returns_zero():
    assert find_reasoning_horizon([]) == 0


@pytest.mark.slow
def test_run_pipeline_no_probe_returns_trace(small_model):
    from src.pipeline.closed_loop import run_pipeline
    trace = run_pipeline(small_model, "The capital of France is", max_tokens=5)
    assert isinstance(trace, PipelineTrace)
    assert len(trace.generated_tokens) > 0
    assert len(trace.labels) == 0
    assert len(trace.confidences) == 0


@pytest.mark.slow
def test_run_pipeline_with_probe_labels_length(small_model, dummy_probe):
    from src.pipeline.closed_loop import run_pipeline
    max_tok = 5
    trace = run_pipeline(
        small_model,
        "Solve: 2 + 2 =",
        probe=dummy_probe,
        max_tokens=max_tok,
    )
    assert len(trace.labels) == len(trace.confidences)
    assert len(trace.labels) <= max_tok


@pytest.mark.slow
def test_steering_events_recorded(small_model, dummy_probe):
    from src.pipeline.closed_loop import run_pipeline
    torch.manual_seed(0)
    v = torch.randn(small_model.cfg.d_model)
    v = v / v.norm()

    trace = run_pipeline(
        small_model,
        "What is 3 times 4?",
        probe=dummy_probe,
        steering_vector=v,
        layer=dummy_probe.layer,
        max_tokens=10,
        threshold=0.0,  # always steer when unfaithful predicted
        alpha=1.0,
    )
    assert isinstance(trace.steering_event_indices, list)
