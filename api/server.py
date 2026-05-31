import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import dataclasses
import os
from pathlib import Path

from src.utils.model_loader import get_model
from src.probing.faithfulness_detector import generate_faithfulness_labels, FaithfulnessProbe
from src.steering.steering_vectors import load_steering_vector
from src.pipeline.closed_loop import run_pipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lazy model loading on startup via FastAPI lifespan.

    Loads the HookedTransformer specified by MODEL_NAME env var, then
    optionally loads a pre-trained probe from PROBE_PATH and a steering
    vector from STEERING_VECTOR_PATH if those files exist.
    """
    model_name = os.getenv("MODEL_NAME", "gpt2")
    device = os.getenv("DEVICE", "cpu")
    _state["model"] = get_model(model_name, device)
    _state["model_name"] = model_name

    probe_path = os.getenv("PROBE_PATH", "")
    if probe_path and Path(probe_path).exists():
        _state["probe"] = FaithfulnessProbe.load(probe_path)
        _state["layer"] = _state["probe"].layer
    else:
        _state["probe"] = None
        _state["layer"] = None

    vector_path = os.getenv("STEERING_VECTOR_PATH", "")
    if vector_path and Path(vector_path).exists():
        vec, _ = load_steering_vector(vector_path)
        _state["steering_vector"] = vec
    else:
        _state["steering_vector"] = None

    yield
    _state.clear()


app = FastAPI(title="CoT-Align API", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    prompt: str
    cot_steps: list[str]


class AlignRequest(BaseModel):
    prompt: str
    max_tokens: int = 200


@app.get("/health")
async def health() -> dict:
    """
    Health check endpoint.

    Returns model name and CUDA availability so callers can verify the
    server loaded the correct model on the expected device.
    """
    return {
        "status": "ok",
        "model": _state.get("model_name", "not loaded"),
        "cuda": torch.cuda.is_available(),
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    """
    Score each CoT step for faithfulness using NLDD.

    Runs generate_faithfulness_labels on the provided prompt and step list,
    returning a per-step faithfulness score. Does not apply any steering.

    Request body:
        prompt: Full prompt string (should contain the cot_steps as substrings).
        cot_steps: List of individual reasoning step strings.

    Returns:
        {'steps': [{'step_idx': int, 'faithful': bool, 'nldd': float}]}
    """
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    labels = generate_faithfulness_labels(
        model=model,
        prompt=req.prompt,
        cot_steps=req.cot_steps,
        strategy=os.getenv("CORRUPTION_STRATEGY", "number_perturbation"),
        nldd_threshold=float(os.getenv("NLDD_THRESHOLD", "0.1")),
    )
    return {
        "steps": [
            {"step_idx": idx, "faithful": bool(faithful), "nldd": float(nldd)}
            for idx, faithful, nldd in labels
        ]
    }


@app.post("/align")
async def align(req: AlignRequest) -> dict:
    """
    Run the full closed-loop alignment pipeline and return a PipelineTrace.

    Runs run_pipeline with the loaded probe and steering vector (if available).
    Without a probe the endpoint still generates and returns the raw token
    sequence. Without a steering vector it probes but never intervenes.

    Request body:
        prompt: Input prompt string.
        max_tokens: Maximum new tokens to generate (default 200).

    Returns:
        Serialized PipelineTrace as a JSON object.
    """
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    trace = run_pipeline(
        model=model,
        prompt=req.prompt,
        probe=_state.get("probe"),
        steering_vector=_state.get("steering_vector"),
        layer=_state.get("layer"),
        max_tokens=req.max_tokens,
        threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7")),
        alpha=float(os.getenv("STEERING_ALPHA", "1.0")),
    )
    return dataclasses.asdict(trace)
