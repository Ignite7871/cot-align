import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
from transformer_lens import HookedTransformer
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.probing.cot_probe import extract_token_activations
from src.probing.faithfulness_detector import FaithfulnessProbe
from src.steering.activation_steering import apply_steering


@dataclass
class PipelineTrace:
    """
    Full record of a single closed-loop pipeline run.

    Captures every token-level event: the input prompt, all generated tokens,
    the faithfulness labels and confidence scores emitted by the probe at each
    step, the NLDD proxy scores, and the indices where steering was triggered.

    Used for post-hoc analysis, visualization (plot_faithfulness_trace), and
    reproducing the Figure 3 results from the paper.
    """

    prompt: str
    generated_tokens: list[int] = field(default_factory=list)
    generated_text: str = ""
    labels: list[bool] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    nldd_scores: list[float] = field(default_factory=list)
    steering_event_indices: list[int] = field(default_factory=list)


def find_reasoning_horizon(
    nldd_scores: list[float],
    method: str = "threshold",
    threshold_factor: float = 0.5,
    window: int = 3,
) -> int:
    """
    Find k*, the step index where faithful reasoning begins to break down.

    The reasoning horizon is the first position in the generation where the
    model's faithfulness signal drops below a critical threshold. This is the
    causal "point of failure" identified in the paper — the earliest step at
    which an intervention is needed to prevent compounding reasoning errors.

    Methods:
        threshold: k* = first index where nldd[k] < threshold_factor * mean(nldd).
            Simple and interpretable; sensitive to the threshold_factor choice.
        gradient: k* = index of the steepest negative gradient in the smoothed
            NLDD curve. Detects the sharpest faithfulness collapse.

    Args:
        nldd_scores: Per-step NLDD or proxy faithfulness scores.
        method: 'threshold' or 'gradient'.
        threshold_factor: For threshold method — fraction of mean NLDD to use
            as the cutoff.
        window: Smoothing window size for gradient method.

    Returns:
        Integer index of the reasoning horizon. Returns 0 for empty input,
        last index if no horizon is found.
    """
    if not nldd_scores:
        return 0

    if method == "threshold":
        threshold = threshold_factor * float(np.mean(nldd_scores))
        for i, score in enumerate(nldd_scores):
            if score < threshold:
                return i
        return len(nldd_scores) - 1

    elif method == "gradient":
        scores = np.array(nldd_scores, dtype=float)
        if len(scores) < 2:
            return 0
        kernel = np.ones(window) / window
        smoothed = np.convolve(scores, kernel, mode="valid")
        if len(smoothed) < 2:
            return 0
        grads = np.diff(smoothed)
        return int(np.argmin(grads))

    return len(nldd_scores) - 1


def run_pipeline(
    model: HookedTransformer,
    prompt: str,
    probe: Optional[FaithfulnessProbe] = None,
    steering_vector: Optional[torch.Tensor] = None,
    layer: Optional[int] = None,
    max_tokens: int = 200,
    threshold: float = 0.7,
    alpha: float = 1.0,
) -> PipelineTrace:
    """
    Run the closed-loop CoT alignment pipeline token-by-token.

    At each generation step the pipeline:
      1. Extracts last-token residual stream activations at probe.layer.
      2. Runs the linear probe to predict faithfulness confidence.
      3. If the probe predicts unfaithful (confidence > threshold) and a
         steering vector is available, applies activation steering for this
         step's logit computation.
      4. Greedily samples the next token (argmax).
      5. Records everything in a PipelineTrace.

    The pipeline degrades gracefully:
      - No probe supplied → generates tokens, no faithfulness labels.
      - Probe supplied, no steering_vector → probes but never steers.
      - Full setup → closed-loop detection + correction.

    Generation stops at EOS or max_tokens, whichever comes first.

    Args:
        model: HookedTransformer instance.
        prompt: Input prompt string.
        probe: Trained FaithfulnessProbe (optional).
        steering_vector: Steering direction tensor, shape (d_model,) (optional).
        layer: Residual stream layer to steer at (required if steering_vector given).
        max_tokens: Maximum new tokens to generate.
        threshold: Unfaithful confidence threshold that triggers steering.
        alpha: Steering intervention strength.

    Returns:
        PipelineTrace with the complete generation record.
    """
    trace = PipelineTrace(prompt=prompt)
    tokens = model.to_tokens(prompt)
    current_tokens = tokens

    is_faithful = True
    confidence = 1.0

    for step_idx in range(max_tokens):
        if probe is not None:
            current_text = model.to_string(current_tokens[0])
            acts = extract_token_activations(model, current_text, token_idx=-1)
            act_vec = acts[probe.layer].float().numpy().reshape(1, -1)
            preds = probe.predict(act_vec)
            is_faithful, confidence = preds[0]
            trace.labels.append(is_faithful)
            trace.confidences.append(confidence)
            trace.nldd_scores.append(confidence if is_faithful else -confidence)

        should_steer = (
            probe is not None
            and steering_vector is not None
            and layer is not None
            and not is_faithful
            and confidence > threshold
        )

        if should_steer:
            trace.steering_event_indices.append(step_idx)
            with apply_steering(model, steering_vector, layer, alpha):
                with torch.no_grad():
                    next_logits = model(current_tokens)[0, -1]
        else:
            with torch.no_grad():
                next_logits = model(current_tokens)[0, -1]

        next_token = next_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        current_tokens = torch.cat([current_tokens, next_token], dim=-1)
        trace.generated_tokens.append(int(next_token.item()))

        if next_token.item() == model.tokenizer.eos_token_id:
            break

    n_prompt = tokens.shape[1]
    trace.generated_text = model.to_string(current_tokens[0, n_prompt:])
    return trace
