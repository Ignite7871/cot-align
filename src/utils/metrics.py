import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import torch.nn.functional as F
import numpy as np


def compute_nldd(clean_log_p: float, corrupted_log_p: float) -> float:
    """
    Compute Normalized Log-Likelihood Difference (NLDD) for one CoT step.

    NLDD measures causal faithfulness: how much the model's answer confidence
    changes when a reasoning step is corrupted.

        NLDD_k = log P(answer | clean CoT) - log P(answer | corrupted step k)

    Positive NLDD → the model's answer depends on this step (faithful).
    Near-zero or negative NLDD → the model ignores this step (unfaithful).

    Args:
        clean_log_p: Log-probability of the correct answer token under the
            unmodified chain-of-thought.
        corrupted_log_p: Log-probability under the corrupted version of step k.

    Returns:
        NLDD score (positive = faithful, negative = unfaithful).
    """
    return clean_log_p - corrupted_log_p


def compute_rsa(clean_acts: dict, corrupted_acts: dict) -> dict:
    """
    Compute Representational Similarity Analysis (RSA) per layer.

    For each layer, measures the cosine similarity between the last-token
    clean activation and the last-token corrupted activation. Using the
    last token correctly handles variable-length clean vs corrupted contexts
    (mean-pooling would mix padding positions when sequence lengths differ).

    Args:
        clean_acts: Dict {layer_idx: Tensor(seq_len, d_model)}.
        corrupted_acts: Dict {layer_idx: Tensor(seq_len, d_model)}.

    Returns:
        Dict {layer_idx: cosine_similarity_score} for shared layers.
    """
    result: dict = {}
    for layer in clean_acts:
        if layer not in corrupted_acts:
            continue
        c = clean_acts[layer].float()[-1]
        r = corrupted_acts[layer].float()[-1]
        sim = float(F.cosine_similarity(c.unsqueeze(0), r.unsqueeze(0)))
        result[layer] = sim
    return result


def compute_tas(clean_acts: dict, corrupted_acts: dict) -> float:
    """
    Compute Trajectory Alignment Score (TAS) across layers.

    Measures the coherence of how a corruption effect propagates through the
    residual stream. Computes per-layer difference vectors (clean - corrupted)
    using the last token position, then returns the mean cosine similarity
    between consecutive layer differences.

    High TAS → corruption ripples coherently through the network (step is
    mechanistically important). Low TAS → incoherent propagation. Returns
    NaN when fewer than 2 layers are available.

    Args:
        clean_acts: Dict {layer_idx: Tensor(seq_len, d_model)}.
        corrupted_acts: Dict {layer_idx: Tensor(seq_len, d_model)}.

    Returns:
        Mean inter-layer trajectory similarity (float), or NaN for <2 layers.
    """
    layers = sorted(set(clean_acts.keys()) & set(corrupted_acts.keys()))
    if len(layers) < 2:
        return float("nan")

    diffs = [
        clean_acts[l].float()[-1] - corrupted_acts[l].float()[-1]
        for l in layers
    ]

    sims = [
        float(F.cosine_similarity(diffs[i].unsqueeze(0), diffs[i + 1].unsqueeze(0)))
        for i in range(len(diffs) - 1)
    ]
    return float(np.mean(sims))


def compute_all_metrics(
    clean_acts: dict,
    corrupted_acts: dict,
    clean_log_p: float,
    corrupted_log_p: float,
) -> dict:
    """
    Compute NLDD, RSA, and TAS in a single call.

    Args:
        clean_acts: Activation dict from the unmodified forward pass.
        corrupted_acts: Activation dict from the corrupted forward pass.
        clean_log_p: Log-prob of correct answer under clean CoT.
        corrupted_log_p: Log-prob of correct answer under corrupted CoT.

    Returns:
        {'nldd': float, 'rsa': {layer_idx: float}, 'tas': float}
    """
    return {
        "nldd": compute_nldd(clean_log_p, corrupted_log_p),
        "rsa": compute_rsa(clean_acts, corrupted_acts),
        "tas": compute_tas(clean_acts, corrupted_acts),
    }
