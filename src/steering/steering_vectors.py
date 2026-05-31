import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
from pathlib import Path
from typing import Optional


def compute_steering_vector(
    faithful_acts: torch.Tensor,
    unfaithful_acts: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Compute a steering vector using the mean-difference method.

    The steering vector is the L2-normalized difference between the mean
    faithful activation and the mean unfaithful activation at a given layer:

        v = normalize(mean(faithful) − mean(unfaithful))

    Adding alpha * v to the residual stream during generation moves the model
    toward the faithful activation cluster. This is the core of activation
    steering for CoT alignment: the geometry of the residual stream encodes
    the model's reasoning "mode", and we can shift it directly.

    Args:
        faithful_acts: shape (n_faithful_samples, d_model).
        unfaithful_acts: shape (n_unfaithful_samples, d_model).
        layer: Layer index (stored in metadata only; not used in computation).

    Returns:
        L2-normalized steering vector of shape (d_model,).
    """
    faithful_mean = faithful_acts.float().mean(dim=0)
    unfaithful_mean = unfaithful_acts.float().mean(dim=0)
    direction = faithful_mean - unfaithful_mean
    return direction / direction.norm()


def compute_steering_vector_from_dicts(
    faithful_dicts: list[dict],
    unfaithful_dicts: list[dict],
    layer: int,
) -> torch.Tensor:
    """
    Compute a steering vector from lists of per-step activation dicts.

    Convenience wrapper around compute_steering_vector for use with the output
    of extract_cot_activations. Stacks the mean token-level activations from
    each dict before computing the mean-difference direction.

    Args:
        faithful_dicts: List of {layer_idx: Tensor(seq_len, d_model)} dicts
            from faithful reasoning steps.
        unfaithful_dicts: List of {layer_idx: Tensor(seq_len, d_model)} dicts
            from unfaithful reasoning steps.
        layer: Which residual stream layer to extract from.

    Returns:
        L2-normalized steering vector of shape (d_model,).
    """
    faithful_acts = torch.stack([d[layer].float().mean(dim=0) for d in faithful_dicts])
    unfaithful_acts = torch.stack([d[layer].float().mean(dim=0) for d in unfaithful_dicts])
    return compute_steering_vector(faithful_acts, unfaithful_acts, layer)


def save_steering_vector(
    vector: torch.Tensor,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save a steering vector and optional metadata to disk.

    Args:
        vector: Steering vector tensor to save.
        path: Output file path (.pt).
        metadata: Optional dict of provenance info (e.g., layer, model_name,
            n_faithful_samples, accuracy).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vector, "metadata": metadata or {}}, path)


def load_steering_vector(path: str) -> tuple[torch.Tensor, dict]:
    """
    Load a steering vector from disk.

    Args:
        path: Path to a .pt file saved by save_steering_vector.

    Returns:
        (vector, metadata) tuple. Vector is on CPU; move to device as needed.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["vector"], data["metadata"]
