import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
from transformer_lens import HookedTransformer
from contextlib import contextmanager
from typing import Generator


@contextmanager
def apply_steering(
    model: HookedTransformer,
    steering_vector: torch.Tensor,
    layer: int,
    alpha: float = 1.0,
) -> Generator:
    """
    Context manager: add a steering vector to the residual stream at one layer.

    Injects a persistent hook at hook_resid_post[layer] that performs:

        h_post = h_post + alpha * v

    for every token in every forward pass executed inside the context block.
    Hooks are removed in the finally clause, guaranteeing cleanup even when
    an exception is raised inside the block.

    This is the minimal mechanistic intervention: we add the faithful-direction
    vector to the model's intermediate representation, nudging its internal
    state toward the faithful activation cluster without retraining any weights.

    Args:
        model: HookedTransformer instance.
        steering_vector: shape (d_model,); will be broadcast across the token
            dimension automatically.
        layer: Which block's hook_resid_post to intervene at.
        alpha: Intervention strength. 0 = no change; 1 = standard; >1 = amplified.
            Larger alpha may degrade fluency; tune empirically.

    Yields:
        Nothing; side-effects are applied to model hooks in-place.
    """
    device = next(model.parameters()).device
    v = steering_vector.to(device)

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        return act + alpha * v.to(act.dtype)

    hook_name = f"blocks.{layer}.hook_resid_post"
    model.add_hook(hook_name, hook_fn)
    try:
        yield
    finally:
        model.reset_hooks()


@contextmanager
def apply_multi_layer_steering(
    model: HookedTransformer,
    vectors_by_layer: dict[int, torch.Tensor],
    alpha: float = 1.0,
) -> Generator:
    """
    Context manager: simultaneously steer at multiple residual stream layers.

    Registers one hook per layer in vectors_by_layer, each applying the
    corresponding steering vector scaled by the shared alpha. Useful for
    comparing single-layer vs. multi-layer interventions and for applying
    probe directions at multiple positions in the residual stream.

    Args:
        model: HookedTransformer instance.
        vectors_by_layer: Dict {layer_idx: steering_vector(d_model,)}.
        alpha: Shared intervention strength across all layers.

    Yields:
        Nothing; side-effects are applied to model hooks in-place.
    """
    device = next(model.parameters()).device

    for layer, v in vectors_by_layer.items():
        v_dev = v.to(device)

        def hook_fn(act: torch.Tensor, hook, vec=v_dev) -> torch.Tensor:
            return act + alpha * vec

        model.add_hook(f"blocks.{layer}.hook_resid_post", hook_fn)

    try:
        yield
    finally:
        model.reset_hooks()


def steered_forward(
    model: HookedTransformer,
    tokens: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    One-shot forward pass with activation steering applied.

    Convenience helper that wraps apply_steering + model forward pass into
    a single call. Useful for quick experiments and the run_pipeline hot path.

    Args:
        model: HookedTransformer instance.
        tokens: shape (batch, seq_len) or (seq_len,).
        steering_vector: shape (d_model,).
        layer: Layer to intervene at.
        alpha: Intervention strength.

    Returns:
        Logits tensor of shape (batch, seq_len, vocab_size).
    """
    with apply_steering(model, steering_vector, layer, alpha):
        with torch.no_grad():
            return model(tokens)
