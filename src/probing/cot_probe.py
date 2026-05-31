import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
from transformer_lens import HookedTransformer
from typing import Optional


def extract_cot_activations(
    model: HookedTransformer,
    prompt: str,
    cot_steps: Optional[list] = None,
) -> dict:
    """
    Extract residual stream activations at every layer for a prompt.

    Runs a single forward pass using TransformerLens hooks that capture
    hook_resid_post at every transformer block. hook_resid_post is the
    residual stream *after* the MLP block — the most informative position
    for probing semantic content in the model's internal representation.

    This is the primary extraction function for the probing pipeline:
    call it once per (prompt, corruption) pair to get the full activation
    table for NLDD and RSA computation.

    Args:
        model: HookedTransformer with hooks available at every layer.
        prompt: Input text string.
        cot_steps: Reserved for future step-level extraction; unused here.

    Returns:
        Dict {layer_idx: Tensor(seq_len, d_model)} for all n_layers.
        Tensors are detached and moved to CPU.
    """
    tokens = model.to_tokens(prompt)
    activations: dict = {}

    hooks = [
        (
            f"blocks.{layer}.hook_resid_post",
            lambda act, hook, l=layer: activations.__setitem__(l, act[0].detach().cpu()),
        )
        for layer in range(model.cfg.n_layers)
    ]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)

    return activations


def extract_token_activations(
    model: HookedTransformer,
    prompt: str,
    token_idx: int = -1,
) -> dict:
    """
    Extract residual stream activations for a single token position.

    Used by the closed-loop pipeline to extract the last-token activation
    at each generation step. The final token's hook_resid_post encodes the
    model's "next-token prediction state" — the representation most directly
    connected to the faithfulness of the next generated token.

    Args:
        model: HookedTransformer instance.
        prompt: Input text string.
        token_idx: Token position to slice (-1 = last token, default).

    Returns:
        Dict {layer_idx: Tensor(d_model,)} for the selected token position.
    """
    all_acts = extract_cot_activations(model, prompt)
    return {layer: acts[token_idx] for layer, acts in all_acts.items()}
