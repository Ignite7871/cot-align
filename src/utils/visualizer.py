from __future__ import annotations

import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.pipeline.closed_loop import PipelineTrace


def plot_faithfulness_trace(
    trace: "PipelineTrace",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of per-token faithfulness confidence over a pipeline run.

    Draws a confidence curve with red dashed vertical lines at every token
    step where the steering intervention was triggered. This lets you see
    whether steering events correlate with drops in the probe's confidence.
    Corresponds to Figure 3 in the paper.

    Args:
        trace: PipelineTrace from run_pipeline.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trace.confidences, label="Faithfulness confidence", color="steelblue", linewidth=1.5)
    for i, idx in enumerate(trace.steering_event_indices):
        label = "Steering event" if i == 0 else None
        ax.axvline(x=idx, color="red", linestyle="--", alpha=0.7, label=label)
    ax.set_xlabel("Token step")
    ax.set_ylabel("Confidence")
    ax.set_title("Faithfulness trace with steering events")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_nldd_by_step(
    nldd_scores: list,
    k_star: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of NLDD per reasoning step, replicating Figure 1 style.

    Blue bars indicate faithful steps (positive NLDD); red bars indicate
    unfaithful steps (negative NLDD). The reasoning horizon k* is marked
    with a dotted vertical line when provided.

    Args:
        nldd_scores: List of NLDD values, one per CoT step.
        k_star: Reasoning horizon index to mark, or None.
        save_path: Optional output path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["steelblue" if s > 0 else "salmon" for s in nldd_scores]
    ax.bar(range(len(nldd_scores)), nldd_scores, color=colors)
    if k_star is not None:
        ax.axvline(x=k_star, color="black", linestyle=":", linewidth=1.5, label=f"k* = {k_star}")
        ax.legend()
    ax.set_xlabel("Reasoning step")
    ax.set_ylabel("NLDD")
    ax.set_title("NLDD per reasoning step (Figure 1)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_probe_accuracy_by_layer(
    layer_scores: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of linear probe accuracy per transformer layer.

    Reveals which layers carry the most faithfulness-discriminative signal.
    Middle layers typically peak because they encode richer semantic content
    than early (syntactic) or late (output-preparation) layers.
    Corresponds to Figure 2a in the paper.

    Args:
        layer_scores: Dict {layer_idx: accuracy_score}.
        save_path: Optional output path.

    Returns:
        matplotlib Figure.
    """
    layers = sorted(layer_scores.keys())
    scores = [layer_scores[l] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, scores, color="steelblue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Faithfulness probe accuracy by layer")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pca_with_direction(
    activations: np.ndarray,
    probe_direction: torch.Tensor,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    PCA scatter of activation vectors with the probe direction overlaid.

    Projects activations to 2D via PCA, then draws an arrow showing where
    the probe's learned direction points in that space. Used to visually
    verify that the faithfulness direction aligns with the main activation
    cluster separation. Corresponds to Figure 2c in the paper.

    Args:
        activations: shape (n_samples, d_model).
        probe_direction: shape (d_model,), the normalized probe weight vector.
        save_path: Optional output path.

    Returns:
        matplotlib Figure.
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(activations)
    direction_2d = pca.transform(probe_direction.detach().numpy().reshape(1, -1))[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=20, c="steelblue")
    scale = float(np.abs(projected).max()) * 0.4
    ax.annotate(
        "",
        xy=direction_2d * scale,
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of activations with probe direction")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
