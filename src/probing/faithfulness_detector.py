import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
from transformer_lens import HookedTransformer
import numpy as np
import random
import re
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from typing import Optional

from src.probing.cot_probe import extract_cot_activations


def corrupt_step(step: str, strategy: str = "number_perturbation", seed: int = 42) -> str:
    """
    Generate a counterfactual version of a CoT step via a corruption strategy.

    Corruption creates a plausible-but-wrong reasoning step. The faithfulness
    signal (NLDD) measures how much the model's answer changes under this
    counterfactual — if the answer changes a lot, the model was actually using
    that step (faithful); if it barely changes, the model was ignoring it.

    Strategies:
        number_perturbation: Perturbs all numeric literals by ±10–50%.
            Best for math reasoning where numbers are causally critical.
        negate: Inserts a negation into the first matching verb phrase.
            Best for logical/deductive reasoning chains.
        shuffle: Randomly shuffles words in the step.
            Destroys semantic content as a stronger baseline.

    Args:
        step: One reasoning step string.
        strategy: 'number_perturbation', 'negate', or 'shuffle'.
        seed: Random seed for reproducibility across runs.

    Returns:
        Corrupted version of the input step.
    """
    rng = random.Random(seed)

    if strategy == "number_perturbation":
        def perturb(match: re.Match) -> str:
            n = float(match.group())
            factor = 1.0 + rng.uniform(0.1, 0.5) * rng.choice([-1, 1])
            result = n * factor
            return str(int(result)) if result == int(result) else f"{result:.2f}"

        return re.sub(r"\b\d+(?:\.\d+)?\b", perturb, step)

    elif strategy == "negate":
        negation_map = {
            " is ": " is not ",
            " are ": " are not ",
            " has ": " does not have ",
            " have ": " do not have ",
            " equals ": " does not equal ",
            " was ": " was not ",
        }
        for phrase, negated in negation_map.items():
            if phrase in step:
                return step.replace(phrase, negated, 1)
        words = step.split()
        if len(words) > 2:
            words.insert(2, "not")
        return " ".join(words)

    elif strategy == "shuffle":
        words = step.split()
        if len(words) > 2:
            rng.shuffle(words)
        return " ".join(words)

    return step


def generate_faithfulness_labels(
    model: HookedTransformer,
    prompt: str,
    cot_steps: list[str],
    strategy: str = "number_perturbation",
    nldd_threshold: float = 0.1,
) -> list[tuple]:
    """
    Label each CoT step as faithful or unfaithful using NLDD.

    Algorithm (per step k):
      1. Run clean forward pass on full prompt → log P(answer | clean CoT).
      2. Replace step k with corrupt_step(step k) in the prompt.
      3. Run corrupted forward pass → log P(answer | corrupted CoT).
      4. NLDD_k = clean_log_p − corrupted_log_p.
      5. Label: faithful if NLDD_k > nldd_threshold.

    The clean log-probability is computed once and reused for all steps,
    since the clean CoT is shared across all corruptions.

    Args:
        model: HookedTransformer instance.
        prompt: Full prompt string (including all CoT steps).
        cot_steps: List of individual reasoning step strings.
        strategy: Corruption strategy for corrupt_step.
        nldd_threshold: NLDD above this value → faithful label.

    Returns:
        List of (step_idx: int, is_faithful: bool, nldd: float).
    """
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        clean_logits = model(tokens)[0, -1]
    clean_log_p = float(torch.log_softmax(clean_logits, dim=-1).max())

    results = []
    for idx, step in enumerate(cot_steps):
        corrupted_step = corrupt_step(step, strategy=strategy, seed=idx)
        corrupted_prompt = prompt.replace(step, corrupted_step, 1)

        corrupted_tokens = model.to_tokens(corrupted_prompt)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)[0, -1]
        corrupted_log_p = float(torch.log_softmax(corrupted_logits, dim=-1).max())

        nldd = clean_log_p - corrupted_log_p
        results.append((idx, nldd > nldd_threshold, nldd))

    return results


class FaithfulnessProbe:
    """
    Linear probe for classifying residual stream activations as faithful/unfaithful.

    Trains a logistic regression classifier on hook_resid_post activation vectors
    at a fixed layer. The learned weight vector (coef_) defines a linear direction
    in activation space separating faithful from unfaithful reasoning states.

    This direction is also usable as a steering vector: adding it to the residual
    stream pushes the model toward the faithful activation cluster. See
    src/steering/steering_vectors.py for the compute_steering_vector approach,
    which uses mean-difference instead of the probe weight directly.
    """

    def __init__(self, layer: int) -> None:
        self.layer = layer
        self._probe = LogisticRegression(max_iter=5000, random_state=42)
        self._trained = False
        self._pca: Optional[PCA] = None

    def train(
        self,
        activations: np.ndarray,
        labels: list[bool],
        pca_components: Optional[int] = None,
    ) -> float:
        """
        Fit the probe on activation vectors with cross-validated accuracy.

        Args:
            activations: shape (n_samples, d_model).
            labels: Boolean faithfulness labels, length n_samples.
            pca_components: If set, reduce activations to this many PCA components
                before training. Fitted PCA is stored as self._pca and applied
                automatically in predict() and direction. Recommended when
                d_model >> n_samples (e.g. LLaMA d_model=4096, n<500).

        Returns:
            Mean cross-validated accuracy (5-fold or n_samples-fold if n < 10).
        """
        y = np.array([int(lbl) for lbl in labels])
        X = activations
        if pca_components is not None and pca_components < X.shape[1]:
            self._pca = PCA(n_components=pca_components, random_state=42)
            X = self._pca.fit_transform(X)
        else:
            self._pca = None
        self._probe.fit(X, y)
        self._trained = True
        n_splits = min(5, len(y))
        scores = cross_val_score(self._probe, X, y, cv=n_splits)
        return float(scores.mean())

    def predict(self, activations: np.ndarray) -> list[tuple[bool, float]]:
        """
        Predict faithfulness labels with per-sample confidence scores.

        Args:
            activations: shape (n_samples, d_model) or (d_model,) for a single vector.
                If the probe was trained with PCA, the transform is applied automatically.

        Returns:
            List of (is_faithful: bool, confidence: float) tuples.
            Confidence is the probability assigned to the predicted class
            (p[1] when faithful, p[0] when unfaithful), so it is always >= 0.5
            and directly comparable against a steering threshold.
        """
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)
        if self._pca is not None:
            activations = self._pca.transform(activations)
        probs = self._probe.predict_proba(activations)
        return [(bool(p[1] > 0.5), float(p[1] if p[1] > 0.5 else p[0])) for p in probs]

    @property
    def direction(self) -> torch.Tensor:
        """
        The probe's learned decision boundary normal as a steering direction.

        When PCA was used, the weight vector is mapped back to the original
        activation space via pca.components_.T @ w before normalising, so the
        returned vector is always in d_model-dimensional space and compatible
        with apply_steering regardless of whether PCA was used.
        """
        w = self._probe.coef_[0]
        if self._pca is not None:
            w = self._pca.components_.T @ w
        v = torch.tensor(w, dtype=torch.float32)
        return v / v.norm()

    def save(self, path: str) -> None:
        """Serialize probe state (including fitted PCA if present) to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"layer": self.layer, "probe": self._probe, "trained": self._trained, "pca": self._pca},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FaithfulnessProbe":
        """Deserialize a saved probe from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(layer=data["layer"])
        obj._probe = data["probe"]
        obj._trained = data["trained"]
        obj._pca = data.get("pca", None)
        return obj


def train_probe(
    activations: np.ndarray,
    labels: list[bool],
    layer: int = 0,
) -> FaithfulnessProbe:
    """
    Functional wrapper: construct and train a FaithfulnessProbe.

    Args:
        activations: shape (n_samples, d_model).
        labels: Boolean faithfulness labels.
        layer: Layer index to associate with this probe.

    Returns:
        Trained FaithfulnessProbe instance.
    """
    probe = FaithfulnessProbe(layer=layer)
    probe.train(activations, labels)
    return probe


def predict_faithfulness(
    probe: FaithfulnessProbe,
    activations: np.ndarray,
) -> list[tuple[bool, float]]:
    """
    Functional wrapper: run predict on an existing probe.

    Args:
        probe: Trained FaithfulnessProbe.
        activations: shape (n_samples, d_model) or (d_model,).

    Returns:
        List of (is_faithful: bool, confidence: float).
    """
    return probe.predict(activations)
