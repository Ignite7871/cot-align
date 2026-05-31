import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import re
import random
from datasets import load_dataset
from typing import Optional


def load_gsm8k(split: str = "train", max_samples: int = 100) -> list[dict]:
    """
    Load GSM8K math reasoning dataset from HuggingFace.

    GSM8K contains grade-school math word problems with step-by-step solutions.
    Each solution is split into individual reasoning steps for NLDD labelling.

    Args:
        split: Dataset split ('train' or 'test').
        max_samples: Maximum number of examples to return.

    Returns:
        List of {'question': str, 'answer': str, 'steps': list[str]}.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    results = []
    for item in ds:
        if len(results) >= max_samples:
            break
        steps = parse_gsm8k_steps(item["answer"])
        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "steps": steps,
        })
    return results


def parse_gsm8k_steps(solution: str) -> list[str]:
    """
    Split a GSM8K solution string into individual reasoning steps.

    GSM8K solutions use newlines and '####' to delimit the final answer.
    This function splits on those boundaries and strips empty lines.

    Args:
        solution: Raw GSM8K solution string.

    Returns:
        List of non-empty step strings, with the final answer line included.
    """
    parts = solution.split("####")
    steps_text = parts[0].strip()
    steps = [s.strip() for s in re.split(r"\n+", steps_text) if s.strip()]
    if len(parts) > 1:
        steps.append(f"#### {parts[1].strip()}")
    return steps


def load_prontoqa(split: str = "train", max_samples: int = 100) -> list[dict]:
    """
    Load ProntoQA logical reasoning dataset from HuggingFace.

    ProntoQA contains deductive reasoning chains over synthetic ontologies.
    Tries 'fengyang0317/prontoqa' first (steps already a list), then falls
    back to legacy sources.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        max_samples: Maximum number of examples to return.

    Returns:
        List of {'question': str, 'answer': str, 'chain': list[str]}.
    """
    for hf_id in ("fengyang0317/prontoqa", "keirp/prontoqa", "andrewzou/prontoqa"):
        try:
            ds = load_dataset(hf_id, split=split)
            results = []
            for item in ds:
                if len(results) >= max_samples:
                    break
                raw = item.get("steps") or item.get("chain_of_thought") or item.get("proof") or ""
                if isinstance(raw, list):
                    chain = [s.strip() for s in raw if str(s).strip()]
                else:
                    chain = [s.strip() for s in re.split(r"\.\s+", str(raw)) if s.strip()]
                results.append({
                    "question": item.get("question", item.get("query", "")),
                    "answer": item.get("answer", item.get("label", "")),
                    "chain": chain,
                })
            return results
        except Exception:
            continue

    return []


def generate_dyck_sequence(n: int = 3, min_depth: int = 2, max_depth: int = 5) -> str:
    """
    Generate a valid Dyck-n bracket sequence of random depth.

    Dyck-n sequences use n types of matching bracket pairs. A valid sequence
    has every opening bracket matched by the correct closing bracket in order.
    These are used as synthetic reasoning benchmarks where each match/unmatch
    step forms one element of the CoT.

    Args:
        n: Number of bracket types (e.g. 3 → {(),[],{}}).
        min_depth: Minimum nesting depth.
        max_depth: Maximum nesting depth.

    Returns:
        A valid Dyck-n bracket sequence string.
    """
    openers = "([{<"[:n]
    closers = ")]}>"[:n]
    rng = random.Random()

    def gen(depth: int) -> str:
        if depth == 0:
            return ""
        i = rng.randrange(n)
        inner = gen(rng.randint(0, depth - 1))
        sibling = gen(depth - 1) if rng.random() < 0.5 else ""
        return f"{openers[i]}{inner}{closers[i]}{sibling}"

    depth = rng.randint(min_depth, max_depth)
    return gen(depth)


def load_dyck_n(n: int = 3, num_samples: int = 100) -> list[dict]:
    """
    Generate a Dyck-n synthetic dataset.

    Each sample is a bracket sequence with one reasoning step per matching
    bracket pair, making the causal structure fully observable for NLDD
    validation experiments.

    Args:
        n: Number of bracket types.
        num_samples: Number of sequences to generate.

    Returns:
        List of {'sequence': str, 'steps': list[str]}.
    """
    results = []
    for _ in range(num_samples):
        seq = generate_dyck_sequence(n=n)
        # Each top-level bracket pair is one reasoning step
        steps = [f"Match bracket at position {i}" for i, c in enumerate(seq) if c in "([{<"[:n]]
        results.append({"sequence": seq, "steps": steps or ["(empty sequence)"]})
    return results
