# CoT-Align

Detecting and correcting unfaithful Chain-of-Thought reasoning at inference time using mechanistic interpretability.

## Key Finding

**CoT faithfulness is domain-specific, not a global model property.**

LLaMA-3.1-8B uses its reasoning chain causally on arithmetic (NLDD = +0.081) but generates decorative CoT on logical deduction (NLDD = −0.051) — despite encoding both reasoning chains in its residual stream (RSA ≥ 0.96 in both cases). This RSA/NLDD dissociation means the model *represents* its proof chain but does not causally use it when answering logical deduction questions. GPT-2 generates decorative CoT on arithmetic regardless of scale.

| Metric | GPT-2 (GSM8K) | LLaMA-3.1-8B (GSM8K) | LLaMA-3.1-8B (PrOntoQA) |
|:---|:---:|:---:|:---:|
| Mean NLDD | −0.126 | **+0.081** | −0.051 |
| Faithful step fraction | 28.1% | 41.4% | 30.0% |
| Mean RSA | 0.996 | 0.966 | 0.982 |
| Mean TAS | 0.824 | 0.687 | 0.677 |
| k\* (normalised) | 0.169 | 0.240 | 0.201 |
| Steps evaluated | 442 | 897 | 1,200 |

**NLDD > 0** = corrupting a CoT step *reduces* model confidence → step is causally influential (faithful).  
**NLDD < 0** = corrupting a step *increases* model confidence → step is decorative (unfaithful).

## Installation

**Requirements:** Python 3.12, CUDA 12.4

```bash
# 1. Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Copy environment template and add your HuggingFace token
cp .env.example .env
# Edit .env — set HF_TOKEN (required only for LLaMA; GPT-2 runs without a token)
```

> **Windows DLL fix:** On Windows, `import pyarrow` must be the **first import** in every entry-point script, before `transformers` or `transformer_lens`. This resolves a DLL load-order conflict caused by CUDA and Arrow competing for the same system DLLs. All source files in this repository already include this header. If you write new scripts, add `import pyarrow` as the very first line.

## Quick Start

```python
import pyarrow 
import torch, sys, os
sys.path.insert(0, '.')

from src.utils.model_loader import get_model
from data.dataset_loader import load_gsm8k
from src.probing.faithfulness_detector import generate_faithfulness_labels

model = get_model('gpt2', device='cuda')
samples = load_gsm8k(split='test', max_samples=5)

for sample in samples:
    question, steps = sample['question'], sample['steps']
    prompt = f"Q: {question} A: {' '.join(steps)}"
    labels = generate_faithfulness_labels(model, prompt, steps, nldd_threshold=0.1)
    for i, (_, _, nldd) in enumerate(labels):
        status = 'faithful' if nldd > 0.1 else 'unfaithful'
        print(f"  Step {i+1}: NLDD={nldd:+.3f}  [{status}]")
```

## Project Structure

```
CoT-Align/
├── experiments/                 # Jupyter notebooks — run in order
│   ├── 01_baseline_cot.ipynb    # GPT-2 NLDD / RSA / TAS (GSM8K + PrOntoQA)
│   ├── 02_probing.ipynb         # GPT-2 faithfulness probe training
│   ├── 03_steering.ipynb        # GPT-2 activation steering
│   ├── 04_llama_baseline.ipynb  # LLaMA-3.1-8B baseline (GSM8K + PrOntoQA)
│   ├── 05_llama_probing.ipynb   # LLaMA faithfulness probe (PCA-50)
│   └── 06_llama_steering.ipynb  # LLaMA activation steering (M1 / M2)
├── src/
│   ├── pipeline/                # Closed-loop detection + correction
│   ├── probing/                 # NLDD labelling and linear probe
│   ├── steering/                # Activation steering vector construction
│   └── utils/                   # Model loader, metrics, visualiser
├── data/
│   └── dataset_loader.py        # GSM8K, PrOntoQA, Dyck-n loaders
├── api/                         # FastAPI inference server
├── tests/                       # pytest suite (31 tests)
├── outputs/                     # Generated figures and probe artefacts (git-ignored)
├── paper/                       # LaTeX source (git-ignored)
├── requirements.txt
├── .env.example
└── LICENSE
```

## Reproducing Results

Notebooks are self-contained and save all figures to `outputs/`.

**GPT-2 baseline** (2 GB VRAM, or CPU with `DEVICE=cpu`):
```bash
jupyter notebook experiments/01_baseline_cot.ipynb
```

**LLaMA-3.1-8B baseline** (≥6 GB VRAM; runs 4-bit quantised via BitsAndBytes):
```bash
HF_TOKEN=hf_your_token jupyter notebook experiments/04_llama_baseline.ipynb
```

**Test suite:**
```bash
pytest tests/ -v
```

## Metrics

| Metric | What it measures |
|:---|:---|
| **NLDD** (Next-token Log-probability Divergence Delta) | Causal influence of each CoT step on the final answer. Positive = faithful, negative = decorative. |
| **RSA** (Representational Similarity Analysis) | How much step corruption changes the residual stream at the answer position. Near 1 = no propagation. |
| **TAS** (Trajectory Alignment Score) | Geometric smoothness of the activation trajectory through the residual stream layers. |
| **k\*** | Normalised reasoning horizon — the step position beyond which faithfulness decays. |

## Citation

```bibtex
@misc{cotalign2026,
  title   = {CoT-Align: Detecting and Correcting Unfaithful Chain-of-Thought Reasoning},
  author  = {[Author]},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

[MIT](LICENSE)
