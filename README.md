# CoT-Align

**Detecting and correcting unfaithful Chain-of-Thought reasoning at inference time using mechanistic interpretability.**

CoT-Align investigates whether a language model's Chain-of-Thought (CoT) reasoning is actually causally used to produce its final answer, rather than merely being correlated with the model's internal computation.

The project combines **causal intervention, activation analysis, probing, and inference-time steering** to detect and correct reasoning steps that appear to be decorative rather than causally influential.

---

## 🔬 Key Finding

**Chain-of-Thought faithfulness is domain-dependent rather than a single global property of a model.**

Experiments on GPT-2 and LLaMA-3.1-8B show that the same model can use its reasoning chain causally on one task while producing largely decorative reasoning on another.

For example:

| Metric                 | GPT-2 (GSM8K) | LLaMA-3.1-8B (GSM8K) | LLaMA-3.1-8B (PrOntoQA) |
| ---------------------- | ------------: | -------------------: | ----------------------: |
| Mean NLDD              |        −0.126 |           **+0.081** |                  −0.051 |
| Faithful step fraction |         28.1% |                41.4% |                   30.0% |
| Mean RSA               |         0.996 |                0.966 |                   0.982 |
| Mean TAS               |         0.824 |                0.687 |                   0.677 |
| Normalised k*          |         0.169 |                0.240 |                   0.201 |
| Steps evaluated        |           442 |                  897 |                   1,200 |

### Interpretation

* **NLDD > 0:** corrupting a CoT step reduces model confidence, indicating that the step is causally influential.
* **NLDD < 0:** corrupting a CoT step increases confidence, indicating that the step behaves more like decorative reasoning.
* High RSA alongside negative NLDD can indicate that a reasoning chain is represented internally without being causally used for the final answer.

This dissociation between **representation and causal use** is the central observation investigated by CoT-Align.

---

## 🧠 What the Project Does

The system follows a closed-loop detection and correction pipeline:

```text
                    ┌─────────────────────┐
                    │   Question + CoT    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Faithfulness Probe │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Causal Intervention │
                    │       (NLDD)        │
                    └──────────┬──────────┘
                               ↓
                 ┌─────────────┴─────────────┐
                 ↓                           ↓
        Representational               Causal Analysis
        Analysis (RSA)                 + Classification
                 │                           │
                 └─────────────┬─────────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Activation Steering │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Corrected Inference │
                    └─────────────────────┘
```

---

## 🧪 Experimental Setup

### Models

* GPT-2
* LLaMA-3.1-8B

### Tasks

* GSM8K
* PrOntoQA
* Dyck-n datasets for supporting experiments

### Techniques

* Causal activation interventions
* Linear probing
* Activation steering
* Representational Similarity Analysis
* Next-token log-probability analysis

---

## 📐 Metrics

| Metric   | Purpose                                                              |
| -------- | -------------------------------------------------------------------- |
| **NLDD** | Measures causal influence of a CoT step on the model's final answer  |
| **RSA**  | Measures representational changes following step corruption          |
| **TAS**  | Measures geometric alignment of the activation trajectory            |
| **k***   | Estimates the effective reasoning horizon before faithfulness decays |

---

## 🏗️ Repository Structure

```text
cot-align/
├── experiments/
│   ├── 01_baseline_cot.ipynb
│   ├── 02_probing.ipynb
│   ├── 03_steering.ipynb
│   ├── 04_llama_baseline.ipynb
│   ├── 05_llama_probing.ipynb
│   └── 06_llama_steering.ipynb
│
├── src/
│   ├── pipeline/       # Closed-loop detection and correction
│   ├── probing/        # Faithfulness labelling and probes
│   ├── steering/       # Activation steering
│   └── utils/          # Model loading, metrics and visualisation
│
├── data/
│   └── dataset_loader.py
│
├── api/                # FastAPI inference server
├── tests/              # Automated tests
├── outputs/            # Generated experiment artefacts
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### Requirements

* Python 3.12
* CUDA 12.4 recommended
* NVIDIA GPU for the LLaMA experiments

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For LLaMA experiments:

```bash
cp .env.example .env
```

Add your Hugging Face token to `.env`.

### Run the test suite

```bash
pytest tests/ -v
```

### Run the GPT-2 experiment

```bash
jupyter notebook experiments/01_baseline_cot.ipynb
```

### Run the LLaMA experiment

```bash
jupyter notebook experiments/04_llama_baseline.ipynb
```

---

## 🖥️ Reproducibility

The repository is organized so that the experiments can be reproduced from the included notebooks and source modules.

All generated figures and probe artefacts are written to the `outputs/` directory.

The repository also includes a test suite covering the core pipeline components.

---

## 📊 Why This Matters

Large language models can generate reasoning traces that appear convincing without those traces necessarily being causally responsible for the final answer.

CoT-Align studies this problem at the level of **internal model computation**, rather than evaluating reasoning solely from the generated text.

This makes the project relevant to research in:

* Mechanistic Interpretability
* LLM Reliability
* AI Safety
* Reasoning Evaluation
* Causal Analysis of Neural Networks
* Inference-Time Model Steering

---

## 📚 Research

CoT-Align is part of my ongoing research into the reliability and internal behaviour of large language models.

Related work includes research on:

* dynamic precision routing
* distributional fragility in LLMs
* retrieval-augmented generation
* privacy-preserving machine learning

---

## ⚠️ Research Status

This repository is a research prototype intended for experimentation and reproducibility rather than production deployment.

Results depend on model architecture, task distribution, intervention methodology, and experimental configuration.

---

## 👤 Author

**Srikar Reddy Gunupati**

B.Tech Computer Science & Engineering (AI & ML)

Research interests: **LLM reliability, mechanistic interpretability, AI security, privacy-preserving ML, and intelligent systems.**

[LinkedIn](https://linkedin.com/in/srikar-reddy-gunupati)
