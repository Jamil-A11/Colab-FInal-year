# LLM Bias Mitigation & Evaluation (QLoRA)

Reducing implicit **gender and religious bias** in a large language model (**Qwen3-4B**) by comparing three mitigation strategies — a base model, prompt engineering, and **QLoRA parameter-efficient fine-tuning** — and evaluating them with a structured human study.

<!-- Optional: add badges once you like. Example:
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-yellow)
-->

---

## Overview

Large language models can reproduce societal biases present in their training data. This project investigates **how much of that implicit bias can be reduced**, and **which mitigation approach works best**, for a mid-sized open model running on limited (consumer/Colab) compute.

I compared three approaches head-to-head:

1. **Base model** — Qwen3-4B out of the box (baseline).
2. **Prompt engineering** — steering behaviour at inference time via carefully designed system/user prompts.
3. **QLoRA fine-tuning** — 4-bit quantised, parameter-efficient fine-tuning (PEFT) on a purpose-built dataset.

## Motivation

Full fine-tuning of a 4B-parameter model is expensive and often out of reach without significant hardware. I wanted to test whether **QLoRA** — which fine-tunes a small number of adapter parameters on top of a 4-bit quantised base — can meaningfully reduce bias **without full retraining**, and whether it outperforms the much cheaper option of prompt engineering.

## Methodology

- **Structured prompt-pair testing:** bias was probed using matched prompt pairs that differ only by a demographic attribute (e.g. gender or religion), so differences in the model's responses isolate implicit bias.
- **Synthetic counterfactual dataset:** I engineered a custom dataset of **400 prompt–response pairs** (JSONL) designed to steer the model toward **demographic neutrality** — counterfactual examples that treat matched demographics equivalently.
- **Efficient fine-tuning:** applied **4-bit quantisation + QLoRA/PEFT** so training runs on limited compute.
- **Human evaluation:** built a structured, numbered questionnaire and ran a human evaluation to rank the three approaches on how neutral/appropriate their responses were.

## Results

In the structured human evaluation, the three approaches ranked:

| Rank | Approach | Notes |
|------|----------|-------|
| 🥇 1st | **QLoRA fine-tuning** | Most consistent reduction in implicit bias |
| 🥈 2nd | Prompt engineering | Helpful and cheap, but less reliable |
| 🥉 3rd | Base model | Baseline |

**Takeaway:** targeted QLoRA fine-tuning on a small counterfactual dataset reduced implicit bias more effectively than prompt engineering alone — achieving this **without full-model retraining** and on limited hardware.

<!-- 👉 If you have exact numbers/plots (agreement scores, bias-score deltas, sample outputs), add them here — quantified results are the single biggest credibility boost for recruiters. -->

## Tech stack

- **Language:** Python
- **ML:** PyTorch, Hugging Face Transformers, PEFT (QLoRA), bitsandbytes (4-bit quantisation)
- **Model:** Qwen3-4B
- **Environment:** Google Colab

## Repository structure

<!-- 👉 Update to match your actual files. -->
```
.
├── data/                # Synthetic counterfactual dataset (JSONL, 400 pairs)
├── notebooks/           # Fine-tuning + evaluation notebooks
├── src/                 # Any scripts (data generation, training, inference)
├── results/             # Evaluation outputs / figures
└── README.md
```

## How to run

<!-- 👉 Adjust to your actual entry points. -->
1. Open the main notebook in Google Colab (GPU runtime recommended).
2. Install dependencies:
   ```bash
   pip install torch transformers peft bitsandbytes datasets accelerate
   ```
3. Load the base model (Qwen3-4B) and the counterfactual dataset from `data/`.
4. Run the fine-tuning notebook to produce the QLoRA adapter.
5. Run the evaluation notebook to reproduce the prompt-pair comparison across the three approaches.

## Key learnings

- End-to-end ownership of an ML problem: framing, dataset design, training, and **designing an evaluation** where no standard benchmark existed.
- Practical experience with **parameter-efficient fine-tuning** and **quantisation** to work within real compute constraints.
- The importance of **counterfactual data** and **matched-pair evaluation** when measuring something as subtle as implicit bias.

## Possible future work

- Expand beyond gender/religion to other demographic axes.
- Add automated bias metrics alongside the human evaluation.
- Test the approach on other open models and adapter configurations.

---

**Author:** Jamil Ahmed · BSc Computer Science, Brunel University London · [github.com/Jamil-A11](https://github.com/Jamil-A11)
