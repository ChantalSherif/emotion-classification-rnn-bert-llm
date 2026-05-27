# Emotion Classification: RNN vs. BERT vs. LLM Prompting

A comparative study of three NLP learning paradigms on a 6-class emotion classification task using the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset.

---

## Overview

This project benchmarks three fundamentally different approaches to text classification:

| Approach | Method | Training Required |
|---|---|---|
| From-Scratch RNN | LSTM/GRU with learned embeddings | Yes — full training |
| Transfer Learning | Fine-tuned DistilBERT | Yes — fine-tuning only |
| LLM Prompting | Phi-3-mini / Llama-3.2-3B (4-bit) | No — zero-shot/few-shot/CoT |

The goal is not just to find the best accuracy, but to understand **when** and **why** each paradigm wins.

---

## Dataset

**[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** — short English texts (tweet-length) labeled with one of six emotions:

`joy` · `sadness` · `anger` · `fear` · `love` · `surprise`

| Split | Examples |
|---|---|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

---

## Project Structure

```
├── emotion_classification.ipynb   # Main notebook (all 6 parts)
└── README.md
```

---

## Parts

### Part 1 — Data Exploration & Preprocessing
- Class distribution, text length statistics
- Custom tokenizer (for RNN) + AutoTokenizer (for BERT/LLM)
- Visualizations: class balance bar chart, token length histogram

### Part 2 — From-Scratch RNN Baseline
- LSTM/GRU with randomly initialized embeddings
- Reports: accuracy, macro F1, per-class F1, confusion matrix
- Measures training time & inference time

### Part 3 — Transfer Learning (DistilBERT)
- Fine-tuned `distilbert-base-uncased` with classification head
- 2–3 epochs at lr ≈ 2e-5
- Same evaluation metrics as Part 2

### Part 4 — LLM Prompting (No Training)
- Model: `microsoft/Phi-3-mini-4k-instruct` with 4-bit quantization
- Three prompting strategies: **zero-shot**, **few-shot (3 & 8 examples)**, **chain-of-thought**
- Includes raw prompt/output examples for each strategy

### Part 5 — Data Efficiency Experiment
- Retrains RNN and DistilBERT on only **500 labeled examples**
- Compares all three approaches at low vs. full data regimes
- Reveals what pretraining actually buys you

### Part 6 — Comparative Analysis
- Summary table: accuracy, macro F1, training time, inference time
- Failure mode analysis via confusion matrices
- Practical guidance: when to use each paradigm in production

---


This project benchmarks three fundamentally different approaches to text classification:

| Approach | Method | Training Required |
|---|---|---|
| From-Scratch RNN | LSTM/GRU with learned embeddings | Yes — full training |
| Transfer Learning | Fine-tuned DistilBERT | Yes — fine-tuning only |
| LLM Prompting | TinyLlama-1.1B (4-bit) | No — zero-shot/few-shot/CoT |

---

## Key Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| From-Scratch RNN | 34.75% | 0.0860 |
| Fine-Tuned DistilBERT | 92.30% | 0.8742 |
| LLM (Zero-Shot) | 32.72% | 0.2456 |
| LLM (Few-Shot 8) | 23.14% | 0.1273 |
| LLM (Chain-of-Thought) | 22.10% | 0.1486 |

---

## Setup & Requirements

```bash
pip install transformers datasets torch bitsandbytes accelerate scikit-learn matplotlib seaborn
```

Run on **Google Colab** (GPU recommended — T4 or better for LLM inference).

---

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `datasets` · `bitsandbytes` · `scikit-learn` · `Matplotlib` · `Seaborn`

---

## Course Info

Advanced Machine Learning — Spring 2026
German International University of Applied Sciences
