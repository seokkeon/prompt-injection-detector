# Training Guide

## Prerequisites

```bash
pip install torch transformers scikit-learn datasets huggingface_hub sentencepiece protobuf loguru beautifulsoup4 lxml python-dotenv
```

Add your HuggingFace token to `.env`:
```
HF_TOKEN=hf_your_token_here
```

---

## Step 1 — Prepare Dataset

Merges three sources:
- `data/raw/data.jsonl` — your uploaded file (benign + auto-labeled injections)
- `deepset/prompt-injections` — HuggingFace labeled dataset
- `JasperLS/prompt-injections` — HuggingFace labeled dataset
- `data/malicious_samples/samples.json` — hand-curated

```bash
# With HuggingFace (recommended)
python -m src.data.prepare_dataset

# Offline only
python -m src.data.prepare_dataset --no-hf
```

Check what was generated:
```bash
cat data/prepared/stats.json
```

---

## Step 2 — Train

```bash
# Fast — CPU-friendly (recommended to start)
python -m src.training.train --model distilbert-base-uncased --epochs 3

# Best accuracy — needs sentencepiece + protobuf + ideally GPU
python -m src.training.train --model microsoft/deberta-v3-base --epochs 5 --batch-size 8
```

Target metrics:
- Val F1 (injection) > 0.85 ✅
- ROC-AUC > 0.90 ✅

---

## Step 3 — Evaluate

```bash
# Full test set report
python -m src.training.evaluate

# Single text
python -m src.training.evaluate --text "Ignore all previous instructions"
```

---

## Step 4 — Use via API

```bash
uvicorn api.main:app --reload
```

Check `/health` — `ml_enabled: true` confirms the trained model loaded.

---

## Model Comparison

| Model | Extra deps | CPU speed | Accuracy |
|-------|-----------|-----------|---------|
| distilbert-base-uncased | None | Fast ⚡ | Good |
| bert-base-uncased | None | Medium | Good |
| microsoft/deberta-v3-base | sentencepiece, protobuf | Slow 🐢 | Best ✅ |

---

## Troubleshooting

**ModuleNotFoundError: No module named 'src'**
→ Always run from the project root directory

**deberta tokenizer error**
→ `pip install sentencepiece protobuf`
→ `rm -rf ~/.cache/huggingface/hub/models--microsoft--deberta-v3-base`

**Training too slow on CPU**
→ Switch to distilbert: `--model distilbert-base-uncased`
→ Or run overnight with: `nohup python -m src.training.train ... &`

**No injection examples found**
→ Run `prepare_dataset.py` first
→ Or check internet access for HuggingFace download

**Low F1 score (< 0.75)**
→ Add more labeled data to `data/malicious_samples/samples.json`
→ Run `prepare_dataset.py` again then retrain
