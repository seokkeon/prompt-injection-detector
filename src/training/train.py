"""
Training pipeline for the prompt injection classifier.

Usage:
  python -m src.training.train
  python -m src.training.train --model distilbert-base-uncased --epochs 3
  python -m src.training.train --model microsoft/deberta-v3-base --epochs 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_hf_token = os.getenv("HF_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
    except ImportError:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

from src.utils.logger import logger

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class InjectionDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 256):
        self.labels = [ex["label"] for ex in examples]
        self.encodings = tokenizer(
            [ex["text"] for ex in examples],
            truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        logger.error(f"File not found: {path}. Run prepare_dataset.py first.")
        sys.exit(1)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(labels, preds, probs) -> Dict:
    metrics = {}
    if SKLEARN_AVAILABLE:
        report = classification_report(labels, preds, target_names=["benign", "injection"], output_dict=True)
        metrics["classification_report"] = report
        metrics["f1_injection"]          = report["injection"]["f1-score"]
        metrics["precision_injection"]   = report["injection"]["precision"]
        metrics["recall_injection"]      = report["injection"]["recall"]
        metrics["accuracy"]              = report["accuracy"]
        metrics["confusion_matrix"]      = confusion_matrix(labels, preds).tolist()
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except Exception:
            pass
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss  = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if (step + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | Loss: {total_loss/(step+1):.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple:
    model.eval()
    total_loss, labels, preds, probs = 0.0, [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(**batch)
        total_loss += out.loss.item()
        p = torch.softmax(out.logits, dim=1)[:, 1].tolist()
        probs.extend(p)
        preds.extend(out.logits.argmax(dim=1).tolist())
        labels.extend(batch["labels"].tolist())
    return total_loss / len(loader), labels, preds, probs


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    data_dir:    str   = "data/prepared",
    output_dir:  str   = "models/trained",
    model_name:  str   = "distilbert-base-uncased",
    epochs:      int   = 3,
    batch_size:  int   = 16,
    lr:          float = 2e-5,
    max_length:  int   = 256,
    warmup_ratio: float = 0.1,
) -> None:

    if not TORCH_AVAILABLE:
        logger.error("PyTorch not installed. Run: pip install torch transformers")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_examples = load_jsonl(f"{data_dir}/train.jsonl")
    val_examples   = load_jsonl(f"{data_dir}/val.jsonl")
    test_examples  = load_jsonl(f"{data_dir}/test.jsonl")
    logger.info(f"Train: {len(train_examples):,} | Val: {len(val_examples):,} | Test: {len(test_examples):,}")

    hf_token  = os.getenv("HF_TOKEN") or None
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, token=hf_token)
    model.to(device)

    train_loader = DataLoader(InjectionDataset(train_examples, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(InjectionDataset(val_examples,   tokenizer, max_length), batch_size=batch_size)
    test_loader  = DataLoader(InjectionDataset(test_examples,  tokenizer, max_length), batch_size=batch_size)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1, best_epoch, log = 0.0, 0, []
    logger.info(f"Starting training — {epochs} epoch(s) | {total_steps} total steps")

    for epoch in range(1, epochs + 1):
        logger.info(f"\n── Epoch {epoch}/{epochs} ──")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_labels, val_preds, val_probs = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_f1  = val_metrics.get("f1_injection", 0.0)
        val_acc = val_metrics.get("accuracy", 0.0)
        logger.info(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        log.append({"epoch": epoch, "train_loss": round(train_loss,4), "val_loss": round(val_loss,4), "val_f1": round(val_f1,4), "val_acc": round(val_acc,4)})

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  ✓ Best model saved (F1={val_f1:.4f}) → {output_dir}")

    logger.info(f"\n── Final evaluation on test set (best epoch: {best_epoch}) ──")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir, token=hf_token).to(device)
    _, test_labels, test_preds, test_probs = evaluate(best_model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    logger.info(f"  Test F1       : {test_metrics.get('f1_injection',0):.4f}")
    logger.info(f"  Test Accuracy : {test_metrics.get('accuracy',0):.4f}")
    logger.info(f"  Test ROC-AUC  : {test_metrics.get('roc_auc',0):.4f}")

    if SKLEARN_AVAILABLE:
        logger.info("\n" + classification_report(test_labels, test_preds, target_names=["benign","injection"]))
        cm = confusion_matrix(test_labels, test_preds)
        logger.info(f"Confusion Matrix:\n  TN={cm[0][0]} FP={cm[0][1]}\n  FN={cm[1][0]} TP={cm[1][1]}")

    results = {
        "model_name": model_name, "best_epoch": best_epoch,
        "best_val_f1": best_val_f1, "training_log": log,
        "test_metrics": {k:v for k,v in test_metrics.items() if k != "classification_report"},
    }
    with open(f"{output_dir}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDone! Model saved → {output_dir}")
    logger.info(f"To use: detector = UnifiedDetector(model_path='{output_dir}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/prepared")
    parser.add_argument("--output-dir", default="models/trained")
    parser.add_argument("--model",      default="distilbert-base-uncased",
                        help="distilbert-base-uncased (fast/CPU) | microsoft/deberta-v3-base (best, needs GPU)")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int,   default=256)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.model, args.epochs, args.batch_size, args.lr, args.max_length)
