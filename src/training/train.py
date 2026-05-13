"""
Training pipeline for the prompt injection classifier.

Usage:
  # Step 1 — Prepare dataset first:
  python -m src.data.prepare_dataset --no-hf

  # Step 2 — Train:
  python -m src.training.train

  # Or with custom args:
  python -m src.training.train --epochs 5 --batch-size 16 --model distilbert-base-uncased
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve project root — works both as script and -m module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.logger import logger

# ── Load .env so HF_TOKEN and other vars are available ───────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — set env vars manually if not installed

# ── Authenticate HuggingFace if token is present ─────────────────────────────
_hf_token = os.getenv("HF_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
        logger.info("HuggingFace authenticated via HF_TOKEN.")
    except ImportError:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token  # fallback

# ── Dependency checks ─────────────────────────────────────────────────────────

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
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        f1_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class InjectionDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 256):
        self.labels = [ex["label"] for ex in examples]
        texts = [ex["text"] for ex in examples]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
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
        logger.error(f"File not found: {path}")
        logger.error("Run prepare_dataset.py first to generate training data.")
        sys.exit(1)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(labels: List[int], preds: List[int], probs: List[float]) -> Dict:
    metrics = {}

    if SKLEARN_AVAILABLE:
        report = classification_report(
            labels, preds,
            target_names=["benign", "injection"],
            output_dict=True,
        )
        metrics["classification_report"] = report
        metrics["f1_injection"] = report["injection"]["f1-score"]
        metrics["precision_injection"] = report["injection"]["precision"]
        metrics["recall_injection"] = report["injection"]["recall"]
        metrics["accuracy"] = report["accuracy"]
        metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except Exception:
            pass

    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            avg = total_loss / (step + 1)
            logger.info(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | Loss: {avg:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, List[int], List[int], List[float]]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds  = []
    all_probs  = []

    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"].tolist()
        outputs = model(**batch)

        total_loss += outputs.loss.item()

        probs = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()
        preds = (outputs.logits.argmax(dim=1)).tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)

    return total_loss / len(loader), all_labels, all_preds, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    data_dir: str   = "data/prepared",
    output_dir: str = "models/trained",
    model_name: str = "distilbert-base-uncased",
    epochs: int     = 3,
    batch_size: int = 16,
    lr: float       = 2e-5,
    max_length: int = 256,
    warmup_ratio: float = 0.1,
) -> None:

    if not TORCH_AVAILABLE:
        logger.error(
            "PyTorch / Transformers not installed.\n"
            "Install with: pip install torch transformers"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load data
    logger.info("Loading datasets...")
    train_examples = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_examples   = load_jsonl(os.path.join(data_dir, "val.jsonl"))
    test_examples  = load_jsonl(os.path.join(data_dir, "test.jsonl"))

    logger.info(
        f"  Train: {len(train_examples):,} | "
        f"Val: {len(val_examples):,} | "
        f"Test: {len(test_examples):,}"
    )

    # ── Tokenizer & model
    hf_token = os.getenv("HF_TOKEN") or None
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, token=hf_token
    )
    model.to(device)

    # ── Datasets & loaders
    train_ds = InjectionDataset(train_examples, tokenizer, max_length)
    val_ds   = InjectionDataset(val_examples,   tokenizer, max_length)
    test_ds  = InjectionDataset(test_examples,  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ── Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop
    best_val_f1   = 0.0
    best_epoch    = 0
    training_log  = []

    logger.info(f"\nStarting training for {epochs} epoch(s)...")
    logger.info(f"  Total steps : {total_steps:,} | Warmup: {warmup_steps}")

    for epoch in range(1, epochs + 1):
        logger.info(f"\n── Epoch {epoch}/{epochs} ──")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_labels, val_preds, val_probs = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)

        val_f1  = val_metrics.get("f1_injection", 0.0)
        val_acc = val_metrics.get("accuracy", 0.0)

        logger.info(
            f"  Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1 (injection): {val_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_f1":     round(val_f1, 4),
            "val_acc":    round(val_acc, 4),
        })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  ✓ New best model saved (F1={val_f1:.4f}) → {output_dir}")

    # ── Final evaluation on test set
    logger.info(f"\n── Final evaluation on test set (best epoch: {best_epoch}) ──")

    # Reload best model
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    _, test_labels, test_preds, test_probs = evaluate(best_model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    logger.info(f"  Test F1 (injection)  : {test_metrics.get('f1_injection', 0):.4f}")
    logger.info(f"  Test Accuracy        : {test_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Test ROC-AUC         : {test_metrics.get('roc_auc', 0):.4f}")

    if "classification_report" in test_metrics:
        logger.info("\nClassification Report:")
        logger.info(
            classification_report(
                test_labels, test_preds,
                target_names=["benign", "injection"]
            )
        )
    if "confusion_matrix" in test_metrics:
        cm = test_metrics["confusion_matrix"]
        logger.info(f"Confusion Matrix:\n  TN={cm[0][0]} FP={cm[0][1]}\n  FN={cm[1][0]} TP={cm[1][1]}")

    # ── Save full results
    results = {
        "model_name":   model_name,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "training_log": training_log,
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if k != "classification_report"
        },
    }
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved → {results_path}")
    logger.info(f"Model saved   → {output_dir}")
    logger.info("\nTo use the trained model:")
    logger.info(f"  classifier = PromptInjectionClassifier(model_path='{output_dir}')")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prompt injection classifier")
    parser.add_argument("--data-dir",   default="data/prepared")
    parser.add_argument("--output-dir", default="models/trained")
    parser.add_argument("--model",      default="distilbert-base-uncased",
                        help="HuggingFace model name. Options: "
                             "distilbert-base-uncased (fast, CPU-friendly, default), "
                             "bert-base-uncased (balanced), "
                             "microsoft/deberta-v3-base (best accuracy, needs sentencepiece + GPU)")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int,   default=256)
    args = parser.parse_args()

    run(
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        model_name  = args.model,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        max_length  = args.max_length,
    )
