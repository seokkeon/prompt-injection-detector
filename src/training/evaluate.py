"""
Evaluate a trained model against the test set or custom input.

Usage:
  python -m src.training.evaluate
  python -m src.training.evaluate --text "Ignore all previous instructions"
  python -m src.training.evaluate --file path/to/custom.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# No os.chdir — use explicit PROJECT_ROOT paths instead

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.logger import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def load_model(model_path: str):
    if not os.path.exists(model_path):
        logger.error(f"No trained model at {model_path}. Run train.py first.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, device) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    inj = probs[1].item()
    return {
        "text":             text[:120] + "..." if len(text) > 120 else text,
        "is_injection":     inj >= 0.5,
        "label":            "INJECTION" if inj >= 0.5 else "benign",
        "injection_prob":   round(inj, 4),
        "benign_prob":      round(probs[0].item(), 4),
    }


def evaluate_file(path: str, tokenizer, model, device) -> None:
    with open(path) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    labels, preds, probs = [], [], []
    for ex in examples:
        r = predict(ex["text"], tokenizer, model, device)
        labels.append(ex["label"])
        preds.append(1 if r["is_injection"] else 0)
        probs.append(r["injection_prob"])

    print("\n=== Evaluation Results ===\n")
    print(classification_report(labels, preds, target_names=["benign", "injection"]))
    cm = confusion_matrix(labels, preds)
    print(f"Confusion Matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")
    try:
        print(f"\nROC-AUC: {roc_auc_score(labels, probs):.4f}")
    except Exception:
        pass


def run(model_path="models/trained", test_file="data/prepared/test.jsonl", text=None, custom_file=None):
    if not AVAILABLE:
        logger.error("Install: pip install torch transformers scikit-learn")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(model_path)
    model.to(device)

    if text:
        r = predict(text, tokenizer, model, device)
        print(f"\nInput : {r['text']}\nLabel : {r['label']}\nInjection prob: {r['injection_prob']:.4f}\nBenign prob:    {r['benign_prob']:.4f}")
    elif custom_file:
        evaluate_file(custom_file, tokenizer, model, device)
    else:
        evaluate_file(test_file, tokenizer, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/trained")
    parser.add_argument("--test-file",  default="data/prepared/test.jsonl")
    parser.add_argument("--text",       default=None)
    parser.add_argument("--file",       default=None)
    args = parser.parse_args()
    run(args.model_path, args.test_file, args.text, args.file)
