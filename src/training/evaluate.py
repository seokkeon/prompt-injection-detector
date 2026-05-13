"""
Evaluate a trained model against the test set or any custom input.

Usage:
  # Evaluate against test set
  python -m src.training.evaluate

  # Evaluate a single string
  python -m src.training.evaluate --text "Ignore all previous instructions"

  # Evaluate a custom JSONL file
  python -m src.training.evaluate --file path/to/custom.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
        logger.error(f"Model not found at {model_path}. Train first.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_single(text: str, tokenizer, model, device) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=256, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    injection_prob = probs[1].item()
    return {
        "text":               text[:120] + "..." if len(text) > 120 else text,
        "is_injection":       injection_prob >= 0.5,
        "injection_prob":     round(injection_prob, 4),
        "benign_prob":        round(probs[0].item(), 4),
        "label":              "INJECTION" if injection_prob >= 0.5 else "benign",
    }


def evaluate_file(file_path: str, tokenizer, model, device) -> None:
    with open(file_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    labels, preds, probs = [], [], []
    for ex in examples:
        result = predict_single(ex["text"], tokenizer, model, device)
        labels.append(ex["label"])
        preds.append(1 if result["is_injection"] else 0)
        probs.append(result["injection_prob"])

    print("\n=== Evaluation Results ===\n")
    print(classification_report(labels, preds, target_names=["benign", "injection"]))

    cm = confusion_matrix(labels, preds)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    try:
        auc = roc_auc_score(labels, probs)
        print(f"\nROC-AUC: {auc:.4f}")
    except Exception:
        pass


def run(
    model_path: str = "models/trained",
    test_file: str  = "data/prepared/test.jsonl",
    text: str       = None,
    custom_file: str = None,
) -> None:

    if not AVAILABLE:
        logger.error("Install torch, transformers, scikit-learn first.")
        sys.exit(1)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(model_path)
    model.to(device)

    if text:
        result = predict_single(text, tokenizer, model, device)
        print(f"\nInput : {result['text']}")
        print(f"Label : {result['label']}")
        print(f"Injection probability : {result['injection_prob']:.4f}")
        print(f"Benign probability    : {result['benign_prob']:.4f}")

    elif custom_file:
        evaluate_file(custom_file, tokenizer, model, device)

    else:
        evaluate_file(test_file, tokenizer, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  default="models/trained")
    parser.add_argument("--test-file",   default="data/prepared/test.jsonl")
    parser.add_argument("--text",        default=None, help="Single text to classify")
    parser.add_argument("--file",        default=None, help="Custom JSONL file to evaluate")
    args = parser.parse_args()

    run(
        model_path  = args.model_path,
        test_file   = args.test_file,
        text        = args.text,
        custom_file = args.file,
    )
