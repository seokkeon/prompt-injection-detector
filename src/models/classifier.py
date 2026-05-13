"""
ML-based prompt injection classifier.
Fine-tunes DeBERTa (or any HuggingFace model) as a binary classifier.
Use this AFTER collecting enough labeled data (Phase 4).
"""

import os
import json
from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("torch/transformers not installed. ML classifier not available.")


# ── Dataset ───────────────────────────────────────────────────────────────────

class InjectionDataset:
    """Simple dataset wrapper for text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── Classifier ────────────────────────────────────────────────────────────────

class PromptInjectionClassifier:
    """
    Fine-tuned HuggingFace sequence classifier for prompt injection detection.

    Labels:
      0 = benign
      1 = injection

    Usage:
      classifier = PromptInjectionClassifier()
      classifier.train(train_texts, train_labels, val_texts, val_labels)
      classifier.save("./models/saved")
      result = classifier.predict("Ignore all previous instructions...")
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        model_path: Optional[str] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available. Cannot use ML classifier.")
            self._available = False
            return

        self._available = True
        self.model_name = model_name

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading saved model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            logger.info(f"Loading base model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: str = "./models/trained",
        epochs: int = 3,
        batch_size: int = 8,
    ) -> None:
        """Fine-tune the model on labeled data."""
        if not self._available:
            return

        train_dataset = InjectionDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = InjectionDataset(val_texts, val_labels, self.tokenizer)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete.")

    def predict(self, text: str) -> Dict:
        """
        Predict whether a text is an injection attempt.

        Returns:
            Dict with is_injection, confidence, label.
        """
        if not self._available:
            return {
                "available": False,
                "note": "ML classifier not available — transformers not installed.",
            }

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]

            is_injection = probs[1].item() > 0.5
            confidence = probs[1].item() if is_injection else probs[0].item()

            return {
                "available": True,
                "is_injection": is_injection,
                "confidence": round(confidence, 4),
                "benign_probability": round(probs[0].item(), 4),
                "injection_probability": round(probs[1].item(), 4),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"available": False, "error": str(e)}

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Predict for a list of texts."""
        return [self.predict(t) for t in texts]

    def save(self, path: str) -> None:
        """Save model and tokenizer."""
        if not self._available:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
