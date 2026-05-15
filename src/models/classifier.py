import os
from typing import Dict, List, Optional
from src.utils.logger import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class InjectionDataset:
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class PromptInjectionClassifier:
    def __init__(self, model_name="microsoft/deberta-v3-base", model_path=None):
        if not TRANSFORMERS_AVAILABLE:
            self._available = False
            return
        self._available = True
        hf_token = os.getenv("HF_TOKEN") or None
        if model_path and os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, token=hf_token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, text: str) -> Dict:
        if not self._available:
            return {"available": False}
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1)[0]
        return {
            "available": True,
            "is_injection": probs[1].item() > 0.5,
            "injection_probability": round(probs[1].item(), 4),
            "benign_probability":    round(probs[0].item(), 4),
        }

    def save(self, path: str):
        if not self._available: return
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
