"""
Dataset preparation pipeline for prompt injection classifier training.

Sources merged:
  1. data/raw/data.jsonl                — your uploaded file (benign from ultrachat etc.)
  2. deepset/prompt-injections           — HuggingFace (662 labeled samples)
  3. JasperLS/prompt-injections          — HuggingFace (~500 labeled samples)
  4. data/malicious_samples/samples.json — hand-curated injections
  5. data/benign_samples/samples.json    — hand-curated benign

Output:
  data/prepared/train.jsonl
  data/prepared/val.jsonl
  data/prepared/test.jsonl
  data/prepared/stats.json

Usage:
  python -m src.data.prepare_dataset           # all sources
  python -m src.data.prepare_dataset --no-hf   # skip HuggingFace
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# No os.chdir — use explicit PROJECT_ROOT paths instead

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.detectors.unified_detector import UnifiedDetector as TextInjectionDetector  # merged
from src.utils.logger import logger

# ── Config ────────────────────────────────────────────────────────────────────

BENIGN_SOURCES        = {"ultrachat", "UltraInteract_pair", "HelpSteer"}
HARMFUL_SOURCE        = "PKU-SafeRLHF"
MAX_BENIGN_FROM_JSONL = 10_000
AUTO_LABEL_THRESHOLD  = 0.65
MAX_BENIGN_TEXT_LEN   = 1_500
MIN_TEXT_LEN          = 10
TRAIN_RATIO           = 0.80
VAL_RATIO             = 0.10
RANDOM_SEED           = 42
OUTPUT_DIR            = "data/prepared"

# All HuggingFace datasets to pull from
HF_DATASETS = [
    {
        "repo":        "deepset/prompt-injections",
        "text_field":  "text",
        "label_field": "label",
        "label_map":   {0: 0, 1: 1},  # 0=benign, 1=injection
    },
    {
        "repo":        "JasperLS/prompt-injections",
        "text_field":  "text",
        "label_field": "label",
        "label_map":   {0: 0, 1: 1},
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_example(text: str, label: int, source: str) -> Dict:
    return {"text": text.strip(), "label": label, "source": source}

def is_valid(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) >= MIN_TEXT_LEN

def dedupe(examples: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for ex in examples:
        key = ex["text"][:200]
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out

def save_jsonl(examples: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"  Saved {len(examples):,} → {path}")

def split_dataset(examples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(RANDOM_SEED)
    data = examples[:]
    random.shuffle(data)
    n_train = int(len(data) * TRAIN_RATIO)
    n_val   = int(len(data) * VAL_RATIO)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]

def balance(benign: List[Dict], injections: List[Dict]) -> List[Dict]:
    random.seed(RANDOM_SEED)
    # Cap benign at 3x injections to avoid heavy imbalance
    target = min(len(benign), len(injections) * 3)
    random.shuffle(benign)
    benign = benign[:target]
    # Oversample injections to match benign count
    if len(injections) < len(benign):
        mul        = (len(benign) // len(injections)) + 1
        injections = (injections * mul)[:len(benign)]
    combined = benign + injections
    random.shuffle(combined)
    counts = Counter(ex["label"] for ex in combined)
    logger.info(f"  After balancing: {counts[0]:,} benign | {counts[1]:,} injections")
    return combined

# ── Source 1: data/raw/data.jsonl ─────────────────────────────────────────────

def load_from_jsonl(path: str, detector: TextInjectionDetector) -> Tuple[List[Dict], List[Dict]]:
    if not os.path.exists(path):
        logger.warning(f"  Not found: {path} — skipping.")
        return [], []

    benign, injections, skipped = [], [], 0
    with open(path) as f:
        for line in f:
            obj    = json.loads(line)
            text   = obj.get("prompt", "")
            source = obj.get("source", "unknown")
            if not is_valid(text):
                skipped += 1
                continue
            if source in BENIGN_SOURCES and len(text) <= MAX_BENIGN_TEXT_LEN:
                benign.append(make_example(text, 0, source))
            elif source == HARMFUL_SOURCE:
                if detector.detect(text).risk_score >= AUTO_LABEL_THRESHOLD:
                    injections.append(make_example(text, 1, f"{source}_autolabeled"))

    random.seed(RANDOM_SEED)
    random.shuffle(benign)
    benign = benign[:MAX_BENIGN_FROM_JSONL]
    logger.info(f"  data.jsonl → {len(benign):,} benign | {len(injections):,} injections | {skipped} skipped")
    return benign, injections

# ── Source 2 & 3: HuggingFace ────────────────────────────────────────────────

def load_from_huggingface() -> Tuple[List[Dict], List[Dict]]:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("  `datasets` not installed. Run: pip install datasets")
        return [], []

    hf_token = os.getenv("HF_TOKEN") or None
    if hf_token:
        logger.info("  HF_TOKEN found — authenticated download.")
    else:
        logger.warning("  No HF_TOKEN — downloads may be slow. Add HF_TOKEN=hf_xxx to .env")

    all_benign, all_injections = [], []

    for cfg in HF_DATASETS:
        repo = cfg["repo"]
        try:
            logger.info(f"  Downloading {repo}...")
            ds = load_dataset(repo, token=hf_token)
            benign, injections = [], []

            for split in ds.values():
                for row in split:
                    text      = row.get(cfg["text_field"], "")
                    raw_label = row.get(cfg["label_field"], -1)
                    label     = cfg["label_map"].get(int(raw_label), -1)
                    if not is_valid(text) or label == -1:
                        continue
                    ex = make_example(text, label, repo)
                    (benign if label == 0 else injections).append(ex)

            logger.info(f"    {repo} → {len(benign):,} benign | {len(injections):,} injections")
            all_benign.extend(benign)
            all_injections.extend(injections)

        except Exception as e:
            logger.warning(f"  Failed to load {repo}: {e}")

    return all_benign, all_injections

# ── Source 4: Hand-curated samples ───────────────────────────────────────────

def load_from_curated(malicious_path: str, benign_path: str) -> Tuple[List[Dict], List[Dict]]:
    benign, injections = [], []
    for path, label, name in [
        (benign_path,    0, "curated_benign"),
        (malicious_path, 1, "curated_malicious"),
    ]:
        if not os.path.exists(path):
            logger.warning(f"  Not found: {path} — skipping.")
            continue
        with open(path) as f:
            for s in json.load(f):
                text = s.get("text", "")
                if is_valid(text):
                    (benign if label == 0 else injections).append(make_example(text, label, name))
    logger.info(f"  Curated → {len(benign):,} benign | {len(injections):,} injections")
    return benign, injections

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    jsonl_path:      str  = "data/raw/data.jsonl",
    malicious_path:  str  = "data/malicious_samples/samples.json",
    benign_path:     str  = "data/benign_samples/samples.json",
    use_huggingface: bool = True,
) -> None:

    detector       = TextInjectionDetector()
    all_benign     = []
    all_injections = []

    logger.info("\n── Source 1: data/raw/data.jsonl ──")
    b, i = load_from_jsonl(jsonl_path, detector)
    all_benign.extend(b); all_injections.extend(i)

    if use_huggingface:
        logger.info("\n── Sources 2 & 3: HuggingFace datasets ──")
        b, i = load_from_huggingface()
        all_benign.extend(b); all_injections.extend(i)

    logger.info("\n── Source 4: Curated samples ──")
    b, i = load_from_curated(malicious_path, benign_path)
    all_benign.extend(b); all_injections.extend(i)

    all_benign     = dedupe(all_benign)
    all_injections = dedupe(all_injections)

    logger.info(f"\n── Totals before balancing ──")
    logger.info(f"  Benign:     {len(all_benign):,}")
    logger.info(f"  Injections: {len(all_injections):,}")

    if not all_injections:
        logger.error("No injection examples found! Check data paths or internet connection.")
        return

    combined = balance(all_benign, all_injections)
    train, val, test = split_dataset(combined)

    logger.info("\n── Saving splits ──")
    save_jsonl(train, f"{OUTPUT_DIR}/train.jsonl")
    save_jsonl(val,   f"{OUTPUT_DIR}/val.jsonl")
    save_jsonl(test,  f"{OUTPUT_DIR}/test.jsonl")

    stats = {
        "total": len(combined), "train": len(train), "val": len(val), "test": len(test),
        "benign_total": len(all_benign), "injection_total": len(all_injections),
        "train_label_dist": dict(Counter(e["label"] for e in train)),
        "val_label_dist":   dict(Counter(e["label"] for e in val)),
        "test_label_dist":  dict(Counter(e["label"] for e in test)),
        "sources": dict(Counter(e["source"] for e in combined)),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "="*50)
    logger.info("Dataset ready!")
    logger.info(f"  Total : {stats['total']:,}")
    logger.info(f"  Train : {stats['train']:,}")
    logger.info(f"  Val   : {stats['val']:,}")
    logger.info(f"  Test  : {stats['test']:,}")
    logger.info(f"  Stats : {OUTPUT_DIR}/stats.json")
    logger.info("="*50)
    logger.info("\nNext: python -m src.training.train --model distilbert-base-uncased")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--jsonl",     default="data/raw/data.jsonl")
    parser.add_argument("--malicious", default="data/malicious_samples/samples.json")
    parser.add_argument("--benign",    default="data/benign_samples/samples.json")
    parser.add_argument("--no-hf",    action="store_true", help="Skip HuggingFace download")
    args = parser.parse_args()
    run(
        jsonl_path      = args.jsonl,
        malicious_path  = args.malicious,
        benign_path     = args.benign,
        use_huggingface = not args.no_hf,
    )
