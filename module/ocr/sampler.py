"""
OCR sample collector for fine-tuning data.
Saves low-confidence or random OCR samples for later human review.
"""
import json
import os
import time
from datetime import datetime

import numpy as np
from PIL import Image


class OcrSampler:
    """
    Hooks into the OCR pipeline to collect training samples.

    Usage:
        sampler = OcrSampler(enabled=True, sample_rate=0.05)
        sampler.save_sample(image, text, cand_alphabet, confidence=0.6)
    """

    def __init__(self, enabled=True, sample_rate=0.02, output_dir="./log/ocr_samples"):
        self.enabled = enabled
        self.sample_rate = sample_rate  # Save this fraction of all OCR calls
        self.output_dir = output_dir
        self._counter = 0
        self._today_dir = ""
        self._index_path = ""
        self._index = []

    def _ensure_dir(self):
        today = datetime.now().strftime("%Y-%m-%d")
        d = os.path.join(self.output_dir, today)
        if d != self._today_dir:
            os.makedirs(d, exist_ok=True)
            self._today_dir = d
            self._index_path = os.path.join(d, "index.json")
            self._counter = 0
            # Load existing index
            if os.path.exists(self._index_path):
                with open(self._index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                self._counter = len(self._index)
            else:
                self._index = []
        return d

    def should_sample(self) -> bool:
        """Random sampling based on sample_rate."""
        if not self.enabled:
            return False
        import random
        return random.random() < self.sample_rate

    def save_sample(self, image: np.ndarray, text: str, cand_alphabet: str = "",
                    confidence: float = 0.0, task: str = ""):
        """
        Save an OCR sample for review.

        Args:
            image: Pre-processed grayscale image array (H, W) or (H, W, 1)
            text: OCR result text
            cand_alphabet: Character whitelist used
            confidence: OCR confidence (if available)
            task: Task name for context
        """
        if not self.enabled:
            return

        d = self._ensure_dir()
        self._counter += 1
        fname = f"{self._counter:06d}.png"
        img_path = os.path.join(d, fname)

        # Save image
        if image.ndim == 3:
            image = image.squeeze(-1)
        img = Image.fromarray(image.astype(np.uint8))
        img.save(img_path)

        # Save metadata
        entry = {
            "id": self._counter,
            "image": fname,
            "text": text,
            "cand_alphabet": cand_alphabet,
            "confidence": confidence,
            "task": task,
            "corrected": False,
            "correct_text": None,
        }
        self._index.append(entry)

        # Write index periodically (every 10 samples)
        if self._counter % 10 == 0:
            self._save_index()

    def _save_index(self):
        if self._index_path:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)

    def close(self):
        self._save_index()

    # ---- Static helpers for the review UI ----

    @staticmethod
    def list_samples(date: str = None):
        """List all available sample dates or samples for a given date."""
        base = "./log/ocr_samples"
        if not os.path.exists(base):
            return []
        if date:
            path = os.path.join(base, date)
            idx = os.path.join(path, "index.json")
            if os.path.exists(idx):
                with open(idx, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))], reverse=True)

    @staticmethod
    def correct_label(date: str, sample_id: int, correct_text: str):
        """Mark a sample as corrected by a human reviewer."""
        path = os.path.join("./log/ocr_samples", date, "index.json")
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for entry in index:
            if entry["id"] == sample_id:
                entry["corrected"] = True
                entry["correct_text"] = correct_text
                break
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        return True

    @staticmethod
    def export_training_data(date: str = None):
        """Export corrected samples as (image_path, text) pairs for training."""
        base = "./log/ocr_samples"
        if date:
            dates = [date]
        else:
            dates = sorted(os.listdir(base), reverse=True)

        pairs = []
        for d in dates:
            path = os.path.join(base, d, "index.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                index = json.load(f)
            for entry in index:
                if entry.get("corrected") and entry.get("correct_text"):
                    pairs.append({
                        "image": os.path.join(base, d, entry["image"]),
                        "label": entry["correct_text"],
                    })
        return pairs


# Global instance for use across the OCR pipeline
SAMPLER = OcrSampler(enabled=True, sample_rate=0.03)
