# core/cache.py — Persistent disk-based JSON cache for expensive LLM calls

import json
from pathlib import Path


# Persistent JSON cache that auto-saves to disk.
class SimpleCache:
    """
    JSON cache that persists to disk.
    Auto-saves on every set() call.
    """

    def __init__(self, cache_file: Path):
        self.cache_file = Path(cache_file)
        self.data: dict = {}
        self.load()

    def load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    def invalidate(self, key: str) -> None:
        self.data.pop(key, None)
        self.save()

    def clear(self) -> None:
        self.data = {}
        self.save()

    def __len__(self) -> int:
        return len(self.data)
