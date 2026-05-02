# core/image_utils.py — Image processing, deduplication, and base64 serialization

import base64
import hashlib
import io
from pathlib import Path

import config

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Image serialization ────────────────────────────────────────────────────────

# Converts PIL Image to base64-encoded JPEG data URL for LLM API.
def pil_to_base64_jpeg(pil_image: "Image.Image", quality: int = None) -> str:
    """
    Converts PIL Image to base64-encoded JPEG data URL for LLM API.
    Resizes if larger than config.TARGET_MAX_SIDE.
    """
    if quality is None:
        quality = config.JPEG_QUALITY

    img = pil_image.convert("RGB")

    max_side = config.TARGET_MAX_SIDE
    if img.width > max_side or img.height > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# Saves PIL Image as JPEG. Returns True on success.
def save_pil_image(pil_image: "Image.Image", output_path: Path, quality: int = None) -> bool:
    """Saves PIL Image as JPEG. Returns True on success."""
    if quality is None:
        quality = config.JPEG_QUALITY
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.convert("RGB").save(str(output_path), "JPEG", quality=quality)
        return True
    except Exception as e:
        print(f"   ⚠️ Bild-Speicherfehler: {e}")
        return False


# Returns MD5 hash of raw pixel data (for cache keys).
def get_image_hash(pil_image: "Image.Image") -> str:
    """MD5 hash of raw pixel data (for cache keys)."""
    return hashlib.md5(pil_image.tobytes()).hexdigest()


# Computes difference hash for perceptual matching. Returns hex string.
def get_dhash(pil_image: "Image.Image", hash_size: int = 8) -> str:
    """Difference hash for perceptual matching. Returns hex string."""
    small = pil_image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(small.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            diff.append(pixels[row * (hash_size + 1) + col] >
                        pixels[row * (hash_size + 1) + col + 1])
    decimal = 0
    hex_str = []
    for i, val in enumerate(diff):
        if val:
            decimal += 2 ** (i % 8)
        if (i % 8) == 7:
            hex_str.append(hex(decimal)[2:].rjust(2, "0"))
            decimal = 0
    return "".join(hex_str)


# ── Duplicate detection ───────────────────────────────────────────────────────

# Two-stage duplicate detection using hash and feature matching.
class AdvancedDeduplicator:
    """
    Two-stage duplicate detection:
    1. Average Hash (Hamming distance ≤ 5 → definite duplicate)
    2. ORB Feature Matching (Hamming distance 5-25 → possible crop/scaling)

    Falls back to simple dHash if cv2 not available.
    """

    def __init__(self):
        self.seen_images: list = []  # (hash_int, descriptors_or_None, img_id)
        if CV2_AVAILABLE:
            self.orb = cv2.ORB_create(nfeatures=500)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self._seen_dhashes: set[str] = set()

    # Returns True if duplicate (skip image), False if new (saves internally).
    def is_duplicate(self, pil_image: "Image.Image", img_id: str = "") -> bool:
        """
        Returns True if duplicate (skip image), False if new (saves internally).
        """
        if not CV2_AVAILABLE:
            return self._simple_dhash_check(pil_image)

        try:
            arr = np.array(pil_image)
            if len(arr.shape) == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr

            curr_hash = self._compute_ahash(gray)
            kp, curr_des = self.orb.detectAndCompute(gray, None)

            if curr_des is None:
                return True  # No content → treat as duplicate/junk

            for old_hash, old_des, _ in self.seen_images:
                h_dist = bin(curr_hash ^ old_hash).count("1")

                if h_dist <= 5:
                    return True  # Definite duplicate

                if h_dist <= 25 and old_des is not None:
                    matches = self.bf.match(curr_des, old_des)
                    matches = sorted(matches, key=lambda x: x.distance)
                    if len(matches) > 10:
                        top = matches[:50]
                        avg_dist = sum(m.distance for m in top) / len(top)
                        ratio = len(matches) / min(len(curr_des), len(old_des))
                        if avg_dist < 50 and ratio > 0.2:
                            return True

            self.seen_images.append((curr_hash, curr_des, img_id))
            return False

        except Exception as e:
            print(f"   ⚠️ Deduplizierungs-Fehler: {e}")
            return False

    # Computes average hash as integer.
    def _compute_ahash(self, gray_cv2) -> int:
        """Average hash as integer."""
        resized = cv2.resize(gray_cv2, (8, 8), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        diff = resized > avg
        val = 0
        for i, b in enumerate(diff.flatten()):
            if b:
                val += 2 ** i
        return val

    # Fallback without cv2: dHash-based duplicate check.
    def _simple_dhash_check(self, pil_image: "Image.Image") -> bool:
        """Fallback without cv2: dHash-based duplicate check."""
        h = get_dhash(pil_image)
        if h in self._seen_dhashes:
            return True
        self._seen_dhashes.add(h)
        return False
