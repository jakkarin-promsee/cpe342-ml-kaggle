"""Deterministic, label-free features from competition screenshots."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern


def extract_image(path: Path, *, engineered: bool = False) -> dict:
    raw = path.read_bytes()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        small = np.asarray(image.resize((16, 8), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        features = {
            "image_sha256": hashlib.sha256(raw).hexdigest(),
            "width": width,
            "height": height,
        }
        features.update({f"px_{i:03d}": value for i, value in enumerate(small.ravel())})
        if not engineered:
            return features

        medium = np.asarray(image.resize((64, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
        gray = np.asarray(image.convert("L").resize((64, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
        hsv = np.asarray(image.convert("HSV").resize((64, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
        for channel, letter in enumerate("rgb"):
            values = medium[:, :, channel]
            features[f"{letter}_mean"] = float(values.mean() / 255)
            features[f"{letter}_std"] = float(values.std() / 255)
            for row in range(2):
                for col in range(4):
                    region = values[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16]
                    features[f"{letter}_pool_{row}_{col}"] = float(region.mean() / 255)
        for channel, letter in enumerate("hsv"):
            histogram = np.histogram(hsv[:, :, channel], bins=16, range=(0, 256))[0]
            features.update({f"{letter}_hist_{i:02d}": float(v / hsv.shape[0] / hsv.shape[1]) for i, v in enumerate(histogram)})
        edges = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm="L2-Hys", feature_vector=True)
        features.update({f"hog_{i:03d}": float(v) for i, v in enumerate(edges)})
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        counts = np.bincount(lbp.astype(np.intp).ravel(), minlength=10)[:10] / lbp.size
        features.update({f"lbp_{i:02d}": float(v) for i, v in enumerate(counts)})
        return features
