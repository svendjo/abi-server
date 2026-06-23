"""Segmentation-free whole-cell digit reader (inference side).

Loads the CNN+CTC model trained by `ctc_train.py` and reads a whole cell crop
straight to its integer value -- no character segmentation. A strike / empty cell
decodes to the empty string -> 0. Enable in the server with USE_CTC=1; the schema
layer then snaps each read to the cell's legal value set.

The preprocessing here MUST match `ctc_train.preprocess_cell` exactly, or the
model sees a different distribution than it trained on.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = "cell-reader0622.keras"
IMG_H, IMG_W = 32, 128
BLANK = 11               # CTC mask index
CHARS = "0123456789-"    # class indices 0-10


def available():
    """True if the trained whole-cell reader is present."""
    return Path(MODEL_PATH).exists()


@lru_cache(maxsize=1)
def _load():
    import keras
    return keras.models.load_model(MODEL_PATH)


def _preprocess(gray):
    """Whole cell -> 32x128 binary, height-normalized (match ctc_train)."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    h, w = bw.shape
    nw = max(1, round(w * IMG_H / h))
    r = cv2.resize(bw, (nw, IMG_H), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_H, IMG_W), "float32")
    nw = min(nw, IMG_W)
    x0 = (IMG_W - nw) // 2
    canvas[:, x0:x0 + nw] = r[:, :nw] / 255.0
    return canvas[..., None]


def read_cells(crops):
    """List of grayscale cell crops -> list of int values (strike/empty -> 0).

    One batched forward pass + greedy CTC decode. Unparseable decodes -> 0.
    """
    if not crops:
        return []
    import keras

    model = _load()
    X = np.array([_preprocess(c) for c in crops], "float32")
    logits = model.predict(X, verbose=0)
    olen = np.full((logits.shape[0],), logits.shape[1], "int32")
    dec, _ = keras.ops.ctc_decode(logits, olen, strategy="greedy", mask_index=BLANK)
    out = []
    for row in np.array(dec[0]):
        s = "".join(CHARS[i] for i in row if 0 <= i < len(CHARS))
        try:
            out.append(int(s) if s not in ("", "-") else 0)
        except ValueError:
            out.append(0)
    return out
