"""Segmentation-free whole-cell digit reader (inference side).

Loads the CNN+CTC model trained by `CellReader.ipynb` and reads a whole cell crop
straight to its integer value -- no character segmentation. A struck cell decodes
to an explicit `x` token (so the model positively identifies strikes rather than
inferring them from emptiness); `x`, `-` and an empty decode all map to value 0.
Enable in the server with USE_CTC=1; the schema layer then snaps each read to the
cell's legal value set.

Runs on the same TFLite runtime as the digit CNN -- no full TensorFlow in prod.
The converted backbone outputs (T, NUM_CLASSES) logits per cell and we do a greedy
CTC decode in numpy. The preprocessing here MUST match `CellReader.ipynb`'s
preprocess_cell exactly, or the model sees a different distribution than it trained on.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Same interpreter the server uses: tflite-runtime in prod, tf.lite for local dev.
try:
    from tflite_runtime.interpreter import Interpreter as _Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter

MODEL_PATH = "cell-reader0627.tflite"   # CTC backbone, converted by CellReader.ipynb
IMG_H, IMG_W = 32, 128
BLANK = 12               # CTC mask index
CHARS = "0123456789-x"   # class indices 0-11 (10='-', 11='x' strike)


def available():
    """True if the trained whole-cell reader is present."""
    return Path(MODEL_PATH).exists()


@lru_cache(maxsize=1)
def _load():
    interp = _Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp


def _preprocess(gray):
    """Whole cell -> 32x128 binary, height-normalized (match CellReader.ipynb)."""
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


def _greedy_decode(logits):
    """Greedy CTC decode of (T, C) logits -> string: argmax per timestep, collapse
    consecutive repeats, drop the blank class."""
    out, prev = [], -1
    for c in logits.argmax(-1):
        c = int(c)
        if c != prev and c != BLANK and 0 <= c < len(CHARS):
            out.append(CHARS[c])
        prev = c
    return "".join(out)


def read_cells(crops):
    """List of grayscale cell crops -> list of int values (x/strike/empty -> 0).

    One greedy CTC decode per cell. Unparseable decodes -> 0.
    """
    if not crops:
        return []
    interp = _load()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    out = []
    for c in crops:
        interp.set_tensor(in_idx, _preprocess(c)[None].astype("float32"))
        interp.invoke()
        s = _greedy_decode(interp.get_tensor(out_idx)[0])
        try:
            out.append(0 if s in ("", "-", "x") else int(s))  # x/-/empty -> strike == 0
        except ValueError:
            out.append(0)
    return out
