"""Segmentation-free whole-cell digit reader (CNN + BiLSTM + CTC).

Instead of splitting a cell into characters and reading each (the old, brittle
pipeline whose blob-count-vs-digit-count mismatches capped accuracy), this reads
the whole cell image straight to its digit string in one shot. CTC handles the
variable length, so "42", "8", "432" and "-4" are all just sequences; a strike /
empty cell is the empty string -> value 0.

Because there's no segmentation filter, it trains on ALL labeled cells (not just
the cleanly-segmented ones). With only 5 sheets it will overfit those hands and
generalize modestly -- the point is the architecture + harness, which improves as
more scorecards are labeled (just re-run export_cells.py then this).

Vocabulary: 0-9 (idx 0-9), '-' (10), CTC blank (11).
Outputs cell-reader<DATE>.keras. Eval prints CTC vs the fine-tuned CNN on a
held-out split.

Usage:  python ctc_train.py            # train + eval + save
        python ctc_train.py --epochs 60 --no-save
"""

import argparse
import csv
import datetime as dt
import os
from collections import Counter

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
import keras
from keras import layers, ops

import server  # for the CNN baseline comparison

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)

DATASET = "../abi-dataset/cells"
IMG_H, IMG_W = 32, 128
BLANK = 11            # CTC mask index; 0-9 digits, 10 = '-'
NUM_CLASSES = 12      # 0-9, '-', blank
CHARS = "0123456789-"


# --- data --------------------------------------------------------------------
def preprocess_cell(gray):
    """Whole cell -> 32x128 binary (white ink on black), height-normalized."""
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


def target_string(value, is_blank):
    """Cell label -> digit string. Strike/empty -> '' (value 0)."""
    return "" if is_blank else str(value)


def load_cells():
    """Return list of (image[32,128,1], target_str, sheet)."""
    rows = list(csv.DictReader(open(f"{DATASET}/labels.csv")))
    out = []
    for r in rows:
        gray = cv2.imread(f"{DATASET}/{r['file']}", cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        s = target_string(int(r["value"]), r["is_blank"] == "1")
        out.append((preprocess_cell(gray), s, r["sheet"]))
    return out


def encode(s, max_len):
    seq = [CHARS.index(c) for c in s]
    return seq + [BLANK] * (max_len - len(seq)), len(seq)


def augment(img):
    ang = np.random.uniform(-7, 7)
    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), ang, scale)
    M[0, 2] += np.random.uniform(-4, 4); M[1, 2] += np.random.uniform(-2, 2)
    return cv2.warpAffine(img[..., 0], M, (IMG_W, IMG_H), borderValue=0.0)[..., None]


# --- model -------------------------------------------------------------------
def build_backbone():
    """Image -> (T=32, NUM_CLASSES) logits."""
    inp = keras.Input((IMG_H, IMG_W, 1))
    x = inp
    for f, pool in [(32, (2, 2)), (64, (2, 2)), (128, (2, 1)), (128, (2, 1)), (128, (2, 1))]:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool)(x)            # height -> 1, width -> 32
    x = layers.Reshape((IMG_W // 4, 128))(x)         # (T=32, feat=128)
    # Temporal context via 1D convs (fully convolutional -> fast on CPU; ample
    # receptive field for short digit strings, no recurrence needed).
    for _ in range(2):
        x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Dropout(0.25)(x)
    logits = layers.Dense(NUM_CLASSES)(x)            # raw logits for ctc_loss
    return keras.Model(inp, logits)


class CTCReader(keras.Model):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, x, training=False):
        return self.backbone(x, training=training)

    def _loss(self, y, ylen, logits):
        T = ops.shape(logits)[1]
        olen = ops.full((ops.shape(logits)[0],), T, dtype="int32")
        return ops.mean(keras.ops.ctc_loss(y, logits, ylen, olen, mask_index=BLANK))

    def train_step(self, data):
        x, y, ylen = data
        with tf.GradientTape() as tape:
            loss = self._loss(y, ylen, self.backbone(x, training=True))
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y, ylen = data
        self.loss_tracker.update_state(self._loss(y, ylen, self.backbone(x, training=False)))
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def decode_logits(logits):
    """(B,T,C) logits -> list of strings (greedy CTC)."""
    B = logits.shape[0]
    olen = np.full((B,), logits.shape[1], "int32")
    dec, _ = keras.ops.ctc_decode(logits, olen, strategy="greedy", mask_index=BLANK)
    dec = np.array(dec[0])  # (B, T), padded with -1
    out = []
    for row in dec:
        out.append("".join(CHARS[i] for i in row if 0 <= i < len(CHARS)))
    return out


def to_value(s):
    try:
        return int(s) if s not in ("", "-") else 0
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--copies", type=int, default=40, help="augmented copies per train cell")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cells = load_cells()
    max_len = max(len(s) for _, s, _ in cells)
    print(f"{len(cells)} cells | max label len {max_len} | "
          f"value-string lengths {dict(sorted(Counter(len(s) for _,s,_ in cells).items()))}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(cells))
    n_val = int(len(cells) * args.val_frac)
    val_idx, tr_idx = set(idx[:n_val].tolist()), idx[n_val:]

    # Training arrays: augmented copies of the train cells.
    Xtr, Ytr, Ltr = [], [], []
    for i in tr_idx:
        img, s, _ = cells[i]
        seq, ln = encode(s, max_len)
        for k in range(args.copies + 1):
            Xtr.append(img if k == 0 else augment(img)); Ytr.append(seq); Ltr.append(ln)
    Xtr = np.array(Xtr, "float32"); Ytr = np.array(Ytr, "int32"); Ltr = np.array(Ltr, "int32")
    p = rng.permutation(len(Xtr)); Xtr, Ytr, Ltr = Xtr[p], Ytr[p], Ltr[p]
    print(f"train cells {len(tr_idx)} -> {len(Xtr)} augmented | val cells {n_val}")

    ds = tf.data.Dataset.from_tensor_slices((Xtr, Ytr, Ltr)).batch(64).prefetch(tf.data.AUTOTUNE)

    model = CTCReader(build_backbone())
    model.compile(optimizer=keras.optimizers.Adam(1e-3))
    model.fit(ds, epochs=args.epochs, verbose=2)

    # --- eval on held-out cells: CTC vs the fine-tuned CNN -------------------
    Xva = np.array([cells[i][0] for i in val_idx], "float32")
    truth = [to_value(cells[i][1]) for i in val_idx]
    logits = model.backbone.predict(Xva, verbose=0)
    ctc_vals = [to_value(s) for s in decode_logits(logits)]

    # CNN baseline reads the same cells loosely (no schema) from their gray crops.
    rows = {(r["sheet"], r["cell"]): r["file"] for r in csv.DictReader(open(f"{DATASET}/labels.csv"))}
    files = [cells[i] for i in val_idx]  # need original gray; reload by index order
    all_rows = list(csv.DictReader(open(f"{DATASET}/labels.csv")))
    cnn_vals = []
    for i in val_idx:
        gray = cv2.imread(f"{DATASET}/{all_rows[i]['file']}", cv2.IMREAD_GRAYSCALE)
        cnn_vals.append(server.read_cell(gray))

    ctc_acc = np.mean([a == b for a, b in zip(ctc_vals, truth)])
    cnn_acc = np.mean([a == b for a, b in zip(cnn_vals, truth)])
    print(f"\nHeld-out cell accuracy ({len(truth)} cells, raw value, no schema):")
    print(f"  CNN (segment+classify, digit-model0622): {cnn_acc:.3f}")
    print(f"  CTC (whole-cell, segmentation-free):     {ctc_acc:.3f}")
    print("  sample (truth / ctc / cnn):",
          [(t, c, n) for t, c, n in zip(truth, ctc_vals, cnn_vals)][:12])

    if args.no_save:
        return
    date = dt.datetime.now().strftime("%m%d")
    path = f"cell-reader{date}.keras"
    model.backbone.save(path)
    print(f"\nSaved {path} (backbone: image -> CTC logits)")


if __name__ == "__main__":
    main()
