"""Fine-tune the digit CNN on real Balut cell handwriting.

The base model (abi-models/digit-model.keras) is trained on MNIST + synthetic
strikes and scores ~99% on MNIST but only ~56% on real Balut digits -- a domain
gap. This script adapts it using the labeled cell crops exported by
`export_cells.py`:

  1. Segment each labeled cell into per-character 28x28 crops with the *same*
     routine the server uses at inference (`server.segment_chars`), keeping only
     cells whose blob count matches the label's digit count (reliable labels).
  2. Hold out a stratified slice of the real chars for validation.
  3. Fine-tune the existing weights on: augmented real chars (oversampled) + an
     MNIST subset + synthetic strikes (so it adapts to Balut handwriting without
     forgetting digits/strikes), at a low learning rate.
  4. Report baseline vs fine-tuned accuracy on the held-out real chars AND on
     MNIST test (catastrophic-forgetting check), and export new artifacts.

Outputs digit-model<DATE>.keras / .tflite next to the base model. Wire the
tflite into the server by pointing MODEL_PATH at it.

Usage:  python finetune.py            # train + report + export
        python finetune.py --epochs 15 --no-export
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
from tensorflow import keras

import server  # for segment_chars: training crops must match inference

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

NUM_CLASSES = 11
STRIKE_CLASS = 10
BASE_MODEL = "../abi-models/digit-model.keras"
DATASET = "../abi-dataset/cells"


# --- 1. Build the real character dataset from labeled cells ------------------
def build_real_chars():
    """Segment labeled cells -> (X, y) of 28x28x1 chars, using only cells whose
    blob count matches the label's digit count (so each blob's label is known)."""
    rows = list(csv.DictReader(open(f"{DATASET}/labels.csv")))
    X, y = [], []
    matched = skipped = 0
    for r in rows:
        cell = cv2.imread(f"{DATASET}/{r['file']}", cv2.IMREAD_GRAYSCALE)
        if cell is None:
            continue
        value, is_blank = int(r["value"]), r["is_blank"] == "1"
        _, chars = server.segment_chars(cell)
        if is_blank:
            if len(chars) == 1:          # a lone strike mark -> strike class
                X.append(chars[0]); y.append(STRIKE_CLASS); matched += 1
            else:
                skipped += 1             # empty (0 blobs) or noisy -> unusable
            continue
        digits = [int(d) for d in str(abs(value))]
        if len(chars) == len(digits):    # clean segmentation -> labels are known
            for ch, d in zip(chars, digits):
                X.append(ch); y.append(d)
            matched += 1
        else:
            skipped += 1
    X = np.array(X).reshape(-1, 28, 28, 1).astype("float32")
    y = np.array(y, dtype=np.int64)
    print(f"Real chars: {len(X)} from {matched} cells ({skipped} cells skipped: "
          "segmentation didn't match the label's digit count)")
    return X, y


def stratified_split(X, y, val_frac=0.2, min_for_val=5):
    """Per-class split. Classes with < min_for_val samples stay entirely in train
    (too few to spare for validation)."""
    tr, va = [], []
    rng = np.random.default_rng(SEED)
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac)) if len(idx) >= min_for_val else 0
        va.extend(idx[:n_val]); tr.extend(idx[n_val:])
    tr, va = np.array(tr), np.array(va)
    return X[tr], y[tr], X[va], y[va]


# --- 2. Augmentation + auxiliary data ----------------------------------------
def augment(img):
    """A small random affine jitter (rotation/scale/shift) on a 28x28 char."""
    ang = np.random.uniform(-12, 12)
    scale = np.random.uniform(0.88, 1.12)
    M = cv2.getRotationMatrix2D((14, 14), ang, scale)
    M[0, 2] += np.random.uniform(-2.5, 2.5)
    M[1, 2] += np.random.uniform(-2.5, 2.5)
    out = cv2.warpAffine(img[..., 0], M, (28, 28), borderValue=0.0)
    return out[..., None]


def oversample_augment(X, y, copies):
    """Return `copies` augmented variants of each sample (plus the originals)."""
    aX, aY = list(X), list(y)
    for _ in range(copies):
        for i in range(len(X)):
            aX.append(augment(X[i])); aY.append(y[i])
    return np.array(aX, "float32"), np.array(aY, np.int64)


BACKSLASH = chr(92)
def make_strike(size=28):
    """Synthetic strike (/, backslash, or x) -> class 10, matching the notebook."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 0); d = ImageDraw.Draw(img)
    t = int(np.random.randint(1, 4)); m = int(np.random.randint(2, 6))
    lo, hi = m, size - 1 - m; j = lambda: int(np.random.randint(-2, 3))
    fwd = lambda: d.line([(lo + j(), hi + j()), (hi + j(), lo + j())], fill=255, width=t)
    back = lambda: d.line([(lo + j(), lo + j()), (hi + j(), hi + j())], fill=255, width=t)
    kind = np.random.choice(["/", BACKSLASH, "x"])
    (fwd() if kind == "/" else back() if kind == BACKSLASH else (fwd(), back()))
    return (np.array(img, "float32") / 255.0)[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--copies", type=int, default=18, help="augmented copies per real char")
    ap.add_argument("--mnist", type=int, default=6000, help="MNIST samples to mix in")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    X_real, y_real = build_real_chars()
    Xtr, ytr, Xva, yva = stratified_split(X_real, y_real)
    print(f"Real split: {len(Xtr)} train / {len(Xva)} val")

    # MNIST subset (keep digit knowledge) + synthetic strikes (keep class 10).
    (xm, ym), (xm_te, ym_te) = keras.datasets.mnist.load_data()
    sel = np.random.default_rng(SEED).choice(len(xm), args.mnist, replace=False)
    Xm = (xm[sel].astype("float32") / 255.0)[..., None]; Ym = ym[sel].astype(np.int64)
    Xst = np.stack([make_strike() for _ in range(args.mnist // 10)])
    Yst = np.full(len(Xst), STRIKE_CLASS, np.int64)
    Xm_te = (xm_te.astype("float32") / 255.0)[..., None]  # MNIST test for forgetting check

    # Oversample + augment the real training chars so they aren't drowned by MNIST.
    Xaug, Yaug = oversample_augment(Xtr, ytr, args.copies)
    X = np.concatenate([Xaug, Xm, Xst]); Y = np.concatenate([Yaug, Ym, Yst])
    perm = np.random.default_rng(SEED).permutation(len(X)); X, Y = X[perm], Y[perm]
    print(f"Training set: {len(X)} ({len(Xaug)} real-aug + {len(Xm)} mnist + {len(Xst)} strikes)")

    model = keras.models.load_model(BASE_MODEL)

    def acc(m, Xe, ye):
        return float((m.predict(Xe, verbose=0).argmax(1) == ye).mean())

    base_real = acc(model, Xva, yva) if len(Xva) else float("nan")
    base_mnist = acc(model, Xm_te, ym_te)
    print(f"\nBASELINE  real-val acc {base_real:.3f}   mnist-test acc {base_mnist:.3f}")

    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    val = (Xva, yva) if len(Xva) else None
    model.fit(X, Y, validation_data=val, epochs=args.epochs, batch_size=128, verbose=2)

    ft_real = acc(model, Xva, yva) if len(Xva) else float("nan")
    ft_mnist = acc(model, Xm_te, ym_te)
    print(f"\nFINE-TUNED real-val acc {ft_real:.3f}   mnist-test acc {ft_mnist:.3f}")
    print(f"  real-val:  {base_real:.3f} -> {ft_real:.3f}  ({ft_real - base_real:+.3f})")
    print(f"  mnist-test:{base_mnist:.3f} -> {ft_mnist:.3f}  ({ft_mnist - base_mnist:+.3f})")

    if args.no_export:
        print("\n--no-export: skipping artifact write."); return
    if not (ft_real > base_real):
        print("\nNo improvement on real-val; NOT exporting. Re-run with tweaks "
              "or more labeled data."); return

    date = dt.datetime.now().strftime("%m%d")
    for out_dir in ("../abi-models", "."):
        kpath = f"{out_dir}/digit-model{date}.keras"
        model.save(kpath)
        sm = f"{out_dir}/digit-model{date}-savedmodel"
        model.export(sm)  # Keras 3 direct TFLite path hits an MLIR bug; go via SavedModel
        tfl = tf.lite.TFLiteConverter.from_saved_model(sm).convert()
        open(f"{out_dir}/digit-model{date}.tflite", "wb").write(tfl)
        print(f"Wrote {kpath} and digit-model{date}.tflite")
    print(f"\nTo deploy: point server MODEL_PATH at digit-model{date}.tflite")


if __name__ == "__main__":
    main()
