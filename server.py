from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import hashlib
import json
import re
try:
    from tflite_runtime.interpreter import Interpreter as tflite
except ImportError:
    # Local dev uses full TensorFlow. tf.lite is lazily loaded, so it must be
    # reached via attribute access -- `from tensorflow.lite import Interpreter`
    # does not trigger the lazy loader and fails on TF 2.16.
    import tensorflow as tf
    tflite = tf.lite.Interpreter
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import cv2
import io
import os

import config
import results_store
import scorecard
import trocr_reader
import ctc_reader

print(f"Environment: APP_ENV={config.APP_ENV}")
app = FastAPI()

# Cell recognizer selection comes from config/<APP_ENV>.yaml (the segment+classify
# CNN is the default; each alternative falls back to the CNN if its model/deps
# aren't available):
#   use_ctc   -> segmentation-free whole-cell CTC reader (cell-reader*.tflite)
#   use_trocr -> local TrOCR handwriting model
USE_CTC = config.USE_CTC and ctc_reader.available()
if config.USE_CTC and not USE_CTC:
    print("WARNING: use_ctc set but no cell-reader model found; using the CNN.")
elif USE_CTC:
    print("Using the segmentation-free CTC reader for cell recognition.")

USE_TROCR = config.USE_TROCR and trocr_reader.available()
if config.USE_TROCR and not USE_TROCR:
    print("WARNING: use_trocr set but torch/transformers not available; using the CNN.")
elif USE_TROCR:
    print("Using TrOCR for cell recognition.")

# Save the numbered slice stages into each result folder to debug slicing.
DEBUG_CROPS = config.DEBUG_CROPS

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# The handwritten sheet is a fixed 10 rows x 8 columns table.
ROWS, COLS = 10, 8
STRIKE_CLASS = 10  # classes 0-9 are digits; 10 means a struck cell (/, \, x) -> value 0

# Fallback dir for slice_sheet's debug images when it's called standalone
# (out_dir=None); real reads write through STORE below.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Per-environment results storage (local dir or S3 bucket; see config + results_store).
STORE = results_store.make_store(config.RESULTS)
print(f"Results store: {config.RESULTS.get('backend', 'local')}")

# Load the digit/symbol recognizer. If it hasn't been trained/copied yet the
# server still boots so the frontend can be developed; /read then 503s.
MODEL_PATH = "digit-model0622.tflite"  # fine-tuned on real Balut cells (finetune.py)
try:
    interpreter = tflite(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Loaded model: {MODEL_PATH}")
except Exception as e:  # noqa: BLE001 - boot anyway, report at request time
    interpreter = None
    input_details = output_details = None
    print(f"WARNING: could not load {MODEL_PATH} ({e}). /read will return 503.")


def classify(char28):
    """Run one preprocessed 28x28x1 character batch through the model.

    Returns the full probability vector over the 11 classes (0-9, strike) so
    callers can do constrained decoding, not just take the argmax.
    """
    interpreter.set_tensor(input_details[0]["index"], char28)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]


MAX_SKEW_DEG = 20.0  # refuse to read a sheet rotated more than this


class ScanError(Exception):
    """The sheet can't be read (e.g. rotated too far); surfaced as a 422."""


def detect_skew(gray):
    """Estimate the sheet's rotation from its long near-horizontal lines.

    Returns degrees to rotate (CCW-positive) to level it, or None if the tilt
    exceeds MAX_SKEW_DEG. Runs on a downscaled copy for speed (angle is scale-
    invariant). 0.0 if no strong lines are found (assume already level).
    """
    h, w = gray.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120,
        minLineLength=int(0.3 * small.shape[1]), maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) <= abs(y2 - y1):
            continue  # keep near-horizontal segments (the table's row lines)
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= 45:
            angles.append(ang)
    if not angles:
        return 0.0
    skew = float(np.median(angles))
    return None if abs(skew) > MAX_SKEW_DEG else skew


def rotate(gray, angle):
    """Rotate `gray` by `angle` degrees (CCW-positive), expanding the canvas."""
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]      # bottom-right has largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right has smallest y-x
    rect[3] = pts[np.argmax(diff)]   # bottom-left has largest y-x
    return rect


def locate_table(gray):
    """Crop to the table by its grid lattice.

    The 10x8 table is the largest connected mesh of long horizontal + vertical
    rules in the frame. We isolate those rules with directional morphology and OR
    them together; the grid (where the rules cross) becomes one big connected
    component whose bounding box is the table. This is robust to the table being
    only part of the sheet and to the surrounding graphics/text (logos, legends,
    titles), which don't form long ruled lines -- unlike "biggest filled contour",
    which the table is only marginally (and which dropped 5.jpg at 0.195 < 0.2).
    Falls back to the full frame if no sufficiently large mesh is found.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    # Long horizontal and vertical runs only (kernels ~1/40 of the frame: long
    # enough to drop handwriting/text, short enough to keep a partial-width table).
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 40), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 40)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    lattice = cv2.bitwise_or(horiz, vert)
    lattice = cv2.dilate(lattice, np.ones((3, 3), np.uint8), iterations=2)  # bridge gaps at crossings
    n, _, stats, _ = cv2.connectedComponentsWithStats((lattice > 0).astype("uint8"), 8)
    if n <= 1:
        return gray

    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # largest mesh = the table
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    bw_, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if bw_ < 0.2 * w or bh < 0.1 * h:  # too small to be the table -> full frame
        return gray
    return gray[y:y + bh, x:x + bw_]


def _trim_to_gridlines(table):
    """Trim the contour box to the table's actual grid extent on all four sides.

    The box from the table contour runs past the grid -- above into the title /
    player header, below into the legend / logos. Those areas have no full-span
    rules, so we trim top/bottom to the first/last horizontal rule, then (on the
    height-trimmed crop, where the vertical rules now span the full height)
    left/right to the first/last vertical rule. Keeping the empty header out also
    stops dewarp from smearing it.
    """
    h = table.shape[0]
    rows = _line_positions(table, 0)
    if len(rows) >= 2 and rows[-1] - rows[0] > 0.3 * h:
        table = table[rows[0]:rows[-1] + 1, :]
    w = table.shape[1]
    cols = _line_positions(table, 1)
    if len(cols) >= 2 and cols[-1] - cols[0] > 0.3 * w:
        table = table[:, cols[0]:cols[-1] + 1]
    return table


def _horizontal_line_curves(table):
    """Detect the table's full-width horizontal rules as curves y(x).

    Returns a list of length-W arrays (one per line; NaN where that line wasn't
    found at that column). Used to measure and undo vertical bowing.
    """
    h, w = table.shape
    bw = cv2.adaptiveThreshold(
        table, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    # Modest kernel: long enough to drop handwriting strokes, short enough that a
    # *curved* rule still survives (a wide kernel erodes bowed lines away).
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_k)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, close_k)  # bridge ink/faint gaps
    n, labels, stats, _ = cv2.connectedComponentsWithStats((horiz > 0).astype("uint8"), 8)
    curves = []
    for i in range(1, n):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if cw < 0.5 * w or ch > 0.15 * h:  # must be wide and thin to be a rule
            continue
        ys, xs = np.where(labels == i)
        sums = np.bincount(xs, weights=ys, minlength=w)
        counts = np.bincount(xs, minlength=w)
        yc = np.full(w, np.nan)
        yc[counts > 0] = sums[counts > 0] / counts[counts > 0]
        curves.append(yc)
    return curves


def _straighten_horizontal(table):
    """Remap so the table's horizontal rules become flat (corrects vertical bow)."""
    h, w = table.shape
    rows = []
    targets = []
    for yc in _horizontal_line_curves(table):
        valid = ~np.isnan(yc)
        if valid.sum() < 0.3 * w:
            continue
        xs = np.arange(w)
        yc = np.interp(xs, xs[valid], yc[valid])  # fill gaps along x
        rows.append(yc)
        targets.append(float(yc.mean()))
    if len(rows) < 2:
        return table  # not enough lines to model the warp
    order = np.argsort(targets)
    curves = np.array(rows)[order]      # [K, w]: source y of each line per column
    targets = np.array(targets)[order]  # [K]: flat (target) y of each line

    # Edge anchors: above the first / below the last detected line, keep the same
    # displacement instead of letting np.interp clamp to the line (which would
    # collapse and ERASE the outer row/column when the border rule isn't detected).
    top_src = curves[0] - targets[0]                    # source y at output row 0
    bot_src = curves[-1] + (h - 1 - targets[-1])        # source y at output row h-1
    t_ext = np.concatenate(([0.0], targets, [h - 1.0]))

    vs = np.arange(h, dtype=np.float32)
    mapy = np.empty((h, w), np.float32)
    for x in range(w):
        src = np.concatenate(([top_src[x]], curves[:, x], [bot_src[x]]))
        mapy[:, x] = np.interp(vs, t_ext, src)
    mapx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    return cv2.remap(table, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def dewarp(table):
    """Flatten the table's curvature: straighten horizontal rules, then vertical
    ones (the latter by transposing, reusing the same routine). Returns the input
    unchanged for whichever axis has too few detectable lines.
    """
    table = _straighten_horizontal(table)
    t = _straighten_horizontal(np.ascontiguousarray(table.T))
    return np.ascontiguousarray(t.T)


def _line_positions(table, axis):
    """Positions of the table's printed rules along `axis`.

    axis=0 -> row rules (full-width horizontal lines), position = mean y.
    axis=1 -> column rules (full-height vertical lines), position = mean x.
    Only near-full-span, thin components count (so handwriting strokes and the
    logo box don't register), and split lines are merged.
    """
    h, w = table.shape
    bw = cv2.adaptiveThreshold(
        table, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    if axis == 0:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1))
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1))
    else:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30)))
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 15)))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_k)
    # Bridge gaps where ink (heavy strikes) or faint printing broke a rule, so it
    # is detected as one full-span line instead of several short pieces.
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_k)
    n, _, stats, cent = cv2.connectedComponentsWithStats((lines > 0).astype("uint8"), 8)

    centers = []
    for i in range(1, n):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if axis == 0 and (cw >= 0.5 * w and ch <= 0.15 * h):
            centers.append(cent[i][1])  # mean y of a wide, thin horizontal rule
        elif axis == 1 and (ch >= 0.5 * h and cw <= 0.15 * w):
            centers.append(cent[i][0])  # mean x of a tall, thin vertical rule
    centers.sort()

    # Merge near-duplicates (a single rule split into two components).
    span = w if axis == 1 else h
    merged = []
    for c in centers:
        if merged and c - merged[-1] < 0.02 * span:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    return [int(round(c)) for c in merged]


def _complete_boundaries(pos, extent):
    """Add the crop edge (0 / extent-1) as a boundary when a border rule was
    missed. The crop is trimmed tight to the grid, so a full-gap-sized space
    between the edge and the first/last detected line means the edge IS a border.
    """
    if len(pos) < 2:
        return pos
    gap = float(np.median(np.diff(pos)))
    pos = list(pos)
    if pos[0] > 0.5 * gap:
        pos = [0] + pos
    if (extent - 1) - pos[-1] > 0.5 * gap:
        pos = pos + [extent - 1]
    return pos


def _schema_rows(rraw):
    """Map detected horizontal rules to the schema's ROWS+1 row boundaries (A..J),
    keyed on the card variant's rule count, or None if the count is unrecognized.

      ROWS+1 (11) rules -- America's Cup / QUITOBAL: the rows are already A..J.
      ROWS+2 (12) rules -- IBF Chicago: an extra Name/UBN/Date row sits on top of
          the grid; drop it so the next rule down (the Score/Jackpot/Points header)
          becomes row A.
      ROWS   (10) rules -- BUNABAL: there is no gridded header row (the
          Score/Jackpot/Points labels are a separate box above the grid) and an
          ID/1st-4th legend row sits at the bottom. Synthesize a one-row-tall
          header band (A) above '4's; B..H and Total then line up, and the bottom
          legend row lands in J (which carries only the grand-total cell).
    """
    n = len(rraw)
    if n == ROWS + 1:
        return list(rraw)
    if n == ROWS + 2:
        return list(rraw[1:])
    if n == ROWS:
        row_h = rraw[1] - rraw[0]
        return [max(0, rraw[0] - row_h)] + list(rraw)
    return None


def grid_boundaries(table):
    """Return (row_bounds, col_bounds, rows_aligned, cols_aligned).

    Maps the detected printed rules (with missing edge borders filled in) onto the
    schema's ROWS+1 x COLS+1 boundaries, handling the three card variants by their
    rule count (see _schema_rows). Falls back to an even split for an axis whose
    rule count is unrecognized; the *_aligned boolean is True when that axis was
    snapped to real rules, False when it was guessed by even split.
    """
    h, w = table.shape
    rraw = _complete_boundaries(_line_positions(table, 0), h)
    craw = _complete_boundaries(_line_positions(table, 1), w)
    rb = _schema_rows(rraw)
    cb = list(craw) if len(craw) == COLS + 1 else None
    rows_aligned, cols_aligned = rb is not None, cb is not None
    if rb is None:
        rb = [round(i * h / ROWS) for i in range(ROWS + 1)]
    if cb is None:
        cb = [round(i * w / COLS) for i in range(COLS + 1)]
    return rb, cb, rows_aligned, cols_aligned


def split_cells(table, rb, cb):
    """Slice into ROWS x COLS cells using the given row/col boundaries, trimming a
    small margin inside each cell so the rules themselves aren't included."""
    cells = []
    for r in range(ROWS):
        y0, y1 = rb[r], rb[r + 1]
        my = int(0.12 * (y1 - y0))
        for c in range(COLS):
            x0, x1 = cb[c], cb[c + 1]
            mx = int(0.12 * (x1 - x0))
            cells.append(table[max(y0 + my, 0):y1 - my, max(x0 + mx, 0):x1 - mx])
    return cells


def preprocess_char(crop):
    """Center a binary (white-on-black) character crop in a 28x28 MNIST-style canvas."""
    h, w = crop.shape
    if h >= w:
        nh, nw = 20, max(1, round(w * 20 / h))
    else:
        nh, nw = max(1, round(h * 20 / w)), 20
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    canvas /= 255.0
    return canvas[np.newaxis, ..., np.newaxis].astype(np.float32)


def segment_chars(cell):
    """Cell -> (sign, [28x28x1 preprocessed char crops], left-to-right).

    Splits a cell into its digit blobs the way the recognizer expects: drop
    specks/grid fragments by area, treat a wide short blob as a leading minus
    sign (the model has no minus class), keep tall blobs as digits ordered left
    to right, and center each in a 28x28 MNIST-style canvas. A blank cell yields
    an empty list. Shared by the recognizer (`_segment`) and the dataset / fine-
    tune tooling, so training crops match exactly what is classified at inference.
    """
    if cell.size == 0:
        return 1, []
    binary = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape
    boxes = [cv2.boundingRect(c) for c in contours]
    # Drop specks and grid-line fragments by area.
    boxes = [b for b in boxes if b[2] * b[3] > 0.015 * H * W]
    # A wide, short blob is a minus sign, not a digit (too short for the height
    # filter below to keep). Detect it before dropping short blobs.
    dashes = [b for b in boxes if b[2] > 1.8 * b[3] and b[3] < 0.35 * H]
    digit_boxes = [b for b in boxes if b not in dashes and b[3] > 0.25 * H]
    digit_boxes.sort(key=lambda b: b[0])  # left to right
    sign = -1 if dashes else 1
    chars = [preprocess_char(binary[y:y + h, x:x + w]) for (x, y, w, h) in digit_boxes]
    return sign, chars


def _segment(cell):
    """Cell -> (sign, [prob_vector per digit blob]). Classifies each blob from
    `segment_chars` with the CNN."""
    sign, chars = segment_chars(cell)
    return sign, [classify(ch) for ch in chars]


def read_cell(cell):
    """Read one cell loosely (argmax per character) -> int value.

    A strike (/, \\, x) and an empty cell both mean value 0. Point cells may be
    negative. Used for the total/checksum cells, which we read independently of
    the schema so check_consistency stays an honest cross-check.
    """
    sign, probs = _segment(cell)
    if not probs:
        return 0  # blank or strike -> 0
    classes = [int(np.argmax(p)) for p in probs]
    if STRIKE_CLASS in classes:
        return 0
    digits = [str(c) for c in classes if c < 10]
    return sign * int("".join(digits)) if digits else 0


def _rank(sign, probs, candidates):
    """Rank `candidates` by the model's likelihood -- most likely value first.

    Works off the per-character probability vectors from `_segment`, so we keep
    the model's 2nd-, 3rd-best readings, not just the argmax. A candidate only
    competes if its sign and digit count match what was segmented; its score is
    the product of the per-digit probabilities (strike / 0 handled specially).
    Returns a list of (value, score) sorted by score descending -- empty if no
    candidate fits the segmented shape.
    """
    n = len(probs)
    scored = []
    for v in candidates:
        if v == 0:  # strike, empty cell, or a written "0"
            if n == 0:
                s = 1.0  # a blank cell is a strike == 0
            elif n == 1:
                s = float(max(probs[0][STRIKE_CLASS], probs[0][0]))
            else:
                continue
        else:
            if (v < 0) != (sign < 0):
                continue  # a detected minus sign must match the candidate's sign
            digits = [int(ch) for ch in str(abs(v))]
            if len(digits) != n:
                continue
            s = 1.0
            for p, d in zip(probs, digits):
                s *= float(p[d])
        scored.append((v, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def read_cell_constrained(cell, candidates):
    """Read a cell as the most likely value from `candidates` (its legal set).

    Keeps the model's full per-character probabilities and picks the legal value
    with the highest likelihood -- so if the most likely reading isn't legal we
    fall through to the next most likely, ranked by the model's own confusion
    rather than numeric distance. `candidates` may include negatives (col 8
    points); the minus sign is read from a separate blob by `_segment`. Falls
    back to a loose read + snap when the segmented digit count fits no candidate.
    """
    sign, probs = _segment(cell)
    ranked = _rank(sign, probs, candidates)
    if ranked:
        return ranked[0][0]

    # No candidate has the segmented digit count -> loose read, then snap.
    if not probs:
        return 0 if 0 in candidates else min(candidates, key=abs)
    classes = [int(np.argmax(p)) for p in probs]
    if STRIKE_CLASS in classes:
        return 0 if 0 in candidates else min(candidates, key=abs)
    digits = [str(c) for c in classes if c < 10]
    value = sign * int("".join(digits)) if digits else 0
    return scorecard._snap(value, candidates)


def ink_fraction(cell):
    """Fraction of the cell that is ink -- used to spot blank cells cheaply."""
    if cell.size == 0:
        return 0.0
    binary = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    return float(np.count_nonzero(binary)) / binary.size


INK_MIN = 0.02  # below this a cell is treated as blank (-> 0)


def _read_cells_cnn(cells, raw):
    """Fill `raw` for every recognized cell using the bundled CNN.

    Constrained cells (entries, score, jackpot, points -- all have a legal set)
    are read by ranking that set with the model's probability vectors and taking
    the top value. The grand totals (I6, I8, J8) have no constraint and are read
    loosely so they stay independent of the schema for the cross-check.
    """
    for r in range(ROWS):
        for c in range(COLS):
            if not scorecard.needs_recognition(r, c):
                continue
            cell = cells[r * COLS + c]
            cands = scorecard.candidates(r, c)
            if not cands:
                raw[r][c] = read_cell(cell)
                continue
            sign, probs = _segment(cell)
            ranked = _rank(sign, probs, cands)
            raw[r][c] = ranked[0][0] if ranked else read_cell_constrained(cell, cands)
            if DEBUG_CROPS:
                top = ", ".join(f"{v}:{s:.3f}" for v, s in ranked[:3]) or "(no fit)"
                print(f"  {chr(ord('A') + r)}{c + 1} -> {raw[r][c]}   top: {top}")


def _read_cells_ctc(cells, raw):
    """Fill `raw` with the segmentation-free CTC reader: read each cell whole, in
    one batch, then snap to the cell's legal set. No blob splitting involved."""
    to_read = [(r, c) for r in range(ROWS) for c in range(COLS)
               if scorecard.needs_recognition(r, c)]
    values = ctc_reader.read_cells([cells[r * COLS + c] for r, c in to_read])
    for (r, c), v in zip(to_read, values):
        cands = scorecard.candidates(r, c)
        raw[r][c] = scorecard._snap(v, cands) if cands else v


def _read_cells_trocr(cells, raw):
    """Fill `raw` using TrOCR: blank cells by ink fraction, the rest in one batch."""
    to_ocr = []  # (r, c) cells that have ink and need OCR
    for r in range(ROWS):
        for c in range(COLS):
            if not scorecard.needs_recognition(r, c):
                continue
            if ink_fraction(cells[r * COLS + c]) < INK_MIN:
                raw[r][c] = 0  # blank == strike == 0
            else:
                to_ocr.append((r, c))

    # Pad each crop with a white border so the digit isn't flush to the edge.
    crops = []
    for r, c in to_ocr:
        cell = cells[r * COLS + c]
        pad = max(4, int(0.2 * min(cell.shape)))
        crops.append(cv2.copyMakeBorder(cell, pad, pad, pad, pad,
                                        cv2.BORDER_CONSTANT, value=255))
    texts = trocr_reader.read_crops(crops)
    for (r, c), text in zip(to_ocr, texts):
        digits = "".join(ch for ch in text if ch.isdigit())
        value = int(digits) if digits else 0  # no digits -> strike/unreadable -> 0
        if value and ("-" in text or "−" in text):  # minus -> negative points
            value = -value
        cands = scorecard.candidates(r, c)
        raw[r][c] = scorecard._snap(value, cands) if cands else value


def _grid_overlay(table, rb, cb):
    """Return a BGR copy of `table` with the row/col boundaries drawn in red."""
    h, w = table.shape
    vis = cv2.cvtColor(table, cv2.COLOR_GRAY2BGR)
    for y in rb:
        cv2.line(vis, (0, int(y)), (w, int(y)), (0, 0, 255), 2)
    for x in cb:
        cv2.line(vis, (int(x), 0), (int(x), h), (0, 0, 255), 2)
    return vis


def slice_sheet(image, debug=False, out_dir=None):
    """Geometry pipeline: PIL image -> (table, cells, rb, cb, rows_aligned, cols_aligned).

    Deskew -> locate -> dewarp -> trim -> grid-snap -> split into 80 cell crops.
    Shared by read_sheet and the dataset-export tool so both slice cells the same
    way. With debug=True, writes the numbered stage images to `out_dir` (the
    per-result folder), falling back to RESULTS_DIR.
    """
    dst = Path(out_dir) if out_dir is not None else RESULTS_DIR
    image = ImageOps.exif_transpose(image)  # honor EXIF orientation (e.g. rotated phone photos)
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if debug:
        cv2.imwrite(str(dst / "1-original.png"), gray)

    # Deskew the whole frame before locating/slicing the table.
    angle = detect_skew(gray)
    if angle is None:
        raise ScanError(
            f"The sheet looks rotated more than {MAX_SKEW_DEG:.0f}°. "
            "Please straighten it (within ±20°) and rescan."
        )
    gray = rotate(gray, angle)
    if debug:
        print(f"Deskew angle: {angle:+.2f}°")
        cv2.imwrite(str(dst / "2-deskew.png"), gray)

    table = locate_table(gray)                      # rough contour crop
    if debug:
        cv2.imwrite(str(dst / "3-crop.png"), table)
    table = dewarp(table)                           # flatten page curvature first
    if debug:
        cv2.imwrite(str(dst / "4-dewarp.png"), table)
    table = _trim_to_gridlines(table)               # then trim to grid extent
    if debug:
        cv2.imwrite(str(dst / "5-trim.png"), table)

    rb, cb, rows_aligned, cols_aligned = grid_boundaries(table)
    cells = split_cells(table, rb, cb)
    if debug:
        aligned = rows_aligned and cols_aligned
        print(f"Grid aligned to schema: rows={rows_aligned} cols={cols_aligned}"
              f"{'' if aligned else '  (even-split fallback on the unaligned axis)'}")
        cv2.imwrite(str(dst / "6-grid.png"), _grid_overlay(table, rb, cb))
    return table, cells, rb, cb, rows_aligned, cols_aligned


def read_sheet(image, out_dir=None):
    """Full pipeline: PIL image -> corrected 10x8 grid.

    Only handwritten cells are recognized; printed labels, blank cells and the
    computed sums are filled in by the scorecard schema (see scorecard.py), so
    the printed text and graphics on the sheet are never read as digits.
    """
    _, cells, _, _, _, _ = slice_sheet(image, debug=DEBUG_CROPS, out_dir=out_dir)
    raw = [[None] * COLS for _ in range(ROWS)]
    if USE_CTC:
        try:
            _read_cells_ctc(cells, raw)
        except Exception as e:  # model/keras issue -> don't 500
            print(f"CTC read failed ({e}); falling back to the CNN.")
            raw = [[None] * COLS for _ in range(ROWS)]
            _read_cells_cnn(cells, raw)
    elif USE_TROCR:
        try:
            _read_cells_trocr(cells, raw)
        except Exception as e:  # broken torch / download failure -> don't 500
            print(f"TrOCR read failed ({e}); falling back to the CNN.")
            raw = [[None] * COLS for _ in range(ROWS)]
            _read_cells_cnn(cells, raw)
    else:
        _read_cells_cnn(cells, raw)
    grid = scorecard.apply_schema(raw)
    warnings = scorecard.check_consistency(raw, grid)
    return grid, warnings


def format_grid(grid):
    """Render the 10x8 grid as an aligned table (rows A-J, cols 1-8) for logging."""
    cells = [["" if v == "" else str(v) for v in row] for row in grid]
    widths = [max(2, *(len(cells[r][c]) for r in range(ROWS))) for c in range(COLS)]
    lines = ["    " + "  ".join(str(c + 1).rjust(widths[c]) for c in range(COLS))]
    for r in range(ROWS):
        row = "  ".join(cells[r][c].rjust(widths[c]) for c in range(COLS))
        lines.append(f"{chr(ord('A') + r)} | {row}")
    return "\n".join(lines)


# Each read gets its own results/ subfolder named YYYYMMDD-xxxxxx, where xxxxxx is
# 6 lowercase hex chars (like a short git SHA). All artifacts for that read --
# input image, debug stages, scorecard CSV, the verdict, and any feedback -- live
# together there.
RESULT_ID_RE = re.compile(r"^\d{8}-[0-9a-f]{6}$")


def _new_result_id():
    return f"{datetime.now():%Y%m%d}-{hashlib.sha256(os.urandom(16)).hexdigest()[:6]}"


def _valid_id(result_id):
    """Validate a client-supplied result id and confirm the read exists in the store.
    The regex guards against path-traversal / crafted storage keys."""
    if not RESULT_ID_RE.fullmatch(result_id or ""):
        raise HTTPException(status_code=400, detail="Invalid result id.")
    if not STORE.exists(result_id):
        raise HTTPException(status_code=404, detail="Unknown result id.")
    return result_id


@app.post("/read")
async def read(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {MODEL_PATH} not loaded. Train it in abi-models and copy it here.",
        )
    result_id = _new_result_id()
    out_dir = STORE.new_working_dir(result_id)  # local dir, or a temp dir staged for S3
    try:
        contents = await file.read()
        (out_dir / "input.jpg").write_bytes(contents)  # keep the original for ground truth
        image = Image.open(io.BytesIO(contents))
        grid, warnings = read_sheet(image, out_dir=out_dir)

        # Always log what was read.
        print("Scorecard read:")
        print(format_grid(grid))

        # The sheet's own written Score/Points totals are read independently and
        # cross-checked against what we compute from the cells. A disagreement is a
        # hint that something was misread, but it's advisory only: we log it and
        # proceed with the computed (best-effort) value rather than rejecting the sheet.
        if warnings:
            print("Cross-check warnings (proceeding with computed values):")
            for w in warnings:
                print(f"  - {w}")

        csv = "\n".join(",".join(str(v) for v in row) for row in grid) + "\n"
        (out_dir / "7-scorecard.csv").write_text(csv)

        # Which cells the correction UI lets the player edit (handwritten cells;
        # this includes I8, which we derive rather than read but is still written).
        editable = [[scorecard.is_editable(r, c) for c in range(COLS)] for r in range(ROWS)]
        saved_as = STORE.describe(result_id)
        response = {
            "id": result_id,
            "grid": grid,
            "editable": editable,
            "csv": csv,
            "saved_as": saved_as,
            "warnings": warnings,
        }
    except ScanError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Persist whatever was written (even a failed scan's input/debug images, for
        # later inspection); best-effort so a storage hiccup doesn't mask the result.
        try:
            STORE.commit(result_id, out_dir)
        except Exception as e:
            print(f"WARNING: failed to persist {result_id}: {e}")

    print(f"Saved {saved_as}")
    return response


@app.post("/accept")
async def accept(id: str = Form(...)):
    """Record a thumbs-up on a read."""
    STORE.put_text(_valid_id(id), "verdict.md", "accepted\n")
    return {"ok": True}


@app.post("/decline")
async def decline(id: str = Form(...)):
    """Record a thumbs-down on a read."""
    STORE.put_text(_valid_id(id), "verdict.md", "declined\n")
    return {"ok": True}


@app.post("/feedback")
async def feedback(id: str = Form(...), grid: str = Form(...)):
    """Store a user-corrected scorecard as ground truth.

    Saved as 8-feedback.csv in the read's own <id>/ folder (which also holds
    input.jpg), so the image and the corrected labels stay together for folding
    into the training set later.
    """
    rid = _valid_id(id)
    try:
        rows = json.loads(grid)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="`grid` must be a JSON array.")
    csv = "\n".join(
        ",".join("" if v is None else str(v) for v in row) for row in rows
    ) + "\n"
    STORE.put_text(rid, "8-feedback.csv", csv)
    saved_as = STORE.describe(rid, "8-feedback.csv")
    print(f"Saved corrected scorecard (ground truth) to {saved_as}")
    return {"ok": True, "saved_as": saved_as}


if __name__ == "__main__":
    import uvicorn

    # `reload: true` (local-dev.yaml) auto-restarts the server when a .py file
    # changes. Reload requires the import-string form ("server:app") rather than the
    # app object, and the `watchfiles` package (pip install watchfiles).
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=config.RELOAD)
