from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from tflite_runtime.interpreter import Interpreter as tflite
except ImportError:
    # Local dev uses full TensorFlow. tf.lite is lazily loaded, so it must be
    # reached via attribute access -- `from tensorflow.lite import Interpreter`
    # does not trigger the lazy loader and fails on TF 2.16.
    import tensorflow as tf
    tflite = tf.lite.Interpreter
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import io
import os

import scorecard
import trocr_reader

app = FastAPI()

# Optionally read cells with the local TrOCR handwriting model instead of the
# bundled CNN. Falls back to the CNN if the deps aren't installed.
USE_TROCR = bool(os.environ.get("USE_TROCR")) and trocr_reader.available()
if os.environ.get("USE_TROCR") and not USE_TROCR:
    print("WARNING: USE_TROCR set but torch/transformers not available; using the CNN.")
elif USE_TROCR:
    print("Using TrOCR for cell recognition.")

# Save the located table and a grid overlay to results/ to debug slicing.
DEBUG_CROPS = bool(os.environ.get("DEBUG_CROPS"))

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

# Where finished CSVs are written.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load the digit/symbol recognizer. If it hasn't been trained/copied yet the
# server still boots so the frontend can be developed; /predict then 503s.
MODEL_PATH = "digit-model0608.tflite"
try:
    interpreter = tflite(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Loaded model: {MODEL_PATH}")
except Exception as e:  # noqa: BLE001 - boot anyway, report at request time
    interpreter = None
    input_details = output_details = None
    print(f"WARNING: could not load {MODEL_PATH} ({e}). /predict will return 503.")


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
    """Crop to the table. The image is already deskewed, so an axis-aligned
    bounding box of the dominant contour is enough. Falls back to the full frame
    if no large contour is found.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    biggest = max(contours, key=cv2.contourArea)
    # Ignore tiny detections (noise) and fall back to the full frame.
    if cv2.contourArea(biggest) < 0.2 * gray.shape[0] * gray.shape[1]:
        return gray

    x, y, w, h = cv2.boundingRect(biggest)
    return gray[y:y + h, x:x + w]


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


def grid_boundaries(table):
    """Return (row_bounds, col_bounds, n_rows_found, n_cols_found).

    Uses the detected printed rules (with missing edge borders filled in) when
    their count matches the schema exactly (ROWS+1 horizontal, COLS+1 vertical);
    otherwise falls back to an even split for that axis. Detected bounds handle
    the uneven label/Points columns.
    """
    h, w = table.shape
    rraw = _complete_boundaries(_line_positions(table, 0), h)
    craw = _complete_boundaries(_line_positions(table, 1), w)
    rb = rraw if len(rraw) == ROWS + 1 else [round(i * h / ROWS) for i in range(ROWS + 1)]
    cb = craw if len(craw) == COLS + 1 else [round(i * w / COLS) for i in range(COLS + 1)]
    return rb, cb, len(rraw), len(craw)


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


def _segment(cell):
    """Cell -> (sign, [prob_vector per digit blob], left-to-right).

    A blank cell yields an empty list. A separate wide, short blob to the left is
    a minus sign and sets sign to -1 (the digit model has no minus class).
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
    probs = [classify(preprocess_char(binary[y:y + h, x:x + w]))
             for (x, y, w, h) in digit_boxes]
    return sign, probs


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


def read_cell_constrained(cell, candidates):
    """Read a cell, choosing the most likely value from `candidates`.

    `candidates` is the cell's legal value set (non-negative ints; 0 covers
    strike/empty). Instead of taking the argmax per character and snapping the
    result afterwards, we keep the model's full per-character probabilities and
    pick the legal value with the highest likelihood -- so the constraint guides
    the read and is ranked by the model's own confusion, not numeric distance.
    """
    _, probs = _segment(cell)
    if not probs:
        return 0 if 0 in candidates else min(candidates, key=abs)
    n = len(probs)

    best, best_score = None, -1.0
    for v in candidates:
        if v == 0:  # strike or a written "0": a single blob
            if n != 1:
                continue
            score = float(max(probs[0][STRIKE_CLASS], probs[0][0]))
        else:
            digits = [int(ch) for ch in str(v)]
            if len(digits) != n:
                continue
            score = 1.0
            for p, d in zip(probs, digits):
                score *= float(p[d])
        if score > best_score:
            best, best_score = v, score
    if best is not None:
        return best

    # No candidate has the segmented digit count -> fall back to a loose read,
    # then snap to the legal set (handles mis-segmentation as best we can).
    classes = [int(np.argmax(p)) for p in probs]
    if STRIKE_CLASS in classes:
        return 0 if 0 in candidates else min(candidates, key=abs)
    digits = [str(c) for c in classes if c < 10]
    value = int("".join(digits)) if digits else 0
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
    """Fill `raw` for every recognized cell using the bundled CNN."""
    for r in range(ROWS):
        for c in range(COLS):
            if not scorecard.needs_recognition(r, c):
                continue
            cell = cells[r * COLS + c]
            cands = scorecard.candidates(r, c)
            # Entries and jackpot have small legal sets -> constrained decode.
            # Totals (col 6, I6, J8) are read loosely to keep the cross-check honest.
            raw[r][c] = read_cell_constrained(cell, cands) if cands else read_cell(cell)


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


def read_sheet(image):
    """Full pipeline: PIL image -> corrected 10x8 grid.

    Only handwritten cells are recognized; printed labels, blank cells and the
    computed sums are filled in by the scorecard schema (see scorecard.py), so
    the printed text and graphics on the sheet are never read as digits.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if DEBUG_CROPS:
        cv2.imwrite(str(RESULTS_DIR / "1-original.png"), gray)

    # Deskew the whole frame before locating/slicing the table.
    angle = detect_skew(gray)
    if angle is None:
        raise ScanError(
            f"The sheet looks rotated more than {MAX_SKEW_DEG:.0f}°. "
            "Please straighten it (within ±20°) and rescan."
        )
    gray = rotate(gray, angle)
    if DEBUG_CROPS:
        print(f"Deskew angle: {angle:+.2f}°")
        cv2.imwrite(str(RESULTS_DIR / "2-deskew.png"), gray)

    table = locate_table(gray)                      # rough contour crop
    if DEBUG_CROPS:
        cv2.imwrite(str(RESULTS_DIR / "3-crop.png"), table)
    table = dewarp(table)                           # flatten page curvature first
    if DEBUG_CROPS:
        cv2.imwrite(str(RESULTS_DIR / "4-dewarp.png"), table)
    table = _trim_to_gridlines(table)               # then trim to grid extent
    if DEBUG_CROPS:
        cv2.imwrite(str(RESULTS_DIR / "5-trim.png"), table)

    rb, cb, nr, nc = grid_boundaries(table)
    cells = split_cells(table, rb, cb)
    if DEBUG_CROPS:
        ok = nr == ROWS + 1 and nc == COLS + 1
        print(f"Grid rules: rows {nr}/{ROWS + 1}, cols {nc}/{COLS + 1}"
              f"{'' if ok else '  (even-split fallback where the count mismatched)'}")
        cv2.imwrite(str(RESULTS_DIR / "6-grid.png"), _grid_overlay(table, rb, cb))
    raw = [[None] * COLS for _ in range(ROWS)]
    if USE_TROCR:
        try:
            _read_cells_trocr(cells, raw)
        except Exception as e:  # broken torch / download failure -> don't 500
            print(f"TrOCR read failed ({e}); falling back to the CNN.")
            raw = [[None] * COLS for _ in range(ROWS)]
            _read_cells_cnn(cells, raw)
    else:
        _read_cells_cnn(cells, raw)
    grid = scorecard.apply_schema(raw)
    problems = scorecard.check_consistency(raw, grid)
    return grid, problems


def format_grid(grid):
    """Render the 10x8 grid as an aligned table (rows A-J, cols 1-8) for logging."""
    cells = [["" if v == "" else str(v) for v in row] for row in grid]
    widths = [max(2, *(len(cells[r][c]) for r in range(ROWS))) for c in range(COLS)]
    lines = ["    " + "  ".join(str(c + 1).rjust(widths[c]) for c in range(COLS))]
    for r in range(ROWS):
        row = "  ".join(cells[r][c].rjust(widths[c]) for c in range(COLS))
        lines.append(f"{chr(ord('A') + r)} | {row}")
    return "\n".join(lines)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {MODEL_PATH} not loaded. Train it in abi-models and copy it here.",
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        grid, problems = read_sheet(image)
    except ScanError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Always log what was read, whether or not it passes validation.
    print("Scorecard read:")
    print(format_grid(grid))

    # The sheet's own written totals must agree with the cells. If they don't,
    # the read is untrustworthy -- return an error (with what we read) instead of
    # a confidently-wrong grid, and don't save it.
    if problems:
        print("Consistency check FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Scorecard failed consistency checks; it was likely misread.",
                "problems": problems,
                "grid": grid,
            },
        )

    csv = "\n".join(",".join(str(v) for v in row) for row in grid) + "\n"
    saved_as = RESULTS_DIR / "7-scorecard.csv"  # overwritten each scan
    saved_as.write_text(csv)
    print(f"Saved {saved_as}")

    return {"grid": grid, "csv": csv, "saved_as": str(saved_as)}


if __name__ == "__main__":
    import uvicorn

    # RELOAD=1 auto-restarts the server when a .py file changes (dev only).
    # Reload requires the import-string form ("server:app") rather than the app
    # object, and the `watchfiles` package (pip install watchfiles).
    reload = bool(os.environ.get("RELOAD"))
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=reload)
