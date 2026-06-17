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
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cv2
import io

import scorecard

app = FastAPI()

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
    """Find the table, perspective-correct it, and return a flat grayscale crop.

    Assumes the table is the dominant roughly-rectangular object in the photo.
    Falls back to the whole image if no quadrilateral contour is found.
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

    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        x, y, w, h = cv2.boundingRect(biggest)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, COLS * 10), max(height, ROWS * 10)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (width, height))


def split_cells(table):
    """Slice the flat table into ROWS x COLS cells (row-major), trimming borders."""
    h, w = table.shape
    cell_h, cell_w = h / ROWS, w / COLS
    my, mx = int(cell_h * 0.12), int(cell_w * 0.12)  # trim grid lines
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            y0, y1 = int(r * cell_h) + my, int((r + 1) * cell_h) - my
            x0, x1 = int(c * cell_w) + mx, int((c + 1) * cell_w) - mx
            cells.append(table[max(y0, 0):y1, max(x0, 0):x1])
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


def read_sheet(image):
    """Full pipeline: PIL image -> corrected 10x8 grid.

    Only handwritten cells are recognized; printed labels, blank cells and the
    computed sums are filled in by the scorecard schema (see scorecard.py), so
    the printed text and graphics on the sheet are never read as digits.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    table = locate_table(gray)
    cells = split_cells(table)
    raw = [[None] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if not scorecard.needs_recognition(r, c):
                continue
            cell = cells[r * COLS + c]
            cands = scorecard.candidates(r, c)
            # Entries and jackpot have small legal sets -> constrained decode.
            # Totals (col 6, I6, J8) are read loosely to keep the cross-check honest.
            raw[r][c] = read_cell_constrained(cell, cands) if cands else read_cell(cell)
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_as = RESULTS_DIR / f"result-{stamp}.csv"
    saved_as.write_text(csv)
    print(f"Saved {saved_as}")

    return {"grid": grid, "csv": csv, "saved_as": str(saved_as)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
