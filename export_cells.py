"""Export labeled cell crops from a scorecard photo + its manual ground truth.

Bootstraps a training/eval dataset for the recognizer. It runs the *same*
geometry pipeline the server uses (`server.slice_sheet`: deskew -> locate ->
dewarp -> trim -> grid-snap -> split), then pairs each value-bearing cell with
its true value from a manual ground-truth CSV (the kind the server writes, e.g.
`results/manual-scorecard.csv`) and saves:

  - one PNG per cell crop, named `<sheet>_<cell>_<value>.png` (e.g. `3_B2_8.png`,
    `3_G5_strike.png` for a strike/empty), and
  - a `labels.csv` (appended to, so you can accumulate many sheets) with columns
    file,sheet,cell,row,col,value,is_blank.

Only handwritten / value-bearing cells are exported (printed labels and the
always-empty cells are skipped, via `scorecard.needs_recognition`).

Usage:
    python export_cells.py --image ../abi-dataset/images/3.jpg --truth results/manual-scorecard.csv
    python export_cells.py --image ../abi-dataset/images/3.jpg --truth results/manual-scorecard.csv \
                           --sheet 3 --out ../abi-dataset/cells

Run it once per (photo, ground-truth) pair to build up ../abi-dataset/cells/.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
from PIL import Image

import server
import scorecard

ROWS, COLS = scorecard.ROWS, scorecard.COLS


def cell_name(r, c):
    """(0-based row, col) -> sheet label like 'B2' (rows A-J, cols 1-8)."""
    return f"{chr(ord('A') + r)}{c + 1}"


def load_truth(path):
    """Read a manual ground-truth CSV into a 10x8 grid of stripped strings.

    Accepts the server's own grid dump (10 data rows A-J, 8 columns, no header).
    Short/missing cells are padded with "".
    """
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f)]
    if len(rows) < ROWS:
        sys.exit(f"Truth CSV {path} has {len(rows)} rows; expected {ROWS} (A-J).")
    grid = []
    for r in range(ROWS):
        cells = [c.strip() for c in rows[r]]
        cells = (cells + [""] * COLS)[:COLS]
        grid.append(cells)
    return grid


# A strike (or empty cell, or the "-" no-jackpot marker) -> value 0. The canonical
# strike glyph in our CSVs is "x", but accept the other strike marks too. A written
# "0" is NOT a strike here -- it's a genuine 0 (e.g. a row score of 0).
STRIKE_TEXT = {"", "-", "x", "X", "/", "\\"}


def truth_value(text):
    """Map a ground-truth cell string to (value:int, is_blank:bool).

    Strike / empty cells -> (0, True); anything else is parsed as an integer.
    """
    if text in STRIKE_TEXT:
        return 0, True
    try:
        return int(text), False
    except ValueError:
        # Non-numeric and not a known strike mark -> skip by signalling None.
        return None, False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="scorecard photo (jpg/png)")
    ap.add_argument("--truth", required=True,
                    help="manual ground-truth CSV (10x8 grid, A-J rows)")
    ap.add_argument("--sheet", default=None,
                    help="sheet id used in filenames/labels (default: image stem)")
    ap.add_argument("--out", default="../abi-dataset/cells", help="output dir")
    args = ap.parse_args()

    image_path = Path(args.image)
    sheet = args.sheet or image_path.stem
    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth(args.truth)
    image = Image.open(image_path)

    # Same slicing the server uses, so crops line up with what gets recognized.
    _, cells, _, _, nr, nc = server.slice_sheet(image, debug=False)
    if (nr, nc) != (ROWS + 1, COLS + 1):
        print(f"WARNING: grid detected {nr}/{ROWS + 1} rows, {nc}/{COLS + 1} cols "
              f"(even-split fallback in play); crops may be misaligned.", file=sys.stderr)

    labels_path = out_dir / "labels.csv"
    new_file = not labels_path.exists()
    exported = skipped = 0
    with open(labels_path, "a", newline="") as lf:
        writer = csv.writer(lf)
        if new_file:
            writer.writerow(["file", "sheet", "cell", "row", "col", "value", "is_blank"])

        for r in range(ROWS):
            for c in range(COLS):
                if not scorecard.needs_recognition(r, c):
                    continue
                value, is_blank = truth_value(truth[r][c])
                if value is None:
                    skipped += 1
                    continue
                crop = cells[r * COLS + c]
                if crop is None or crop.size == 0:
                    print(f"  skip {cell_name(r, c)}: empty crop", file=sys.stderr)
                    skipped += 1
                    continue

                tag = "strike" if is_blank else str(value).replace("-", "m")
                fname = f"{sheet}_{cell_name(r, c)}_{tag}.png"
                cv2.imwrite(str(crops_dir / fname), crop)
                writer.writerow([f"crops/{fname}", sheet, cell_name(r, c),
                                 r, c, value, int(is_blank)])
                exported += 1

    print(f"Exported {exported} cell crops from {image_path.name} "
          f"(sheet '{sheet}') to {crops_dir}/  [skipped {skipped}]")
    print(f"Labels -> {labels_path}")


if __name__ == "__main__":
    main()
