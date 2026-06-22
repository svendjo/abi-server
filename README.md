# abi-server
Web service for Balut Eye.

Takes an uploaded photo of a 10×8 table of handwritten numbers, locates the table
with OpenCV, slices it into 80 cells, recognizes each cell with the TFLite model
from `abi-models`, and returns the result as a 10×8 grid. A copy of every result
is saved server-side as a CSV under `results/`.

`POST /predict` (multipart `file`) →
`{ "grid": [[..8..], …10 rows], "csv": "<10 lines>", "saved_as": "results/result-<ts>.csv" }`

A cell containing `/`, `\`, or `x` is read as `0`.

## Scorecard schema
The sheet is not just any 10×8 grid — it is the IBF Balut card, and most cells are
constrained. `scorecard.py` encodes those rules and `read_sheet` applies them:

- **Printed text and graphics are ignored.** Column 1 (`4's`, `5's`, `Straight`, …) and the
  row-A headers (`Score`, `Jackpot`, `Points`) are not handwritten, so they are never
  digit-recognized — the schema fills them in. Always-empty cells (`H7`, `I7`, `A1`–`A5`) are
  left blank.
- **Handwritten cells are constrained to their legal set.** e.g. a `4's` game cell must be one of
  `{4, 8, 12, 16, 20, strike}`, a jackpot must be `16` or a strike. Rather than read a cell
  freely and snap afterwards, `read_cell_constrained` keeps the model's full per-character
  probabilities and picks the legal value with the highest likelihood (so `"76"` → `16`, not the
  numerically-nearest `20`). Points cells aren't read at all — they're derived (see below).
- **Sums are computed, not read.** Each row's Score (col 6) is the sum of its four game cells,
  `I6` is the grand total of scores, and `J8` is the grand total of points.
- **Points (col 8) are derived.** B–G from the jackpot/incentive rules, H (Balut) from the number
  of baluts scored. Negative values come from the rules, not from reading a minus sign.
- **The written totals are cross-checked.** Each row Score and the two grand totals are *also* read
  (loosely, to stay independent) and compared to the computed values; a mismatch makes `/predict`
  return `422` instead of a likely-wrong grid.

The returned grid therefore mixes the printed labels (strings) with the corrected numbers.

## Recognizers
Two interchangeable ways to read a cell; both feed the same `scorecard.py` schema layer:

- **Bundled CNN (default).** The small TFLite digit/strike model from `abi-models`, with
  per-character segmentation and probability-constrained decoding. No extra deps.
- **TrOCR (optional, local).** `microsoft/trocr-base-handwritten` reads a whole cell crop in
  one shot — no segmentation — which is more robust on real photos. Enable it with:

  ```
  pip install -r requirements-trocr.txt   # heavy: torch + transformers (~GBs)
  USE_TROCR=1 python server.py            # model (~1.3 GB) downloads on first run
  ```

  Blank cells are caught by an ink-fraction check; the rest are batch-read by TrOCR, parsed to
  digits, and snapped to each cell's legal set. If the deps aren't installed the server logs a
  warning and falls back to the CNN. (TrOCR is trained on English handwriting lines, not isolated
  digits, and is slow on CPU — batched but still ~seconds per sheet; a GPU helps a lot.)

## Local
Create a virtual environment.

`pyenv virtualenv 3.12.7 venv-abi-server`

Activate the environment.

`pyenv local venv-abi-server`

Install requirements.

`pip install setuptools`
`pip install -r requirements-dev.txt`

Train the model in `abi-models` (run `Recognition.ipynb`) and copy the result
here:

`cp ../abi-models/digit-model.tflite .`

Run the server.

`python server.py`

Test it.

`curl -X POST http://localhost:8080/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/sheet.jpg"`

## Docker
Build a Docker image.

`docker build -t balut-docker:latest .`

Run the Docker container locally. Remember to expose the port.

`docker run -p 8080:8080 balut-docker:latest`

## Tag and push it to AWS ECR
`aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com`

`docker tag balut-docker:latest <account>.dkr.ecr.us-west-2.amazonaws.com/balut-repository:latest`

`docker push <account>.dkr.ecr.us-west-2.amazonaws.com/balut-repository:latest`

## Deploy AWS App Runner
Go to AWS App Runner and deploy balut-repository:latest.
Now go build & deploy the frontend.
