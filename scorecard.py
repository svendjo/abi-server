"""Balut Eye scorecard schema and correction.

The sheet is a fixed 10x8 table. Rows are A-J (top to bottom), columns 1-8
(left to right). Column 1 holds printed category labels and row A holds printed
column headers ("Score", "Jackpot", "Points"); those cells are NOT handwritten,
so we never run digit recognition on them -- the schema supplies their text
directly. The remaining (handwritten) cells are each constrained to a small set
of legal values, which we use to correct the raw model output, and several cells
are sums of others, which we compute rather than read.

A `strike` (a printed /, \\ or x) scores 0, so it is represented here as the
value 0. Point cells (column 8) may be negative.
"""

ROWS, COLS = 10, 8
STRIKE = 0  # a struck cell (/, \, x) -> value 0


def _idx(row, col):
    """('B', 3) -> (row_index, col_index) into a 0-based 10x8 grid."""
    return ord(row) - ord("A"), col - 1


# --- Printed labels (column 1 + the row-A headers). Never recognized. ---------
LABELS = {
    ("A", 6): "Score", ("A", 7): "Jackpot", ("A", 8): "Points",
    ("B", 1): "4's", ("C", 1): "5's", ("D", 1): "6's",
    ("E", 1): "Straight", ("F", 1): "Full House",
    ("G", 1): "Choice", ("H", 1): "Balut",
    ("I", 1): "Total Score", ("J", 1): "Points - Grand Total",
}

# --- Cells that are always blank. ---------------------------------------------
EMPTY = {("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5), ("H", 7), ("I", 7)}

# Full House scores the sum of five dice making a full house: three of value a
# plus two of value b (a != b), i.e. 3a + 2b. (Notably 10 and 25 are impossible.)
FULL_HOUSE = {3 * a + 2 * b for a in range(1, 7) for b in range(1, 7) if a != b}
# Choice scores the sum of five six-sided dice: anything in [5, 30].
CHOICE = set(range(5, 31))

# Balut (row H) has no jackpot; its points are the number of baluts scored
# (non-strike cells in H2-5): 0->0, 1->3, 2->8, 3->12, 4->16.
BALUT_POINTS = {0: 0, 1: 3, 2: 8, 3: 12, 4: 16}

# --- Per-row rules for the seven scoring rows B-H. ----------------------------
# entries : legal values for the four game cells (cols 2-5); None = free.
#           STRIKE (0) is always additionally allowed (a strike may also be
#           written as a plain "0").
# jackpot : the single legal jackpot value (col 7); STRIKE also allowed.
#           None = no jackpot cell (always empty).
# jackpot   : the achievement value shown in col 7 (or None for H, no jackpot).
# jp_points : points scored in col 8 for the jackpot -- +jp_points if achieved,
#             -jp_points if struck. (Distinct from `jackpot`: e.g. achieving the
#             4's jackpot shows "16" in col 7 but scores 4 points.)
# incentive : points earned when no jackpot was attempted and the incentive
#             condition holds; `threshold` is a minimum row score, or "all"
#             meaning every game cell must be non-strike.
# multiple  : the row score (col 6) must be a multiple of this (None = no rule).
# Col 8 (Points) is always DERIVED, never read: B-G from the jackpot/incentive
# (see _points), H from the number of baluts (see BALUT_POINTS). H has no jackpot.
ROW_SPEC = {
    "B": dict(entries={4, 8, 12, 16, 20},       jackpot=16,   jp_points=4, incentive=2, threshold=52,    multiple=4),
    "C": dict(entries={5, 10, 15, 20, 25},      jackpot=20,   jp_points=4, incentive=2, threshold=65,    multiple=5),
    "D": dict(entries={6, 12, 18, 24},          jackpot=24,   jp_points=4, incentive=2, threshold=78,    multiple=6),
    "E": dict(entries={15, 20},                 jackpot=20,   jp_points=8, incentive=4, threshold="all", multiple=None),
    "F": dict(entries=FULL_HOUSE,               jackpot=22,   jp_points=6, incentive=3, threshold="all", multiple=None),
    "G": dict(entries=CHOICE,                   jackpot=25,   jp_points=4, incentive=2, threshold=100,    multiple=None),
    "H": dict(entries={25, 30, 35, 40, 45, 50}, jackpot=None,                                            multiple=None),
}
SCORE_ROWS = list(ROW_SPEC)  # B..H, in order

# Cells whose handwritten value duplicates something the schema computes (each
# row's Score, and the two grand totals). We read them anyway and cross-check
# the written number against the computed one to catch misreads.
CHECK_CELLS = [(row, 6) for row in SCORE_ROWS] + [("I", 6), ("J", 8)]


def needs_recognition(r, c):
    """True iff the (0-based) cell is handwritten and must be read by the model.

    Printed labels and always-empty cells are excluded so the server never tries
    to digit-recognize printed text. The computed sums (col 6, I6, J8) are read
    too -- not for the output grid, but so check_consistency can cross-check them.
    """
    row, col = chr(ord("A") + r), c + 1
    if (row, col) in LABELS or (row, col) in EMPTY:
        return False
    if (row, col) in CHECK_CELLS:  # written totals, read only to cross-check
        return True
    if row in ROW_SPEC:
        if col in (2, 3, 4, 5):  # game cells
            return True
        if col == 7:  # jackpot (None for H, which has no jackpot)
            return ROW_SPEC[row]["jackpot"] is not None
        if col == 8:  # points: derived, but read too as a cross-check
            return True
        return False  # col 6 score is read via CHECK_CELLS
    if (row, col) == ("I", 8):  # a point value that feeds J8
        return True
    return False  # I6 and J8 are computed; everything else is blank


def score_candidates(row):
    """Every value the row's score (col 6) can legally take: a sum of its four
    game cells, each of which is a legal entry or a strike (0)."""
    entries = ROW_SPEC[row]["entries"] | {STRIKE}
    sums = {0}
    for _ in range(4):
        sums = {s + e for s in sums for e in entries}
    return sums


def points_candidates(row):
    """The row's small legal set of point values (col 8). For B-G this is
    {0, incentive, +jp_points, -jp_points}; for H (Balut) the balut-count points.
    """
    spec = ROW_SPEC[row]
    if spec["jackpot"] is None:  # H: points come from the number of baluts
        return set(BALUT_POINTS.values())
    jp = spec["jp_points"]
    return {0, spec["incentive"], jp, -jp}


def candidates(r, c):
    """Legal value set for a CONSTRAINED read of cell (r, c), or None.

    The reader keeps the model's full probability vectors and picks the legal
    value with the highest likelihood. We constrain:
      - entry cells (cols 2-5) to their per-row value set,
      - the jackpot (col 7) to {jackpot value, strike},
      - the row score (col 6) to its achievable sums, and
      - the points (col 8) to the row's small legal set.
    Col 6 and col 8 are *also* computed from the schema; the constrained read is
    an independent cross-check (read from the cell's own pixels, not the entries).
    Returns None for the grand totals (I6, I8, J8), which are read loosely so they
    stay fully independent of the schema.
    """
    row, col = chr(ord("A") + r), c + 1
    if row in ROW_SPEC:
        if col in (2, 3, 4, 5):  # game cells
            return ROW_SPEC[row]["entries"] | {STRIKE}
        if col == 6:  # row score: a sum of the four legal entries
            return score_candidates(row)
        if col == 7 and ROW_SPEC[row]["jackpot"] is not None:  # jackpot
            return {ROW_SPEC[row]["jackpot"], STRIKE}
        if col == 8:  # points (also derived)
            return points_candidates(row)
    return None


def _snap(value, allowed):
    """Nearest legal value to `value` (ties resolve to the smaller magnitude)."""
    return min(allowed, key=lambda a: (abs(a - value), abs(a)))


def _incentive_met(spec, entries, score):
    """Whether the row's incentive condition holds (a strike == empty == 0)."""
    if spec["threshold"] == "all":
        return all(e > 0 for e in entries)  # every game cell non-strike
    return score >= spec["threshold"]


def _points(spec, entries, score, achieved):
    """Derive a B-G points cell.

    The incentive condition decides the sign in both cases. If the jackpot was
    achieved (col 7 holds its value) the row scores +/- jp_points; otherwise
    (col 7 struck or empty -- the same thing) it scores the incentive or 0.
    """
    met = _incentive_met(spec, entries, score)
    if achieved:
        return spec["jp_points"] if met else -spec["jp_points"]
    return spec["incentive"] if met else 0


def apply_schema(raw):
    """Correct a raw 10x8 grid into the final grid.

    `raw[r][c]` is the model's integer reading for handwritten cells and None
    elsewhere. Returns a new 10x8 grid where label cells hold their printed text,
    blank cells hold "", handwritten cells are snapped to their legal set, and
    the score column, I6 and J8 are computed from the schema.
    """
    out = [["" for _ in range(COLS)] for _ in range(ROWS)]

    # Labels (printed text) come straight from the schema.
    for (row, col), text in LABELS.items():
        r, c = _idx(row, col)
        out[r][c] = text

    # Scoring rows B-H: snap entries, compute the score, snap jackpot/points.
    for row in SCORE_ROWS:
        spec = ROW_SPEC[row]
        r = ord(row) - ord("A")

        entries = []
        for col in (2, 3, 4, 5):
            v = _snap(raw[r][col - 1] or 0, spec["entries"] | {STRIKE})
            entries.append(v)
            out[r][col - 1] = v

        score = sum(entries)
        out[r][5] = score  # col 6 = row score

        if spec["jackpot"] is None:  # H (Balut): no jackpot; col 7 stays "-"
            out[r][6] = "-"
            baluts = sum(1 for e in entries if e > 0)  # non-strike H2-5 cells
            out[r][7] = BALUT_POINTS[baluts]  # col 8
            continue

        # Col 7 is the jackpot: its achievement value, or struck/empty (== 0).
        jpv = spec["jackpot"]
        achieved = _snap(raw[r][6] or 0, {jpv, STRIKE}) == jpv
        out[r][6] = jpv if achieved else ""
        out[r][7] = _points(spec, entries, score, achieved)  # col 8 (derived)

    # I8 is a read point value; keep it (used by the J8 grand total).
    i8 = raw[_idx("I", 8)[0]][_idx("I", 8)[1]]
    out[_idx("I", 8)[0]][_idx("I", 8)[1]] = i8 if i8 is not None else 0

    # I6 = sum of the score column over B..H.
    out[_idx("I", 6)[0]][_idx("I", 6)[1]] = sum(
        out[ord(row) - ord("A")][5] for row in SCORE_ROWS
    )

    # J8 = sum of the point column B8..I8.
    out[_idx("J", 8)[0]][_idx("J", 8)[1]] = sum(
        out[ord(row) - ord("A")][7] for row in SCORE_ROWS + ["I"]
    )

    return out


def check_consistency(raw, grid):
    """Cross-check the player's handwritten totals against the computed ones.

    Returns a list of human-readable WARNING strings (empty if the sheet is
    self-consistent). The player writes every row Score (col 6) and Points
    (col 8) plus the two grand totals; we read those independently and flag any
    that disagree with what the schema computes from the cells. These are
    advisory only -- the caller logs them and proceeds with the computed value
    (the best-effort read), it does not reject the sheet.
    """
    warnings = []

    def written(row, col):
        return raw[_idx(row, col)[0]][_idx(row, col)[1]]
    def computed(row, col):
        return grid[_idx(row, col)[0]][_idx(row, col)[1]]

    for row in SCORE_ROWS:
        wr6 = written(row, 6)
        if wr6 is not None and wr6 != computed(row, 6):
            warnings.append(
                f"Row {row}: read score {wr6} != computed {computed(row, 6)}; using computed."
            )
        wr8 = written(row, 8)
        if wr8 is not None and wr8 != computed(row, 8):
            warnings.append(
                f"Row {row}: read points {wr8} != computed {computed(row, 8)}; using computed."
            )

    wr_i6 = written("I", 6)
    if wr_i6 is not None and wr_i6 != computed("I", 6):
        warnings.append(
            f"Total Score (I6): read {wr_i6} != computed {computed('I', 6)}; using computed."
        )

    wr_j8 = written("J", 8)
    if wr_j8 is not None and wr_j8 != computed("J", 8):
        warnings.append(
            f"Grand Total points (J8): read {wr_j8} != computed {computed('J', 8)}; using computed."
        )

    return warnings
