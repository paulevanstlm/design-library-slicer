"""Find horizontal module boundaries in a full-page screenshot.

Pure Python so the logic can be tested without NumPy; slice.py feeds it rows sampled with NumPy.
`rows` is a list of rows, each a list of (r, g, b) samples taken at a fixed x step across the page.

Method (brief section 5.4, pixel pass):
  1. Band changes: the background colour of the left and right margins changes and stays changed.
  2. Whitespace gaps: long runs of near-uniform rows, cut at the middle of the run.
Candidates are then merged so no module is shorter than MIN_HEIGHT (a short slice joins the section below it,
so headings stay with their content), oversized slices are split at their widest internal gap, and the weakest
cuts are dropped until the page has at most MAX_MODULES modules. A fixed sidebar menu is trimmed off first.
"""

from __future__ import annotations

MIN_HEIGHT = 140         # px; shorter slices merge into the section below (headings stay with their content)
TOP_MIN_HEIGHT = 60      # the first slice (site header) may be shorter: it's a module of its own
MAX_HEIGHT = 2600        # px; taller slices are split at their widest internal gap
SPLIT_GAP_MIN = 28       # px of uniform rows that may serve as a split point inside an oversized slice
SIDEBAR_SHARE = 0.5      # share of rows where one edge is chrome-coloured and the other is page background
MAX_MODULES = 15
EDGE_SAMPLES = 6         # samples per side used as the margin strip
BAND_DELTA = 28          # colour distance that counts as a background change
SOLID_DELTA = 6          # smaller change that counts when both rows are solid colour
REPEAT_MIN = 3           # this many similar consecutive bands merge into one "repeating" module
REPEAT_TOLERANCE = 0.15  # height difference allowed between repeating bands
BAND_HOLD = 12           # rows the new background must hold for
UNIFORM_SPREAD = 10      # max channel range within a row to call it uniform
GAP_MIN = 56             # px of uniform rows that count as a section gap


def _dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def _mean(px):
    n = len(px)
    return (sum(p[0] for p in px) / n, sum(p[1] for p in px) / n, sum(p[2] for p in px) / n)


def _spread(px):
    return max(max(p[c] for p in px) - min(p[c] for p in px) for c in range(3))


def _trim_sidebar(rows):
    """Drops a fixed sidebar menu (Murdoch Clarke) so the margins reflect the page, not the chrome.
    Detects an edge whose strip differs from the opposite edge on most rows while staying one colour."""
    if not rows or len(rows[0]) < 40:
        return rows
    w = len(rows[0])
    for side in ("left", "right"):
        strip = (lambda r: r[:EDGE_SAMPLES]) if side == "left" else (lambda r: r[-EDGE_SAMPLES:])
        other = (lambda r: r[-EDGE_SAMPLES:]) if side == "left" else (lambda r: r[:EDGE_SAMPLES])
        differs = [r for r in rows if _dist(_mean(strip(r)), _mean(other(r))) > 40]
        if len(differs) < SIDEBAR_SHARE * len(rows):
            continue
        ref = _mean(strip(differs[0]))
        if sum(_dist(_mean(strip(r)), ref) < 12 for r in differs) < 0.9 * len(differs):
            continue
        # Width: walk inwards while the column stays chrome-coloured on the differing rows (up to a third of the page).
        width = EDGE_SAMPLES
        while width < w // 3:
            col = [r[width] if side == "left" else r[w - 1 - width] for r in differs[:: max(1, len(differs) // 200)]]
            if sum(_dist(c, ref) < 20 for c in col) < 0.8 * len(col):
                break
            width += 1
        width += 2
        rows = [r[width:] for r in rows] if side == "left" else [r[: w - width] for r in rows]
        w = len(rows[0])
    return rows


def signatures(rows):
    sig = []
    for px in rows:
        left = _mean(px[:EDGE_SAMPLES])
        right = _mean(px[-EDGE_SAMPLES:])
        spread = _spread(px)
        sig.append({"left": left, "right": right, "uniform": spread <= UNIFORM_SPREAD, "mean": _mean(px), "spread": spread})
    return sig


def _band_cuts(sig):
    """Rows where both margins change colour and the new colour holds. Strength = colour distance."""
    cuts = []
    h = len(sig)
    y = 1
    while y < h - BAND_HOLD:
        dl = _dist(sig[y]["left"], sig[y - 1]["left"])
        dr = _dist(sig[y]["right"], sig[y - 1]["right"])
        # Solid band to a different solid band: tints like #F4F8FC -> #FFFFFF are faint but always a section edge.
        solid = sig[y]["uniform"] and sig[y - 1]["uniform"] and _dist(sig[y]["mean"], sig[y - 1]["mean"]) >= SOLID_DELTA
        if solid:
            held = all(sig[y + k]["uniform"] is False or _dist(sig[y + k]["mean"], sig[y]["mean"]) < SOLID_DELTA for k in range(1, BAND_HOLD))
            if held:
                cuts.append((y, 1000 + BAND_DELTA))
                y += BAND_HOLD
                continue
        if min(dl, dr) >= BAND_DELTA:
            held = all(
                _dist(sig[y + k]["left"], sig[y]["left"]) < BAND_DELTA and _dist(sig[y + k]["right"], sig[y]["right"]) < BAND_DELTA
                for k in range(1, BAND_HOLD)
            )
            if held:
                cuts.append((y, 1000 + min(dl, dr)))
                y += BAND_HOLD
                continue
        y += 1
    return cuts


def _gap_cuts(sig):
    """Middles of long runs of uniform rows of one colour. Strength = run length."""
    cuts = []
    h = len(sig)
    y = 0
    while y < h:
        if not sig[y]["uniform"]:
            y += 1
            continue
        start = y
        while y + 1 < h and sig[y + 1]["uniform"] and _dist(sig[y + 1]["mean"], sig[start]["mean"]) < 8:
            y += 1
        run = y - start + 1
        if run >= GAP_MIN and start > 0 and y < h - 1:
            cuts.append(((start + y + 1) // 2, run))
        y += 1
    return cuts


def _merge_repeats(cuts, sig, h):
    """Collapse runs of similar-height bands whose backgrounds alternate (A, B, A...) into one module,
    e.g. a stack of feature rows on alternating tints. Designers read these as one repeating module."""
    edges = [0] + cuts + [h]
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bg = [sig[min(a + 4, b - 1)]["left"] for a, b in segs]

    def similar(i, j):
        hi, hj = segs[i][1] - segs[i][0], segs[j][1] - segs[j][0]
        return abs(hi - hj) <= REPEAT_TOLERANCE * max(hi, hj)

    drop = set()
    i = 0
    while i < len(segs):
        j = i
        while j + 1 < len(segs) and similar(j, j + 1) and (j == i or _dist(bg[j + 1], bg[j - 1]) < BAND_DELTA):
            j += 1
        if j - i + 1 >= REPEAT_MIN:
            drop.update(segs[k][1] for k in range(i, j))
        i = j + 1
    return [c for c in cuts if c not in drop]


def _quietest(sig, lo, hi, centre, window=24):
    """Row (centre of a `window`-row band) with the least colour variation between lo and hi, favouring the centre."""
    if hi - lo < window * 2:
        return None
    best, best_score = None, None
    for y in range(lo, hi - window, 4):
        busy = sum(sig[k]["spread"] for k in range(y, y + window, 4)) / (window // 4)
        score = busy + abs(y + window // 2 - centre) / 50
        if best_score is None or score < best_score:
            best, best_score = y + window // 2, score
    return best


def _split_tall(cuts, sig, h, cands):
    """Splits slices taller than MAX_HEIGHT at their widest internal whitespace run (pilot: 17,000px news lists)."""
    changed = True
    while changed:
        changed = False
        edges = [0] + cuts + [h]
        for a, b in zip(edges, edges[1:]):
            if b - a <= MAX_HEIGHT:
                continue
            best, best_run, y = None, 0, a + MIN_HEIGHT
            while y < b - MIN_HEIGHT:
                if not sig[y]["uniform"]:
                    y += 1
                    continue
                start = y
                while y + 1 < b and sig[y + 1]["uniform"] and _dist(sig[y + 1]["mean"], sig[start]["mean"]) < 8:
                    y += 1
                run = y - start + 1
                mid = (start + y + 1) // 2
                # Prefer wide gaps, then gaps near the middle of the slice.
                score = run - abs(mid - (a + b) / 2) / 200
                if run >= SPLIT_GAP_MIN and mid - a >= MIN_HEIGHT and b - mid >= MIN_HEIGHT and (best is None or score > best_run):
                    best, best_run = mid, score
                y += 1
            if best is None:
                # No blank gap (a long list of cards or articles): split at the quietest band of rows near the middle,
                # which usually falls between two items.
                best = _quietest(sig, a + MIN_HEIGHT, b - MIN_HEIGHT, (a + b) // 2)
                best_run = 1
            if best is not None:
                cuts.append(best)
                cands[best] = max(1, int(best_run))
                cuts.sort()
                changed = True
                break
    return cuts


def find_boundaries(rows):
    """Returns cut rows (exclusive y of each module but the last), sorted ascending."""
    sig = signatures(_trim_sidebar(rows))
    h = len(sig)
    cands = {}
    for y, s in _band_cuts(sig) + _gap_cuts(sig):
        # A gap cut close to a band cut is the same boundary; keep the stronger one.
        near = [k for k in cands if abs(k - y) < MIN_HEIGHT // 2]
        if near:
            k = near[0]
            if s > cands[k]:
                del cands[k]
                cands[y] = s
        else:
            cands[y] = s

    cuts = sorted(cands)

    def segments(cs):
        edges = [0] + cs + [h]
        return [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

    # Merge slivers into the section below (a heading belongs with what it introduces). The header bar at the top
    # may be shorter; the last slice merges upwards because there's nothing below it.
    changed = True
    while changed and cuts:
        changed = False
        segs = segments(cuts)
        for i, seg in enumerate(segs):
            limit = TOP_MIN_HEIGHT if i == 0 else MIN_HEIGHT
            if seg < limit:
                cuts.remove(cuts[i] if i < len(cuts) else cuts[i - 1])
                changed = True
                break

    cuts = _merge_repeats(cuts, sig, h)
    cuts = _split_tall(cuts, sig, h, cands)

    while len(cuts) + 1 > MAX_MODULES:
        cuts.remove(min(cuts, key=lambda c: cands[c]))

    return cuts
