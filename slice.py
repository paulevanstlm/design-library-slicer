"""Slice full-page screenshots into modules. Runs in GitHub Actions (see .github/workflows/slice.yml).

Flow: ask the Worker for pending slice jobs, download each screenshot, find module boundaries,
then post the slices, thumbnails and per-module colours back. All calls use PIPELINE_SHARED_SECRET.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from boundaries import find_boundaries

Image.MAX_IMAGE_PIXELS = None  # full-page screenshots are legitimately huge

SAMPLE_STEP = 4
THUMB_WIDTH = 900


def call(method, path, body=None, content_type=None):
    worker = os.environ["WORKER_URL"].rstrip("/")
    req = urllib.request.Request(f"{worker}{path}", method=method, data=body)
    req.add_header("authorization", f"Bearer {os.environ['PIPELINE_SHARED_SECRET']}")
    # Cloudflare's browser integrity check rejects urllib's default user agent (error 1010).
    req.add_header("user-agent", "design-library-slicer/1.0 (+https://github.com/paulevanstlm/design-library-slicer)")
    if content_type:
        req.add_header("content-type", content_type)
    with urllib.request.urlopen(req, timeout=120) as res:
        return res.read()


def sampled_rows(img: Image.Image):
    arr = np.asarray(img.convert("RGB"))[:, ::SAMPLE_STEP, :]
    return [[tuple(int(c) for c in px) for px in row] for row in arr]


def dominant_colours(img: Image.Image, n=5):
    small = img.convert("RGB")
    small.thumbnail((240, 240))
    q = small.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(c for c, _ in counts)
    out = []
    for count, idx in counts:
        r, g, b = pal[idx * 3: idx * 3 + 3]
        out.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "share": round(count / total, 3)})
    return out


def dhash(img: Image.Image) -> str:
    """64-bit difference hash: the same header or footer on two pages hashes to (nearly) the same value."""
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def encode(img: Image.Image, fmt: str, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def thumb(img: Image.Image):
    t = img.convert("RGB")
    if t.width > THUMB_WIDTH:
        t = t.resize((THUMB_WIDTH, round(t.height * THUMB_WIDTH / t.width)), Image.LANCZOS)
    return encode(t, "JPEG", quality=80, optimize=True, progressive=True)


class BlankCapture(Exception):
    """The page hadn't rendered when it was shot (pilot: Coote). The Worker re-captures it with more time."""


def is_blank(rows):
    """Mostly empty near-white rows on a page taller than one screen."""
    if len(rows) < 1200:
        return False
    step = max(1, len(rows) // 400)
    sample = rows[::step]
    empty = 0
    for r in sample:
        lo = min(min(p) for p in r)
        if lo >= 235:
            empty += 1
    return empty / len(sample) > 0.85


def upload(capture_id, name, data, ctype):
    q = urllib.parse.urlencode({"capture": capture_id, "name": name})
    call("PUT", f"/api/pipeline/slice-upload?{q}", data, ctype)


def process(job):
    raw = call("GET", f"/api/pipeline/captures/{job['captureId']}/image")
    page = Image.open(io.BytesIO(raw))
    page.load()
    # An admin's adjusted boundaries win over detection (brief 4.5).
    manual = [c for c in (job.get("cuts") or []) if 0 < c < page.height]
    rows = sampled_rows(page)
    if not manual and is_blank(rows):
        raise BlankCapture("Screenshot is blank: the page hadn't rendered")
    cuts = sorted(set(manual)) if manual else find_boundaries(rows)
    edges = [0] + cuts + [page.height]

    cid = job["captureId"]
    modules = []
    for i in range(len(edges) - 1):
        y0, y1 = edges[i], edges[i + 1]
        part = page.crop((0, y0, page.width, y1))
        # One request per image keeps the Worker's memory small on long pages.
        upload(cid, f"modules/{i}.png", encode(part, "PNG", optimize=True), "image/png")
        upload(cid, f"thumbs/{i}.jpg", thumb(part), "image/jpeg")
        modules.append({"index": i, "yStart": y0, "yEnd": y1, "colours": dominant_colours(part), "hash": dhash(part)})

    # Page card thumbnail: the first screen (16:9 at full width).
    top = page.crop((0, 0, page.width, min(page.height, round(page.width * 9 / 16))))
    upload(cid, "thumbs/page.jpg", thumb(top), "image/jpeg")

    payload = {"jobId": job["jobId"], "captureId": cid, "width": page.width, "height": page.height, "modules": modules}
    call("POST", "/api/pipeline/slice-result", json.dumps(payload).encode(), "application/json")
    return len(modules)


def report_error(job, message, blank=False):
    try:
        body = json.dumps({"jobId": job["jobId"], "error": message[:500], "blank": blank}).encode()
        call("POST", "/api/pipeline/slice-error", body, "application/json")
    except Exception as e:
        print(f"  could not report failure: {e}", file=sys.stderr)


def backfill_hashes():
    """Fingerprints modules sliced before hashing existed, without re-slicing (which would drop their tags)."""
    todo = json.loads(call("GET", "/api/pipeline/hash-backfill"))
    if not todo:
        return
    out = []
    for m in todo:
        try:
            img = Image.open(io.BytesIO(call("GET", f"/api/pipeline/modules/{urllib.parse.quote(m['id'])}/image")))
            out.append({"id": m["id"], "hash": dhash(img)})
        except Exception as e:
            print(f"  hash {m['id']}: {e}", file=sys.stderr)
    call("POST", "/api/pipeline/hash-backfill", json.dumps({"hashes": out}).encode(), "application/json")
    print(f"fingerprinted {len(out)} existing module(s)")


def main():
    try:
        backfill_hashes()
    except Exception as e:
        print(f"hash backfill failed: {e}", file=sys.stderr)
    # Keep taking work until the queue is empty or the run is near its 30-minute timeout.
    deadline = time.time() + 25 * 60
    failed = 0
    total = 0
    while time.time() < deadline:
        jobs = json.loads(call("GET", "/api/pipeline/slice-jobs"))
        if not jobs:
            break
        total += len(jobs)
        print(f"{len(jobs)} capture(s) to slice")
        failed += slice_all(jobs)
    print(f"done: {total} capture(s) this run")
    if failed:
        sys.exit(1)


# Pages slice in parallel: most of each page's time is waiting on downloads and uploads, not pixel work.
PARALLEL = 4


def slice_one(job):
    """Slices one capture; returns 1 when it failed, 0 otherwise."""
    try:
        n = process(job)
        print(f"  {job['captureId']}: {n} modules")
    except BlankCapture as e:
        print(f"  {job['captureId']}: blank, asking for a slow re-capture")
        report_error(job, str(e), blank=True)
    except urllib.error.HTTPError as e:
        if e.code == 409:  # stale: the page was re-captured meanwhile; the new capture has its own job
            print(f"  {job['captureId']}: skipped (stale)")
            return 0
        print(f"  {job['captureId']}: FAILED HTTP {e.code}", file=sys.stderr)
        report_error(job, f"HTTP {e.code} from the Worker")
        return 1
    except Exception as e:  # report and carry on so one bad page doesn't block the batch
        print(f"  {job['captureId']}: FAILED {e}", file=sys.stderr)
        report_error(job, str(e))
        return 1
    return 0


def slice_all(jobs):
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        return sum(pool.map(slice_one, jobs))


if __name__ == "__main__":
    main()
