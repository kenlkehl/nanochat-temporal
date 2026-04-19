"""
scripts/build_pretrain_corpus.py

Build the pre-1985 contamination-controlled pretraining corpus as parquet shards
in the same format as ClimbMix (single 'text' column, ~250 MB per shard, zstd
compression, row group size 1024). Output goes to nanochat/dataset_pre1985.DATA_DIR
by default.

Sources (mixed by weight, configurable):
  - books_ia:        Pre-1929 books from storytracer/internet_archive_books_en
  - books_loc:       Pre-1929 books from storytracer/LoC-PD-Books
  - books_gutenberg: Pre-1929 books from sedthh/gutenberg_english
  - pubmed:          Pre-1985 MEDLINE/PubMed abstracts from NLM baseline FTP

Idempotent: skips shards already on disk; resumable.
Streaming: low memory footprint.

Example usage:

    # Default mix: 70% books / 30% PubMed, target ~30B tokens
    python -m scripts.build_pretrain_corpus

    # Custom mix and smaller target (e.g. for a d12 model)
    python -m scripts.build_pretrain_corpus \\
        --target-tokens 5_000_000_000 \\
        --mix "books_ia:0.6,books_gutenberg:0.1,pubmed:0.3"

    # Quick smoke test — cap each stream at 200 docs
    python -m scripts.build_pretrain_corpus --max-docs-per-stream 200 --target-tokens 0
"""

from __future__ import annotations

import os
import re
import gzip
import time
import json
import random
import argparse
import urllib.request
import urllib.error
from typing import Iterator, Optional, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset_pre1985 import DATA_DIR, index_to_filename

CUTOFF_BOOK_YEAR = 1929    # US public-domain
CUTOFF_SCIENCE_YEAR = 1985  # contamination cutoff

# ---------------------------------------------------------------------------
# OCR cleanup (light touch — preserve archaic style)
# ---------------------------------------------------------------------------

LIGATURE_MAP = str.maketrans({
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "ſ": "s",  # long-s in older books
})


def clean_ocr_light(text: Optional[str]) -> Optional[str]:
    """Light-touch OCR cleanup. Returns None if text is too low-quality."""
    if not text or len(text) < 1000:
        return None
    text = text.translate(LIGATURE_MAP)
    text = re.sub(r"\n{3,}", "\n\n", text)
    out_lines = []
    for line in text.split("\n"):
        if len(line) > 20:
            alpha_frac = sum(1 for c in line if c.isalpha()) / len(line)
            if alpha_frac < 0.30:
                continue
        out_lines.append(line)
    text = "\n".join(out_lines)
    if len(text) < 1000:
        return None
    words = text.split()
    if len(words) < 200:
        return None
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.15:
        return None
    return text


# ---------------------------------------------------------------------------
# Date parsing (HF metadata is a mess; be defensive)
# ---------------------------------------------------------------------------

_YEAR_REGEX = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def parse_year(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 2100 else None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    m = _YEAR_REGEX.search(s)
    if m:
        y = int(m.group(1))
        return y if 1000 <= y <= 2100 else None
    return None


def _safe_get(d, *keys):
    """Try multiple keys, returning the first non-None value."""
    for k in keys:
        if d is None:
            return None
        v = d.get(k) if isinstance(d, dict) else None
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Source streams
# ---------------------------------------------------------------------------

def _try_load_streaming(name: str, **kwargs):
    """Try to load an HF dataset in streaming mode; return None on failure."""
    try:
        from datasets import load_dataset
        return load_dataset(name, split="train", streaming=True, **kwargs)
    except Exception as e:
        print(f"[warn] could not load HF dataset {name!r}: {type(e).__name__}: {e}")
        return None


def stream_ia_books(max_docs: Optional[int] = None) -> Iterator[str]:
    ds = _try_load_streaming("storytracer/internet_archive_books_en")
    if ds is None:
        return
    n = 0
    for ex in ds:
        meta = ex.get("metadata") if isinstance(ex.get("metadata"), dict) else {}
        year = (
            parse_year(ex.get("year"))
            or parse_year(ex.get("date"))
            or parse_year(ex.get("publish_year"))
            or parse_year(_safe_get(meta, "date"))
            or parse_year(_safe_get(meta, "year"))
            or parse_year(_safe_get(meta, "publish_year"))
        )
        if year is None or year >= CUTOFF_BOOK_YEAR:
            continue
        text = ex.get("text") or ex.get("ocr") or ex.get("content")
        cleaned = clean_ocr_light(text)
        if cleaned is None:
            continue
        yield cleaned
        n += 1
        if max_docs and n >= max_docs:
            return


def stream_loc_books(max_docs: Optional[int] = None) -> Iterator[str]:
    ds = _try_load_streaming("storytracer/LoC-PD-Books")
    if ds is None:
        return
    n = 0
    for ex in ds:
        meta = ex.get("metadata") if isinstance(ex.get("metadata"), dict) else {}
        year = (
            parse_year(ex.get("year"))
            or parse_year(ex.get("date"))
            or parse_year(_safe_get(meta, "date"))
            or parse_year(_safe_get(meta, "year"))
        )
        if year is None or year >= CUTOFF_BOOK_YEAR:
            continue
        text = ex.get("text") or ex.get("ocr")
        cleaned = clean_ocr_light(text)
        if cleaned is None:
            continue
        yield cleaned
        n += 1
        if max_docs and n >= max_docs:
            return


def stream_gutenberg(max_docs: Optional[int] = None) -> Iterator[str]:
    ds = _try_load_streaming("sedthh/gutenberg_english")
    if ds is None:
        return
    n = 0
    for ex in ds:
        meta = ex.get("METADATA") or ex.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        year = (
            parse_year(_safe_get(meta, "publication_year"))
            or parse_year(_safe_get(meta, "date"))
            or parse_year(_safe_get(meta, "year"))
            or parse_year(ex.get("year"))
            or parse_year(ex.get("publication_year"))
        )
        if year is None or year >= CUTOFF_BOOK_YEAR:
            continue
        text = ex.get("TEXT") or ex.get("text")
        cleaned = clean_ocr_light(text)
        if cleaned is None:
            continue
        yield cleaned
        n += 1
        if max_docs and n >= max_docs:
            return


# ---------------------------------------------------------------------------
# PubMed (NLM baseline FTP, gzipped XML)
# ---------------------------------------------------------------------------

PUBMED_BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
_PUBMED_FILE_REGEX = re.compile(r'(pubmed\d+n\d+\.xml\.gz)')


def list_pubmed_baseline_files(cache_dir: str) -> list[str]:
    """List all .xml.gz files in the latest PubMed baseline. Returns full URLs."""
    os.makedirs(cache_dir, exist_ok=True)
    listing_path = os.path.join(cache_dir, "_listing.html")
    if not os.path.exists(listing_path):
        try:
            with urllib.request.urlopen(PUBMED_BASELINE_URL, timeout=60) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            with open(listing_path, "w", encoding="utf-8") as f:
                f.write(content)
        except urllib.error.URLError as e:
            print(f"[warn] could not list PubMed baseline: {e}")
            return []
    else:
        with open(listing_path, "r", encoding="utf-8") as f:
            content = f.read()
    files = sorted(set(_PUBMED_FILE_REGEX.findall(content)))
    return [PUBMED_BASELINE_URL + f for f in files]


def download_pubmed_file(url: str, cache_dir: str, max_attempts: int = 5) -> Optional[str]:
    fname = url.rsplit("/", 1)[-1]
    out_path = os.path.join(cache_dir, fname)
    if os.path.exists(out_path):
        return out_path
    tmp_path = out_path + ".tmp"
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  downloading {fname} (attempt {attempt})...")
            with urllib.request.urlopen(url, timeout=120) as resp:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            os.rename(tmp_path, out_path)
            return out_path
        except (urllib.error.URLError, OSError) as e:
            print(f"  attempt {attempt} failed: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt == max_attempts:
                return None
            time.sleep(2 ** attempt)
    return None


def iter_pubmed_xml(path: str) -> Iterator[tuple[Optional[int], str, str]]:
    """Yield (year, title, abstract) for each PubmedArticle in one .xml.gz file."""
    from lxml import etree
    with gzip.open(path, "rb") as f:
        ctx = etree.iterparse(f, events=("end",), tag="PubmedArticle")
        for _, article in ctx:
            year = None
            year_el = article.find(".//PubDate/Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    year = None
            if year is None:
                med_el = article.find(".//PubDate/MedlineDate")
                if med_el is not None and med_el.text:
                    year = parse_year(med_el.text)
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abstract_parts = []
            for ab in article.findall(".//AbstractText"):
                txt = "".join(ab.itertext()).strip()
                if txt:
                    label = ab.get("Label")
                    abstract_parts.append(f"{label}: {txt}" if label else txt)
            abstract = "\n".join(abstract_parts).strip()
            yield year, title, abstract
            article.clear()
            while article.getprevious() is not None:
                del article.getparent()[0]


def stream_pubmed_pre1985(cache_dir: str, max_docs: Optional[int] = None) -> Iterator[str]:
    files = list_pubmed_baseline_files(cache_dir)
    if not files:
        print("[warn] no PubMed baseline files; skipping pubmed stream.")
        return
    n = 0
    for url in files:
        local = download_pubmed_file(url, cache_dir)
        if local is None:
            continue
        try:
            for year, title, abstract in iter_pubmed_xml(local):
                if year is None or year >= CUTOFF_SCIENCE_YEAR:
                    continue
                if abstract:
                    text = f"{title}\n\n{abstract}".strip()
                else:
                    text = title.strip()
                if len(text) < 50:
                    continue
                yield text
                n += 1
                if max_docs and n >= max_docs:
                    return
        except Exception as e:
            print(f"[warn] error parsing {local}: {e}")
            continue


# ---------------------------------------------------------------------------
# Round-robin scheduler (weighted, fair)
# ---------------------------------------------------------------------------

def round_robin_weighted(streams: dict[str, Iterator[str]], weights: dict[str, float]) -> Iterator[str]:
    """Yield documents from streams keeping the proportion close to `weights`.
    Stops when all streams are exhausted."""
    keys = list(streams.keys())
    iters = {k: iter(streams[k]) for k in keys}
    counts = {k: 0 for k in keys}
    exhausted: set[str] = set()
    while len(exhausted) < len(keys):
        live = [k for k in keys if k not in exhausted]
        if not live:
            break
        # Pick the live key with the smallest count/weight ratio
        live.sort(key=lambda k: counts[k] / max(weights.get(k, 1e-9), 1e-9))
        for k in live:
            try:
                doc = next(iters[k])
            except StopIteration:
                exhausted.add(k)
                continue
            counts[k] += 1
            yield doc
            break
        else:
            # fell through without yielding — all live streams exhausted this round
            break


# ---------------------------------------------------------------------------
# Shard writer (idempotent, ClimbMix-compatible format)
# ---------------------------------------------------------------------------

CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024


def existing_shard_count(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    existing = sorted(
        f for f in os.listdir(out_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    if not existing:
        return 0
    m = re.search(r"shard_(\d+)\.parquet", existing[-1])
    if m is None:
        return 0
    return int(m.group(1)) + 1


def write_shards(doc_iter: Iterable[str], out_dir: str, target_chars: Optional[int] = None,
                 chars_per_shard: int = CHARS_PER_SHARD, row_group_size: int = ROW_GROUP_SIZE) -> int:
    """Consume documents and write parquet shards. Returns total chars written.
    Stops when target_chars is reached (if given). Resumable across runs."""
    os.makedirs(out_dir, exist_ok=True)
    shard_index = existing_shard_count(out_dir)
    if shard_index > 0:
        print(f"[resume] {shard_index} shards already exist, continuing from shard_{shard_index:05d}")

    shard_docs: list[str] = []
    shard_chars = 0
    total_chars_written = 0
    t0 = time.time()
    docs_seen = 0

    def flush():
        nonlocal shard_index, shard_docs, shard_chars, total_chars_written, t0
        if not shard_docs:
            return
        shard_path = os.path.join(out_dir, index_to_filename(shard_index))
        tmp_path = shard_path + ".tmp"
        tbl = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            tbl, tmp_path,
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        os.rename(tmp_path, shard_path)
        t1 = time.time()
        print(f"  wrote {shard_path} | {len(shard_docs):,} docs | {shard_chars/1e6:.1f}M chars | {t1-t0:.1f}s")
        total_chars_written += shard_chars
        shard_docs = []
        shard_chars = 0
        shard_index += 1
        t0 = t1

    for doc in doc_iter:
        shard_docs.append(doc)
        shard_chars += len(doc)
        docs_seen += 1
        if shard_chars >= chars_per_shard and len(shard_docs) % row_group_size == 0:
            flush()
            if target_chars is not None and total_chars_written >= target_chars:
                print(f"[done] reached target {target_chars/1e9:.2f}B chars")
                return total_chars_written

    # final partial shard
    flush()
    return total_chars_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_mix(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in (x.strip() for x in s.split(",") if x.strip()):
        k, v = p.split(":")
        out[k.strip()] = float(v.strip())
    return out


STREAM_FACTORIES = {
    "books_ia": lambda max_docs, **_: stream_ia_books(max_docs),
    "books_loc": lambda max_docs, **_: stream_loc_books(max_docs),
    "books_gutenberg": lambda max_docs, **_: stream_gutenberg(max_docs),
    "pubmed": lambda max_docs, pubmed_cache, **_: stream_pubmed_pre1985(pubmed_cache, max_docs),
}


def main():
    parser = argparse.ArgumentParser(description="Build pre-1985 contamination-controlled pretraining corpus.")
    parser.add_argument(
        "--target-tokens", type=int, default=30_000_000_000,
        help="Approximate target total tokens (assumes ~4 chars/token). 0 = no cap.",
    )
    parser.add_argument(
        "--mix", type=str,
        default="books_ia:0.55,books_loc:0.10,books_gutenberg:0.05,pubmed:0.30",
        help="comma-separated stream:weight pairs. Available streams: "
             + ",".join(STREAM_FACTORIES.keys()),
    )
    parser.add_argument("--out-dir", type=str, default=DATA_DIR,
                        help=f"Output directory for parquet shards (default: {DATA_DIR}).")
    parser.add_argument("--pubmed-cache-dir", type=str, default=None,
                        help="Cache dir for downloaded PubMed XML.gz files (default: <out-dir>/../pubmed_baseline).")
    parser.add_argument("--max-docs-per-stream", type=int, default=None,
                        help="Cap docs per stream (mostly for smoke-testing).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights = parse_mix(args.mix)
    unknown = set(weights) - set(STREAM_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown streams in --mix: {unknown}. Available: {list(STREAM_FACTORIES.keys())}")
    pubmed_cache = args.pubmed_cache_dir or os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "pubmed_baseline")

    print(f"Output dir:        {args.out_dir}")
    print(f"PubMed cache:      {pubmed_cache}")
    print(f"Mix:               {weights}")
    print(f"Target tokens:     {args.target_tokens:,}" if args.target_tokens else "Target tokens:     no cap")
    print(f"Max docs/stream:   {args.max_docs_per_stream or 'no cap'}")

    streams = {
        k: STREAM_FACTORIES[k](max_docs=args.max_docs_per_stream, pubmed_cache=pubmed_cache)
        for k in weights
    }
    doc_iter = round_robin_weighted(streams, weights)
    target_chars = args.target_tokens * 4 if args.target_tokens else None
    written = write_shards(doc_iter, args.out_dir, target_chars=target_chars)
    print(f"\nDone. Total chars written: {written:,} (~{written/4/1e9:.2f}B tokens estimated)")


if __name__ == "__main__":
    main()
