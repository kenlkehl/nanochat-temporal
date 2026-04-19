"""
Constants for the pre-1985 contamination-controlled pretraining dataset.

This corpus is built locally by `scripts/build_pretrain_corpus.py` from
pre-1929 public-domain books (Internet Archive, Library of Congress,
Project Gutenberg) plus pre-1985 PubMed/MEDLINE abstracts (NLM baseline).

Unlike the ClimbMix dataset which is downloaded from HuggingFace, this
dataset is built locally and not hosted anywhere, so BASE_URL is None
and download_single_file is a no-op (will exit with a hint).

Output shards live at ~/.cache/nanochat/base_data_pre1985 in the same
parquet format as ClimbMix (single `text` column, ~250MB per shard,
zstd compression, row group size 1024) so dataloader.py and tok_train.py
work without modification.
"""

import os
from nanochat.common import get_base_dir

BASE_URL = None  # Not hosted; built locally
MAX_SHARD = None  # Determined dynamically from disk at training time

def index_to_filename(index):
    return f"shard_{index:05d}.parquet"

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_pre1985")
