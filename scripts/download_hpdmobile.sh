#!/usr/bin/env bash
# Download the HPDmobile residential occupancy dataset (figshare, CC0):
# audio, ground truth, environmental data and metadata for homes H1-H6.
# Image data is skipped because the first study uses audio only.
#
# Usage:   bash scripts/download_hpdmobile.sh [DEST_DIR]
# Default: data/raw/hpdmobile
# Re-running skips files whose MD5 already matches and resumes partial files.
set -euo pipefail

DEST="${1:-data/raw/hpdmobile}"
mkdir -p "$DEST"
cd "$DEST"

# figshare article IDs (collection 5364449)
ARTICLES=(
  14368802                    # metadata
  14368472 14368460 14368457  # H1: audio, ground truth, environmental
  14368529 14368541 14368538  # H2
  14368565 14368574 14368580  # H3
  14368622 14368613 14368619  # H4
  14368661 14368658 14368670  # H5
  14368694 14368706 14368703  # H6
)

if command -v md5sum >/dev/null; then
  md5_of() { md5sum "$1" | cut -d' ' -f1; }
else
  md5_of() { md5 -q "$1"; }
fi

# About 13.6 GB in total; require 20 GB free.
avail_kb=$(df -Pk . | awk 'NR==2 {print $4}')
if [ "$avail_kb" -lt $((20 * 1024 * 1024)) ]; then
  echo "Not enough free space: $((avail_kb / 1024 / 1024)) GB available, 20 GB needed." >&2
  exit 1
fi

# File list: name, size, md5, download URL
python3 - "${ARTICLES[@]}" > files.tsv <<'PY'
import json, sys, urllib.request
for aid in sys.argv[1:]:
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{aid}/files") as r:
        for f in json.load(r):
            md5 = f.get("supplied_md5") or f.get("computed_md5")
            print(f["name"], f["size"], md5, f["download_url"], sep="\t")
PY

echo "Files to download: $(wc -l < files.tsv | tr -d ' ')"

while IFS=$'\t' read -r name size sum url; do
  if [ -f "$name" ] && [ "$(md5_of "$name")" = "$sum" ]; then
    echo "Already downloaded: $name"
    continue
  fi
  echo "Downloading: $name ($((size / 1000000)) MB)"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$name" "$url"
  if [ "$(md5_of "$name")" != "$sum" ]; then
    echo "MD5 mismatch: $name (re-run the script to retry)" >&2
    exit 1
  fi
done < files.tsv

echo "Done: $(pwd)"
