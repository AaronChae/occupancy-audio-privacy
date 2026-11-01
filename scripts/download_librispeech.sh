#!/usr/bin/env bash
# Download and extract LibriSpeech subsets (openslr resource 12, CC BY 4.0).
#
# Usage: bash scripts/download_librispeech.sh [DEST_DIR] [SUBSET ...]
# Default subsets: dev-clean test-clean train-clean-100
set -euo pipefail

DEST="${1:-data/raw/librispeech}"
[ $# -gt 0 ] && shift
SUBSETS="${*:-dev-clean test-clean train-clean-100}"
mkdir -p "$DEST"
cd "$DEST"

for s in $SUBSETS; do
  if [ -f "$s.extracted" ]; then echo "Already extracted: $s"; continue; fi
  curl -sSfL --retry 5 --retry-delay 10 -C - -o "$s.tar.gz" "https://www.openslr.org/resources/12/$s.tar.gz"
  tar -xzf "$s.tar.gz"
  touch "$s.extracted"
  echo "Done: $s"
done
