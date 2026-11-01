#!/usr/bin/env bash
# Parallel version of download_hpdmobile.sh: downloads up to JOBS files at once.
# Reads data/raw/hpdmobile/files.tsv (written by download_hpdmobile.sh),
# skips files whose MD5 already matches, and resumes partial files.
# A transfer that stalls (under 10 KB/s for 60 s) is restarted from where it stopped.
#
# Usage: caffeinate -i bash scripts/download_hpdmobile_parallel.sh [DEST_DIR]
#        JOBS=6 bash scripts/download_hpdmobile_parallel.sh   # more parallel downloads
set -euo pipefail

DEST="${1:-data/raw/hpdmobile}"
JOBS="${JOBS:-4}"
cd "$DEST"
[ -f files.tsv ] || { echo "files.tsv not found. Run download_hpdmobile.sh first." >&2; exit 1; }

md5_of() {
  if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

download_one() {
  name=$1; size=$2; sum=$3; url=$4
  if [ -f "$name" ] && [ "$(md5_of "$name")" = "$sum" ]; then
    echo "Already downloaded: $name"
    return 0
  fi
  echo "Start: $name ($((size / 1000000)) MB)"
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [ -f "$name" ] && [ "$(wc -c < "$name" | tr -d ' ')" -ge "$size" ]; then break; fi
    if curl -sSfL --speed-limit 10240 --speed-time 60 -C - -o "$name" "$url"; then break; fi
    echo "Retrying: $name (attempt $attempt)" >&2
    sleep 10
  done
  if [ "$(md5_of "$name")" = "$sum" ]; then
    echo "Done: $name"
  else
    echo "MD5 mismatch: $name" >&2
    return 1
  fi
}
export -f md5_of download_one

if xargs -P "$JOBS" -L 1 bash -c 'download_one "$@"' _ < files.tsv; then
  echo "All files downloaded: $(pwd)"
else
  echo "Some files failed. Re-run this script to retry." >&2
  exit 1
fi
