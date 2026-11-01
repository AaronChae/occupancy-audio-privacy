#!/usr/bin/env python3
"""Download files of a Zenodo record and check their MD5.

The record's license id is saved next to the files as license.txt.

Example:
    python3 scripts/download_zenodo.py 1227121 --dest data/raw/demand \
        --files OOFFICE_16k.zip OMEETING_16k.zip DEMAND.pdf
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--files", nargs="*", help="only these file names (default: all files)")
    args = ap.parse_args()
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{args.record}", timeout=60) as r:
        meta = json.load(r)
    args.dest.mkdir(parents=True, exist_ok=True)
    license_id = (meta["metadata"].get("license") or {}).get("id", "not stated")
    (args.dest / "license.txt").write_text(f"zenodo record {args.record}: {license_id}\n")
    for f in meta["files"]:
        if args.files and f["key"] not in args.files:
            continue
        path, want = args.dest / f["key"], f["checksum"].split(":", 1)[1]
        if path.exists() and md5(path) == want:
            print(f"Already downloaded: {f['key']}", flush=True)
            continue
        with urllib.request.urlopen(f["links"]["self"], timeout=120) as r, open(path, "wb") as w:
            shutil.copyfileobj(r, w, 1 << 20)
        status = "Done" if md5(path) == want else "MD5 mismatch"
        print(f"{status}: {f['key']} ({f['size'] // 1000000} MB)", flush=True)


if __name__ == "__main__":
    main()
