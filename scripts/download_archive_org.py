#!/usr/bin/env python3
"""Download the original files of archive.org items and check their MD5.

The item's license URL is saved next to the files as license.txt.

Example:
    python3 scripts/download_archive_org.py --dest data/raw/dcase2016_task2 \
        dcase2016_task2_train_dev dcase2016_task2_test_public
"""
import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

SKIP = ("_meta.xml", "_files.xml", "_meta.sqlite", "_archive.torrent", "__ia_thumb.jpg")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+", help="archive.org item identifiers")
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    for item in args.items:
        with urllib.request.urlopen(f"https://archive.org/metadata/{item}", timeout=60) as r:
            meta = json.load(r)
        out = args.dest / item
        out.mkdir(parents=True, exist_ok=True)
        (out / "license.txt").write_text(meta["metadata"].get("licenseurl", "not stated") + "\n")
        for f in meta["files"]:
            name = f["name"]
            if f.get("source") != "original" or name.endswith(SKIP):
                continue
            path = out / name
            if path.exists() and md5(path) == f["md5"]:
                print(f"Already downloaded: {item}/{name}", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://archive.org/download/{item}/{urllib.parse.quote(name)}"
            with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as w:
                shutil.copyfileobj(r, w, 1 << 20)
            status = "Done" if md5(path) == f["md5"] else "MD5 mismatch"
            print(f"{status}: {item}/{name} ({int(f['size']) // 1000000} MB)", flush=True)


if __name__ == "__main__":
    main()
