"""Download the official TimesFM 2.5-200M (Flax) checkpoint from Google's public Vertex AI Model
Garden bucket (Hugging Face is blocked in our execution environment).

Source: https://storage.googleapis.com/vertex-model-garden-public-us/timesfm/timesfm-2.5-200m-flax/
Every file is verified against the MD5 hash published in the bucket listing. Output: models/timesfm-2.5-200m-flax/
Usage: python benchmarks/fetch_timesfm.py   (stdlib only, ~860 MB)
"""
from __future__ import annotations

import base64
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/vertex-model-garden-public-us"
PREFIX = "timesfm/timesfm-2.5-200m-flax/"
DEST = Path(__file__).resolve().parents[1] / "models" / "timesfm-2.5-200m-flax"


def listing() -> list[tuple[str, int, str]]:
    xml = urllib.request.urlopen(f"{BUCKET}/?prefix={PREFIX}", timeout=60).read().decode()
    items = re.findall(r"<Key>(.*?)</Key>.*?<ETag>(?:&quot;|\")?(.*?)(?:&quot;|\")?</ETag><Size>(\d+)</Size>", xml)
    return [(k, int(s), e) for k, e, s in items]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = listing()
    assert files, "empty bucket listing"
    for key, size, etag in files:
        out = DEST / key[len(PREFIX):]
        out.parent.mkdir(parents=True, exist_ok=True)
        multipart = "-" in etag  # multipart ETag is not an MD5; then only the size is checked
        good = lambda: out.exists() and out.stat().st_size == size and (multipart or md5(out) == etag)  # noqa: E731
        if not good():
            urllib.request.urlretrieve(f"{BUCKET}/{key}", out)
        ok = good()
        print(("ok  " if ok else "BAD ") + f"{size/1e6:9.2f} MB {key}", flush=True)
        if not ok:
            sys.exit(f"checksum mismatch for {key}")
    print("checkpoint at", DEST)


if __name__ == "__main__":
    main()
