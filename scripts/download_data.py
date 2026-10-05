"""Download the first trajectories of each split of DeepMind's ``cylinder_flow`` dataset.

    python scripts/download_data.py --train 60 --valid 10 --test 20
    python scripts/download_data.py --train all --valid all --test all   # 16.4 GB

The splits are single TFRecord files with one record per trajectory (about 13.6 MB
each). The upstream dataset class reads the first ``num_samples`` records of a file,
so a file cut after N complete records is a valid input for ``num_samples <= N``.
This script streams a file and stops at a record boundary; it does not reorder,
decode or modify the records. ``manifest.json`` records what was written.
"""

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

from _bootstrap import ROOT

BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/"
# Sizes of the complete files, from the Content-Length headers on 2026-10-05.
FULL_BYTES = {"train": 13_645_805_387, "valid": 1_363_987_289, "test": 1_355_376_404}
FULL_RECORDS = {"train": 1000, "valid": 100, "test": 100}


def _read_exact(stream, n: int) -> bytes:
    chunks, remaining = [], n
    while remaining:
        chunk = stream.read(min(remaining, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def copy_records(stream, out, max_records: int | None) -> tuple[int, int, str]:
    """Copy whole TFRecord records from ``stream`` to ``out``.

    A record is: uint64 length, uint32 CRC of the length, payload, uint32 CRC of
    the payload. Returns (records, bytes, sha256 of the bytes written).
    """
    digest = hashlib.sha256()
    records = written = 0
    while max_records is None or records < max_records:
        header = _read_exact(stream, 12)
        if not header:
            break
        if len(header) != 12:
            raise IOError("stream ended inside a record header")
        (length,) = struct.unpack("<Q", header[:8])
        body = _read_exact(stream, length + 4)
        if len(body) != length + 4:
            raise IOError("stream ended inside a record")
        for part in (header, body):
            out.write(part)
            digest.update(part)
        records += 1
        written += 12 + length + 4
    return records, written, digest.hexdigest()


def download_split(split: str, count: int | None, data_dir: Path) -> dict:
    url = BASE_URL + f"{split}.tfrecord"
    path = data_dir / f"{split}.tfrecord"
    tmp = path.with_suffix(".tfrecord.part")
    with urllib.request.urlopen(url) as stream, open(tmp, "wb") as out:
        records, written, sha = copy_records(stream, out, count)
    tmp.replace(path)
    return {
        "url": url,
        "records": records,
        "bytes": written,
        "sha256": sha,
        "complete_file": written == FULL_BYTES[split],
        "complete_file_bytes": FULL_BYTES[split],
        "complete_file_records": FULL_RECORDS[split],
    }


def _count(value: str) -> int | None:
    return None if value == "all" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "cylinder_flow"))
    for split in ("train", "valid", "test"):
        parser.add_argument(f"--{split}", type=_count, default=0, help="number of trajectories, or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="print the expected size and exit")
    args = parser.parse_args()

    requested = {s: getattr(args, s) for s in ("train", "valid", "test")}
    expected = sum(
        FULL_BYTES[s] if n is None else FULL_BYTES[s] / FULL_RECORDS[s] * n for s, n in requested.items()
    )
    print(f"Expected download: {expected / 1e9:.2f} GB into {args.data_dir}", flush=True)
    if args.dry_run:
        return

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": BASE_URL, "splits": {}}

    with urllib.request.urlopen(BASE_URL + "meta.json") as stream:
        (data_dir / "meta.json").write_bytes(stream.read())

    for split, count in requested.items():
        if count == 0:
            continue
        have = manifest["splits"].get(split)
        if have and (path := data_dir / f"{split}.tfrecord").exists() and path.stat().st_size == have["bytes"]:
            if have["complete_file"] or (count is not None and have["records"] >= count):
                print(f"{split}: {have['records']} trajectories already present", flush=True)
                continue
        entry = download_split(split, count, data_dir)
        manifest["splits"][split] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{split}: {entry['records']} trajectories, {entry['bytes'] / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
