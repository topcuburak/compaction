#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    repeats = (length + len(digest) - 1) // len(digest)
    return (digest * repeats)[:length]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    plaintext = bytes(a ^ b for a, b in zip(encrypted, key))
    return plaintext.decode("utf-8")


def download_csv(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        data = response.read()
    dest_path.write_bytes(data)


def convert_csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as in_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as out_file:
        reader = csv.DictReader(in_file)
        required_cols = {"canary", "problem", "answer"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV missing required columns {sorted(required_cols)}. Found: {reader.fieldnames or []}"
            )

        for idx, row in enumerate(reader):
            canary = row["canary"]
            question = decrypt(row["problem"], canary)
            answer = decrypt(row["answer"], canary)
            record = {
                "id": str(idx),
                "question": question,
                "answer": answer,
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare BrowseComp dataset into benchmark JSONL format")
    parser.add_argument("--url", default=BROWSECOMP_CSV_URL, help="BrowseComp CSV URL")
    parser.add_argument(
        "--csv-path",
        default="data/browse_comp_test_set.csv",
        help="Path to save/read BrowseComp CSV",
    )
    parser.add_argument(
        "--output",
        default="data/browsecomp.jsonl",
        help="Output JSONL path for run_browsecomp_langchain.py",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and only convert existing --csv-path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_path = Path(args.output)

    if not args.skip_download:
        print(f"Downloading BrowseComp CSV from {args.url} -> {csv_path}")
        download_csv(args.url, csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Converting {csv_path} -> {output_path}")
    count = convert_csv_to_jsonl(csv_path, output_path)
    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
