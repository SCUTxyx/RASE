#!/usr/bin/env python3
"""Robust fixed-revision Hugging Face file download using verified byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import threading
import time
from pathlib import Path

import requests


CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", default="model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        actual_size = output.stat().st_size
        actual_sha = sha256(output)
        if actual_size == args.expected_size and actual_sha == args.expected_sha256:
            print(f"RANGE_DOWNLOAD already_verified path={output}", flush=True)
            return 0
        raise ValueError(
            f"existing output has wrong identity: size={actual_size} sha256={actual_sha}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".range.part")
    markers = output.with_name(output.name + ".range.markers")
    markers.mkdir(exist_ok=True)

    with part.open("ab") as handle:
        handle.truncate(args.expected_size)

    chunk = args.chunk_mib << 20
    ranges = [
        (index, start, min(args.expected_size - 1, start + chunk - 1))
        for index, start in enumerate(range(0, args.expected_size, chunk))
    ]
    url = (
        f"{args.endpoint.rstrip('/')}/{args.repo}/resolve/"
        f"{args.revision}/{args.filename}"
    )
    progress_lock = threading.Lock()
    completed = sum((markers / f"{index:05d}.done").is_file() for index, _, _ in ranges)
    print(
        f"RANGE_DOWNLOAD start repo={args.repo} chunks={len(ranges)} "
        f"completed={completed} workers={args.workers}",
        flush=True,
    )

    def fetch(item: tuple[int, int, int]) -> int:
        nonlocal completed
        index, start, end = item
        marker = markers / f"{index:05d}.done"
        if marker.is_file():
            return index
        expected = end - start + 1
        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(20, 120),
                )
                match = CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
                if response.status_code != 206 or match is None:
                    raise RuntimeError(
                        f"range {index} HTTP {response.status_code} "
                        f"content-range={response.headers.get('content-range')!r}"
                    )
                returned_start, returned_end, total = map(int, match.groups())
                if (returned_start, returned_end, total) != (
                    start, end, args.expected_size
                ):
                    raise RuntimeError(
                        f"range {index} mismatch: {(returned_start, returned_end, total)}"
                    )
                payload = response.content
                if len(payload) != expected:
                    raise RuntimeError(
                        f"range {index} short body: {len(payload)} != {expected}"
                    )
                descriptor = os.open(part, os.O_WRONLY)
                try:
                    written = os.pwrite(descriptor, payload, start)
                    if written != expected:
                        raise RuntimeError(
                            f"range {index} short write: {written} != {expected}"
                        )
                    os.fdatasync(descriptor)
                finally:
                    os.close(descriptor)
                marker.write_text(f"{start}-{end}\n")
                with progress_lock:
                    completed += 1
                    print(
                        f"RANGE_DOWNLOAD chunk={completed}/{len(ranges)} "
                        f"index={index} attempts={attempt}",
                        flush=True,
                    )
                return index
            except Exception as exc:
                delay = min(30, 2 ** min(attempt, 5))
                print(
                    f"RANGE_RETRY index={index} attempt={attempt} "
                    f"delay={delay}s error={exc}",
                    flush=True,
                )
                time.sleep(delay)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fetch, ranges))

    actual_size = part.stat().st_size
    if actual_size != args.expected_size:
        raise RuntimeError(f"assembled size {actual_size} != {args.expected_size}")
    actual_sha = sha256(part)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(
            f"assembled sha256 {actual_sha} != {args.expected_sha256}"
        )
    part.replace(output)
    print(
        f"RANGE_DOWNLOAD verified path={output} size={actual_size} sha256={actual_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
