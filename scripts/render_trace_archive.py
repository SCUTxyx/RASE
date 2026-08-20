#!/usr/bin/env python3
"""Encode a portable rollout trace archive (JPEG frames + trace.json) to MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Directory with trace.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    trace_dir = args.trace.resolve()
    payload = json.loads((trace_dir / "trace.json").read_text(encoding="utf-8"))
    records = list(payload.get("frames") or [])
    if not records:
        raise SystemExit(f"trace has no frames: {trace_dir}")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("install video extras: pip install -e '.[video]'") from exc

    output = args.output.resolve() if args.output else trace_dir / "rollout.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=args.fps, codec="libx264") as writer:
        for record in records:
            writer.append_data(imageio.imread(trace_dir / record["file"]))
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
