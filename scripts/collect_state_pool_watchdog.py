#!/usr/bin/env python3
"""Restart collect after native crashes; only skip the crashing episode for signals.

Python exceptions (exit code 1) are NOT treated as native crashes — they usually mean
a resume/idempotency bug or config error. Only process deaths by signal (SIGFPE,
SIGSEGV, …) mark the current episode for skip.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.collect.pipeline import (  # noqa: E402
    clear_current_episode,
    read_current_episode,
    record_skip_index,
)


def _pool_from_config(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    out = Path(cfg["collection"]["output_dir"])
    return out if out.is_absolute() else ROOT / out


def _signal_name(code: int) -> str:
    if code < 0:
        sig = -code
    elif code >= 128:
        sig = code - 128
    else:
        return ""
    try:
        return signal.Signals(sig).name
    except (ValueError, AttributeError):
        return f"signal_{sig}"


def _is_native_signal_death(code: int) -> bool:
    """True when the child was killed by a fatal signal (not a Python SystemExit)."""
    return code < 0 or code >= 128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--max-restarts", type=int, default=0, help="0 = unlimited")
    parser.add_argument(
        "--skip-on-any-nonzero",
        action="store_true",
        help="Legacy: also skip current episode on Python exit code 1 (not recommended)",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    pool_root = _pool_from_config(args.config)
    attempt = 0
    print(
        "WATCHDOG_READY only auto-skips on signal death (SIGFPE/SIGSEGV/…); "
        "Python FileExistsError / exit 1 will stop the watchdog so you can fix it.",
        flush=True,
    )
    while True:
        attempt += 1
        if args.max_restarts and attempt > args.max_restarts:
            print(f"WATCHDOG_STOP max_restarts={args.max_restarts}", flush=True)
            return 1
        print(f"WATCHDOG_START attempt={attempt}", flush=True)
        print(
            "WATCHDOG_WAIT loading policy / first episodes may take several minutes; "
            "do not Ctrl-C unless you intend to stop. Look for COLLECT_EPISODE_START.",
            flush=True,
        )
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "collect_state_pool.py"),
            "--config",
            args.config,
        ]
        try:
            proc = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
        except KeyboardInterrupt:
            print("WATCHDOG_INTERRUPT stopping without skip", flush=True)
            return 130
        code = int(proc.returncode)
        if code == 0:
            print("WATCHDOG_DONE collect exited 0", flush=True)
            return 0

        marker = read_current_episode(pool_root)
        sig = _signal_name(code)
        should_skip = _is_native_signal_death(code) or (
            args.skip_on_any_nonzero and code != 0
        )

        if should_skip and marker is not None:
            record_skip_index(
                pool_root,
                int(marker["index"]),
                reason=(
                    f"watchdog: {_signal_name(code) or f'exit_{code}'} "
                    "while episode current"
                ),
            )
            clear_current_episode(pool_root)
            print(
                f"WATCHDOG_SKIP index={marker['index']} signal={sig or 'n/a'} code={code}",
                flush=True,
            )
            print(
                f"WATCHDOG_CRASH code={code} skipped_index={marker['index']}; "
                f"restarting in {args.sleep}s",
                flush=True,
            )
            time.sleep(args.sleep)
            continue

        print(
            f"WATCHDOG_ABORT code={code} signal={sig or 'none'} "
            f"current={marker}; not skipping (fix the error, then restart watchdog).",
            flush=True,
        )
        return code


if __name__ == "__main__":
    raise SystemExit(main())
