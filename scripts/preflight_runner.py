#!/usr/bin/env python3
"""Read-only environment and artifact checks for W4/OFT runners."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, env=env, check=False
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _gpu_capacity_failure(
    free_mib: int | None, *, minimum_mib: int, allow_busy: bool
) -> str | None:
    if allow_busy or minimum_mib <= 0:
        return None
    if free_mib is None:
        return "cannot verify free GPU memory while a minimum is required"
    if free_mib < minimum_mib:
        return f"GPU free memory {free_mib} MiB is below required {minimum_mib} MiB"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-root", type=Path, default=Path("/root/miniconda3"))
    parser.add_argument("--smolvla-env", default="smolvla")
    parser.add_argument("--oft-env", default="oft")
    parser.add_argument(
        "--libero-plus-root",
        type=Path,
        default=Path("/root/autodl-tmp/src/LIBERO-plus"),
    )
    parser.add_argument("--checkpoints-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=0)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="bypass the free-memory hard gate (unsafe for OFT startup)",
    )
    args = parser.parse_args()

    conda = args.conda_root.expanduser().resolve() / "bin" / "conda"
    libero_root = args.libero_plus_root.expanduser().resolve()
    checkpoints_root = args.checkpoints_root.expanduser().resolve()
    failures: list[str] = []

    if not conda.is_file():
        failures.append(f"conda executable missing: {conda}")
    else:
        for env_name in (args.smolvla_env, args.oft_env):
            result = _run(
                [
                    str(conda),
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "-c",
                    "import zmq; print(zmq.__version__)",
                ]
            )
            if result.returncode:
                failures.append(
                    f"pyzmq unavailable in env {env_name}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            else:
                print(f"OK pyzmq env={env_name} version={result.stdout.strip()}")

        benchmark_code = """
import contextlib
import io
import json
from pathlib import Path
from libero.libero import benchmark
root = Path(__import__("os").environ["EXPECTED_LIBERO_ROOT"]).resolve()
module_path = Path(benchmark.__file__).resolve()
if root not in module_path.parents:
    raise RuntimeError(f"benchmark path {module_path} is not under {root}")
suite_map = benchmark.get_benchmark_dict()
result = {"benchmark_file": str(module_path), "tasks": {}}
for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    if name not in suite_map:
        raise RuntimeError(f"missing suite {name}")
    with contextlib.redirect_stdout(io.StringIO()):
        count = int(suite_map[name]().get_num_tasks())
    if count <= 10:
        raise RuntimeError(f"{name} has only {count} tasks; expected LIBERO-Plus (>10)")
    result["tasks"][name] = count
print(json.dumps(result, sort_keys=True))
""".strip()
        env = os.environ.copy()
        env["EXPECTED_LIBERO_ROOT"] = str(libero_root)
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(libero_root), env.get("PYTHONPATH", "")) if item
        )
        result = _run(
            [
                str(conda),
                "run",
                "-n",
                args.smolvla_env,
                "python",
                "-c",
                benchmark_code,
            ],
            env=env,
        )
        if result.returncode:
            failures.append(
                "LIBERO-Plus benchmark check failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        else:
            print(f"OK LIBERO-Plus {result.stdout.strip()}")

    checkpoint_names = {
        "libero_spatial": "oft_spatial",
        "libero_object": "oft_object",
        "libero_goal": "oft_goal",
        "libero_10": "oft_10",
    }
    for suite, directory_name in checkpoint_names.items():
        checkpoint = checkpoints_root / directory_name
        stats_path = checkpoint / "dataset_statistics.json"
        if not (checkpoint / "config.json").is_file():
            failures.append(f"checkpoint config missing: {checkpoint / 'config.json'}")
            continue
        if not stats_path.is_file():
            failures.append(f"checkpoint dataset statistics missing: {stats_path}")
            continue
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {stats_path}: {exc}")
            continue
        matching = [name for name in stats if name.startswith(f"{suite}_")]
        if not matching:
            failures.append(
                f"checkpoint/suite mismatch: {checkpoint} has no {suite}_* statistics"
            )
        else:
            print(f"OK checkpoint suite={suite} path={checkpoint}")

    gpu = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu.returncode:
        print("WARN GPU occupancy unavailable (nvidia-smi failed)")
    elif gpu.stdout.strip():
        print("WARN GPU already has compute processes:")
        print(gpu.stdout.strip())
    else:
        print("OK GPU has no reported compute processes")

    free = _run(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    free_mib: int | None = None
    if free.returncode == 0:
        try:
            free_mib = int(free.stdout.strip().splitlines()[0])
            print(
                f"OK GPU index={args.gpu_index} free_mib={free_mib} "
                f"required_mib={args.min_free_gpu_mib}"
            )
        except (IndexError, ValueError):
            pass
    capacity_failure = _gpu_capacity_failure(
        free_mib,
        minimum_mib=args.min_free_gpu_mib,
        allow_busy=args.allow_busy_gpu,
    )
    if capacity_failure:
        failures.append(capacity_failure)

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        print(f"PREFLIGHT_FAILED checks={len(failures)}", file=sys.stderr)
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
