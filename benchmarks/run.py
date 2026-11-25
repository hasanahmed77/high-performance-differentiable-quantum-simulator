"""Run the benchmark suite (QS-1102) and write a JSON result file under ``benchmarks/results/``.

Examples::

    uv run python -m benchmarks.run                       # everything, CPU, published settings
    uv run python -m benchmarks.run --device gpu          # the JAX runners on the GPU
    uv run python -m benchmarks.run --benchmarks qft --sizes 8,12 --quick

Each case runs in its own subprocess, ``settings.processes`` times (ADR-0008 D3). Only the
default settings produce numbers for publishing; ``--quick`` and ``--smoke`` runs are written as
``unpublished-*.json``, which git ignores.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from benchmarks import harness
from benchmarks.cases import DEVICES, PRECISIONS, RUNNERS, SMOKE_SIZE, Case, cases, prepare
from benchmarks.circuits import BENCHMARKS

ROOT: Final = Path(__file__).resolve().parent.parent
RESULTS: Final = ROOT / "benchmarks" / "results"


def run_case(case: Case, settings: harness.Settings) -> dict[str, Any]:
    """Prepare and time ``case`` in this process; return its timing and facts."""
    prepared = prepare(case)
    timing = harness.measure(prepared.call, prepared.check, settings)
    return {**timing.to_dict(), "pid": os.getpid(), "info": prepared.info}


def _worker(case_json: str, settings_json: str) -> int:
    case = Case(**json.loads(case_json))
    settings = harness.Settings(**json.loads(settings_json))
    print(json.dumps(run_case(case, settings)))
    return 0


def _in_subprocess(case: Case, settings: harness.Settings) -> dict[str, Any]:
    env = {**os.environ, "JAX_PLATFORMS": "cpu" if case.device == "cpu" else "cuda"}
    command = [
        sys.executable,
        "-m",
        "benchmarks.run",
        "--worker",
        json.dumps(asdict(case)),
        "--settings",
        json.dumps(asdict(settings)),
    ]
    done = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{case.id} failed:\n{done.stderr[-4000:]}")
    result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def run(
    selected: Sequence[Case], settings: harness.Settings, machine: dict[str, Any]
) -> dict[str, Any]:
    """Time every case, each in ``settings.processes`` fresh processes, and return the results."""
    results = []
    for i, case in enumerate(selected, start=1):
        start = time.perf_counter()
        timings = [_in_subprocess(case, settings) for _ in range(settings.processes)]
        info = timings[0].pop("info")
        for timing in timings[1:]:
            timing.pop("info")
        combined = harness.combine(timings)
        results.append(
            {
                "id": case.id,
                **asdict(case),
                **info,
                **combined,
                "dispatch_bound": case.device == "gpu"
                and combined["median_s"] < harness.DISPATCH_BOUND_S,
                "processes": timings,
            }
        )
        print(
            f"[{i}/{len(selected)}] {case.id}: median {combined['median_s'] * 1e3:.3f} ms, "
            f"first call {combined['first_call_s'] * 1e3:.1f} ms "
            f"({time.perf_counter() - start:.0f} s)",
            flush=True,
        )
    data = {
        "format": harness.FORMAT,
        "version": harness.VERSION,
        "machine": machine,
        "settings": asdict(settings),
        "cases": results,
    }
    harness.validate(data)
    return data


def _list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.run", description=__doc__)
    parser.add_argument("--benchmarks", type=_list, default=list(BENCHMARKS))
    parser.add_argument("--runners", type=_list, default=list(RUNNERS))
    parser.add_argument(
        "--sizes",
        type=lambda s: [int(n) for n in _list(s)],
        default=None,
        help="qubit counts for every benchmark (default: each benchmark's own)",
    )
    parser.add_argument("--device", choices=DEVICES, default="cpu")
    parser.add_argument("--precision", choices=PRECISIONS, default="complex128")
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--quick", action="store_true", help="1 process, 5 samples; unpublished")
    speed.add_argument(
        "--smoke",
        action="store_true",
        help="1 sample of 1 call, at 3 qubits unless --sizes says; unpublished",
    )
    parser.add_argument("--out", type=Path, default=None, help="result file to write")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--settings", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmarks the command line selects and write their JSON."""
    args = _parser().parse_args(argv)
    if args.worker is not None:
        return _worker(args.worker, args.settings)
    settings = harness.SMOKE if args.smoke else harness.QUICK if args.quick else harness.FULL
    sizes = [SMOKE_SIZE] if args.smoke and args.sizes is None else args.sizes
    selected = cases(args.benchmarks, args.runners, sizes, args.device, args.precision)
    data = run(selected, settings, harness.machine_record(ROOT))
    date = datetime.date.today().isoformat()
    prefix = "" if settings.published else "unpublished-"
    out: Path = args.out or RESULTS / f"{prefix}{date}-{args.device}-{args.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(data['cases'])} cases to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
