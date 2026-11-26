"""How many qubits the JAX backend fits on a device (QS-1108).

Example, in the GPU workstation image::

    uv run python -m benchmarks.capacity --device gpu

For each workload and precision, circuits of growing size are run, each in a fresh process, until
one runs out of memory. A workload is the final state of the ``random`` benchmark circuit, or the
gradient of the ``vqe-grad`` benchmark (a 4-layer ansatz and an Ising energy) with backprop or
adjoint. Each size is checked by one call, after compiling; its time and the device's peak memory
are recorded. The result, with the machine record, is written to
``benchmarks/capacity/<date>-<device>.json``.

Backprop keeps every intermediate state, so its capacity depends on the circuit's depth; adjoint's
doesn't (ADR-0007).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from benchmarks import harness
from benchmarks.circuits import random_circuit, vqe_ansatz

ROOT: Final = Path(__file__).resolve().parent.parent
OUT: Final = ROOT / "benchmarks" / "capacity"
WORKLOADS: Final = ("state", "adjoint", "backprop")
PRECISIONS: Final = ("complex64", "complex128")
TIMEOUT_S: Final = 1800


def attempt(workload: str, precision: str, n: int) -> dict[str, Any]:
    """Run one workload at ``n`` qubits in this process; return its time and peak memory."""
    import jax  # noqa: PLC0415 - the capacity tool needs JAX
    import jax.numpy as jnp  # noqa: PLC0415

    from ketjet.backends import jax_backend  # noqa: PLC0415

    jax.config.update("jax_enable_x64", precision == "complex128")  # type: ignore[no-untyped-call]
    real = jnp.float64 if precision == "complex128" else jnp.float32
    if workload == "state":
        circuit = random_circuit(n).bound()
        run = jax_backend.compile_circuit(circuit, dtype=precision)
        x = jnp.asarray(jax_backend.parameter_values(circuit), dtype=real)
        call: Any = lambda: run(x).block_until_ready()  # noqa: E731
    else:
        spec = vqe_ansatz(n)
        assert spec.observable is not None
        method: jax_backend.DiffMethod = "adjoint" if workload == "adjoint" else "backprop"
        f = jax_backend.expval_fn(
            spec.circuit, [spec.observable], dtype=precision, diff_method=method
        )
        grad = jax.jit(jax.grad(lambda v: f(v)[0]))
        weights = jnp.asarray(spec.values, dtype=real)
        call = lambda: grad(weights).block_until_ready()  # noqa: E731
    start = time.perf_counter()
    call()
    first = time.perf_counter() - start
    start = time.perf_counter()
    call()
    second = time.perf_counter() - start
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "ok": True,
        "first_call_s": first,
        "call_s": second,
        "peak_bytes": stats.get("peak_bytes_in_use"),
        "device": str(jax.devices()[0]),
    }


def _in_subprocess(workload: str, precision: str, n: int, device: str) -> dict[str, Any]:
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu" if device == "cpu" else "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    command = [sys.executable, "-m", "benchmarks.capacity", "--worker", workload, precision, str(n)]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
            check=False,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {TIMEOUT_S} s"}
    if done.returncode == 0:
        result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
        return result
    tail = done.stderr.strip().splitlines()[-3:]
    memory = any("RESOURCE_EXHAUSTED" in line or "out of memory" in line.lower() for line in tail)
    return {"ok": False, "out_of_memory": memory, "exit_code": done.returncode, "error": tail}


def search(
    workload: str, precision: str, start: int, stop: int, device: str
) -> tuple[int | None, list[dict[str, Any]]]:
    """Grow the circuit from ``start`` until it fails; return the largest size that ran."""
    largest = None
    attempts = []
    for n in range(start, stop + 1):
        result = {"workload": workload, "precision": precision, "n_qubits": n}
        result |= _in_subprocess(workload, precision, n, device)
        attempts.append(result)
        status = "ok" if result["ok"] else f"failed: {result.get('error')}"
        print(f"{workload} {precision} n={n}: {status}", flush=True)
        if not result["ok"]:
            break
        largest = n
    return largest, attempts


def main(argv: Sequence[str] | None = None) -> int:
    """Find each workload's capacity and write the JSON."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.capacity", description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument("--start", type=int, default=22)
    parser.add_argument("--stop", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        workload, precision, n = args.worker
        print(json.dumps(attempt(workload, precision, int(n))))
        return 0
    results = []
    for precision in args.precisions.split(","):
        for workload in args.workloads.split(","):
            largest, attempts = search(workload, precision, args.start, args.stop, args.device)
            results.append(
                {
                    "workload": workload,
                    "precision": precision,
                    "largest": largest,
                    "attempts": attempts,
                }
            )
    data = {
        "format": "ketjet-capacity",
        "version": 1,
        "machine": harness.machine_record(ROOT),
        "device": args.device,
        "results": results,
    }
    out: Path = args.out or OUT / f"{datetime.date.today().isoformat()}-{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    for r in results:
        print(f"{r['workload']:>8} {r['precision']}: largest {r['largest']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
