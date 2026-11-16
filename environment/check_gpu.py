"""Check that the GPU image sees the GPU and that ketjet runs on it (QS-112).

Run inside the container (``docker run --rm --gpus all ketjet-gpu``, the image's default command)::

    python environment/check_gpu.py --expect-name "RTX 4060"

It prints a report to paste into the setup record (docs/dev/gpu-workstation.md) and exits non-zero
if a check fails:

1. ``nvidia-smi`` sees the GPU, and the driver supports the bundled CUDA runtime;
2. ``jax.devices()`` lists it as a CUDA device;
3. a compiled ketjet circuit and its ``jax.grad`` run on it and match the NumPy oracle.

``--allow-cpu`` skips the GPU requirements, so CI can smoke-test the image on a runner without
a GPU.
"""

import argparse
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import ketjet
from ketjet.backends import get_backend
from ketjet.backends.jax_backend import compile_circuit, expectation, parameter_map
from ketjet.circuit import Circuit
from ketjet.observables import Z
from ketjet.operations import Parameter

N_QUBITS = 16
LAYERS = 8
# complex64 on a GPU sums in a different order than on the CPU: allow float32 rounding over the
# circuit's ~250 gates, as the test suite does (plan.md §6.3).
TOLERANCE = {"complex128": 1e-10, "complex64": 1e-4}


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _ansatz() -> Circuit:
    """Return a layered ansatz: every qubit entangled, every parameter symbolic."""
    circuit = Circuit(N_QUBITS)
    k = 0
    for _ in range(LAYERS):
        for w in range(N_QUBITS):
            circuit.ry(Parameter(f"t{k}"), w)
            k += 1
        for w in range(N_QUBITS - 1):
            circuit.cnot(w, w + 1)
        circuit.crz(Parameter(f"t{k}"), N_QUBITS - 1, 0)
        k += 1
    return circuit


def driver_cuda_version(header: str) -> tuple[int, int] | None:
    """Return the highest CUDA version the driver supports, from ``nvidia-smi``'s header.

    Older drivers print ``CUDA Version: 12.4``; newer ones, such as 610.88, print
    ``CUDA UMD Version: 13.3`` next to ``KMD Version``. Returns None if neither is present.
    """
    match = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", header)
    return (int(match[1]), int(match[2])) if match else None


def check_driver(failures: list[str], allow_cpu: bool) -> None:
    """Check that nvidia-smi sees a GPU whose driver supports the bundled CUDA runtime."""
    print("== nvidia-smi")
    if shutil.which("nvidia-smi") is None:
        msg = "nvidia-smi isn't on PATH: run with --gpus all on a host with the NVIDIA runtime"
        print(f"   {msg}")
        if not allow_cpu:
            failures.append(msg)
        return
    query = "name,driver_version,memory.total,compute_cap"
    print("  ", _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]))
    supported = driver_cuda_version(_run(["nvidia-smi"]))
    runtime = _version("nvidia-cuda-runtime-cu12")
    shown = "?" if supported is None else f"{supported[0]}.{supported[1]}"
    print(f"   driver supports CUDA {shown}; bundled CUDA runtime {runtime}")
    if supported is None:
        # Don't pass a check that didn't run: a new nvidia-smi header format must be noticed.
        failures.append("Couldn't read the driver's CUDA version from nvidia-smi's header")
        return
    try:
        needed = tuple(int(x) for x in runtime.split(".")[:2])
    except ValueError:
        failures.append(f"Couldn't read the bundled CUDA runtime's version: {runtime!r}")
        return
    if supported < needed:
        failures.append(
            f"The driver supports CUDA {shown}, older than the bundled runtime {runtime}: "
            "update the NVIDIA driver on Windows"
        )


def check_jax(failures: list[str], expect_name: str | None, allow_cpu: bool) -> None:
    """Check that JAX lists a CUDA device, with the expected name if one is given."""
    print("== jax.devices()")
    devices = jax.devices()
    for device in devices:
        print(f"   {device.platform}:{device.id} {device.device_kind}")
    gpus = [d for d in devices if d.platform == "gpu"]
    if not gpus and not allow_cpu:
        failures.append(f"JAX sees no GPU, only {[d.platform for d in devices]}")
    if gpus and expect_name and not any(expect_name in d.device_kind for d in gpus):
        failures.append(f"No GPU named {expect_name!r}: {[d.device_kind for d in gpus]}")


def check_ketjet(failures: list[str]) -> None:
    """Check that a compiled circuit and its gradient run and match the NumPy oracle."""
    print(f"== ketjet on {jax.default_backend()}: {N_QUBITS} qubits, {LAYERS} layers")
    circuit = _ansatz()
    n_params = len(circuit.parameters)
    x = np.random.default_rng(0).uniform(-np.pi, np.pi, n_params)
    bound = circuit.bind({p.name: v for p, v in zip(circuit.parameters, x, strict=True)})
    oracle = bound.state(backend=get_backend("numpy"))

    for dtype in ("complex128", "complex64"):
        run, to_slots = compile_circuit(circuit, dtype), parameter_map(circuit)
        start = time.perf_counter()
        state = run(to_slots(x)).block_until_ready()
        compile_s = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(10):
            state = run(to_slots(x)).block_until_ready()
        run_ms = (time.perf_counter() - start) / 10 * 1e3
        error = float(np.abs(np.asarray(state) - oracle).max())
        ok = error <= TOLERANCE[dtype]
        print(
            f"   {dtype}: first call {compile_s:.2f} s, then {run_ms:.2f} ms; "
            f"max |state - NumPy| = {error:.1e} {'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"{dtype} state differs from the NumPy oracle by {error:.1e}")

    run, to_slots = compile_circuit(circuit), parameter_map(circuit)
    grad = jax.jit(jax.grad(lambda v: expectation(run(to_slots(v)), Z(0) @ Z(N_QUBITS - 1))))
    g = np.asarray(grad(jnp.asarray(x)))
    finite = bool(np.isfinite(g).all())
    print(f"   jax.jit(jax.grad): {n_params} parameters, |grad| = {np.linalg.norm(g):.4f}")
    if not finite:
        failures.append("The gradient has non-finite entries")


def main() -> int:
    """Run every check, print the report and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect-name", help='a substring of the GPU name, e.g. "RTX 4060"')
    parser.add_argument("--allow-cpu", action="store_true", help="don't require a GPU (CI smoke)")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]  # ADR-0006 D5
    print("== versions")
    print(f"   Python {platform.python_version()}, {platform.platform()}")
    print(f"   ketjet {ketjet.__version__}, jax {jax.__version__}, numpy {np.__version__}")
    print(f"   jax-cuda12-plugin {_version('jax-cuda12-plugin')}")

    failures: list[str] = []
    check_driver(failures, args.allow_cpu)
    check_jax(failures, args.expect_name, args.allow_cpu)
    check_ketjet(failures)

    print("== result")
    for failure in failures:
        print(f"   FAIL: {failure}")
    print("   all checks passed" if not failures else f"   {len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
