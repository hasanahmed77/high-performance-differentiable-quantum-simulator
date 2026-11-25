"""The timing harness (QS-1102): ADR-0008 D1 to D4 and D6.

:func:`measure` times one call: the first call apart, a correctness check, two discarded warm-up
calls, then samples of at least ``min_sample_s`` each. :func:`machine_record` describes where
the numbers came from, and :func:`validate` rejects a result file without that description.
"""

import datetime
import importlib.metadata
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

FORMAT: Final = "ketjet-benchmarks"
VERSION: Final = 1

DISPATCH_BOUND_S: Final = 0.010
"""GPU calls faster than this are labelled dispatch-bound (ADR-0008 D4)."""

NO_CLEAR_DIFFERENCE: Final = (0.9, 1.1)
"""Speedups inside this range are reported as "no clear difference" (ADR-0008 D4)."""

LIBRARIES: Final = (
    "ketjet",
    "numpy",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "nvidia-cuda-runtime-cu12",
    "pennylane",
    "pennylane-lightning",
    "pennylane-lightning-gpu",
    "qiskit",
    "qiskit-aer",
    "qiskit-aer-gpu",
    "qulacs",
)
"""Distributions whose versions go into the machine record, when installed."""

THREAD_VARIABLES: Final = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
)
"""Environment variables that change threads or devices, recorded as set (ADR-0008 D6)."""

REQUIRED_MACHINE: Final = ("date", "hardware", "software", "settings")
REQUIRED_HARDWARE: Final = ("cpu", "cores", "threads")
REQUIRED_SOFTWARE: Final = ("os", "python", "ketjet", "git_commit", "libraries")
REQUIRED_CASE: Final = (
    "id",
    "benchmark",
    "runner",
    "device",
    "precision",
    "n_qubits",
    "median_s",
    "q1_s",
    "q3_s",
    "min_s",
    "first_call_s",
    "processes",
)


@dataclass(frozen=True)
class Settings:
    """How many samples to take, and how long each lasts (ADR-0008 D2, D3)."""

    min_sample_s: float = 0.2
    """A sample repeats the call until it lasts at least this long."""

    min_samples: int = 15
    """Samples per process."""

    slow_samples: int = 5
    """Samples per process for a call slower than ``slow_call_s``."""

    slow_call_s: float = 10.0
    """Calls slower than this take ``slow_samples`` samples."""

    warmup: int = 2
    """Calls made and discarded after the first call, before sampling."""

    processes: int = 3
    """Separate processes per case; the reported time is the median of their medians."""

    published: bool = True
    """Whether these settings may produce published numbers (ADR-0008: quick runs may not)."""


FULL: Final = Settings()
"""The published settings (ADR-0008 D2, D3)."""

QUICK: Final = Settings(min_samples=5, slow_samples=5, processes=1, published=False)
"""For development: one process and five samples. Not for publishing."""

SMOKE: Final = Settings(
    min_sample_s=0.0, min_samples=1, slow_samples=1, warmup=1, processes=1, published=False
)
"""For CI: one sample of one call, to show every benchmark still runs."""


@dataclass(frozen=True)
class Timing:
    """One process's measurement of one case, in seconds."""

    first_call_s: float
    warmup_s: float
    number: int
    samples: tuple[float, ...]

    @property
    def median_s(self) -> float:
        """The median sample: the reported time (ADR-0008 D4)."""
        return statistics.median(self.samples)

    @property
    def quartiles_s(self) -> tuple[float, float]:
        """The first and third quartiles; their difference is the reported spread."""
        if len(self.samples) < 2:
            return self.samples[0], self.samples[0]
        q1, _, q3 = statistics.quantiles(self.samples, n=4, method="inclusive")
        return q1, q3

    def to_dict(self) -> dict[str, Any]:
        """Return the timing, with its statistics, as JSON-compatible data."""
        q1, q3 = self.quartiles_s
        return {
            **asdict(self),
            "samples": list(self.samples),
            "median_s": self.median_s,
            "q1_s": q1,
            "q3_s": q3,
            "min_s": min(self.samples),
        }


def measure(
    call: Callable[[], Any],
    check: Callable[[Any], None],
    settings: Settings = FULL,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """Time ``call`` as ADR-0008 D1 to D3 describe.

    Parameters
    ----------
    call
        One call as a user makes it, returning its result only once it's finished: JAX results
        must already have been through ``block_until_ready``.
    check
        Raises if the first call's result is wrong; nothing is timed after a failure (D1).
    settings
        Sample counts and durations.
    clock
        The timer, replaceable in tests.
    """
    start = clock()
    result = call()
    first = clock() - start
    check(result)
    start = clock()
    for _ in range(settings.warmup):
        call()
    warmup = (clock() - start) / settings.warmup if settings.warmup else first
    per_call = max(warmup, 1e-9)
    number = max(1, math.ceil(settings.min_sample_s / per_call))
    count = settings.slow_samples if per_call > settings.slow_call_s else settings.min_samples
    samples = []
    for _ in range(count):
        start = clock()
        for _ in range(number):
            call()
        samples.append((clock() - start) / number)
    return Timing(first, warmup, number, tuple(samples))


def combine(timings: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine processes' timings: the median of their medians, quartiles and first calls (D3)."""

    def middle(key: str) -> float:
        return float(statistics.median(t[key] for t in timings))

    return {
        "median_s": middle("median_s"),
        "q1_s": middle("q1_s"),
        "q3_s": middle("q3_s"),
        "min_s": min(t["min_s"] for t in timings),
        "first_call_s": middle("first_call_s"),
        "process_spread": (
            (max(t["median_s"] for t in timings) - min(t["median_s"] for t in timings))
            / middle("median_s")
        ),
    }


def compare(ours_s: float, theirs_s: float) -> str:
    """Describe ``theirs / ours`` as a speedup, a loss, or no clear difference (D4)."""
    ratio = theirs_s / ours_s
    low, high = NO_CLEAR_DIFFERENCE
    if low <= ratio <= high:
        return f"no clear difference ({ratio:.2f}x)"
    return f"{ratio:.2f}x faster" if ratio > 1 else f"{1 / ratio:.2f}x slower"


# --------------------------------------------------------------------------- machine record


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _run(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _cpu() -> tuple[str, int | None]:
    """Return the CPU model and its physical core count, where the OS says."""
    info = _read("/proc/cpuinfo") or ""
    fields = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in info.splitlines())
        if value
    }
    model = fields.get("model name") or platform.processor() or platform.machine()
    cores = int(fields["cpu cores"]) if fields.get("cpu cores", "").isdigit() else None
    return model, cores


def _memory_bytes() -> int | None:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _memory_cap() -> str | None:
    cap = _read("/sys/fs/cgroup/memory.max")
    return cap.strip() if cap else None


def _gpus() -> list[dict[str, str]]:
    out = _run(
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
    )
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        name, memory, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "memory": memory, "driver": driver})
    return gpus


def _git(root: Path) -> tuple[str | None, bool | None]:
    """Return the checked-out commit and whether the tree is clean, without needing ``git``."""
    commit = _run("git", "-C", str(root), "rev-parse", "HEAD")
    status = _run("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no")
    if commit:
        return commit, status == ""
    head = _read(str(root / ".git" / "HEAD"))
    if head is None:
        return None, None
    head = head.strip()
    if not head.startswith("ref: "):
        return head, None
    ref = head.removeprefix("ref: ")
    loose = _read(str(root / ".git" / ref))
    if loose:
        return loose.strip(), None
    for line in (_read(str(root / ".git" / "packed-refs")) or "").splitlines():
        sha, _, name = line.partition(" ")
        if name == ref:
            return sha, None
    return None, None


def library_versions() -> dict[str, str]:
    """Return the versions of the installed :data:`LIBRARIES`."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def machine_record(root: Path, environ: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """Describe the machine, software and settings a run used (ADR-0008 D6)."""
    cpu, cores = _cpu()
    commit, clean = _git(root)
    kernel = platform.release()
    proc_version = _read("/proc/version") or ""
    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "hardware": {
            "cpu": cpu,
            "cores": cores,
            "threads": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
            "gpus": _gpus(),
        },
        "software": {
            "os": platform.platform(),
            "kernel": kernel,
            "wsl": "WSL2" if "microsoft" in (kernel + proc_version).lower() else None,
            "python": sys.version.split()[0],
            "ketjet": library_versions().get("ketjet"),
            "git_commit": commit,
            "git_clean": clean,
            "container_image": environ.get("KETJET_IMAGE_ID"),
            "libraries": library_versions(),
        },
        "settings": {
            "environment": {name: environ[name] for name in THREAD_VARIABLES if name in environ},
            "memory_cap": _memory_cap(),
        },
    }


def validate(data: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``data`` is a complete result file with its machine record.

    ADR-0008 D6: a result without its record isn't published.
    """
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError(f"Not a {FORMAT} version {VERSION} result file.")
    machine = data.get("machine")
    if not isinstance(machine, Mapping):
        raise ValueError("The result file has no machine record (ADR-0008 D6).")
    for key in REQUIRED_MACHINE:
        if key not in machine:
            raise ValueError(f"The machine record has no {key!r}.")
    for part, keys in (("hardware", REQUIRED_HARDWARE), ("software", REQUIRED_SOFTWARE)):
        missing = [key for key in keys if key not in machine[part]]
        if missing:
            raise ValueError(f"The machine record's {part} has no {', '.join(missing)}.")
    if "settings" not in data:
        raise ValueError("The result file doesn't say which sample settings it used.")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The result file has no cases.")
    for case in cases:
        missing = [key for key in REQUIRED_CASE if key not in case]
        if missing:
            raise ValueError(f"Case {case.get('id', '?')} has no {', '.join(missing)}.")
