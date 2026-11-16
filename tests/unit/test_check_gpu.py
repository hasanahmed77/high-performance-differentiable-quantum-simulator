"""The GPU image's driver check reads both nvidia-smi header formats (QS-112).

The script under test is environment/check_gpu.py.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[2] / "environment" / "check_gpu.py"

# The header line as older drivers print it, and as newer ones do (the workstation's 610.88).
OLD_HEADER = (
    "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |"
)
NEW_HEADER = (
    "| NVIDIA-SMI 610.88                 KMD Version: 610.88        CUDA UMD Version: 13.3     |"
)


@pytest.fixture(scope="module")
def check_gpu() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_gpu", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("header", "expected"),
    [(OLD_HEADER, (12, 4)), (NEW_HEADER, (13, 3)), ("| NVIDIA-SMI 999 | no version here |", None)],
)
def test_driver_cuda_version_reads_both_header_formats(
    check_gpu: ModuleType, header: str, expected: tuple[int, int] | None
) -> None:
    assert check_gpu.driver_cuda_version(f"Mon Nov 16 2026\n+---+\n{header}\n+---+") == expected


def _check_driver(
    check_gpu: ModuleType, monkeypatch: pytest.MonkeyPatch, header: str, runtime: str
) -> list[str]:
    monkeypatch.setattr(check_gpu.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        check_gpu, "_run", lambda cmd: "RTX 4060, 610.88" if len(cmd) > 1 else header
    )
    monkeypatch.setattr(check_gpu, "_version", lambda _: runtime)
    failures: list[str] = []
    check_gpu.check_driver(failures, allow_cpu=False)
    return failures


@pytest.mark.parametrize("header", [OLD_HEADER.replace("12.4", "12.9"), NEW_HEADER])
def test_a_driver_new_enough_passes(
    check_gpu: ModuleType, monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    assert _check_driver(check_gpu, monkeypatch, header, "12.9.79") == []


@pytest.mark.parametrize("header", [OLD_HEADER, NEW_HEADER.replace("13.3", "12.4")])
def test_a_driver_older_than_the_runtime_fails(
    check_gpu: ModuleType, monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    (failure,) = _check_driver(check_gpu, monkeypatch, header, "12.9.79")
    assert "older than the bundled runtime" in failure


def test_an_unreadable_header_fails_rather_than_skipping_the_check(
    check_gpu: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    (failure,) = _check_driver(check_gpu, monkeypatch, "| NVIDIA-SMI 999 |", "12.9.79")
    assert "Couldn't read the driver's CUDA version" in failure
