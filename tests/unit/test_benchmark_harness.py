"""The benchmark harness follows ADR-0008 D1 to D4 and D6 (QS-1102)."""

import copy
from pathlib import Path
from typing import Any

import pytest
from benchmarks import harness
from benchmarks.harness import Settings, Timing


class FakeClock:
    """A clock that a fake call advances by its own duration."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def fake_call(clock: FakeClock, first_s: float, then_s: float) -> tuple[Any, list[float]]:
    """Return a call that takes ``first_s`` once, then ``then_s``, and a log of its durations."""
    log: list[float] = []

    def call() -> str:
        duration = first_s if not log else then_s
        log.append(duration)
        clock.now += duration
        return "result"

    return call, log


def test_full_settings_are_adr_0008() -> None:
    """D2: two warm-up calls. D3: 0.2 s samples, 15 of them (5 over 10 s), 3 processes."""
    assert (
        Settings(
            min_sample_s=0.2,
            min_samples=15,
            slow_samples=5,
            slow_call_s=10.0,
            warmup=2,
            processes=3,
            published=True,
        )
        == harness.FULL
    )
    assert not harness.QUICK.published
    assert (harness.QUICK.processes, harness.QUICK.min_samples) == (1, 5)
    assert not harness.SMOKE.published


def test_first_call_is_reported_apart() -> None:
    """D1: a slow first call (a compilation) is its own number and stays out of the samples."""
    clock = FakeClock()
    call, _ = fake_call(clock, first_s=5.0, then_s=0.001)
    timing = harness.measure(call, lambda _: None, clock=clock)
    assert timing.first_call_s == pytest.approx(5.0)
    assert timing.median_s == pytest.approx(0.001)
    assert max(timing.samples) == pytest.approx(0.001)


def test_samples_last_at_least_the_minimum() -> None:
    """D3: fast calls are repeated within a sample until it lasts at least 0.2 s."""
    clock = FakeClock()
    call, log = fake_call(clock, first_s=0.003, then_s=0.003)
    timing = harness.measure(call, lambda _: None, clock=clock)
    assert timing.number * 0.003 >= 0.2
    assert (timing.number - 1) * 0.003 < 0.2
    assert len(timing.samples) == 15
    assert len(log) == 1 + 2 + 15 * timing.number  # first call, warm-up, samples


def test_slow_calls_take_fewer_samples() -> None:
    """D3: at least 5 samples for a call over 10 s, one call each."""
    clock = FakeClock()
    call, log = fake_call(clock, first_s=12.0, then_s=11.0)
    timing = harness.measure(call, lambda _: None, clock=clock)
    assert (timing.number, len(timing.samples), len(log)) == (1, 5, 1 + 2 + 5)


def test_a_wrong_result_is_not_timed() -> None:
    """D1: the first call's result is checked before anything else is timed."""
    clock = FakeClock()
    call, log = fake_call(clock, first_s=0.1, then_s=0.1)
    seen = []

    def check(result: Any) -> None:
        seen.append(result)
        raise AssertionError("wrong")

    with pytest.raises(AssertionError, match="wrong"):
        harness.measure(call, check, clock=clock)
    assert seen == ["result"]
    assert len(log) == 1


def test_timing_statistics() -> None:
    """D4: the median is reported with the interquartile range; the minimum is kept too."""
    timing = Timing(first_call_s=1.0, warmup_s=0.5, number=1, samples=(5.0, 1.0, 3.0, 2.0, 4.0))
    data = timing.to_dict()
    assert (data["median_s"], data["q1_s"], data["q3_s"], data["min_s"]) == (3.0, 2.0, 4.0, 1.0)
    assert data["samples"] == [5.0, 1.0, 3.0, 2.0, 4.0]
    single = Timing(first_call_s=1.0, warmup_s=1.0, number=1, samples=(2.0,))
    assert single.quartiles_s == (2.0, 2.0)


def test_processes_combine_as_the_median_of_medians() -> None:
    """D3: headline numbers are the median of the processes' medians."""
    timings = [
        {"median_s": m, "q1_s": m - 1, "q3_s": m + 1, "min_s": m - 2, "first_call_s": 10 * m}
        for m in (10.0, 11.0, 12.0)
    ]
    combined = harness.combine(timings)
    assert combined["median_s"] == 11.0
    assert (combined["q1_s"], combined["q3_s"], combined["min_s"]) == (10.0, 12.0, 8.0)
    assert combined["first_call_s"] == 110.0
    assert combined["process_spread"] == pytest.approx(2 / 11)


@pytest.mark.parametrize(
    ("theirs", "expected"),
    [
        (1.05, "no clear difference (1.05x)"),
        (0.92, "no clear difference (0.92x)"),
        (2.0, "2.00x faster"),
        (0.5, "2.00x slower"),
    ],
)
def test_speedups_inside_ten_percent_are_no_clear_difference(theirs: float, expected: str) -> None:
    """D4: a ratio between 0.9 and 1.1 is neither a win nor a loss."""
    assert harness.compare(1.0, theirs) == expected


def _result_file() -> dict[str, Any]:
    case = dict.fromkeys(harness.REQUIRED_CASE, 1.0)
    return {
        "format": harness.FORMAT,
        "version": harness.VERSION,
        "machine": harness.machine_record(Path.cwd()),
        "settings": {},
        "cases": [case],
    }


def test_the_machine_record_has_what_adr_0008_d6_lists() -> None:
    """D6: hardware, software and settings, with the ketjet version and git commit."""
    record = harness.machine_record(Path.cwd(), environ={"OMP_NUM_THREADS": "4", "HOME": "/x"})
    assert record["hardware"]["threads"] >= 1
    assert record["software"]["ketjet"] is not None
    assert "numpy" in record["software"]["libraries"]
    assert record["settings"]["environment"] == {"OMP_NUM_THREADS": "4"}
    harness.validate(_result_file())


def test_a_result_without_a_machine_record_is_rejected() -> None:
    """D6: a result without its record isn't published."""
    data = _result_file()
    del data["machine"]
    with pytest.raises(ValueError, match="no machine record"):
        harness.validate(data)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("format",), "Not a ketjet-benchmarks"),
        (("machine", "date"), "has no 'date'"),
        (("machine", "hardware", "cpu"), "hardware has no cpu"),
        (("machine", "software", "git_commit"), "software has no git_commit"),
        (("settings",), "sample settings"),
        (("cases",), "no cases"),
        (("cases", 0, "median_s"), "has no median_s"),
    ],
)
def test_incomplete_result_files_are_rejected(path: tuple[Any, ...], message: str) -> None:
    data = copy.deepcopy(_result_file())
    parent: Any = data
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(ValueError, match=message):
        harness.validate(data)


def test_git_commit_is_read_without_git(tmp_path: Path) -> None:
    """The image has no git, so the commit comes from ``.git`` itself; cleanliness is unknown."""
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    assert harness._git(tmp_path) == (None, None)
    (git / "packed-refs").write_text("# pack-refs\n" + "a" * 40 + " refs/heads/main\n")
    assert harness._git(tmp_path) == ("a" * 40, None)
    (git / "refs" / "heads" / "main").write_text("b" * 40 + "\n")
    assert harness._git(tmp_path) == ("b" * 40, None)
    (git / "HEAD").write_text("c" * 40 + "\n")
    assert harness._git(tmp_path) == ("c" * 40, None)
    assert harness._git(tmp_path / "missing") == (None, None)
