# Contributing to ketjet

Thanks for your interest in ketjet. This guide covers how to set up a development environment, the
workflow every change follows, and the quality bar a change must meet before it is merged.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug:** open an issue with the *Bug report* template and include a minimal reproducer.
- **Propose a feature:** open an issue with the *User story* template. Changes to public APIs, the backend
  protocol or the circuit IR also need an ADR (see below).
- **Pick up work:** issues labelled `good first issue` or `help wanted` are ready to start. Comment on the issue
  first so work isn't duplicated.
- **Security issues:** don't open a public issue. See [SECURITY.md](SECURITY.md).

## Development setup

ketjet uses [uv](https://docs.astral.sh/uv/) for Python versions, environments and the lockfile.

```bash
git clone https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator.git
cd high-performance-differentiable-quantum-simulator
uv sync                      # creates .venv with the package (editable) and dev tools
uv run pre-commit install    # runs ruff and mypy on every commit
```

Supported Python versions: **3.12–3.15**.

## Running the checks

CI runs exactly these commands, so a change that passes locally should pass CI.

| Check | Command |
|---|---|
| Lint | `uv run ruff check` |
| Format | `uv run ruff format --check` |
| Types | `uv run mypy` (strict mode) |
| Tests + coverage | `uv run pytest --cov` (fails below 85%) |
| All hooks | `uv run pre-commit run --all-files` |
| Build | `uv build` |

Test markers: `slow`, `gpu`, `native`, `pennylane` and `nightly` are excluded from the default run.
Select them explicitly, e.g. `uv run pytest -m gpu`.

## Workflow

1. **One issue per change.** Every PR links to an issue (`Closes #N`) and covers one story.
2. **Branch** from `main` using `<type>/QS-<id>-<slug>`, for example `feat/QS-203-apply-matrix`.
3. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `perf:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`.
4. **Open a PR** using the template. CI must be green on the full matrix.
5. **Review and merge:** the maintainer reviews and squash-merges. `main` is protected, so direct pushes are rejected.

## Definition of Done

A change is ready to merge when:

- [ ] Code is fully type-hinted; `mypy --strict` and `ruff` pass
- [ ] Unit tests are written, plus integration tests where the change crosses a component boundary
- [ ] Changed lines have ≥90% coverage and the project stays ≥85%
- [ ] Bug fixes include a regression test that fails before the fix
- [ ] Public APIs have NumPy-style docstrings
- [ ] CI is green on every OS and Python version

The full definitions (story, sprint and release level) are in [plan.md §5](plan.md#5-definitions-of-ready-and-done).

## Architecture Decision Records

Significant design decisions are recorded as ADRs in `docs/adr/`. A change to a public API, the backend
protocol or the circuit IR needs an approved ADR **before** implementation. Copy `docs/adr/0000-template.md`,
take the next number, and open it as a PR for discussion.

## Numerical tolerances

Tests compare floating-point results using the shared tolerances in `tests/tolerances.py` (introduced with the
first numerical code in Sprint 1). Don't loosen a
tolerance inline. If a looser bound is genuinely needed, explain why in an ADR.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
