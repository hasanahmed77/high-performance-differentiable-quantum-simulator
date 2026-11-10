# Sprint 00: Foundations

| | |
|---|---|
| **Status** | Done: accepted by the PO in #102 |
| **Started** | 2026-11-09 |
| **Milestone** | `Sprint 00 – Foundations` |
| **Sprint goal** | An empty but production-grade repository in which every automated quality gate (QB3, QB4, QB5) is already enforced, and the delivery process (board, templates, protection) is in place. |

## Sprint backlog

| ID | Story | Pts | Delivered in |
|---|---|---|---|
| QS-101 | `src/` layout package with `pyproject.toml` (hatchling) | 2 | PR: bootstrap |
| QS-102 | ruff + mypy `--strict` + pre-commit | 2 | PR: bootstrap |
| QS-103 | CI matrix: 3 OSes × Python 3.12–3.15 | 3 | PR: bootstrap |
| QS-104 | Coverage gate at 85% | 1 | PR: bootstrap |
| QS-105 | GitHub Projects board, fields, labels, backlog imported | 3 | Board setup (no PR) |
| QS-106 | Branch protection + PR template | 1 | PR: process templates + repo settings |
| QS-107 | CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, LICENSE | 2 | PR: governance |
| QS-109 | ADR and sprint-report templates | 1 | PR: process templates |
| QS-111 | Prepare the `ketjet` PyPI release (publishers + pipeline) | 1 | PR: release pipeline (needs PO action, see below) |
| QS-1201 | Docs site skeleton (MkDocs Material) | 2 of 5 | PR: docs skeleton |
| QS-1203 | README skeleton | 1 of 2 | PR: docs skeleton |
| | **Total** | **19** | |

### Process exception: the bootstrap PR

QS-101 to QS-104 ship as one PR. The CI gates (QS-102–104) can't exist before the package (QS-101),
and none of the four is testable on its own. Every later PR follows the one-story-per-PR rule.

## Test plan

| ID | Verification |
|---|---|
| QS-101 | CI job builds sdist + wheel with `uv build`; wheel installed into a clean venv; `import ketjet` and `ketjet.__version__` checked |
| QS-102 | `ruff check`, `ruff format --check` and `mypy --strict` run as required CI jobs; pre-commit config validated in CI |
| QS-103 | All 12 matrix cells green (ubuntu, macos, windows × 3.12, 3.13, 3.14, 3.15) |
| QS-104 | `pytest --cov-fail-under=85` (project) and `diff-cover --fail-under=90` (patch) in CI; verified by a deliberately under-tested change failing locally |
| QS-105 | Board exists with the §4.3 fields; all 95 backlog stories exist as issues with epic/priority/type labels and sprint milestones |
| QS-106 | A direct push to `main` is rejected; a PR with red CI can't be merged |
| QS-107 | Files present; license text is Apache-2.0; GitHub detects the license |
| QS-109 | Templates render on GitHub; the issue forms validate |
| QS-111 | Pending trusted publishers registered on PyPI and TestPyPI; `release.yml` merged and lint-clean; release logic tested locally. End-to-end run happens with the first functional release. |
| QS-1201 | `mkdocs build --strict` passes in CI; site deployed to GitHub Pages |
| QS-1203 | README renders with project summary, status and badges |

## Actions needed from the PO

1. **PyPI (QS-111).** Create accounts on pypi.org and test.pypi.org (with 2FA), then add a *pending trusted publisher*:
   - project name `ketjet`, owner `hasanahmed77`, repository `high-performance-differentiable-quantum-simulator`,
     workflow `release.yml`, environment `pypi` (and `testpypi` on TestPyPI).
   - A pending publisher does **not** reserve the name. We deliberately **don't** upload an empty placeholder:
     PEP 541 classes a package with no functionality as name squatting. The name is claimed by the first functional release.
2. **Board views and automations (QS-105).** GitHub's API can't create Project views, column limits or built-in workflows,
   so these are a short manual checklist (given at the end of this sprint's board setup).
3. **Review and merge** each Sprint 0 PR.

## Decisions taken during planning

- **PR approval for a solo maintainer.** GitHub doesn't let a PR author approve their own PR, and all commits are
  pushed from the PO's account. Branch protection therefore requires a PR and green status checks; the PO's
  review and merge is the approval of record. plan.md §4.4 is updated to match.
- **Coverage enforcement without third-party accounts.** The 85% gate is enforced by `pytest-cov`
  (`--cov-fail-under`, project ≥85%) and `diff-cover` (changed lines ≥90%) inside CI. Codecov is optional, for badges only.
- **D4 revised: Python 3.12–3.15.** NumPy ≥2.5 and JAX ≥0.12 require Python ≥3.12, and Python 3.15 is released.
  Approved by the PO during planning.
- **Tooling.** `uv` manages Python versions, the virtual environment and the lockfile (`uv.lock`), which is the
  pinned lockfile referenced in §4.5.

- **No placeholder upload (PEP 541).** Publishing an empty 0.0.0 to reserve the name would be name squatting under
  PEP 541. QS-111 is re-scoped to "publishers + pipeline ready"; the first PyPI upload carries real functionality.

## Sprint review

**Goal met?** **Yes, with one manual item outstanding.** Every automated quality gate (QB3 types and lint, QB4 the
CI matrix, QB5 coverage) is enforced on a protected `main`, and the delivery process is live. The only open work is
QS-105's board views and column limits, which GitHub's API can't create.

| ID | Outcome |
|---|---|
| QS-101 | ✅ Done (#96): `src/` layout, hatchling, `py.typed`, `uv.lock` |
| QS-102 | ✅ Done (#96): ruff + `mypy --strict` + pre-commit |
| QS-103 | ✅ Done (#96): 3 OSes × Python 3.12–3.15, all 12 cells green |
| QS-104 | ✅ Done (#96): project ≥85% and patch ≥90% gates; verified that a drop to 27% fails |
| QS-105 | ⏳ Board, fields, labels, milestones and 95 issues done; **views and column limits pending (PO, manual)** |
| QS-106 | ✅ Done (#98 + repo settings): direct push to `main` verified rejected (`GH006`) |
| QS-107 | ✅ Done (#97) |
| QS-109 | ✅ Done (#98) |
| QS-111 | ✅ Done, re-scoped (#101): publishers registered, pipeline merged; no placeholder upload (PEP 541) |
| QS-1201 | ✅ Sprint 0 portion (#100): site live; remainder in Sprint 15 |
| QS-1203 | ✅ Sprint 0 portion (#100): README skeleton; remainder in Sprint 4 |

### Test results (sprint-level DoD, plan.md §5.3)

| Suite | Result |
|---|---|
| Unit | ✅ 4/4 passing on all 12 matrix cells |
| Integration / property / differential / regression / PennyLane | n/a: no simulator code yet (start in Sprints 1–4) |
| Benchmarks | n/a until Sprint 7 |
| Coverage (project / patch) | ✅ 100% / gate active (diff-cover ≥90%) |
| Build + wheel smoke test | ✅ wheel installs and imports in a clean venv |
| Docs build | ✅ `mkdocs build --strict`; deployed to GitHub Pages (HTTP 200) |
| `main` CI after every merge | ✅ all CI and Docs runs green |
| Release pipeline | ⏳ logic tested locally and actionlint-clean; end-to-end run with the first functional release |

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 16 of 19 points done (QS-105, 3 points, pending manual views) |
| Median PR open → merge (review wait + CI) | 55 min (max 65 min) |
| Sprint duration | Same day (2026-11-09), about 3 hours |
| Escaped defects | 0 |

### Demo

- A direct push to `main` is rejected: `GH006: Protected branch update failed`.
- An under-tested change fails the gate: `FAIL Required test coverage of 85.0% not reached. Total coverage: 27.27%`.
- Docs site: <https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/>
- Board: <https://github.com/users/hasanahmed77/projects/2>

## Retrospective

| Keep | Change | Try |
|---|---|---|
| Verifying gates by forcing them to fail (coverage drop, direct push) | **Stacked PRs:** retarget a stacked PR to `main` *before* deleting its base branch. Deleting it auto-closed #99, which had to be replaced by #100. | Fewer, slightly larger PRs where stories can't be tested alone, to cut repeated CI cycles |
| SHA-pinned actions and the single `CI success` aggregate check | **WIP limits were exceeded** (9 items In Review vs a limit of 3). The bootstrap made it unavoidable, but from Sprint 1 a new PR opens only once In Review is ≤3. | Check facts (dependency support windows, PyPI policy) **before** recommending a decision: D4 and the 0.0.0 placeholder both had to be revised |
| Checking policies before publishing (PEP 541) | **"Require up-to-date branches" costs a full CI rerun per queued PR**, about 10 minutes each with free macOS runners | Merge PRs in dependency order as soon as each is green, to keep the rebase queue short |

## Forecast update

Not enough data yet; the first throughput-based forecast follows Sprint 2 (plan.md §8).
