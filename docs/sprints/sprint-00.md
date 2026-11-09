# Sprint 00: Foundations

| | |
|---|---|
| **Status** | In progress |
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
| QS-111 | Reserve the `ketjet` name on PyPI | 1 | PR: release bootstrap (needs PO action, see below) |
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
| QS-111 | `ketjet` 0.0.0 appears on PyPI, published from CI via Trusted Publishing |
| QS-1201 | `mkdocs build --strict` passes in CI; site deployed to GitHub Pages |
| QS-1203 | README renders with project summary, status and badges |

## Actions needed from the PO

1. **PyPI (QS-111).** Create accounts on pypi.org and test.pypi.org (with 2FA), then add a *pending trusted publisher*:
   - project name `ketjet`, owner `hasanahmed77`, repository `high-performance-differentiable-quantum-simulator`,
     workflow `release.yml`, environment `pypi` (and `testpypi` on TestPyPI).
   - A pending publisher does **not** reserve the name. The name is reserved when CI publishes the 0.0.0 placeholder.
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

## Sprint review

*(Completed at sprint close: goal met?, stories done/carried, test results, flow metrics.)*

## Retrospective

*(Keep / change / try.)*
