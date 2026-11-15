# Security Policy

## Supported versions

ketjet is pre-1.0. Security fixes are applied to the latest released minor version only.

| Version | Supported |
|---|---|
| Latest `0.x` release | ✅ |
| Older releases | ❌ |

From v1.0.0 onward, this table will list the supported release lines.

## Reporting a vulnerability

**Don't report security vulnerabilities in public issues, discussions or pull requests.**

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/security/advisories/new)
(*Security* tab → *Report a vulnerability*). Please include:

- the affected version(s) and platform,
- a description of the issue and its impact,
- steps or a minimal script to reproduce it.

## What to expect

| Step | Target |
|---|---|
| Acknowledgement of the report | within 3 days |
| Initial assessment and severity | within 7 days |
| Fix or mitigation for confirmed high-severity issues | within 30 days |

Reporters are credited in the security advisory unless they prefer to remain anonymous.

## How dependencies and code are checked

| Check | When | Blocks a merge |
|---|---|---|
| `pip-audit` on every locked dependency (`uv.lock`, both resolutions), against PyPI and OSV advisories | every PR and push, and weekly | yes, on any known vulnerability |
| CodeQL (`security-extended` queries) on the Python code and the GitHub Actions workflows | every PR and push, and weekly | yes, on high or critical severity (CVSS ≥ 7.0) |
| Dependabot: grouped weekly updates of `uv.lock` and of the pinned GitHub Actions, and security updates as soon as an advisory is published | weekly, and on advisories | n/a: it opens PRs, which pass the same checks |

Dependabot refreshes `uv.lock` within the version ranges in `pyproject.toml` and never widens them. Raising an
upper bound, such as PennyLane's, is a reviewed change with its own compatibility run. New releases wait seven days
before Dependabot proposes them.

## Scope

In scope: the `ketjet` package, its native extensions, and the build and release pipeline in this repository.
Note that ketjet is a numerical library: incorrect simulation results are bugs, but they are not security
issues unless they can be triggered to cause memory corruption, code execution or a denial of service.
