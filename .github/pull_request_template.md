## Summary
<!-- What does this change do, and why? One story per PR. -->

Closes #

## Changes
-

## Design notes for review
<!-- Trade-offs, alternatives considered, anything the reviewer should look at first. Link the ADR if one applies. -->

## Test plan
<!-- Which test types cover this change? Paste key results (benchmarks, coverage) where relevant. -->
- [ ] Unit
- [ ] Integration
- [ ] Property-based / differential
- [ ] Regression test (bug fixes: fails before the fix, passes after)
- [ ] Performance (benchmark before/after)

## Definition of Done (plan.md §5.2)
- [ ] Fully type-hinted; `mypy --strict` and `ruff` pass
- [ ] Unit tests written; integration tests where the change crosses a component boundary
- [ ] Changed lines ≥90% covered; project coverage ≥85%
- [ ] Bug fixes include a regression test
- [ ] CI green on the full matrix
- [ ] Public API documented (NumPy-style docstrings) and in the API reference
- [ ] Changelog fragment added (or `skip-changelog` label)
- [ ] ADR approved (if this touches a public API, the backend protocol or the IR)
