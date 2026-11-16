# Changelog fragments

Every PR that changes user-visible behaviour adds one fragment here. [towncrier](https://towncrier.readthedocs.io)
compiles them into `CHANGELOG.md` at release time, so PRs never conflict on the changelog.

**File name:** `<issue>.<type>.md`, where `<issue>` is the GitHub issue the PR closes and `<type>` is one of
`added`, `changed`, `deprecated`, `removed`, `fixed`, `performance`, `security`. Use `+<slug>.<type>.md` when there
is no issue.

**Content:** one or two sentences, in the past tense, written for users. For example:

```
`Circuit.probs(wires)` returns marginal probabilities in PennyLane's wire order.
```

CI fails a PR that has no new fragment, unless the PR carries the `skip-changelog` label (for docs-only or
CI-only changes). Adding or removing a label doesn't start CI. The changelog check reads the labels when it runs,
so add the label when you open the PR, or re-run the **Changelog fragment** job after adding it.
