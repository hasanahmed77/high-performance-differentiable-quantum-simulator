#!/usr/bin/env bash
# Audit every locked dependency set against the PyPI / OSV advisory databases (QS-110).
#
# The conformance group conflicts with the dev groups (it pins an older JAX), so the lockfile
# holds two resolutions; each is exported and audited on its own. --require-hashes audits the
# exact locked artifacts, and --disable-pip skips building a throwaway environment.
set -euo pipefail

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

uv export --frozen --quiet --all-extras --group dev --group docs --no-emit-project \
  --output-file "$out/main.txt"
uv export --frozen --quiet --all-extras --no-default-groups --group conformance --no-emit-project \
  --output-file "$out/conformance.txt"

status=0
for set in main conformance; do
  echo "::group::pip-audit: $set dependencies"
  uv run --no-sync pip-audit --requirement "$out/$set.txt" --require-hashes --disable-pip \
    --progress-spinner off --desc on || status=1
  echo "::endgroup::"
done
exit "$status"
