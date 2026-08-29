#!/usr/bin/env bash
# Force-sync this clone to GitHub (fixes "pull says up to date" / missing new scripts).
#
# Usage (from repo root):
#   bash scripts/update_from_github.sh
#
# WARNING: discards uncommitted local changes in the repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${1:-feature/latent-history-attention}"

echo "remote=$(git remote get-url origin)"
echo "branch=${BRANCH}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git reset --hard "origin/${BRANCH}"

echo "HEAD=$(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "scripts present:"
ls -1 scripts/run_canonical*.sh 2>/dev/null || true
