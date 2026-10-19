#!/usr/bin/env bash
# Creates the GitHub repo and pushes this skeleton.
# Requires the GitHub CLI (https://cli.github.com) and `gh auth login` first.
# Usage: ./scripts/init_github.sh <owner> [repo-name]   e.g. ./scripts/init_github.sh code-coogs aip
set -euo pipefail
OWNER="${1:?give the GitHub user or org}"
REPO="${2:-aip}"
cd "$(dirname "$0")/.."
git init -b main 2>/dev/null || true
git add .
git commit -m "Initial project skeleton" 2>/dev/null || echo "nothing new to commit"
gh repo create "$OWNER/$REPO" --private --source=. --remote=origin --push
# Protect main: require a PR with one review and passing CI
gh api -X PUT "repos/$OWNER/$REPO/branches/main/protection" \
  -H "Accept: application/vnd.github+json" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=C++ build and tests (CPU only)" \
  -F "enforce_admins=false" \
  -F "required_pull_request_reviews[required_approving_review_count]=1" \
  -F "restrictions=null" || echo "Branch protection needs a paid plan for private repos on personal accounts; set it later in Settings."
echo "Done: https://github.com/$OWNER/$REPO"
