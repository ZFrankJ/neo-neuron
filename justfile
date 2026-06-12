set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
  @just --list

help:
  @just --list

status:
  git status --short --branch

check:
  make check

test:
  make test

mlx-parity:
  make mlx-parity

mps-probe:
  make mps-probe

pr-status:
  gh pr status

sync-main:
  #!/usr/bin/env bash
  set -euo pipefail
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[ERROR] Working tree is dirty. Commit/stash changes before syncing main." >&2
    exit 1
  fi
  git fetch --prune origin
  git switch main
  git pull --ff-only origin main
  git status --short --branch

start-branch name:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="{{name}}"
  if [[ -z "$branch_name" ]]; then
    echo "[ERROR] Provide name=<branch-name>." >&2
    exit 1
  fi
  if [[ ! "$branch_name" =~ ^codex/(feat|fix|maint|docs|ci|test)/.+$ ]]; then
    echo "[ERROR] Branch name should match codex/<type>/<topic>." >&2
    exit 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[ERROR] Working tree is dirty. Commit/stash changes before starting a branch." >&2
    exit 1
  fi
  git fetch --prune origin
  git switch main
  git pull --ff-only origin main
  git switch -c "$branch_name"
  git status --short --branch

refresh-branch:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] refresh-branch is for topic branches. Use just sync-main on main." >&2
    exit 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[ERROR] Working tree is dirty. Commit/stash changes before rebasing." >&2
    exit 1
  fi
  git fetch --prune origin
  git rebase origin/main
  git status --short --branch

stage-all:
  git add -A

stage +paths:
  git add -- {{paths}}

unstage +paths:
  git restore --staged -- {{paths}}

commit msg:
  #!/usr/bin/env bash
  set -euo pipefail
  subject="{{msg}}"
  branch_name="$(git branch --show-current)"
  if [[ -z "$subject" ]]; then
    echo "[ERROR] Provide msg=\"type(scope): summary\"." >&2
    exit 1
  fi
  if ! printf '%s\n' "$subject" | grep -Eq '^(feat|fix|docs|refactor|test|chore|build|ci|perf|revert)(\([[:alnum:]_.-]+\))?: .+'; then
    echo "[ERROR] Commit subject should follow conventional style, e.g. docs: add roadmap." >&2
    exit 1
  fi
  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] Refusing branch commit on main. Use just commit-main for explicit direct-main work." >&2
    exit 1
  fi
  if git diff --cached --quiet; then
    echo "[ERROR] No staged changes. Run just stage-all or git add first." >&2
    exit 1
  fi
  commit_file="$(mktemp)"
  trap 'rm -f "$commit_file"' EXIT
  printf '%s\n' "$subject" > "$commit_file"
  if [[ -n "${CODEX_COMMIT_BODY:-}" ]]; then
    printf '\n%s\n' "$CODEX_COMMIT_BODY" >> "$commit_file"
  fi
  if [[ -n "${CODEX_THREAD_ID:-}" ]]; then
    printf '\nCodex-Thread: %s\nCodex-Branch: %s\n' "$CODEX_THREAD_ID" "$branch_name" >> "$commit_file"
  fi
  git commit -F "$commit_file"

commit-main msg:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" != "main" ]]; then
    echo "[ERROR] commit-main is only for explicit direct-main work." >&2
    exit 1
  fi
  subject="{{msg}}"
  if [[ -z "$subject" ]]; then
    echo "[ERROR] Provide a conventional subject, e.g. docs: add workflow harness." >&2
    exit 1
  fi
  if ! printf '%s\n' "$subject" | grep -Eq '^(feat|fix|docs|refactor|test|chore|build|ci|perf|revert)(\([[:alnum:]_.-]+\))?: .+'; then
    echo "[ERROR] Commit subject should follow conventional style." >&2
    exit 1
  fi
  if git diff --cached --quiet; then
    echo "[ERROR] No staged changes. Run just stage-all or git add first." >&2
    exit 1
  fi
  git commit -m "$subject"

push:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] Refusing to push main. Use just push-main for explicit direct-main work." >&2
    exit 1
  fi
  if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git push
  else
    git push -u origin "$branch_name"
  fi

push-main:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" != "main" ]]; then
    echo "[ERROR] push-main is only for explicit direct-main work." >&2
    exit 1
  fi
  git push

finish-branch:
  #!/usr/bin/env bash
  set -euo pipefail
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[ERROR] Working tree is dirty. Commit/stash changes before finishing the branch." >&2
    exit 1
  fi
  current_branch="$(git branch --show-current)"
  git fetch --prune origin
  if [[ "$current_branch" != "main" ]]; then
    git switch main
  fi
  git pull --ff-only origin main
  if [[ "$current_branch" != "main" ]] && git show-ref --verify --quiet "refs/heads/$current_branch"; then
    if git merge-base --is-ancestor "$current_branch" main; then
      git branch -d "$current_branch"
    else
      echo "[WARN] Kept local branch $current_branch because it is not merged into main." >&2
    fi
  fi
  git status --short --branch

pr-body-template path:
  #!/usr/bin/env bash
  set -euo pipefail
  output_path="{{path}}"
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] PR body templates are for topic branches, not main." >&2
    exit 1
  fi
  {
    if [[ -n "${CODEX_THREAD_ID:-}" ]]; then
      printf '%s\n' "Thread: $CODEX_THREAD_ID"
    fi
    printf '%s\n' "Branch: $branch_name" ""
    printf '%s\n\n' "Follow-up to #2."
    printf '%s\n\n' "## Queue Item"
    printf '%s\n\n' "- PR queue item: "
    printf '%s\n\n' "## Summary"
    printf '%s\n\n' "- "
    printf '%s\n\n' "## Scope"
    printf '%s\n' "- MLX runtime semantics unchanged."
    printf '%s\n' "- No WT103/main-machine/result/neo.csv changes."
    printf '%s\n\n' "- Trusted PyTorch parity path remains use_checkpoint=false."
    printf '%s\n\n' "## Verification"
    printf '%s\n' "- python3 -m pytest -q tests/test_mlx_reference_parity.py: "
    printf '%s\n' "- python3 -m pytest -q: "
    printf '%s\n' "- git diff --check: "
    printf '%s\n\n' "- GitHub Actions: "
    printf '%s\n\n' "## Risks / Follow-ups"
    printf '%s\n' "- "
  } > "$output_path"
  echo "[OK] Wrote PR body template to $output_path"

pr-create-from-file title body_file:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] PR creation helpers are for topic branches, not main." >&2
    exit 1
  fi
  title="{{title}}"
  body_file="{{body_file}}"
  if [[ ! -f "$body_file" ]]; then
    echo "[ERROR] PR body file not found: $body_file" >&2
    exit 1
  fi
  gh pr create --base main --title "$title" --body-file "$body_file"

pr-create-draft-from-file title body_file:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="$(git branch --show-current)"
  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] PR creation helpers are for topic branches, not main." >&2
    exit 1
  fi
  title="{{title}}"
  body_file="{{body_file}}"
  if [[ ! -f "$body_file" ]]; then
    echo "[ERROR] PR body file not found: $body_file" >&2
    exit 1
  fi
  gh pr create --draft --base main --title "$title" --body-file "$body_file"

prune-branches:
  #!/usr/bin/env bash
  set -euo pipefail
  git fetch --prune origin
  current_branch="$(git branch --show-current)"
  git branch --format='%(refname:short)' --merged main \
    | grep -v '^main$' \
    | grep -v "^${current_branch}$" \
    || true

delete-branch name:
  #!/usr/bin/env bash
  set -euo pipefail
  branch_name="{{name}}"
  if [[ -z "$branch_name" || "$branch_name" == "main" ]]; then
    echo "[ERROR] Refusing to delete branch: $branch_name" >&2
    exit 1
  fi
  current_branch="$(git branch --show-current)"
  if [[ "$current_branch" == "$branch_name" ]]; then
    git switch main
  fi
  git branch -d "$branch_name"
