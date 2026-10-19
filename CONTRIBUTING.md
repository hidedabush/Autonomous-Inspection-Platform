# Contributing to AIP

## Workflow

1. Pick a ticket on your team's Jira board and move it to **In Progress**.
2. Branch from `main`: `<JIRA-KEY>-short-description`, for example `GPU-12-sobel-kernel`.
3. Commit messages start with the ticket key: `GPU-12: add shared-memory Sobel kernel`. Jira links the commit automatically when the GitHub for Jira app is installed.
4. Open a pull request using the template. At least one reviewer, ideally from another team when the change touches an interface in `docs/interfaces.md`.
5. CI must pass. Squash-merge into `main`, then move the ticket to **Done**.

## Rules

- `main` always builds. A broken build is fixed before new work starts.
- Never commit datasets, model weights, videos or build folders. Put large files in the shared Google Drive and record the version in `data/README.md`.
- Changing an interface in `docs/interfaces.md` needs sign-off from every team that uses it.
- Big technical decisions get a short record in `docs/adr/` (copy `0000-template.md`).

## Code style

- C++17, formatted with `.clang-format` (run `clang-format -i` on changed files).
- Python formatted with `ruff format`, linted with `ruff check`.
- CUDA kernels: one kernel per `.cu` file where practical, with a matching CPU version in `src/cpu` for benchmarking.
