# ADR-0001: One repository, CMake, Jira, Drive for large files

- Date: 2026-10-04
- Status: proposed
- Deciders: team lead

## Context
Four sub-teams build parts that must integrate weekly.

## Decision
- One GitHub repository for C++, CUDA, ML, firmware and hardware files.
- CMake build; CUDA optional so CI and non-GPU machines still build.
- Jira for tasks; branch names carry the Jira key.
- Datasets, weights and videos live in Google Drive, not Git.

## Consequences
Integration breaks show up early; large-file growth in Git is avoided.
