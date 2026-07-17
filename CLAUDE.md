# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Dataset Payload Hygiene for Agents

This repo's dataset (`data/`) contains **raw jailbreak / prompt-injection attack text** (hackaprompt, BIPIA, InjecAgent, struq, …). If that text enters an AI assistant's conversation context, provider safety classifiers can reject the API request (AUP error). These rules keep it out:

1. **Never Read/cat/print raw dataset rows.** Do not `Read` files under `data/`, and do not print `text`/`input`/`rendered_input`/`prompt` fields of dataset records. Work with **aggregates only**: counts, ratios, lengths, hashes, label/channel/source breakdowns.
2. **Use the safe inspector** instead of `head`/`jq`/pandas display:
   ```bash
   PYTHONPATH=. python scripts/inspect_samples.py data/splits/train.jsonl --limit 20
   PYTHONPATH=. python scripts/inspect_samples.py data/eval_proposal/eval.jsonl --summary
   ```
   It prints id/source/channel/label/family/length/sha256 — never the text.
3. **`scripts/demo_pipeline.py` is redacted by default.** `--show-payloads` prints raw attack text and is for plain human terminals only — never run it inside an AI-assistant session.
4. **Notebook cell outputs count as context.** Do not leave raw dataset text in saved `.ipynb` outputs (e.g. "inspect disagreements"-style cells). Clear such outputs before saving, or wrap prints with a hash/length redaction. Optional hardening: strip all outputs on commit with [nbstripout](https://github.com/kynan/nbstripout):
   ```bash
   pip install nbstripout && nbstripout --install   # writes .gitattributes filter
   ```
5. **If an AUP/safety error hits mid-session**, the offending content is stuck in the conversation history — `/clear` (or start a new session) rather than retrying. Persistent false positives can be reported to usersafety@anthropic.com.

## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

_Add your project-specific conventions here_
