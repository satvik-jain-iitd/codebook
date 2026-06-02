## AUTOMATIC GUARD — DO NOT REMOVE

Run this check before EVERY operation:
```bash
python3 -c "
import os
cwd = os.path.realpath('.')
expected = os.path.realpath(os.path.expanduser('~/Codebook/codebook'))
assert cwd == expected or cwd.startswith(expected + '/'), f'WRONG PROJECT: cwd={cwd}, expected prefix={expected}'
print(f'✅ Project verified: {cwd}')
"
```

# Gemini CLI Instructions — Codebook Project

## CRITICAL: Project Directory

You are working on the **Codebook** project ONLY.

**Working directory:** `~/Codebook/codebook`

If your current workspace shows any other directory (e.g., `sync-resume-engine`, `linkright`, or any other project), you are in the WRONG project. Stop immediately and alert the user.

**Never edit files outside of `~/Codebook/codebook`.**

## Project Context

This is `codebookx` — a Python CLI tool that builds a Knowledge Graph of codebases and enables LLM-powered Q&A. See `CLAUDE.md` for full project details and team workflow.

## Key Files

- `codebookx/` — main package
- `codebookx/cli.py` — CLI entry point
- `codebookx/core.py` — command runners
- `codebookx/engine/` — KG indexer, parser, graph
- `codebookx/webapp/app.py` — Flask web UI
- `plan.md` — all planning, proposals, sign-offs (append only, never delete)
- `slack.md` — team notifications (Romanized Hindi)
