# plan.md — Codebook-X Single Planning File

> **TEAM RULE:** Yeh single file hai jahan sab planning, discussion, proposals, feedback, aur sign-offs hote hain.
> - Nayi planning markdown files kabhi mat banao.
> - Agar kuch review karwana ho → yahan add karo, phir `slack.md` mein link karke @mention karo.
> - Agar codebase mein already kuch likha hai → directly file path + line number reference karo. Context assume mat karo.

---

## [ARCHIVED] Sprints 1–7 — Closed

Sprint 7 ✅ SHIPPED: S7.1 `pyproject.toml` renamed `codebookx`→`codebook` + `[tool.setuptools.packages.find]` added, v3.0.0 wheel built (`dist/codebook-3.0.0-py3-none-any.whl`). S7.2 `extract_ts_relations` + `resolve_ts_relative_imports` in `parser.py`, Phase 3 wired in `indexer.py`, 4/4 tests pass. PyPI publish blocked on Satvik API token. Commit `86c4686`. 
Sprint 6 ✅ SHIPPED: S6.1 chat disk save, S6.2 help text fix, S6.3 IMPORTS cross-file resolution. 0 bugs. Commit `c1b1573`. PR #1 merged.
Sprints 1–5 ✅ all closed. Full history in git log + claude-mem observations.

---

## Sprint 8 Backlog (Satvik to prioritize)

| # | Item | Priority | Notes |
|---|------|----------|-------|
| ~~S8.1~~ | ~~Publish v3.0.0 to PyPI~~ | ~~🔴 High~~ | ~~DONE — published as `codebookx` (name `codebook` taken). Live: pypi.org/project/codebookx/3.0.0/~~ |
| ~~S8.2~~ | ~~Fix flask hard-import in `codebookx/cli.py`~~ | ~~🔴 High~~ | ~~DONE — lazy import inside `view` handler. Commit `6e1fb12`. v3.0.1 live on PyPI.~~ |
| S8.3 | PEP 420 namespace packages | 🟢 Low | `resolve_ts_relative_imports` stops at dirs without `__init__.py` |
| S8.4 | `ask -c --resume` — load previous `chat_*.md` | 🟢 Low | Continue terminated chat session |

**S8.2 quick fix (Sanika to implement when Sprint 8 starts):**
```python
# codebookx/cli.py — lazy import instead of module-level
def run_webapp():
    from .webapp.app import run_server
    run_server()
```
