import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import questionary
from tqdm import tqdm

from .llm import check_server, stream_chat_completion
from .engine.indexer import Indexer
from .engine.parser import extract_snippets
from .engine.vendor.claude_mem_lite import generate_code_skeleton
from .engine.vendor.repomix_lite import pack_repo
from .engine.graph import KnowledgeGraph
from .prompts import PCTF_SYSTEM_PROMPT, AATD_SYSTEM_PROMPT, ENHANCE_SYSTEM_PROMPT, QA_SYSTEM_PROMPT

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You explain code like a friendly senior engineer talking to someone who just started learning to code. Think of it like explaining to a smart 7th grader — no jargon, no lectures, just clear and casual conversation.

YOUR RULES:
1. Always start with one sentence: what does this code actually DO? Plain English, no tech words.
2. Then walk through the important parts step by step. For each part, say what it does in everyday language first, then mention the technical term in parentheses if needed.
3. Use a real-world analogy whenever you can. The best analogies are things anyone would recognize — kitchens, libraries, phone calls, to-do lists.
4. Keep it short. 6 to 10 lines max. No padding, no filler.
5. Never use a technical word without explaining it first. If you must use one, explain it right there in plain English.
6. Write like you're texting a smart friend — casual, warm, and direct. Not a textbook. Not a lecture.

TONE: Friendly, clear, conversational. Like a helpful older sibling who happens to know how to code."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "File: math.py | Snippet: add_numbers\n\n```python\ndef add_numbers(a, b):\n    result = a + b\n    return result\n```",
    },
    {
        "role": "assistant",
        "content": "This takes two numbers, adds them together, and gives you back the answer.\n\nHere's what happens inside:\n- `add_numbers(a, b)` — the function takes two numbers as input. Think of `a` and `b` as two blank boxes you fill in when you use it.\n- `result = a + b` — it adds them and stores the answer in a variable called `result`. A variable is just a labeled box that holds a value.\n- `return result` — it hands the answer back to whoever called it. `return` is basically saying \"here's your answer, I'm done.\"\n\nReal-world version: it's a calculator. You punch in two numbers, it spits out the sum.",
    },
]

def load_config(args, root: Path) -> dict:
    defaults = {
        "url": "http://localhost:1234/v1",
        "model": None,
        "output": "CODEBASE_BOOK.md",
        "skip_extensions": [],
        "skip_dirs": ["node_modules", ".next", "__pycache__", ".git", "dist", "build", ".beads"],
        "prompt_lang": "simple English",
        "max_context": 30000,
    }
    config_file = root / "codebook.toml"
    file_config = {}
    if config_file.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        try:
            with open(config_file, "rb") as f:
                file_config = tomllib.load(f)
        except Exception as e:
            print(f"Warning: Could not read {config_file}: {e}")

    config = {**defaults, **file_config}
    config["url"] = os.environ.get("CODEBOOKX_URL") or config["url"]
    if hasattr(args, 'url') and args.url: config["url"] = args.url
    if hasattr(args, 'model') and args.model: config["model"] = args.model
    if hasattr(args, 'output') and args.output: config["output"] = args.output
    if hasattr(args, 'prompt_lang') and args.prompt_lang: config["prompt_lang"] = args.prompt_lang
    return config

# ──────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def get_all_extensions(root: Path, skip_dirs: set[str]) -> dict[str, list[str]]:
    import os
    code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".cpp", ".cc", ".c", ".h", ".cs", ".rb", ".php", ".swift"}
    doc_extensions = {".md", ".markdown", ".rst", ".txt"}
    config_extensions = {".json", ".yaml", ".yml", ".xml", ".toml"}
    categories = {"CODE": set(), "DOCS": set(), "CONFIG": set(), "OTHER": set()}
    binary_extensions = {".exe", ".dll", ".so", ".zip", ".tar", ".gz", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg"}

    try:
        for r, dirs, filenames in os.walk(root):
            # Prune skip_dirs in-place
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in filenames:
                ext = Path(f).suffix.lower()
                if not ext or ext in binary_extensions: continue
                if ext in code_extensions: categories["CODE"].add(ext)
                elif ext in doc_extensions: categories["DOCS"].add(ext)
                elif ext in config_extensions: categories["CONFIG"].add(ext)
                else: categories["OTHER"].add(ext)
    except Exception as e: print(f"Error: {e}")
    return {k: sorted(list(v)) for k, v in categories.items() if v}

def run_setup_wizard(root: Path, config: dict) -> dict:
    print("\n📚 Codebase-X — Setup\n")
    ext_by_category = get_all_extensions(root, set(config["skip_dirs"]))
    choices = []
    for category, exts in ext_by_category.items():
        for ext in exts: choices.append(f"{ext} ({category})")
    
    if not choices:
        return config
        
    skip_choices = questionary.checkbox(
        "Which file types should be SKIPPED?",
        choices=choices
    ).ask()
    
    config["skip_extensions"] = [c.split()[0] for c in skip_choices] if skip_choices else []
    config["prompt_lang"] = "simple English"
    return config

def get_files(root: Path, skip_extensions: list[str], skip_dirs: set[str], single: Optional[str] = None) -> list[Path]:
    if single:
        p = Path(single)
        return [p] if p.exists() else []
    
    import os
    skip_ext_lower = set(ext.lower() for ext in skip_extensions)
    files = []
    try:
        for r, dirs, filenames in os.walk(root):
            # Prune skip_dirs in-place
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in filenames:
                file_path = Path(r) / f
                if file_path.suffix.lower() in skip_ext_lower:
                    continue
                files.append(file_path)
    except Exception as e: print(f"Error: {e}")
    return sorted(files)

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT (LEGACY)
# ──────────────────────────────────────────────────────────────────────────────

def already_done(name: str, path: str, output_file: Path) -> bool:
    if not output_file.exists(): return False
    content = output_file.read_text(errors="ignore")
    return f"`{name}` — {path}" in content

def init_book(output_file: Path, root: Path, model: str) -> None:
    if output_file.exists(): return
    header = f"# Codebase Book\n\nAuto-generated by [Codebase-X].\nModel: {model} | Generated: {datetime.now()}\n\n---\n\n"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(header)

def append_to_book(file_path: str, snippet: dict, explanation: str, output_file: Path) -> None:
    entry = f"## `{snippet['name']}` — {file_path}\n\n**Lines {snippet['start']}–{snippet['end']}**\n\n{explanation}\n\n---\n\n"
    with open(output_file, "a", encoding="utf-8") as f: f.write(entry)

# ──────────────────────────────────────────────────────────────────────────────
# COMMAND RUNNERS
# ──────────────────────────────────────────────────────────────────────────────

def run_ask(args):
    """Run Q&A about the codebase (KG-backed)."""
    root = Path(args.target_dir or ".").resolve()
    config = load_config(args, root)
    
    question = args.question
    if not question.strip():
        sys.exit("⚠️ Please provide a question. Usage: codebookx ask 'What does X do?'")
    
    teardown_path = root / "CODEBASE_TEARDOWN.md"
    if not teardown_path.exists():
        sys.exit("⚠️ No teardown found at CODEBASE_TEARDOWN.md. Please run 'codebookx analyze' first.")
    
    context = teardown_path.read_text(errors="ignore")
    if not context.strip():
        sys.exit("⚠️ Teardown file is empty. Run 'codebookx analyze' first.")
    
    # NEW: KG-backed symbol context (graceful fallback if KG missing)
    db_path = root / ".codebook_cache.db"
    symbol_context = ""
    if db_path.exists():
        try:
            kg = KnowledgeGraph(str(db_path))
            symbol_context = kg.get_symbol_context_for_question(question)
        except Exception:
            pass  # Graceful fallback — teardown-only is fine

    if symbol_context:
        full_context = f"# Codebase Teardown\n{context}\n\n# Symbols\n{symbol_context}"
    else:
        full_context = context  # P0.6 fallback — teardown only

    is_running, model = check_server(config["url"])
    if not is_running:
        print(f"⚠️ LLM server not running at {config['url']}. Start your server and try again.")
        return
    
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Codebase Context:\n{full_context}\n\nQuestion: {question}"}
    ]
    answer = stream_chat_completion(config["url"], model or "unknown", messages)
    print(f"\n{answer}")

    # S5.2: Save Q&A to log file
    ask_dir = args.dir or os.environ.get("CODEBOOK_ASK_DIR") or str(root / "ask_history")
    try:
        Path(ask_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        sys.exit(f"⚠️ {ask_dir} exists but is not a directory.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(ask_dir) / f"ask_{timestamp}.md"
    log_file.write_text(f"# Question\n\n{question}\n\n# Answer\n\n{answer}")
    print(f"💾 Saved to {log_file}")

def run_ask_chat(args):
    """Interactive chat mode — multi-turn with context carry-over."""
    root = Path(args.target_dir or ".").resolve()
    config = load_config(args, root)

    ask_dir = getattr(args, "dir", None) or os.environ.get("CODEBOOK_ASK_DIR") or str(root / "ask_history")
    try:
        Path(ask_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        sys.exit(f"⚠️ {ask_dir} exists but is not a directory.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(ask_dir) / f"chat_{timestamp}.md"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"# Chat Session — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root: {root}\n\n")

    question = args.question
    if not question.strip():
        sys.exit("⚠️ Please provide a question. Usage: codebookx ask -c 'What does X do?'")

    teardown_path = root / "CODEBASE_TEARDOWN.md"
    if not teardown_path.exists():
        sys.exit("⚠️ No teardown found. Please run 'codebookx analyze' first.")

    context = teardown_path.read_text(errors="ignore")
    if not context.strip():
        sys.exit("⚠️ Teardown is empty.")

    # KG context (graceful fallback)
    db_path = root / ".codebook_cache.db"
    symbol_context = ""
    if db_path.exists():
        try:
            kg = KnowledgeGraph(str(db_path))
            symbol_context = kg.get_symbol_context_for_question(question)
        except Exception:
            pass

    if symbol_context:
        full_context = f"# Codebase Teardown\n{context}\n\n# Symbols\n{symbol_context}"
    else:
        full_context = context

    is_running, model = check_server(config["url"])
    if not is_running:
        print(f"⚠️ LLM server not running at {config['url']}. Start your server and try again.")
        return

    # First turn: inject full context + question
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Codebase Context:\n{full_context}\n\nQuestion: {question}"}
    ]
    print(f"\n🤖 {model or 'unknown'}")
    answer = stream_chat_completion(config["url"], model or "unknown", messages)
    print(f"\n{answer}")
    messages.append({"role": "assistant", "content": answer})

    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"## Question\n\n{question}\n\n## Answer\n\n{answer}\n\n")

    # Follow-up turns: just the new question (full history in messages)
    try:
        while True:
            user_input = input("\n💬 prompt> ").strip()
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                with log_file.open("a", encoding="utf-8") as f:
                    f.write("---\n*Session ended.*\n")
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            answer = stream_chat_completion(config["url"], model or "unknown", messages)
            print(f"\n{answer}")
            messages.append({"role": "assistant", "content": answer})
            
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"## Question\n\n{user_input}\n\n## Answer\n\n{answer}\n\n")

    except (KeyboardInterrupt, EOFError):
        with log_file.open("a", encoding="utf-8") as f:
            f.write("---\n*Session ended (interrupted).*\n")
        print("\n👋 Goodbye!")

def run_generate(args):
    """The original Codebook documentation generation logic."""
    root = Path(args.target_dir or ".").resolve()
    config = load_config(args, root)
    print("📚 Codebase-X — Checking LLM server...")
    is_running, detected_model = check_server(config["url"])
    if not is_running:
        print(f"⚠️ LLM server not running at {config['url']}. Start your server and try again.")
        return
    if not config["model"]:
        config["model"] = detected_model or "unknown"
    
    print(f"✓ Server ready. Model: {config['model']}\n")
    config = run_setup_wizard(root, config)
    
    single_file = args.file if hasattr(args, 'file') else None
    files = get_files(root, config["skip_extensions"], set(config["skip_dirs"]), single_file)
    
    total_functions = 0
    file_snippets = {}
    for file_path in files:
        try:
            source = file_path.read_text(errors="ignore")
            snippets = extract_snippets(file_path, source)
            if snippets:
                file_snippets[file_path] = snippets
                total_functions += len(snippets)
        except Exception as e: print(f"Error: {e}")

    if total_functions == 0:
        print("No functions found."); return

    output_file = root / config["output"]
    init_book(output_file, root, config["model"])
    print(f"Found {len(files)} files with {total_functions} functions\n")

    with tqdm(total=total_functions, file=sys.stderr, ncols=80, colour="green") as pbar:
        for file_path, snippets in file_snippets.items():
            rel_path = file_path.relative_to(root)
            for snippet in snippets:
                if already_done(snippet["name"], str(rel_path), output_file):
                    pbar.update(1); continue
                
                print(f"\n  {snippet['name']} ({rel_path}:{snippet['start']}-{snippet['end']})")
                print("  " + "─" * 60)
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *FEW_SHOT_EXAMPLES,
                    {"role": "user", "content": f"File: {rel_path} | Snippet: {snippet['name']}\n\n```\n{snippet['code']}\n```"}
                ]
                explanation = stream_chat_completion(config["url"], config["model"], messages)
                append_to_book(str(rel_path), snippet, explanation, output_file)
                pbar.update(1)
    
    print(f"\n✅ Done! Generated: {output_file}")

def run_analyze(args):
    """Run PCTF Codebase Teardown."""
    root = Path(args.target_dir or ".").resolve()
    db_path = str(root / ".codebook_cache.db")
    indexer = Indexer(str(root), db_path)
    indexer.index(force=args.force)
    config = load_config(args, root)
    is_running, model = check_server(config["url"])
    if not is_running: return
    print("\n🔍 Generating Codebase Teardown...")
    readme_content = ""
    readme_path = root / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(errors="ignore")[:2000]
    folder_tree = [f"- {d.name}/" for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
    context = f"README:\n{readme_content}\n\nFolder Structure:\n" + "\n".join(folder_tree)
    messages = [{"role": "system", "content": PCTF_SYSTEM_PROMPT}, {"role": "user", "content": f"Analyze this codebase:\n\n{context}"}]
    report = stream_chat_completion(config["url"], model or "unknown", messages)
    output_path = root / "CODEBASE_TEARDOWN.md"
    output_path.write_text(report)
    print(f"\n✅ Teardown saved to {output_path}")

def run_decompose(args):
    """Run AATD Feature Decomposition."""
    root = Path(args.target_dir or ".").resolve()
    config = load_config(args, root)
    is_running, model = check_server(config["url"])
    if not is_running:
        print(f"⚠️ LLM server not running at {config['url']}. Start your server and try again.")
        return
    
    context_source = "Teardown"
    teardown_path = root / "CODEBASE_TEARDOWN.md"
    teardown_context = ""
    if teardown_path.exists():
        teardown_context = teardown_path.read_text(errors="ignore")

    # Wire --deep flag
    if args.deep:
        print("📦 Packing repository for deep context...")
        deep_context = pack_repo(root)
        if len(deep_context) > 100000:
            print("⚠️ Deep context too large (>100k chars). Falling back to teardown context.")
            context_source = "Teardown (fallback: deep context too large)"
        else:
            teardown_context = deep_context
            context_source = "Deep Repomix Pack"

    if not teardown_context and not args.deep:
        sys.exit("❌ No codebase context found. Please run 'codebook analyze' first to generate a teardown, or use the '--deep' flag.")
        
    if args.deep and not teardown_context and context_source == "Teardown (fallback: deep context too large)":
        sys.exit("❌ Context too large for deep pack and no teardown found. Please run 'codebook analyze' first.")

    print(f"\n🔨 Decomposing feature: {args.feature} (Context: {context_source})")
    messages = [{"role": "system", "content": AATD_SYSTEM_PROMPT}, {"role": "user", "content": f"Codebase Context:\n{teardown_context}\n\nFeature Request: {args.feature}"}]
    tasks_json = stream_chat_completion(config["url"], model or "unknown", messages)
    output_path = root / "FEATURE_TASKS.md"
    output_path.write_text(f"# Feature Tasks: {args.feature}\n\n{tasks_json}")
    print(f"\n✅ Tasks saved to {output_path}")

def run_enhance(args):
    """Run Prompt Enhancement (KG-backed)."""
    root = Path(args.target_dir or ".").resolve()
    config = load_config(args, root)

    db_path = root / ".codebook_cache.db"
    if not db_path.exists():
        sys.exit("⚠️ No Knowledge Graph found. Please run 'codebookx analyze' first to generate codebase context.")

    is_running, model = check_server(config["url"])
    if not is_running:
        print(f"⚠️ LLM server not running at {config['url']}. Start your server and try again.")
        return
    
    print(f"\n✨ Enhancing prompt for: {args.feature}")
    kg = KnowledgeGraph(str(db_path))
    context = kg.get_all_symbol_context()
    
    if not context.strip():
        sys.exit("⚠️ Knowledge Graph is empty. Run 'codebookx analyze' first.")
            
    messages = [
        {"role": "system", "content": ENHANCE_SYSTEM_PROMPT}, 
        {"role": "user", "content": f"Codebase Structure:\n{context}\n\nUser Request: {args.feature}\n\nGenerate an enhanced prompt."}
    ]
    enhanced_prompt = stream_chat_completion(config["url"], model or "unknown", messages)
    output_path = root / "ENHANCED_PROMPT.txt"
    output_path.write_text(enhanced_prompt)
    print(f"\n✅ Enhanced prompt saved to {output_path}")
