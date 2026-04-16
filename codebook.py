#!/usr/bin/env python3
"""
codebook — Generate documentation for any codebase using local LM Studio.

Scans Python/TypeScript files, extracts functions, sends each to LM Studio,
generates a human-readable CODEBASE_BOOK.md with real-time streaming output
and progress tracking.

Usage:
    python codebook.py                          # scan current directory
    python codebook.py /path/to/repo            # scan specific path
    python codebook.py --url http://192.168.1.8:1234/v1  # custom LM Studio URL
    python codebook.py --check                  # verify LM Studio is running
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import questionary
import requests
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_config(args: argparse.Namespace, root: Path) -> dict:
    """
    Merge configuration from CLI args, optional codebook.toml, and defaults.
    CLI args always take precedence.
    """
    defaults = {
        "url": "http://localhost:1234/v1",
        "model": None,  # auto-detect from server
        "output": "CODEBASE_BOOK.md",
        "skip_extensions": [],  # will be set by wizard
        "skip_dirs": ["node_modules", ".next", "__pycache__", ".git", "dist", "build", ".beads"],
        "prompt_lang": "Romanized Hindi",
    }

    # Try to load codebook.toml
    config_file = root / "codebook.toml"
    file_config = {}
    if config_file.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # fallback for Python < 3.11

        try:
            with open(config_file, "rb") as f:
                file_config = tomllib.load(f)
        except Exception as e:
            print(f"Warning: Could not read {config_file}: {e}")

    # Merge: defaults < file_config < CLI args
    config = {**defaults, **file_config}

    if args.url:
        config["url"] = args.url
    if args.model:
        config["model"] = args.model
    if args.output:
        config["output"] = args.output
    if args.skip_dirs:
        # Filter out empty strings
        skip_dirs_list = [d.strip() for d in args.skip_dirs.split(",") if d.strip()]
        config["skip_dirs"] = skip_dirs_list if skip_dirs_list else []
    if args.prompt_lang:
        config["prompt_lang"] = args.prompt_lang

    return config


# ──────────────────────────────────────────────────────────────────────────────
# CODE PARSING
# ──────────────────────────────────────────────────────────────────────────────


def extract_python_functions(source: str) -> list[dict]:
    """Extract functions and classes from Python source using AST."""
    snippets = []
    try:
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                snippets.append(
                    {
                        "name": node.name,
                        "start": node.lineno,
                        "end": node.end_lineno,
                        "code": "\n".join(lines[start:end]),
                        "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    }
                )
    except SyntaxError:
        pass
    return snippets


def extract_ts_functions(source: str) -> list[dict]:
    """Extract functions from TypeScript/TSX source using regex (40-line window)."""
    snippets = []
    lines = source.splitlines()
    pattern = re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*[(<]"
        r"|^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(",
        re.MULTILINE,
    )
    for match in pattern.finditer(source):
        name = match.group(1) or match.group(2)
        start_line = source[: match.start()].count("\n")
        end_line = min(start_line + 40, len(lines))
        snippets.append(
            {
                "name": name,
                "start": start_line + 1,
                "end": end_line,
                "code": "\n".join(lines[start_line:end_line]),
                "type": "function",
            }
        )
    return snippets


def extract_snippets(file_path: Path, source: str) -> list[dict]:
    """Dispatch to appropriate extractor based on file type."""
    if file_path.suffix == ".py":
        return extract_python_functions(source)
    elif file_path.suffix in (".ts", ".tsx"):
        return extract_ts_functions(source)
    # For other file types, try to extract simple function-like patterns
    return extract_generic_functions(source, file_path.suffix)


def extract_generic_functions(source: str, ext: str) -> list[dict]:
    """Extract function-like patterns from any text file (generic fallback)."""
    snippets = []
    lines = source.splitlines()

    # Simple pattern: look for lines with "function", "def", "func", etc.
    patterns = [
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",  # JS/TS functions
        r"^def\s+(\w+)\s*\(",  # Python defs
        r"^class\s+(\w+)",  # Class definitions
        r"^pub\s+(?:async\s+)?fn\s+(\w+)",  # Rust functions
        r"^func\s+(?:\w+\s+)?(\w+)",  # Go functions
    ]

    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.MULTILINE)
        for match in pattern.finditer(source):
            name = match.group(1)
            start_line = source[: match.start()].count("\n")
            end_line = min(start_line + 30, len(lines))
            snippets.append(
                {
                    "name": name,
                    "start": start_line + 1,
                    "end": end_line,
                    "code": "\n".join(lines[start_line:end_line]),
                    "type": "function",
                }
            )

    return snippets


# ──────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────


def get_all_extensions(root: Path, skip_dirs: set[str]) -> dict[str, list[str]]:
    """
    Scan directory and find all unique file extensions.
    Groups them by category for display.
    Returns dict with categories as keys and list of extensions as values.
    """
    # Extension categories
    code_extensions = {
        ".py", ".ts", ".tsx", ".js", ".jsx",  # Python, TypeScript, JavaScript
        ".go", ".rs", ".java", ".kt", ".scala",  # Go, Rust, Java, Kotlin, Scala
        ".cpp", ".cc", ".c", ".h", ".hpp",  # C/C++
        ".cs", ".vb", ".fs", ".fsx",  # C#, VB.NET, F#
        ".rb", ".php", ".swift", ".m", ".mm",  # Ruby, PHP, Swift, Objective-C
        ".sh", ".bash", ".zsh", ".ps1",  # Shell scripts
        ".sql", ".r", ".lua", ".pl", ".pm",  # SQL, R, Lua, Perl
    }

    doc_extensions = {
        ".md", ".markdown", ".rst", ".txt", ".doc", ".docx",  # Docs
    }

    config_extensions = {
        ".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg",  # Config
    }

    image_extensions = {
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp",  # Images
    }

    # True binaries to never scan
    binary_extensions = {
        ".exe", ".dll", ".so", ".zip", ".tar", ".gz", ".pdf",  # Binary
    }

    categories = {
        "CODE": set(),
        "DOCS": set(),
        "CONFIG": set(),
        "IMAGES": set(),
        "OTHER": set(),
    }

    try:
        for f in root.rglob("*"):
            if f.is_file() and not any(skip in f.parts for skip in skip_dirs):
                ext = f.suffix.lower()
                if not ext or ext in binary_extensions:
                    continue

                if ext in code_extensions:
                    categories["CODE"].add(ext)
                elif ext in doc_extensions:
                    categories["DOCS"].add(ext)
                elif ext in config_extensions:
                    categories["CONFIG"].add(ext)
                elif ext in image_extensions:
                    categories["IMAGES"].add(ext)
                else:
                    categories["OTHER"].add(ext)
    except (PermissionError, RecursionError):
        pass

    # Convert sets to sorted lists
    result = {k: sorted(list(v)) for k, v in categories.items() if v}
    return result if result else {"CODE": [".py", ".ts", ".tsx", ".js"]}


def run_setup_wizard(root: Path, config: dict) -> dict:
    """
    Interactive setup wizard: choose which extensions to skip, set language.
    Returns updated config with skip_extensions.
    """
    print("\n📚 codebook — Setup\n")

    # Scan for all extensions
    ext_by_category = get_all_extensions(root, set(config["skip_dirs"]))

    # Flatten for questionary
    all_extensions = []
    for category, exts in ext_by_category.items():
        for ext in exts:
            all_extensions.append((ext, category))

    # Build checkbox choices
    choices = []
    for ext, category in sorted(all_extensions):
        choices.append(f"{ext} ({category})")

    # Ask which to skip (default: none)
    print(f"Found {len(all_extensions)} file types.\n")
    skip_choices = questionary.checkbox(
        "Which file types should be SKIPPED? (Space to select, Enter to confirm)",
        choices=choices,
        validate=lambda x: True,  # Allow empty selection (skip nothing)
    ).ask()

    # Extract extensions from choices
    skip_extensions = []
    if skip_choices:
        skip_extensions = [choice.split()[0] for choice in skip_choices]

    # Language is fixed to Romanized Hindi
    language = "Romanized Hindi"

    # Show confirmation
    skip_set = set(ext.lower() for ext in skip_extensions)
    will_scan = [ext for ext, _ in sorted(all_extensions) if ext.lower() not in skip_set]
    will_scan_str = ", ".join(will_scan) if will_scan else "(nothing)"
    skip_exts_str = ", ".join(skip_extensions) if skip_extensions else "(none)"

    print("\n" + "─" * 50)
    print(f"  Will scan:   {will_scan_str}")
    print(f"  Will skip:   {skip_exts_str}")
    print(f"  Language:    {language}")
    print(f"  Output:      {config['output']}")
    print("─" * 50 + "\n")

    confirm = questionary.confirm("Looks good?").ask()
    if not confirm:
        print("Cancelled.")
        sys.exit(0)

    # Update config
    config["skip_extensions"] = skip_extensions
    config["prompt_lang"] = language

    return config


def get_files(
    root: Path, skip_extensions: list[str], skip_dirs: set[str], single: Optional[str] = None
) -> list[Path]:
    """Recursively discover files, excluding skip_dirs and skip_extensions."""
    if single:
        return [Path(single)]

    skip_ext_lower = set(ext.lower() for ext in skip_extensions)
    files = []
    try:
        for f in root.rglob("*"):
            if f.is_file():
                # Check if any skip_dir component is in the path
                if any(skip in f.parts for skip in skip_dirs):
                    continue
                # Skip if extension is in skip list
                if f.suffix.lower() in skip_ext_lower:
                    continue
                files.append(f)
    except (PermissionError, RecursionError):
        pass

    return sorted(files)


# ──────────────────────────────────────────────────────────────────────────────
# LM STUDIO COMMUNICATION
# ──────────────────────────────────────────────────────────────────────────────


def check_server(url: str) -> tuple[bool, Optional[str]]:
    """Check if LM Studio is running and return (is_running, model_name)."""
    try:
        r = requests.get(f"{url.rstrip('/')}/models", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            if models:
                model_name = models[0].get("id", "unknown")
                return True, model_name
            return True, None
    except Exception:
        pass
    return False, None


def strip_think_tokens(text: str) -> str:
    """Remove <think>...</think> blocks (used by reasoning models like DeepSeek-R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


SYSTEM_PROMPT = """Tu ek senior software engineer hai jo apne junior dost ko code explain kar raha hai — bilkul waise jaise chai pe casually baat karte hain.

TERE RULES:
1. Sirf Romanized Hindi mein likho. English sentences bilkul mat likho.
   - Theek hai: variable names, function names, aur code snippets English mein
   - Galat hai: "This function checks if the user is logged in" jaisi poori English line
2. Pehle ek line mein batao — YEH KYA KARTA HAI
3. Phir step-by-step explain karo — har important line ka matlab
4. Technical terms seedhe mat thoko — pehle simple shabdon mein samjhao, phir term batao
   - Example: "return matlab — yeh value bahar bhejo, kaam ho gaya"
5. Jahan possible ho, real-life analogy do
6. Max 8-10 lines mein raho. Zyada bakwaas mat karo.

TONE: Dost jaise, casual, encouraging. Koi lecture nahi."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """File: auth.py | Lines: 1-4 | Function: add_numbers

```python
def add_numbers(a, b):
    result = a + b
    return result
```""",
    },
    {
        "role": "assistant",
        "content": """YEH KYA KARTA HAI: Do numbers ko jodta hai aur answer deta hai.

KAISE KAAM KARTA HAI:
- `add_numbers` function do cheezein maangta hai — `a` aur `b` (yeh do numbers hain jo tum doge)
- `result = a + b` — dono ko jodta hai aur ek variable mein rakhta hai (`result` matlab — answer ka dabba)
- `return result` — woh answer bahar bhejta hai (`return` matlab — "le bhai, yeh raha jawab")

ANALOGY: Calculator pe 3 + 5 press karo — exactly wahi kaam.""",
    },
    {
        "role": "user",
        "content": """File: users.py | Lines: 10-18 | Function: find_active_users

```python
def find_active_users(users):
    active = []
    for user in users:
        if user['is_active']:
            active.append(user)
    return active
```""",
    },
    {
        "role": "assistant",
        "content": """YEH KYA KARTA HAI: Ek badi list mein se sirf woh users nikalti hai jo abhi active hain.

KAISE KAAM KARTA HAI:
- `users` — yeh input hai, matlab saare users ki list (active + inactive dono milaake)
- `active = []` — ek khali dabba banaya jisme sirf active wale rakhenge
- `for user in users:` — ek ek karke har user ko check kar raha hai (jaise attendance lena)
- `if user['is_active']:` — pooch raha hai — "yeh active hai?" Agar haan, toh dabba mein daalo
- `return active` — filtering ke baad jo bache, unki list bahar bhejo

ANALOGY: Class ki attendance sheet se sirf present logo ke naam nikalana.""",
    },
    {
        "role": "user",
        "content": """File: api.py | Lines: 5-22 | Function: fetch_data

```python
def fetch_data(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
```""",
    },
    {
        "role": "assistant",
        "content": """YEH KYA KARTA HAI: Kisi bhi URL se data fetch karta hai — aur agar failure aaye toh dobara try karta hai.

KAISE KAAM KARTA HAI:
- `url` — woh address jahan se data lana hai; `retries=3` — kitni baar try karna hai (default: 3)
- `for attempt in range(retries):` — 3 baar try karne ka loop (`attempt` — yeh kaunwi baar hai, 0 se start)
- `try:` — yeh block keh raha hai "koshish karo, agar kuch toot jaaye toh sambhal lenge"
- `requests.get(url, timeout=10)` — URL pe request bheji, 10 seconds mein jawab nahi aaya toh fail
- `response.raise_for_status()` — agar server ne error bheja (404, 500 etc.) toh shor macha do
- `return response.json()` — data successfully mila, JSON format mein bahar bhejo
- `except requests.RequestException as e:` — kuch bhi galat hua? Yahan pakad lo
- `if attempt == retries - 1: raise` — agar yeh aakhri try thi, toh error aage bhejo (haath khade karo)
- `time.sleep(2 ** attempt)` — dobara try karne se pehle thoda ruko — 1s, 2s, 4s (har baar double)

ANALOGY: Phone pe kisi ko call karo — nahi uthaya toh 3 baar try karo, har baar thoda zyada wait karo.""",
    },
]


def annotate_streaming(
    file_path: str, snippet: dict, config: dict, pbar: tqdm
) -> str:
    """
    Send snippet to LM Studio and stream the response live to terminal.
    Returns the full explanation text (with <think> tokens stripped).
    """
    user_message = (
        f"File: {file_path} | Lines: {snippet['start']}-{snippet['end']} "
        f"| Function/Class: {snippet['name']}\n\n"
        f"```\n{snippet['code']}\n```"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message},
    ]

    try:
        response = requests.post(
            f"{config['url'].rstrip('/')}/chat/completions",
            json={
                "model": config["model"],
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 800,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_text = ""
        content_type = response.headers.get("content-type", "")

        # Handle streaming response
        if "application/json" in content_type:
            # Non-streaming JSON response
            data = response.json()
            full_text = data["choices"][0]["message"]["content"]
        else:
            # Streaming SSE response
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_text += delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        print()  # newline after streaming
        pbar.update(1)
        return strip_think_tokens(full_text)

    except requests.exceptions.RequestException as e:
        tqdm.write(f"  [Error: {e}]")
        pbar.update(1)
        return f"[Error: {e}]"


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────


def already_done(name: str, path: str, output_file: Path) -> bool:
    """Check if function is already documented."""
    if not output_file.exists():
        return False
    content = output_file.read_text(errors="ignore")
    # Check for both the function name and file path to avoid false positives
    return f"`{name}` — {path}" in content


def init_book(output_file: Path, root: Path, model: str) -> None:
    """Create or reset the CODEBASE_BOOK.md header."""
    if output_file.exists():
        return  # Don't overwrite, append only

    header = f"""# Codebase Book

Auto-generated by [codebook](https://github.com/satvikjain/codebook).
Model: {model} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(header)


def append_to_book(file_path: str, snippet: dict, explanation: str, output_file: Path) -> None:
    """Append function documentation to the book."""
    entry = f"""## `{snippet['name']}` — {file_path}

**Lines {snippet['start']}–{snippet['end']}**

{explanation}

---

"""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(entry)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for any codebase using local LM Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python codebook.py                           # scan current directory
  python codebook.py /path/to/repo             # scan specific repo
  python codebook.py --check                   # verify LM Studio is running
  python codebook.py --file src/auth.py        # single file only
  python codebook.py --url http://192.168.1.8:1234/v1  # custom LM URL
        """,
    )

    parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan (default: current)")
    parser.add_argument("--url", help="LM Studio base URL (default: http://localhost:1234/v1)")
    parser.add_argument("--model", help="Model name (auto-detect if not specified)")
    parser.add_argument("--output", help="Output file path (default: CODEBASE_BOOK.md)")
    parser.add_argument("--skip-dirs", help="Comma-separated dirs to skip")
    parser.add_argument(
        "--prompt-lang",
        help="Language/style for explanations (default: plain English)",
    )
    parser.add_argument("--file", help="Annotate single file only")
    parser.add_argument("--check", action="store_true", help="Check server status and exit")

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    root = Path(args.target_dir).resolve()
    if not root.exists():
        print(f"Error: Directory not found: {root}")
        sys.exit(1)

    config = load_config(args, root)

    # Health check
    print("📚 codebook — Checking LM Studio...")
    is_running, detected_model = check_server(config["url"])

    if not is_running:
        print(f"Error: LM Studio not running at {config['url']}")
        print("\nOpen LM Studio → Developer tab → Start Server")
        sys.exit(1)

    # Auto-detect model if not specified
    if not config["model"]:
        if detected_model:
            config["model"] = detected_model
        else:
            print("Error: Could not detect model. Specify with --model")
            sys.exit(1)

    print(f"✓ Server ready. Model: {config['model']}\n")

    # Exit early if --check
    if args.check:
        sys.exit(0)

    # Run setup wizard
    config = run_setup_wizard(root, config)

    # Discover files
    print(f"\nScanning {root}...")
    # If --file is provided, resolve it relative to root if it's not absolute
    single_file = None
    if args.file:
        single_path = Path(args.file)
        if not single_path.is_absolute():
            single_path = root / single_path
        single_file = str(single_path.resolve())

    files = get_files(root, config["skip_extensions"], set(config["skip_dirs"]), single_file)

    if not files:
        print("No files found.")
        sys.exit(0)

    # Count total functions
    total_functions = 0
    file_snippets = {}  # file_path -> [snippets]
    for file_path in files:
        try:
            source = file_path.read_text(errors="ignore")
            snippets = extract_snippets(file_path, source)
            if snippets:
                file_snippets[file_path] = snippets
                total_functions += len(snippets)
        except Exception as e:
            pass

    if total_functions == 0:
        print("No functions found.")
        sys.exit(0)

    # Prepare output
    output_file = root / config["output"]
    init_book(output_file, root, config["model"])

    print(f"Found {len(files)} files with {total_functions} functions")
    print(f"Output: {output_file}\n")

    # Process with progress bar
    with tqdm(total=total_functions, file=sys.stderr, ncols=80, colour="green") as pbar:
        for file_path in files:
            if file_path not in file_snippets:
                continue

            snippets = file_snippets[file_path]
            # Compute relative path, handling cases where file is outside root
            try:
                rel_path = file_path.relative_to(root)
            except ValueError:
                # File is outside root (e.g., --file with absolute path), use file name
                rel_path = file_path

            for snippet in snippets:
                # Skip if already done
                if already_done(snippet["name"], str(rel_path), output_file):
                    tqdm.write(f"  ⊘ {snippet['name']} — already done")
                    pbar.update(1)
                    continue

                # Show function being processed
                tqdm.write(f"\n  {snippet['name']} ({rel_path}:{snippet['start']}-{snippet['end']})")
                tqdm.write("  " + "─" * 60)

                # Get explanation (streaming)
                explanation = annotate_streaming(str(rel_path), snippet, config, pbar)

                # Save to file
                append_to_book(str(rel_path), snippet, explanation, output_file)

                time.sleep(0.2)  # Light throttle between requests

    print(f"\n✅ Done! Generated: {output_file}")


if __name__ == "__main__":
    main()
