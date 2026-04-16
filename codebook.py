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
        "extensions": [".py", ".ts", ".tsx"],
        "skip_dirs": ["node_modules", ".next", "__pycache__", ".git", "dist", "build", ".beads"],
        "prompt_lang": "plain English",
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
    if args.extensions:
        config["extensions"] = args.extensions.split(",")
    if args.skip_dirs:
        config["skip_dirs"] = args.skip_dirs.split(",")
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
    return []


# ──────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────


def get_files(
    root: Path, extensions: set[str], skip_dirs: set[str], single: Optional[str] = None
) -> list[Path]:
    """Recursively discover files matching extensions, excluding skip_dirs."""
    if single:
        return [Path(single)]

    files = []
    try:
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in extensions:
                # Check if any skip_dir component is in the path
                if not any(skip in f.parts for skip in skip_dirs):
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


def annotate_streaming(
    file_path: str, snippet: dict, config: dict, pbar: tqdm
) -> str:
    """
    Send snippet to LM Studio and stream the response live to terminal.
    Returns the full explanation text (with <think> tokens stripped).
    """
    prompt = f"""Explain this code in {config['prompt_lang']} for someone learning to code.
Line by line if possible. Keep it concise but clear.

File: {file_path}
Lines: {snippet['start']}-{snippet['end']}
Function/Class: {snippet['name']}

```
{snippet['code']}
```

Explanation:"""

    try:
        response = requests.post(
            f"{config['url'].rstrip('/')}/chat/completions",
            json={
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
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
    parser.add_argument("--extensions", help="Comma-separated extensions (default: .py,.ts,.tsx)")
    parser.add_argument("--skip-dirs", help="Comma-separated dirs to skip")
    parser.add_argument(
        "--prompt-lang",
        help="Language/style for explanations (default: plain English)",
    )
    parser.add_argument("--file", help="Annotate single file only")
    parser.add_argument("--check", action="store_true", help="Check server status and exit")
    parser.add_argument(
        "--no-skip-done",
        action="store_true",
        help="Re-annotate already-done functions",
    )

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

    # Discover files
    print(f"Scanning {root}...")
    # If --file is provided, resolve it relative to root if it's not absolute
    single_file = None
    if args.file:
        single_path = Path(args.file)
        if not single_path.is_absolute():
            single_path = root / single_path
        single_file = str(single_path.resolve())

    files = get_files(root, set(config["extensions"]), set(config["skip_dirs"]), single_file)

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
                # Skip if already done (unless --no-skip-done)
                if not args.no_skip_done and already_done(snippet["name"], str(rel_path), output_file):
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
