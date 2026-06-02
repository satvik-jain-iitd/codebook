import argparse
import sys
from pathlib import Path
from .core import run_generate, run_analyze, run_decompose, run_enhance, run_ask, run_ask_chat
from .webapp.app import run_server

def main():
    parser = argparse.ArgumentParser(
        prog="codebookx",
        description="Codebase-X: Evolution of Codebook for offline code comprehension."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate (Classic Codebook)
    gen_parser = subparsers.add_parser("generate", help="Generate classic CODEBASE_BOOK.md")
    gen_parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan")
    gen_parser.add_argument("--url", help="LM Studio base URL")
    gen_parser.add_argument("--model", help="Model name")
    gen_parser.add_argument("--output", help="Output file path")
    gen_parser.add_argument("--prompt-lang", help="Language for explanations")
    gen_parser.add_argument("--file", help="Annotate single file only")

    # Analyze (PCTF)
    analyze_parser = subparsers.add_parser("analyze", help="Run PCTF Codebase Teardown")
    analyze_parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan")
    analyze_parser.add_argument("--force", action="store_true", help="Force re-analysis")

    # Decompose (AATD)
    decompose_parser = subparsers.add_parser("decompose", help="Run AATD Feature Decomposition")
    decompose_parser.add_argument("feature", help="Feature description to decompose")
    decompose_parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan")
    decompose_parser.add_argument("--deep", action="store_true", help="Use deep context (repomix)")

    # Enhance (New: Prompt Enhancement)
    enhance_parser = subparsers.add_parser("enhance", help="Enhance a prompt with codebase context (requires: codebook analyze first)")
    enhance_parser.add_argument("feature", help="Feature description to enhance")
    enhance_parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan")

    # Ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the codebase (requires: codebook analyze first)")
    ask_parser.add_argument("question", help="The question to ask about your codebase")
    ask_parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan")
    ask_parser.add_argument("--dir", default=None,
        help="Directory to save Q&A log (default: $CODEBOOK_ASK_DIR or ./ask_history)")
    ask_parser.add_argument("-c", "--chat", action="store_true",
        help="Interactive chat mode (multi-turn, context carry-over) (requires: codebook analyze first)")

    # View
    view_parser = subparsers.add_parser("view", help="Launch local Knowledge Graph web UI")
    view_parser.add_argument("target_dir", nargs="?", default=".", help="Directory to scan")
    view_parser.add_argument("--port", type=int, default=8050, help="Port to run the UI on")

    # Backward compatibility: Intercept sys.argv
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in ["generate", "analyze", "decompose", "ask", "enhance", "view", "-h", "--help"]:
        sys.argv.insert(1, "generate")
    elif len(sys.argv) == 1:
        sys.argv.append("generate")

    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "decompose":
        run_decompose(args)
    elif args.command == "enhance":
        run_enhance(args)
    elif args.command == "ask":
        if getattr(args, "chat", False):
            run_ask_chat(args)
        else:
            run_ask(args)
    elif args.command == "view":
        root = Path(args.target_dir or ".").resolve()
        db_path = root / ".codebook_cache.db"
        print(f"Starting local UI on http://localhost:{args.port}...")
        print(f"Using database: {db_path}")
        run_server(port=args.port, db_path=str(db_path))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
