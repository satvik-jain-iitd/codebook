import hashlib
from pathlib import Path
from typing import List, Dict, Any
from .graph import KnowledgeGraph
from .parser import extract_snippets, extract_python_relations, resolve_relative_imports
from .vendor.claude_mem_lite import generate_code_skeleton

class Indexer:
    def __init__(self, root_path: str, db_path: str):
        self.root = Path(root_path).resolve()
        self.kg = KnowledgeGraph(db_path)
        self.skip_dirs = {".git", "node_modules", "__pycache__", "dist", "build"}

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def index(self, force: bool = False):
        """Run the multi-phase indexing pipeline."""
        print(f"🔍 Indexing {self.root}...")
        
        # Phase 1: Discovery
        files = self._discover_files()
        
        # Phase 2: Parsing & Ingestion
        for file_path in files:
            rel_path = str(file_path.relative_to(self.root))
            file_hash = self.get_file_hash(file_path)
            
            # Check if file changed or force re-index
            existing_hash = self.kg.get_file_hash(rel_path)
            if not force and existing_hash == file_hash:
                continue
            
            # TODO: Replace with RETURNING id when SQLite minimum version allows
            file_id = self.kg.add_file(rel_path, file_hash)
            self.kg.clear_file_symbols(file_id)
            
            # Use core extraction for now (Phase 1 legacy)
            source = file_path.read_text(errors="ignore")
            snippets = extract_snippets(file_path, source)
            
            # P1.1: Skeleton fallback for non-Python/JS languages
            if not snippets and file_path.suffix in (".go", ".rs", ".java", ".cpp", ".cs"):
                skeleton = generate_code_skeleton(file_path)
                if skeleton:
                    snippets = [{
                        "name": file_path.stem,
                        "start": 1,
                        "end": skeleton.count("\n") + 1,
                        "code": skeleton,
                        "type": "module",
                    }]
            
            # Map FQN to DB ID for relations
            symbol_ids = {}
            for snip in snippets:
                sym_id = self.kg.add_symbol(
                    file_id, 
                    snip["name"], 
                    snip["type"], 
                    snip["start"], 
                    snip["end"], 
                    snip["code"]
                )
                symbol_ids[snip["name"]] = sym_id
                
                # Wire relations if parent exists
                parent_name = snip.get("parent")
                if parent_name and parent_name in symbol_ids:
                    self.kg.add_relation(symbol_ids[parent_name], sym_id, "CONTAINS")

        # P2.1 Phase 3: Two-pass post-processing for CALLS/IMPORTS (Python only)
        py_files = [f for f in files if f.suffix == ".py"]
        if py_files:
            print("  Resolving CALLS/IMPORTS (Python only)...")
            
            # Pass 1: Build map of all Python symbol names -> id
            all_symbols = {}
            for file_path in py_files:
                source = file_path.read_text(errors="ignore")
                snippets = extract_snippets(file_path, source)
                for snip in snippets:
                    sid = self.kg.get_symbol_id_by_name(snip["name"])
                    if sid:
                        all_symbols[snip["name"]] = sid
            
            # Pass 2: Parse each Python file's AST for CALLS/IMPORTS
            for file_path in py_files:
                source = file_path.read_text(errors="ignore")
                rels = extract_python_relations(source)
                rels = resolve_relative_imports(file_path, rels)
                for rel in rels:
                    target_name = rel["target"]
                    # For IMPORTS like "core.run_analyze", try bare name too
                    bare_name = target_name.split(".")[-1]
                    resolved = all_symbols.get(target_name) or all_symbols.get(bare_name)
                    if resolved:
                        from_ids = self.kg.get_symbol_ids_by_file(
                            str(file_path.relative_to(self.root))
                        )
                        for fid in from_ids:
                            self.kg.add_relation(fid, resolved, rel["type"])

        print(f"✅ Indexing complete. Knowledge Graph updated.")

    def _discover_files(self) -> List[Path]:
        import os
        files = []
        for root, dirs, filenames in os.walk(self.root):
            # Prune skip_dirs in-place
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            for f in filenames:
                file_path = Path(root) / f
                if file_path.suffix in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".cpp", ".cs"):
                    files.append(file_path)
        return files
