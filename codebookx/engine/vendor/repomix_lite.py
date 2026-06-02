import os
from pathlib import Path
from typing import List

def pack_repo(root_path: Path, include_extensions: List[str] = None) -> str:
    """Pack the repository into a single text block, respecting some ignores."""
    if include_extensions is None:
        include_extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt"]
    
    skip_dirs = {".git", "node_modules", "__pycache__", "dist", "build"}
    packed_output = []
    
    for root, dirs, files in os.walk(root_path):
        # Filter directories in-place to skip them
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in include_extensions:
                try:
                    rel_path = file_path.relative_to(root_path)
                    content = file_path.read_text(errors="ignore")
                    
                    packed_output.append(f"--- BEGIN FILE: {rel_path} ---")
                    packed_output.append(content)
                    packed_output.append(f"--- END FILE: {rel_path} ---\n")
                except Exception:
                    continue
                    
    return "\n".join(packed_output)
