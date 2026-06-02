import ast
from pathlib import Path

def generate_code_skeleton(file_path: Path) -> str:
    """Generate a compact skeleton of a file (functions and classes only)."""
    if not file_path.exists():
        return ""
    
    source = file_path.read_text(errors="ignore")
    skeleton = [f"File: {file_path.name}"]
    
    try:
        if file_path.suffix == ".py":
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    skeleton.append(f"  Class: {node.name}")
                    for subnode in node.body:
                        if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            skeleton.append(f"    Method: {subnode.name}")
                        elif isinstance(subnode, ast.ClassDef):
                            skeleton.append(f"    Nested Class: {subnode.name}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    skeleton.append(f"  Function: {node.name}")
        elif file_path.suffix in (".ts", ".tsx", ".js", ".jsx"):
            # Simple regex-based skeleton for JS/TS
            import re
            patterns = [
                r"export\s+(?:async\s+)?function\s+(\w+)",
                r"export\s+class\s+(\w+)",
                r"export\s+const\s+(\w+)\s*=",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, source):
                    skeleton.append(f"  Symbol: {match.group(1)}")
    except Exception as e:
        print(f"Skeleton generation error: {e}")
        
    return "\n".join(skeleton)
