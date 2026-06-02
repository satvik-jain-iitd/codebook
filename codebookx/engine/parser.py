import ast
import re
from pathlib import Path

def extract_python_functions(source: str) -> list[dict]:
    snippets = []
    try:
        tree = ast.parse(source)
        lines = source.splitlines()
        
        def traverse(node, parent_name=None):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    fqn = f"{parent_name}.{child.name}" if parent_name else child.name
                    start = child.lineno - 1
                    end = child.end_lineno
                    snippets.append({
                        "name": fqn,
                        "start": child.lineno,
                        "end": child.end_lineno,
                        "code": "\n".join(lines[start:end]),
                        "type": "class" if isinstance(child, ast.ClassDef) else "function",
                        "parent": parent_name
                    })
                    # Recurse into the scope of this class/function
                    traverse(child, fqn)
                else:
                    # Generic traversal for non-scope nodes (If, Try, With, etc.)
                    # Recurse without changing scope
                    traverse(child, parent_name)
                    
        traverse(tree)
    except SyntaxError: pass
    return snippets

def extract_python_relations(source: str) -> list[dict]:
    """Extract CALLS and IMPORTS from Python AST.
    Returns list of {type, target, line, level} dicts."""
    relations = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = node.names
                if isinstance(node, ast.Import):
                    for alias in names:
                        relations.append({"type": "IMPORTS", "target": alias.name, "line": node.lineno, "level": 0})
                else:  # ImportFrom
                    module = node.module or ""
                    rel_level = getattr(node, 'level', 0)
                    for alias in names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        relations.append({"type": "IMPORTS", "target": full_name, "line": node.lineno, "level": rel_level})
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    relations.append({"type": "CALLS", "target": node.func.id, "line": node.lineno})
                elif isinstance(node.func, ast.Attribute):
                    # e.g., obj.method() → target = "method"
                    relations.append({"type": "CALLS", "target": node.func.attr, "line": node.lineno})
    except (SyntaxError, Exception):
        pass
    return relations

def resolve_relative_imports(file_path: Path, rels: list[dict]) -> list[dict]:
    """
    Resolves relative imports (level > 0) to absolute dotted module paths 
    by walking up the package tree.
    """
    resolved_rels = []
    for rel in rels:
        if rel["type"] != "IMPORTS" or rel.get("level", 0) == 0:
            resolved_rels.append(rel)
            continue

        level = rel["level"]
        target = rel["target"]
        
        # Determine package base directory
        # level 1 = same dir, level 2 = parent dir, etc.
        base_dir = file_path.parent
        for _ in range(level - 1):
            if base_dir.parent == base_dir: # Reached root
                break
            base_dir = base_dir.parent

        # Discover package prefix by walking UP from base_dir as long as __init__.py exists
        prefix_parts = []
        walk_dir = base_dir
        while (walk_dir / "__init__.py").exists():
            prefix_parts.insert(0, walk_dir.name)
            if walk_dir.parent == walk_dir:
                break
            walk_dir = walk_dir.parent
            
        # Build resolved target
        target_parts = target.split(".")
        resolved_name = ".".join(prefix_parts + target_parts)
        
        new_rel = rel.copy()
        new_rel["target"] = resolved_name
        new_rel["level"] = 0 # Now absolute
        resolved_rels.append(new_rel)
        
    return resolved_rels

def extract_ts_functions(source: str) -> list[dict]:
    snippets = []
    lines = source.splitlines()
    pattern = re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:"
        r"function\s*\*?\s+(\w+)"
        r"|class\s+(\w+)"
        r"|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)=>"
        r"|(?!(?:if|for|while|switch|catch)\b)(\w+)\s*\([^)]*\)\s*\{"
        r")",
        re.MULTILINE,
    )
    
    for match in pattern.finditer(source):
        name = next(g for g in match.groups() if g is not None)
        start_line = source[: match.start()].count("\n")
        
        # Forward Depth State Machine
        depth = 0
        end_line = start_line
        in_string = False
        string_char = None
        started = False
        
        for i in range(start_line, min(start_line + 500, len(lines))):
            line = lines[i]
            j = 0
            while j < len(line):
                char = line[j]
                
                # Handle comments
                if not in_string and char == '/' and j + 1 < len(line) and line[j+1] == '/':
                    break # Skip rest of line
                    
                # Handle strings/templates
                if char in ("'", '"', '`'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif string_char == char:
                        in_string = False
                        string_char = None
                
                if not in_string:
                    if char == '{':
                        depth += 1
                        started = True
                    elif char == '}':
                        depth -= 1
                
                if started and depth == 0:
                    end_line = i + 1
                    break
                j += 1
            if started and depth == 0:
                break
        
        snippets.append({
            "name": name,
            "start": start_line + 1,
            "end": end_line,
            "code": "\n".join(lines[start_line:end_line]),
            "type": "function"
        })
    return snippets

def extract_snippets(file_path: Path, source: str) -> list[dict]:
    if file_path.suffix == ".py":
        return extract_python_functions(source)
    elif file_path.suffix in (".js", ".ts", ".jsx", ".tsx"):
        return extract_ts_functions(source)
    return []
