import pytest
from pathlib import Path
from codebookx.engine.parser import resolve_relative_imports, extract_python_relations

def test_resolve_relative_imports(tmp_path):
    # Setup fixture directory structure:
    # pkg/
    #   __init__.py
    #   mod.py
    #   sub/
    #     __init__.py
    #     sibling.py
    #     inner.py
    
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    (pkg / "mod.py").touch()
    
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").touch()
    (sub / "sibling.py").touch()
    inner_file = sub / "inner.py"
    inner_file.touch()

    # 1. Sibling import: from . import sibling
    rels = [{"type": "IMPORTS", "target": "sibling", "level": 1, "line": 1}]
    resolved = resolve_relative_imports(inner_file, rels)
    assert resolved[0]["target"] == "pkg.sub.sibling"
    assert resolved[0]["level"] == 0

    # 2. Sub-module import: from .utils import helper
    rels = [{"type": "IMPORTS", "target": "utils.helper", "level": 1, "line": 2}]
    resolved = resolve_relative_imports(inner_file, rels)
    assert resolved[0]["target"] == "pkg.sub.utils.helper"

    # 3. Parent import: from ..mod import func
    rels = [{"type": "IMPORTS", "target": "mod.func", "level": 2, "line": 3}]
    resolved = resolve_relative_imports(inner_file, rels)
    assert resolved[0]["target"] == "pkg.mod.func"

    # 4. Absolute import: import os (level 0)
    rels = [{"type": "IMPORTS", "target": "os", "level": 0, "line": 4}]
    resolved = resolve_relative_imports(inner_file, rels)
    assert resolved[0]["target"] == "os"
    assert resolved[0]["level"] == 0

    # 5. Future import: from __future__ import annotations (level 0)
    rels = [{"type": "IMPORTS", "target": "__future__.annotations", "level": 0, "line": 5}]
    resolved = resolve_relative_imports(inner_file, rels)
    assert resolved[0]["target"] == "__future__.annotations"

    # 6. Absolute from import: from pkg.mod import func
    rels = [{"type": "IMPORTS", "target": "pkg.mod.func", "level": 0, "line": 6}]
    resolved = resolve_relative_imports(inner_file, rels)
    assert resolved[0]["target"] == "pkg.mod.func"

def test_extract_ts_relations():
    source = """
import { X } from './module';
import Y from '../parent';
// import { Z } from './comment';
/*
import { A } from './block';
*/
const s = "import { B } from './string'";
import './side-effect';
export { C } from './export';
import('./dynamic');
"""
    from codebookx.engine.parser import extract_ts_relations
    rels = extract_ts_relations(source)
    targets = [r["target"] for r in rels]
    
    assert "./module" in targets
    assert "../parent" in targets
    assert "./side-effect" in targets
    assert "./export" in targets
    assert "./dynamic" in targets
    
    # Should ignore comments and strings
    assert "./comment" not in targets
    assert "./block" not in targets
    assert "./string" not in targets

    # Level checks
    levels = {r["target"]: r["level"] for r in rels}
    assert levels["./module"] == 1
    assert levels["../parent"] == 2

def test_resolve_ts_relative_imports(tmp_path):
    # Setup TS project structure
    # src/
    #   app.ts
    #   utils/
    #     helper.ts
    #     index.ts
    #   ui/
    #     button.tsx
    
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.ts").touch()
    
    utils = src / "utils"
    utils.mkdir()
    (utils / "helper.ts").touch()
    (utils / "index.ts").touch()
    
    ui = src / "ui"
    ui.mkdir()
    (ui / "button.tsx").touch()
    
    from codebookx.engine.parser import resolve_ts_relative_imports
    
    app_file = src / "app.ts"
    
    # 1. Sibling probing .ts
    rels = [{"type": "IMPORTS", "target": "./utils/helper", "level": 1}]
    resolved = resolve_ts_relative_imports(app_file, rels)
    assert resolved[0]["target"] == "src.utils.helper"
    
    # 2. Directory index probing
    rels = [{"type": "IMPORTS", "target": "./utils", "level": 1}]
    resolved = resolve_ts_relative_imports(app_file, rels)
    assert resolved[0]["target"] == "src.utils"
    
    # 3. .tsx probing
    rels = [{"type": "IMPORTS", "target": "./ui/button", "level": 1}]
    resolved = resolve_ts_relative_imports(app_file, rels)
    assert resolved[0]["target"] == "src.ui.button"
    
    # 4. Parent probing
    helper_file = utils / "helper.ts"
    rels = [{"type": "IMPORTS", "target": "../app", "level": 2}]
    resolved = resolve_ts_relative_imports(helper_file, rels)
    assert resolved[0]["target"] == "src.app"

def test_extract_python_relations_with_level():
    source = """
import os
from . import sibling
from ..parent import func
from __future__ import annotations
"""
    rels = extract_python_relations(source)
    # Filter for IMPORTS
    imports = [r for r in rels if r["type"] == "IMPORTS"]
    
    # 1. import os
    assert imports[0]["target"] == "os"
    assert imports[0]["level"] == 0
    
    # 2. from . import sibling
    assert imports[1]["target"] == "sibling"
    assert imports[1]["level"] == 1
    
    # 3. from ..parent import func
    assert imports[2]["target"] == "parent.func"
    assert imports[2]["level"] == 2
    
    # 4. from __future__ import annotations
    assert imports[3]["target"] == "__future__.annotations"
    assert imports[3]["level"] == 0
