import sqlite3
from pathlib import Path
from typing import Optional

class KnowledgeGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    name TEXT,
                    type TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    code TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY,
                    from_id INTEGER,
                    to_id INTEGER,
                    type TEXT, -- CALLS, IMPORTS, CONTAINS
                    FOREIGN KEY(from_id) REFERENCES symbols(id) ON DELETE CASCADE,
                    FOREIGN KEY(to_id) REFERENCES symbols(id) ON DELETE CASCADE
                )
            """)

    def get_file_hash(self, path: str) -> Optional[str]:
        with self._get_conn() as conn:
            res = conn.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
            return res[0] if res else None

    def clear_file_symbols(self, file_id: int):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM symbols WHERE file_id=?", (file_id,))

    def add_file(self, path: str, file_hash: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO files (path, hash) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET hash=excluded.hash", 
                (path, file_hash)
            )
            # Fetch the id for the path
            return conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]

    def add_symbol(self, file_id: int, name: str, sym_type: str, start: int, end: int, code: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO symbols (file_id, name, type, start_line, end_line, code) VALUES (?, ?, ?, ?, ?, ?)",
                (file_id, name, sym_type, start, end, code)
            )
            return cursor.lastrowid

    def add_relation(self, from_id: int, to_id: int, rel_type: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO relations (from_id, to_id, type) VALUES (?, ?, ?)",
                (from_id, to_id, rel_type)
            )

    def get_all_symbol_context(self) -> str:
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT s.name, s.type, f.path, s.code
                FROM symbols s
                JOIN files f ON s.file_id = f.id
                ORDER BY f.path, s.name
            """).fetchall()
        parts = []
        for name, sym_type, path, code in rows:
            parts.append(f"{sym_type}: {name} ({path})")
            if code:
                parts.append(f"  ```\n{code[:500]}\n  ```")
        return "\n".join(parts)

    def get_symbol_context_for_question(self, question: str, top_n: int = 30) -> str:
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "how", "what",
                      "when", "where", "why", "which", "who", "this", "that",
                      "these", "those", "it", "its", "in", "on", "at", "to",
                      "for", "of", "with", "by", "from", "and", "or", "not",
                      "please", "tell", "me", "about", "work", "explain"}
        tokens = set(
            t.lower().rstrip("?.!,;:") for t in question.split()
            if t.lower().rstrip("?.!,;:") not in stop_words
            and len(t.rstrip("?.!,;:")) > 1
        )
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT s.name, s.type, f.path, s.code
                FROM symbols s
                JOIN files f ON s.file_id = f.id
                ORDER BY s.name
            """).fetchall()
        scored = []
        for name, sym_type, path, code in rows:
            name_lower = name.lower()
            score = sum(1 for t in tokens if t in name_lower)
            if score > 0:
                scored.append((score, sym_type, name, path, code))
        scored.sort(key=lambda x: -x[0])
        parts = []
        for _, sym_type, name, path, code in scored[:top_n]:
            parts.append(f"{sym_type}: {name} ({path})")
            if code:
                parts.append(f"  ```\n{code[:500]}\n  ```")
        return "\n".join(parts)

    def get_symbol_id_by_name(self, name: str) -> Optional[int]:
        with self._get_conn() as conn:
            res = conn.execute(
                "SELECT id FROM symbols WHERE name=? LIMIT 1", (name,)
            ).fetchone()
            return res[0] if res else None

    def get_symbol_ids_by_file(self, file_path: str) -> list[int]:
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT s.id FROM symbols s
                JOIN files f ON s.file_id = f.id
                WHERE f.path=?
            """, (file_path,)).fetchall()
            return [r[0] for r in rows]
