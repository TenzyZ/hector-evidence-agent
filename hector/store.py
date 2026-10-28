"""SQLite FTS5 corpus storage; untrusted query syntax never reaches MATCH."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

from . import config
from .ingest import Chunk, IngestError


class StoreError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def query_terms(query):
    return [term.lower() for term in re.findall(r"[^\W_]+", query)[:config.QUERY_MAX_TERMS]]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    filename: str
    start: int
    end: int
    text: str
    score: float


@dataclass(frozen=True)
class Passage(Chunk):
    prev_id: str | None
    next_id: str | None


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    filename: str
    media_type: str
    sha256: str
    char_count: int
    ingested_at: str


class Store:
    def __init__(self, path, *, create=True):
        if create or path == ":memory:":
            self.connection = sqlite3.connect(path)
        else:
            self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True)
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            try:
                self.connection.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(text)")
                self.connection.execute("DROP TABLE temp.fts5_probe")
            except sqlite3.OperationalError:
                raise StoreError("FTS5_UNAVAILABLE", "SQLite FTS5 is required but unavailable.") from None
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS documents(
                    doc_id TEXT PRIMARY KEY, filename TEXT NOT NULL, media_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL, char_count INTEGER NOT NULL, text TEXT NOT NULL,
                    ingested_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks(
                    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                    idx INTEGER NOT NULL, start INTEGER NOT NULL, "end" INTEGER NOT NULL, text TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text, chunk_id UNINDEXED, tokenize='porter unicode61 remove_diacritics 2');
            ''')
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.connection.close()

    def add(self, document, chunks):
        with self.connection:
            # Serialize the limit check and insertion, including across CLI processes.
            self.connection.execute("BEGIN IMMEDIATE")
            existing = self.connection.execute(
                "SELECT sha256 FROM documents WHERE doc_id = ?", (document.doc_id,)).fetchone()
            if existing:
                if existing[0] != document.sha256:
                    raise sqlite3.IntegrityError("Document ID collision; corpus left unchanged.")
                return
            count, size = self.connection.execute(
                "SELECT count(*), coalesce(sum(length(CAST(text AS BLOB))), 0) FROM documents"
            ).fetchone()
            if count >= config.MAX_DOCUMENTS or size + len(document.text.encode("utf-8")) > config.MAX_CORPUS_BYTES:
                raise IngestError("CORPUS_LIMIT", "Corpus document count or normalized UTF-8 size exceeded.")
            self.connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (document.doc_id, document.filename, document.media_type,
                                     document.sha256, document.char_count, document.text,
                                     datetime.now(timezone.utc).isoformat()))
            self.connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                                        [(c.chunk_id, c.doc_id, c.index, c.start, c.end, c.text) for c in chunks])
            self.connection.executemany("INSERT INTO chunks_fts(text, chunk_id) VALUES (?, ?)",
                                        [(c.text, c.chunk_id) for c in chunks])

    def search(self, query, k=config.SEARCH_K_DEFAULT):
        if type(k) is not int or not 1 <= k <= config.SEARCH_K_MAX:
            raise StoreError("INVALID_K", "k must be an integer from 1 to SEARCH_K_MAX.")
        terms = query_terms(query)
        if not terms:
            raise StoreError("EMPTY_QUERY", "Query must contain at least one word token.")
        match = " OR ".join('"' + term + '"' for term in terms)
        rows = self.connection.execute('''
            SELECT c.chunk_id, c.doc_id, d.filename, c.start, c."end", c.text,
                   -bm25(chunks_fts) AS score
            FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts), c.chunk_id LIMIT ?
        ''', (match, k)).fetchall()
        return [Hit(*row) for row in rows]

    def get_chunk(self, chunk_id):
        row = self.connection.execute('''
            SELECT chunk_id, doc_id, idx, start, "end", text FROM chunks WHERE chunk_id = ?
        ''', (chunk_id,)).fetchone()
        if row is None:
            raise StoreError("UNKNOWN_CHUNK", "Chunk is not in this corpus.")
        neighbors = dict(self.connection.execute(
            "SELECT idx, chunk_id FROM chunks WHERE doc_id = ? AND idx IN (?, ?)",
            (row[1], row[2] - 1, row[2] + 1)))
        return Passage(*row, neighbors.get(row[2] - 1), neighbors.get(row[2] + 1))

    def documents(self):
        return [DocumentSummary(*row) for row in self.connection.execute('''
            SELECT doc_id, filename, media_type, sha256, char_count, ingested_at
            FROM documents ORDER BY doc_id
        ''')]


def open_store(path=":memory:", *, create=True) -> Store:
    return Store(path, create=create)
