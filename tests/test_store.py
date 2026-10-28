"""FTS safety, ranking, persistence, limits and transaction boundaries."""

from dataclasses import replace
import sqlite3
import unittest
from unittest.mock import Mock, patch

from hector import config
from hector.ingest import ingest
from hector.store import StoreError, open_store
from test_ingest import workspace_tempdir


class StoreTests(unittest.TestCase):
    def test_stemming_source_metadata_persistence_and_duplicates(self):
        with workspace_tempdir() as root:
            path = root / "test.db"
            document, chunks = ingest("ledger.md", b"Verified records require signed ledgers.")
            with open_store(path) as store:
                store.add(document, chunks)
                store.add(*ingest("renamed.txt", document.text.encode()))
                self.assertEqual(len(store.documents()), 1)
                self.assertEqual(store.connection.execute("SELECT count(*) FROM chunks").fetchone()[0], 1)
                self.assertEqual(store.connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0], 1)
                hit, = store.search("verify")
                self.assertEqual((hit.doc_id, hit.filename, hit.start, hit.end, hit.text),
                                 (document.doc_id, "ledger.md", 0, len(document.text), document.text))
                self.assertGreater(hit.score, 0)
            with open_store(path) as store:
                self.assertEqual(store.search("verify")[0].chunk_id, chunks[0].chunk_id)
                self.assertEqual(store.documents()[0].sha256, document.sha256)
                self.assertTrue(store.documents()[0].ingested_at)

    def test_unavailable_fts5_is_explicit_and_connection_closes(self):
        real_connect = sqlite3.connect
        class NoFTS(sqlite3.Connection):
            closed = False
            def execute(self, statement, *args):
                if "fts5_probe" in statement:
                    raise sqlite3.OperationalError("no such module: fts5")
                return super().execute(statement, *args)
            def close(self):
                self.closed = True
                super().close()
        connection = real_connect(":memory:", factory=NoFTS)
        with patch("hector.store.sqlite3.connect", return_value=connection):
            with self.assertRaises(StoreError) as error:
                open_store()
        self.assertEqual(error.exception.code, "FTS5_UNAVAILABLE")
        self.assertIn("FTS5", str(error.exception))
        self.assertTrue(connection.closed)

    def test_existing_only_store_never_creates_a_file_and_quotes_uri_paths(self):
        with workspace_tempdir() as root:
            missing = root / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                open_store(missing, create=False)
            self.assertFalse(missing.exists())
            path = root / "search # 100%.db"
            with open_store(path) as store:
                store.add(*ingest("a.md", b"verified records"))
            with open_store(path, create=False) as store:
                self.assertEqual(store.search("verify")[0].filename, "a.md")
        with open_store(":memory:", create=False) as store:
            self.assertEqual(store.documents(), [])

    def test_malformed_match_and_sql_injection_never_execute_syntax(self):
        queries = ['"', 'NEAR(', '*', '-x', 'text:foo', '^a', 'a AND', '()',
                   '" * - ( ) : NEAR AND OR NOT ^', "x'; DROP TABLE documents; --", '" OR 1=1 --']
        with open_store() as store:
            store.add(*ingest("a.txt", b"x foo a verified records"))
            for query in queries:
                with self.subTest(query=query):
                    try:
                        store.search(query)
                    except StoreError as error:
                        self.assertEqual(error.code, "EMPTY_QUERY")
            self.assertEqual(len(store.documents()), 1)
            self.assertEqual(len(store.search("verify")), 1)
            for query in ("", " \t", "*()^:\"", "---", "*", "*__", "_", "___"):
                with self.assertRaises(StoreError) as error:
                    store.search(query)
                self.assertEqual(error.exception.code, "EMPTY_QUERY")
            self.assertEqual(store.search("__verified__")[0].filename, "a.txt")

    def test_query_terms_are_quoted_lowercased_and_capped(self):
        with open_store() as store:
            store.add(*ingest("last.txt", b"lastterm"))
            terms = [f"word{i}" for i in range(config.QUERY_MAX_TERMS)]
            self.assertEqual(store.search(" ".join(terms + ["lastterm"])), [])
            self.assertTrue(store.search("LASTTERM"))
            self.assertTrue(store.search("missing OR lastterm"))
            self.assertEqual(store.search("missing AND absent"), [])

    def test_tokenization_precedes_unicode_lowercasing(self):
        with open_store() as store:
            connection = store.connection
            store.connection = Mock(wraps=connection)
            store.search("\u0130STANBUL " + "word " * config.QUERY_MAX_TERMS)
            match, k = store.connection.execute.call_args.args[1]
            self.assertEqual(match.split(" OR ")[0], '"i\u0307stanbul"')
            self.assertEqual(len(match.split(" OR ")), config.QUERY_MAX_TERMS)
            self.assertEqual(k, config.SEARCH_K_DEFAULT)

    def test_bm25_order_and_k_boundaries(self):
        with open_store() as store:
            for i in range(10):
                store.add(*ingest(f"{i}.txt", (("quartz " * (i + 1)) + ("filler " * (10 - i))).encode()))
            hits = store.search("quartz", config.SEARCH_K_MAX)
            self.assertEqual(len(hits), config.SEARCH_K_MAX)
            self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))
            self.assertEqual(hits[0].filename, "9.txt")
            raw = store.connection.execute(
                "SELECT chunk_id, -bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts), chunk_id LIMIT ?", ('"quartz"', config.SEARCH_K_MAX)).fetchall()
            self.assertEqual([(h.chunk_id, h.score) for h in hits], raw)
            self.assertEqual(len(store.search("quartz")), config.SEARCH_K_DEFAULT)
            self.assertEqual(len(store.search("quartz", 1)), 1)
            for k in (0, -1, 9, True, 1.5, "5"):
                with self.subTest(k=k), self.assertRaises(StoreError) as error:
                    store.search("quartz", k)
                self.assertEqual(error.exception.code, "INVALID_K")

    def test_chunk_neighbors_and_unknown_chunk(self):
        document, chunks = ingest("big.txt", b"x" * (config.CHUNK_MAX_CHARS * 2 + 1))
        with open_store() as store:
            store.add(document, chunks)
            for i, chunk in enumerate(chunks):
                passage = store.get_chunk(chunk.chunk_id)
                self.assertEqual(passage.text, chunk.text)
                self.assertEqual(passage.index, i)
                self.assertEqual(passage.prev_id, chunks[i - 1].chunk_id if i else None)
                self.assertEqual(passage.next_id, chunks[i + 1].chunk_id if i + 1 < len(chunks) else None)
            with self.assertRaises(StoreError) as error:
                store.get_chunk("unknown' OR 1=1")
            self.assertEqual(error.exception.code, "UNKNOWN_CHUNK")

    def test_failed_insert_rolls_back_document_chunks_and_fts(self):
        document, chunks = ingest("a.txt", b"x" * (config.CHUNK_MAX_CHARS + 1))
        with open_store() as store:
            with self.assertRaises(sqlite3.IntegrityError):
                store.add(document, [chunks[0], replace(chunks[1], chunk_id=chunks[0].chunk_id)])
            for table in ("documents", "chunks", "chunks_fts"):
                # Table names are fixed test data, never user input.
                self.assertEqual(store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            store.add(document, chunks)
            self.assertEqual(len(store.documents()), 1)

    def test_document_id_collision_never_overwrites_existing_source(self):
        document, chunks = ingest("a.txt", b"verified records")
        with open_store() as store:
            store.add(document, chunks)
            with self.assertRaises(sqlite3.IntegrityError):
                store.add(replace(document, sha256="0" * 64), chunks)
            self.assertEqual(store.documents()[0].sha256, document.sha256)
            self.assertEqual(store.search("verify")[0].text, document.text)
