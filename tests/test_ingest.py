"""Normalization, exact coverage, stable identity and all M1 rejection codes."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import random
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import uuid

from hector import config
from hector.ingest import IngestError, ingest
from hector.store import open_store

FIXTURES = Path(__file__).parent / "fixtures"


@contextmanager
def workspace_tempdir():
    # Python 3.14 tempfile ACLs exclude the restricted Windows test token.
    parent = Path(tempfile.gettempdir()) / "hector-phase02-tests"
    path = parent / ("phase02-test-" + uuid.uuid4().hex)
    path.mkdir(parents=True)

    def remove_readonly(function, filename, error):
        if not isinstance(error, PermissionError):
            raise error
        Path(filename).chmod(stat.S_IWRITE)
        function(filename)

    try:
        yield path
    finally:
        if not path.resolve().is_relative_to(parent.resolve()):
            raise AssertionError("Cleanup escaped the test results directory")
        shutil.rmtree(path, onexc=remove_readonly)


class IngestTests(unittest.TestCase):
    def assert_invariants(self, document, chunks):
        covered = set()
        previous = 0
        self.assertTrue(chunks)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.index, index)
            self.assertEqual(chunk.chunk_id, f"{document.doc_id}:{index:04d}")
            self.assertEqual(chunk.doc_id, document.doc_id)
            self.assertLessEqual(previous, chunk.start)
            self.assertLess(chunk.start, chunk.end)
            self.assertEqual(document.text[chunk.start:chunk.end], chunk.text)
            self.assertEqual(chunk.text, chunk.text.strip())
            self.assertLessEqual(len(chunk.text), config.CHUNK_MAX_CHARS)
            covered.update(range(chunk.start, chunk.end))
            previous = chunk.end
        self.assertTrue(all(i in covered for i, c in enumerate(document.text) if not c.isspace()))

    def test_valid_fixtures_and_content_addresses(self):
        for path in (FIXTURES / "corpus").iterdir():
            with self.subTest(path=path.name):
                document, chunks = ingest(str(path), path.read_bytes())
                self.assert_invariants(document, chunks)
                digest = hashlib.sha256(document.text.encode()).hexdigest()
                self.assertEqual(document.sha256, digest)
                self.assertEqual(document.doc_id, "d_" + digest[:16])
                self.assertEqual(document.char_count, len(document.text))
                self.assertEqual(document.filename, path.name)
                self.assertEqual(ingest("renamed.TXT", path.read_bytes())[0].doc_id, document.doc_id)

    def test_normalization_and_exact_offsets(self):
        raw = b"\xef\xbb\xbf  Cafe\xcc\x81\r\n\tverified\r\rnext\x01\x0b\x1f!  "
        document, chunks = ingest("folder/example.MD", raw)
        self.assertEqual(document.text, "  Caf\u00e9\n\tverified\n\nnext!  ")
        self.assertEqual(document.media_type, "text/markdown")
        self.assert_invariants(document, chunks)
        self.assertEqual(document.doc_id, ingest("nfc.txt", document.text.encode())[0].doc_id)

    def test_each_file_failure_code(self):
        fixtures = [("bad.pdf", b"ok", "UNSUPPORTED_TYPE"),
                    ("big.txt", b"x" * (config.MAX_FILE_BYTES + 1), "FILE_TOO_LARGE"),
                    ("binary.md", b"a\0b", "BINARY_CONTENT"),
                    ("bad.txt", b"\xff", "INVALID_UTF8"),
                    ("blank.md", b"\xef\xbb\xbf \r\n\t\x01", "EMPTY_DOCUMENT")]
        for name, data, code in fixtures:
            with self.subTest(code=code), self.assertRaises(IngestError) as error:
                ingest(name, data)
            self.assertEqual(error.exception.code, code)

    def test_removing_controls_leaves_nfc_and_idempotent_content(self):
        document, chunks = ingest("controls.txt", "e\x01\u0301".encode())
        self.assertEqual(document.text, "\u00e9")
        self.assertEqual(ingest("again.txt", document.text.encode())[0].doc_id, document.doc_id)
        self.assert_invariants(document, chunks)

    def test_corpus_count_and_utf8_size_failure_codes(self):
        for limits in ({"MAX_DOCUMENTS": 1}, {"MAX_CORPUS_BYTES": 3}):
            with self.subTest(limits=limits), patch.multiple(config, **limits), open_store() as store:
                store.add(*ingest("a.txt", "\u00e9".encode()))
                with self.assertRaises(IngestError) as error:
                    store.add(*ingest("b.txt", b"ab"))
                self.assertEqual(error.exception.code, "CORPUS_LIMIT")
                self.assertEqual(len(store.documents()), 1)

    def test_file_and_corpus_size_exact_boundary(self):
        document, chunks = ingest("limit.txt", b"x" * config.MAX_FILE_BYTES)
        self.assert_invariants(document, chunks)
        with patch.multiple(config, MAX_FILE_BYTES=4, MAX_CORPUS_BYTES=4), open_store() as store:
            store.add(*ingest("a.txt", b"abcd"))
            store.add(*ingest("copy.txt", b"abcd"))
            self.assertEqual(len(store.documents()), 1)
            with self.assertRaises(IngestError) as error:
                store.add(*ingest("b.txt", b"e"))
            self.assertEqual(error.exception.code, "CORPUS_LIMIT")

    def test_greedy_paragraph_packing_and_whitespace_split(self):
        first, second, third = "a" * 500, "b" * 500, "c" * 500
        document, chunks = ingest("p.txt", f"  {first}\n\n{second}\n\n{third}  ".encode())
        self.assertEqual([c.text for c in chunks], [first + "\n\n" + second, third])
        self.assert_invariants(document, chunks)
        paragraph = "a" * 800 + " " + "b" * 800
        document, chunks = ingest("p.txt", paragraph.encode())
        self.assertEqual([c.text for c in chunks], ["a" * 800, "b" * 800])
        self.assert_invariants(document, chunks)

    def test_hard_cut_and_oversized_paragraph_remainder_packs(self):
        limit = config.CHUNK_MAX_CHARS
        document, chunks = ingest("long.txt", ("x" * (limit + 10) + "\n\nlast").encode())
        self.assertEqual([len(c.text) for c in chunks], [limit, 16])
        self.assert_invariants(document, chunks)

    def test_trimming_paragraphs_and_split_chunks_preserves_offsets(self):
        limit = config.CHUNK_MAX_CHARS
        text = " \t\n\n  " + "a" * limit + " \t\n\n \t " + "b" * limit + "  \n\t"
        document, chunks = ingest("trim.txt", text.encode())
        self.assertEqual([chunk.text for chunk in chunks], ["a" * limit, "b" * limit])
        self.assert_invariants(document, chunks)

    def test_seeded_chunking_edges_preserve_all_nonwhitespace(self):
        rng = random.Random(2)
        for _ in range(40):
            text = "".join(rng.choice(("word ", "\n\n", "\t", "\u00e9", "\U0001f680", "z" * 1301))
                           for _ in range(20))
            document, chunks = ingest("random.txt", text.encode())
            self.assert_invariants(document, chunks)
