"""Real module subprocesses exercise PowerShell-style unexpanded globs and JSON errors."""

import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

from test_ingest import FIXTURES, workspace_tempdir

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def cli(self, *args, success=True, cwd=ROOT):
        process = subprocess.run([sys.executable, "-B", "-m", "hector", *map(str, args)],
                                 cwd=cwd, env=os.environ | {"PYTHONPATH": str(ROOT)},
                                 capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(process.stderr, "")
        self.assertEqual(process.returncode == 0, success, process.stdout)
        result = json.loads(process.stdout)
        if not success:
            self.assertIn("code", result["error"])
        return result

    def test_ingest_glob_search_and_duplicate_cli(self):
        with workspace_tempdir() as root:
            database = root / "cli.db"
            pattern = FIXTURES / "corpus" / "*.md"
            self.assertEqual(len(self.cli("ingest", "--db", database, pattern)["documents"]), 3)
            self.cli("ingest", "--db", database, pattern)
            result = self.cli("search", "--db", database, "-k", 5, "verify")
            self.assertEqual(len(result["hits"]), 1)
            self.assertEqual(result["hits"][0]["filename"], "handbook.md")

    def test_validate_and_run_json(self):
        options = ("--corpus", FIXTURES / "corpus", "--questions", FIXTURES / "questions.jsonl")
        self.assertTrue(self.cli("eval", "validate", *options)["valid"])
        with workspace_tempdir() as root:
            result = self.cli("eval", "run", "--config", "A", *options, "--split", "dev", cwd=root)
            directory = (root / result["output_dir"]).resolve()
            self.assertTrue(directory.is_relative_to(root.resolve()))
            self.assertTrue((directory / "runs.jsonl").is_file())
            self.assertEqual(result["summary"]["metrics"]["R1"]["rate"], 7 / 8)
            self.assertEqual(result["summary"]["metrics"]["R2"]["value"], 4.5 / 6)

    def test_partial_ingest_reports_basename_and_saved_documents(self):
        with workspace_tempdir() as root:
            (root / "a.md").write_bytes(b"alpha verified records")
            (root / "b.md").write_bytes(b"\xff")
            (root / "c.md").write_bytes(b"charlie records")
            database = root / "partial.db"
            result = self.cli("ingest", "--db", database, root / "*.md", success=False)
            self.assertEqual(result["error"]["code"], "INVALID_UTF8")
            self.assertEqual(result["error"]["filename"], "b.md")
            self.assertEqual([doc["filename"] for doc in result["documents"]], ["a.md"])
            self.assertNotIn(str(root), json.dumps(result))
            hits = self.cli("search", "--db", database, "alpha")["hits"]
            self.assertEqual(hits[0]["doc_id"], result["documents"][0]["doc_id"])
            self.assertEqual(self.cli("search", "--db", database, "charlie")["hits"], [])

    def test_search_missing_database_fails_without_creating_it(self):
        with workspace_tempdir() as root:
            database = root / "typo.db"
            self.cli("search", "--db", database, "verify", success=False)
            self.assertFalse(database.exists())

    def test_literal_wildcard_and_bracket_ingest_paths(self):
        with workspace_tempdir() as root:
            directory = root / "files with spaces"
            directory.mkdir()
            names = ("literal.md", "bracket[1].md", "part1.md", "part2.md")
            for name in names:
                (directory / name).write_text(name, encoding="utf-8")
            database = root / "patterns.db"
            self.assertEqual(self.cli("ingest", "--db", database, directory / "literal.md")
                             ["documents"][0]["filename"], "literal.md")
            self.assertEqual(self.cli("ingest", "--db", database, glob.escape(str(directory / "bracket[1].md")))
                             ["documents"][0]["filename"], "bracket[1].md")
            for pattern in ("part[12].md", "part?.md", "part*.md"):
                with self.subTest(pattern=pattern):
                    result = self.cli("ingest", "--db", database, directory / pattern)
                    self.assertEqual([doc["filename"] for doc in result["documents"]], ["part1.md", "part2.md"])

    def test_json_nonzero_failure_paths(self):
        with workspace_tempdir() as root:
            database = root / "errors.db"
            cases = [(), ("missing",), ("ingest",), ("search", "--db", database),
                     ("search", "--db", database, "-k", "bad", "verify"),
                     ("search", "--db", database, "-k", 0, "verify"),
                     ("search", "--db", database, "()"),
                     ("ingest", "--db", database, root / "absent*.md"),
                     ("ingest", "--db", database, root / "missing.md"),
                     ("eval", "validate"),
                     ("eval", "run", "--config", "B", "--split", "dev"),
                     ("eval", "run", "--config", "A", "--split", "test")]
            for args in cases:
                with self.subTest(args=args):
                    self.cli(*args, success=False)
            for name, raw, code in (("bad.txt", b"\xff", "INVALID_UTF8"),
                                    ("bad.pdf", b"text", "UNSUPPORTED_TYPE")):
                path = root / name
                path.write_bytes(raw)
                result = self.cli("ingest", "--db", database, path, success=False)
                self.assertEqual(result["error"]["code"], code)

    def test_package_metadata_is_stdlib_only(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["name"], "hector")
        self.assertEqual(project["requires-python"], ">=3.12")
        self.assertEqual(project["dependencies"], [])
