"""Schema, corpus integrity, gold spans and hand-calculated Configuration A metrics."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from hector import config
from hector.evaluate import EvaluationError, run, validate
from hector.ingest import IngestError
from test_ingest import FIXTURES, workspace_tempdir

CORPUS = FIXTURES / "corpus"
QUESTIONS = FIXTURES / "questions.jsonl"


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]

    def write_items(self, root, items):
        path = root / "questions.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        return path

    def assert_invalid(self, items, code="INVALID_ITEM"):
        with workspace_tempdir() as root:
            path = self.write_items(root, items)
            with self.assertRaises(EvaluationError) as error:
                validate(CORPUS, path)
            self.assertEqual(error.exception.code, code)

    def test_accepts_fixture_schema_and_exact_file_identities(self):
        dataset = validate(CORPUS, QUESTIONS)
        self.assertEqual(len(dataset.documents), 4)
        self.assertEqual(len(dataset.items), 8)
        self.assertEqual(dataset.identities["questions_sha256"], hashlib.sha256(QUESTIONS.read_bytes()).hexdigest())
        for name, digest in dataset.identities["corpus_files_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((CORPUS / name).read_bytes()).hexdigest())
        self.assertEqual(len(dataset.identities["corpus_sha256"]), 64)
        self.assertEqual(dataset.identities, validate(CORPUS, QUESTIONS).identities)

    def test_duplicate_id_and_every_missing_field(self):
        self.assert_invalid([self.items[0], self.items[0]], "DUPLICATE_ID")
        for field in self.items[0]:
            with self.subTest(field=field):
                item = copy.deepcopy(self.items[0])
                del item[field]
                self.assert_invalid([item])

    def test_unexpected_top_level_fields_are_rejected(self):
        item = copy.deepcopy(self.items[0])
        item["unexpected"] = "extra"
        self.assert_invalid([item])

    def test_nonsearchable_questions_fail_validation(self):
        for question in ("???", "***", "*__", "___"):
            with self.subTest(question=question):
                item = copy.deepcopy(self.items[0])
                item["question"] = question
                self.assert_invalid([item])
        with workspace_tempdir() as root:
            item = copy.deepcopy(self.items[0])
            item["question"] = "__verify__"
            self.assertEqual(validate(CORPUS, self.write_items(root, [item])).items, [item])

    def test_bad_enums_and_schema_types(self):
        changes = {"id": [None, 1, ""], "split": ["other", [], None],
                   "category": ["other", {}], "question": ["", 3, "x" * (config.QUESTION_CHARS[1] + 1)],
                   "expected_outcome": ["SUPPORTED", None], "gold_evidence": [None, {}],
                   "key_facts": [None, "fact", [3], [""]], "trap_fact": [3], "canary": [True],
                   "notes": [None, []]}
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    item = copy.deepcopy(self.items[0])
                    item[field] = value
                    self.assert_invalid([item])
        self.assert_invalid([[]])

    def test_absent_wrong_document_and_malformed_gold(self):
        golds = [[{"doc": "handbook.md", "quote": "No such quotation"}],
                 [{"doc": "archive.md", "quote": self.items[0]["gold_evidence"][0]["quote"]}],
                 [{"doc": "../handbook.md", "quote": "Verified records"}],
                 [{"doc": "handbook.md", "quote": ""}], [{"doc": "handbook.md"}],
                 [{"doc": [], "quote": "text"}], ["quote"]]
        for gold in golds:
            with self.subTest(gold=gold):
                item = copy.deepcopy(self.items[0])
                item["gold_evidence"] = gold
                self.assert_invalid([item], "INVALID_GOLD")

    def test_gold_quote_crossing_chunk_boundary_rejected(self):
        with workspace_tempdir() as root:
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "boundary.txt").write_text("x" * config.CHUNK_MAX_CHARS + "y" * 30, encoding="utf-8")
            item = copy.deepcopy(self.items[0])
            item["gold_evidence"] = [{"doc": "boundary.txt", "quote": "xy"}]
            with self.assertRaises(EvaluationError) as error:
                validate(corpus, self.write_items(root, [item]))
            self.assertEqual(error.exception.code, "GOLD_CROSSES_CHUNKS")

    def test_gold_is_checked_against_normalized_text_and_literal_jsonl_delimiters(self):
        with workspace_tempdir() as root:
            corpus = root / "corpus"
            corpus.mkdir()
            raw = b"\xef\xbb\xbfCafe\xcc\x81\r\nverified records"
            (corpus / "normal.MD").write_bytes(raw)
            item = copy.deepcopy(self.items[0])
            item["notes"] = "one\u2028two\u0085three\u2029four"
            item["gold_evidence"] = [{"doc": "normal.MD", "quote": "Caf\u00e9\nverified records"}]
            questions = self.write_items(root, [item])
            self.assertEqual(validate(corpus, questions).items, [item])
            questions.write_bytes(questions.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(validate(corpus, questions).items, [item])

    def test_invalid_json_duplicate_keys_and_empty_inputs(self):
        with workspace_tempdir() as root:
            questions = root / "bad.jsonl"
            for raw in (b"", b"\n", b"{}\n\n", b"\xff", b"{", b'{"id":"a","id":"b"}', b'{"x":NaN}'):
                with self.subTest(raw=raw):
                    questions.write_bytes(raw)
                    with self.assertRaises(EvaluationError):
                        validate(CORPUS, questions)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(EvaluationError) as error:
                validate(empty, QUESTIONS)
            self.assertEqual(error.exception.code, "EMPTY_CORPUS")
            with self.assertRaises(EvaluationError):
                validate(root / "absent", QUESTIONS)
            (empty / "bad.txt").write_bytes(b"a\0b")
            with self.assertRaises(IngestError) as error:
                validate(empty, QUESTIONS)
            self.assertEqual(error.exception.code, "BINARY_CONTENT")

    def test_corpus_limits_and_integrity_hash_changes(self):
        with workspace_tempdir() as root:
            corpus = root / "corpus"
            corpus.mkdir()
            for name in ("a.txt", "b.txt"):
                (corpus / name).write_text(name, encoding="utf-8")
            with patch.object(config, "MAX_DOCUMENTS", 1), self.assertRaises(IngestError) as error:
                validate(corpus, QUESTIONS)
            self.assertEqual(error.exception.code, "CORPUS_LIMIT")
            item = copy.deepcopy(self.items[0])
            item["gold_evidence"] = []
            questions = self.write_items(root, [item])
            before = validate(corpus, questions).identities
            (corpus / "a.txt").write_text("changed", encoding="utf-8")
            after = validate(corpus, questions).identities
            self.assertNotEqual(before["corpus_sha256"], after["corpus_sha256"])
            self.assertNotEqual(before["dataset_sha256"], after["dataset_sha256"])
            item["notes"] = "changed"
            self.write_items(root, [item])
            self.assertNotEqual(after["dataset_sha256"], validate(corpus, questions).identities["dataset_sha256"])

    def test_fixture_metrics_match_hand_calculation_and_trace_contract(self):
        with workspace_tempdir() as root:
            result = run(CORPUS, QUESTIONS, output_root=root)
            directory = Path(result["output_dir"])
            self.assertEqual({p.name for p in directory.iterdir()}, {"runs.jsonl", "summary.json", "summary.md"})
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            records = [json.loads(line) for line in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
            metrics = summary["metrics"]
            # Eight gold quotes; six eligible items; rank sum: 1+1+1+0+1+1/2.
            self.assertEqual((metrics["R1"]["numerator"], metrics["R1"]["denominator"]), (7, 8))
            self.assertEqual(metrics["R1"]["rate"], 7 / 8)
            self.assertEqual(metrics["R2"]["value"], 4.5 / 6)
            self.assertEqual(metrics["R2"]["denominator"], 6)
            self.assertEqual(metrics["O1"]["rate"], 1 / 2)
            self.assertEqual((metrics["O1"]["numerator"], metrics["O1"]["denominator"]), (1, 2))
            self.assertEqual(records[7]["reciprocal_rank"], 0.5)
            self.assertEqual(records[2]["outcome"], "ANSWERED")
            self.assertEqual(records[6]["outcome"], "ABSTAINED")
            for record in records:
                self.assertEqual(record["claims"], [])
                self.assertEqual(record["usage"]["model_calls"], 0)
                self.assertEqual(record["steps"][0]["arguments"]["k"], config.PRESEED_K)
            for name in ("R1", "O1"):
                self.assertEqual(len(metrics[name]["wilson95"]), 2)
            for metric in metrics.values():
                self.assertEqual(metric["mean"], metric["min"])
                self.assertEqual(metric["mean"], metric["max"])
            self.assertEqual(len(summary["git_commit"]), 40)
            self.assertIsInstance(summary["dirty"], bool)
            self.assertEqual(summary["total_tokens"], 0)
            self.assertEqual(summary["total_cost_usd"], 0)
            self.assertEqual(summary["limits"]["CHUNK_MAX_CHARS"], config.CHUNK_MAX_CHARS)
            self.assertIsNone(summary["model"])
            self.assertIn("Small synthetic", (directory / "summary.md").read_text(encoding="utf-8"))

    def test_empty_metric_denominators_are_null(self):
        with workspace_tempdir() as root:
            item = copy.deepcopy(self.items[0])
            item["gold_evidence"] = []
            questions = self.write_items(root, [item])
            summary = run(CORPUS, questions, output_root=root)["summary"]
            self.assertIsNone(summary["metrics"]["R1"]["rate"])
            self.assertIsNone(summary["metrics"]["O1"]["rate"])
            self.assertIsNone(summary["metrics"]["R2"]["value"])
            self.assertEqual(summary["metrics"]["R2"]["denominator"], 0)
            for key in ("mean", "min", "max"):
                self.assertIsNone(summary["metrics"]["R2"][key])

    def test_git_attributes_keep_evaluation_checkout_hashes_lf(self):
        root = FIXTURES.resolve().parents[1]
        with workspace_tempdir() as scratch:
            repository = scratch / "checkout"
            subprocess.run(["git", "init", "-q", str(repository)], check=True, capture_output=True)
            (repository / ".gitattributes").write_bytes((root / ".gitattributes").read_bytes())

            def git(*arguments, input=None):
                return subprocess.run(["git", "-C", str(repository), *arguments], input=input,
                                      check=True, capture_output=True).stdout

            inputs = {path.relative_to(root).as_posix(): path.read_bytes()
                      for path in FIXTURES.rglob("*") if path.is_file()}
            inputs.update({"eval/corpus/example.md": b"alpha\nbeta\n", "eval/questions.jsonl": b"{}\n"})
            for relative, data in inputs.items():
                self.assertNotIn(b"\r", data)
                for raw in (data, data.replace(b"\n", b"\r\n")):
                    oid = git("hash-object", "-w", "--path=" + relative, "--stdin", input=raw).decode().strip()
                    for autocrlf, eol in (("true", "crlf"), ("false", "lf")):
                        with self.subTest(path=relative, autocrlf=autocrlf):
                            checkout = git("-c", "core.autocrlf=" + autocrlf, "-c", "core.eol=" + eol,
                                           "cat-file", "--filters", "--path=" + relative, oid)
                            self.assertEqual(hashlib.sha256(checkout).digest(), hashlib.sha256(data).digest())
            protected = ("eval/pilot-b0/questions.jsonl", "eval/pilot-b0/corpus/example.md",
                         "docs/eval/pilot-b0/summary.json", "Hector-B0-Evidence/example.json")
            attributes = git("check-attr", "text", "eol", "--", *protected).decode().splitlines()
            self.assertTrue(all(line.endswith(": unspecified") for line in attributes))

    def test_retrieval_metrics_exclude_sixth_hit_and_wrong_source(self):
        with workspace_tempdir() as root:
            corpus = root / "corpus"
            corpus.mkdir()
            for i in range(6):
                (corpus / f"{i}.txt").write_text("target " * (i + 1) + "filler " * (10 - i), encoding="utf-8")
            item = copy.deepcopy(self.items[0])
            item["question"] = "target"
            item["gold_evidence"] = [{"doc": "0.txt", "quote": "target filler"}]
            questions = self.write_items(root, [item])
            result = run(corpus, questions, output_root=root)
            self.assertEqual(result["summary"]["metrics"]["R1"]["rate"], 0)
            self.assertEqual(result["summary"]["metrics"]["R2"]["value"], 0)
            record = json.loads((Path(result["output_dir"]) / "runs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(len(record["hits"]), config.PRESEED_K)
            self.assertNotIn("0.txt", [hit["filename"] for hit in record["hits"]])
            # The quote exists in retrieved documents, but only its named source counts.
            self.assertTrue(any("target filler" in hit["text"] for hit in record["hits"]))

    def test_missing_git_provenance_prevents_output_creation(self):
        with workspace_tempdir() as root, patch("hector.evaluate.subprocess.run", side_effect=OSError):
            with self.assertRaises(EvaluationError) as error:
                run(CORPUS, QUESTIONS, output_root=root)
            self.assertEqual(error.exception.code, "PROVENANCE_UNAVAILABLE")
            self.assertEqual(list(root.iterdir()), [])

    def test_other_config_test_split_and_empty_dev_are_not_run(self):
        with workspace_tempdir() as root:
            for kwargs in ({"split": "test"}, {"configuration": "B"}):
                with self.assertRaises(EvaluationError) as error:
                    run(CORPUS, QUESTIONS, output_root=root, **kwargs)
                self.assertEqual(error.exception.code, "UNAUTHORIZED_EVALUATION")
            item = copy.deepcopy(self.items[0])
            item["split"] = "test"
            with self.assertRaises(EvaluationError) as error:
                run(CORPUS, self.write_items(root, [item]), output_root=root)
            self.assertEqual(error.exception.code, "EMPTY_SPLIT")

    def test_network_and_ml_are_not_needed(self):
        with workspace_tempdir() as root, patch("socket.create_connection", side_effect=AssertionError("network")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("network")), \
                patch.dict("sys.modules", {"torch": None, "transformers": None}):
            self.assertEqual(run(CORPUS, QUESTIONS, output_root=root)["summary"]["item_count"], 8)
