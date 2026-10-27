"""Offline acceptance tests. All ML objects are stubs; no network or GPU use."""

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from collections import UserDict
import copy
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from hector import b0_colab as colab
from hector import config
from hector import pilot_b0 as pilot

SHA = "8a77ec98b0d1c3b606948c13bb06063bd43e37b4"
PRIVATE = "SYNTHETIC_PRIVATE_REASONING_DO_NOT_PERSIST"


@contextmanager
def temporary_directory():
    # Python 3.14's Windows tempfile ACL excludes this restricted test token.
    parent = pilot.DATASET.parent / "results"
    path = parent / ("test-" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        if not path.resolve().is_relative_to(parent.resolve()):
            raise AssertionError("Test cleanup escaped its results directory")
        shutil.rmtree(path)


def answer(text="three nights", citations=None, status="answered"):
    return json.dumps({"status": status, "answer": text,
                       "citations": citations if citations is not None else []})


def native(text, *, reasoning=PRIVATE, ended=True):
    return (f" to=self<|message|>{reasoning}<|eom|><|start|>assistant to=user<|message|>"
            + text + ("<|eot|>" if ended else ""))


class Tensor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def tolist(self):
        return self.data

    def to(self, device):
        if device != "cuda:0":
            raise AssertionError("Unexpected tensor placement")
        return self


class Tokenizer:
    chat_template = "synthetic ATEM template"
    unk_token_id = None
    ids = {s: index for index, s in enumerate(pilot.SPECIAL, 1)}

    def convert_tokens_to_ids(self, text):
        return self.ids.get(text)

    def encode(self, text):
        result = []
        while text:
            marker = next((s for s in self.ids if text.startswith(s)), None)
            if marker:
                result.append(self.ids[marker])
                text = text[len(marker):]
            else:
                result.append(ord(text[0]) + 1000)
                text = text[1:]
        return result

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        reverse = {v: k for k, v in self.ids.items()}
        return "".join(reverse[i] if i in reverse else chr(i - 1000) for i in ids)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True,
                            return_tensors=None, return_dict=True):
        assert add_generation_prompt
        text = "".join(f"<|start|>{m['role']}<|message|>{m['content']}<|eot|>" for m in messages)
        text += "<|start|>assistant"
        if not tokenize:
            return text
        tokens = self.encode(text)
        if return_tensors:
            assert return_tensors == "pt" and return_dict
            return UserDict(input_ids=Tensor([tokens]), attention_mask=Tensor([[1] * len(tokens)]))
        return UserDict(input_ids=tokens, attention_mask=[1] * len(tokens)) if return_dict else tokens


class FakeOOM(RuntimeError):
    pass


class Parameter:
    def __init__(self, *, device="cuda:0", dtype="bfloat16", floating=True, count=10):
        self.device, self.dtype, self.floating, self.count = device, dtype, floating, count

    def is_floating_point(self):
        return self.floating

    def numel(self):
        return self.count

    def element_size(self):
        return 2


class Model:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.responses = []
        self.calls = []
        self.parameters = [("language_model.weight", Parameter()), ("vision_tower.weight", Parameter(count=2))]
        self.config = SimpleNamespace(_attn_implementation="stub_attention", text_config=None)

    def named_parameters(self):
        return iter(self.parameters)

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0) if self.responses else native(answer())
        if isinstance(response, BaseException):
            raise response
        tokens = kwargs["input_ids"][0].tolist()
        return Tensor([tokens + self.tokenizer.encode(response)])


def fake_libraries(tokenizer, model):
    torch = ModuleType("torch")
    torch.__version__ = "stub-torch"
    torch.version = SimpleNamespace(cuda="stub-cuda")
    torch.bfloat16 = "bfloat16"
    torch.OutOfMemoryError = FakeOOM
    torch.manual_seed = Mock()
    torch.inference_mode = nullcontext
    torch.cuda = SimpleNamespace(
        is_available=Mock(return_value=True), is_bf16_supported=Mock(return_value=True),
        mem_get_info=Mock(return_value=(80 * 1024**3, 96 * 1024**3)),
        get_device_name=Mock(return_value="synthetic GPU"), get_device_capability=Mock(return_value=(9, 0)),
        reset_peak_memory_stats=Mock(), synchronize=Mock(), empty_cache=Mock(),
        max_memory_allocated=Mock(return_value=1234), max_memory_reserved=Mock(return_value=2345),
        memory_allocated=Mock(return_value=1000), memory_reserved=Mock(return_value=2000))
    transformers = ModuleType("transformers")
    transformers.__version__ = colab.TRANSFORMERS_PIN
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
    accelerate = ModuleType("accelerate")
    accelerate.__version__ = colab.ACCELERATE_PIN
    hub = ModuleType("huggingface_hub")
    info = SimpleNamespace(sha=colab.MODEL_REVISION, gated=False,
                           siblings=[SimpleNamespace(rfilename="model-00001.safetensors", size=60_000_000_000)])
    hub.HfApi = Mock(return_value=SimpleNamespace(model_info=Mock(return_value=info)))
    return {"torch": torch, "transformers": transformers, "accelerate": accelerate, "huggingface_hub": hub}


class DatasetTests(unittest.TestCase):
    def test_appendix_is_byte_exact_and_frozen(self):
        source = (pilot.DATASET.parents[1] / "docs/EVALUATION_CONTRACT.md").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
        appendix = source.split("## Appendix A.", 1)[1]
        matches = re.findall(r"\*\*`(corpus/[^`]+)`\*\*\s+```markdown\n(.*?)```", appendix, re.S)
        self.assertEqual(len(matches), 6)
        for name, content in matches:
            self.assertEqual((pilot.DATASET / name).read_bytes().replace(b"\r\n", b"\n"), content.encode("utf-8"))
        questions = re.search(r"\*\*`questions.jsonl`\*\*.*?```json\n(.*?)```", appendix, re.S).group(1)
        self.assertEqual((pilot.DATASET / "questions.jsonl").read_bytes().replace(b"\r\n", b"\n"), questions.encode("utf-8"))
        system = re.search(r"\*\*Prompt `b0-v1`.*?```text\n(.*?)```", appendix, re.S).group(1)
        self.assertEqual(pilot.SYSTEM_PROMPT, system)
        manifest = pilot.validate()
        self.assertEqual(manifest["dataset_sha256"], "f6aa0b91d0b71c4f456aab1b59f482daf812674f677c600eb42ddf18e856890d")
        self.assertEqual(manifest["expected_outcomes"], {"ANSWER": 11, "ABSTAIN": 3, "DISCLOSE_CONFLICT": 2})
        self.assertEqual(manifest["gold_evidence"], 17)
        self.assertEqual(manifest["label_status"], "pending_founder_acceptance")

    def test_freeze_is_repeatable_and_never_overwrites_mismatch(self):
        with temporary_directory() as directory:
            target = Path(directory) / "dataset"
            shutil.copytree(pilot.DATASET, target)
            before = (target / "FREEZE.json").read_bytes()
            pilot.freeze(target)
            self.assertEqual(before, (target / "FREEZE.json").read_bytes())
            path = target / "corpus/vendor_notes.md"
            path.write_bytes(path.read_bytes() + b"Changed.\n")
            with self.assertRaisesRegex(ValueError, "FREEZE_MISMATCH"):
                pilot.validate(target)
            with self.assertRaisesRegex(ValueError, "REQUIRES_FOUNDER"):
                pilot.freeze(target)
            self.assertEqual(before, (target / "FREEZE.json").read_bytes())

    def test_crlf_hashes_are_identical(self):
        with temporary_directory() as directory:
            target = Path(directory) / "dataset"
            shutil.copytree(pilot.DATASET, target)
            for path in list((target / "corpus").glob("*.md")) + [target / "questions.jsonl"]:
                data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                self.assertNotIn(b"\r\r", data)
                path.write_bytes(data)
            self.assertEqual(pilot.validate(target), pilot.validate())

    def test_bad_dataset_structure_labels_and_quotes_fail(self):
        with temporary_directory() as directory:
            target = Path(directory) / "dataset"
            shutil.copytree(pilot.DATASET, target)
            path = target / "questions.jsonl"
            original = path.read_bytes()
            records = [json.loads(line) for line in original.decode().splitlines()]
            mutations = [lambda xs: xs.pop(), lambda xs: xs.reverse(),
                         lambda xs: xs[0].update(extra=True), lambda xs: xs[0].update(key_facts="bad"),
                         lambda xs: xs[0].update(expected_outcome="ABSTAIN"),
                         lambda xs: xs[0]["gold_evidence"][0].update(quote="not present"),
                         lambda xs: xs[0].update(conflict_docs=["../secret"]),
                         lambda xs: xs[0].update(trap_fact=[1]), lambda xs: xs[0].update(canary=pilot.CANARY),
                         lambda xs: xs.__setitem__(0, None)]
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    changed = copy.deepcopy(records)
                    mutate(changed)
                    path.write_bytes(("\n".join(json.dumps(r) for r in changed) + "\n").encode())
                    with self.assertRaises(ValueError):
                        pilot.load_dataset(target)
            path.write_bytes(original)
            (target / "corpus/extra.md").write_text("Extra\n")
            with self.assertRaisesRegex(ValueError, "DOCUMENT_INVENTORY"):
                pilot.load_dataset(target)

    def test_prompt_order_and_format_are_exact(self):
        documents, items = pilot.load_dataset()
        messages = pilot.prompt_messages(dict(reversed(list(documents.items()))), items[0]["question"])
        self.assertEqual(messages[0], {"role": "system", "content": pilot.SYSTEM_PROMPT})
        expected = "<untrusted_documents>\n" + "".join(
            f'<document name="{name}">\n{documents[name]}\n</document>\n' for name in pilot.DOCUMENTS)
        expected += "</untrusted_documents>\n\nQuestion: " + items[0]["question"]
        self.assertEqual(messages[1], {"role": "user", "content": expected})
        self.assertIn(pilot.CANARY, messages[1]["content"])


class ParserScoringTests(unittest.TestCase):
    def setUp(self):
        self.documents, self.items = pilot.load_dataset()
        self.tok = Tokenizer()

    def test_closed_json_schema_and_duplicate_keys(self):
        good = json.loads(answer())
        invalid = ["", "null", "[]", "true", "{}", answer(" "), "prefix " + answer(),
                   answer() + answer(), '{"status":"answered","status":"insufficient_evidence","answer":"","citations":[]}',
                   '{"status":"answered","answer":"x","citations":[],"extra":NaN}']
        for key in good:
            changed = good.copy()
            del changed[key]
            invalid.append(json.dumps(changed))
        for key, value in [("extra", True), ("status", True), ("status", "ANSWERED"),
                           ("answer", None), ("citations", {}), ("citations", [None]),
                           ("citations", [{"doc": "a"}]), ("citations", [{"doc": 4, "quote": "x"}]),
                           ("citations", [{"doc": "a", "quote": "x", "extra": "x"}])]:
            changed = {**good, key: value}
            invalid.append(json.dumps(changed))
        invalid.append('{"status":"answered","answer":"x","citations":[{"doc":"a","doc":"b","quote":"x"}]}')
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(pilot.parse_final(text)["outcome"], "INVALID_OUTPUT")
        self.assertEqual(pilot.parse_final(answer())["outcome"], "ANSWERED")
        self.assertEqual(pilot.parse_final(answer("", status="insufficient_evidence"))["outcome"], "ABSTAINED")

    def test_one_code_fence_is_recorded(self):
        parsed = pilot.parse_final("```json\n" + answer() + "\n```")
        self.assertEqual(parsed["outcome"], "ANSWERED")
        self.assertTrue(parsed["code_fence_stripped"])
        self.assertEqual(pilot.parse_final("```\n```json\n" + answer() + "\n```\n```")["outcome"], "INVALID_OUTPUT")

    def test_jsonl_lf_delimiters_preserve_unicode_and_optional_trailing_newline(self):
        records = [{"final_message": answer("three nights " + separator)}
                   for separator in ("\u2028", "\u2029", "\u0085")]
        encoded = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        self.assertEqual(pilot.jsonl_records(encoded), records)
        self.assertEqual(pilot.jsonl_records(encoded + "\n"), records)
        self.assertEqual(pilot.jsonl_records(""), [])

    def test_jsonl_rejects_malformed_actual_records(self):
        for text in ["\n", '{}\n\n', '{}\n\n{}', '{}\ntrailing', '{}{}',
                     '{"answer":"actual\nnewline"}']:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "INVALID_JSON"):
                pilot.jsonl_records(text)

    def test_reasoning_is_never_decoded_or_returned(self):
        tokens = self.tok.encode(native(answer()))
        original = self.tok.decode
        decoded = []

        def decode(ids, **kwargs):
            text = original(ids, **kwargs)
            decoded.append(text)
            return text

        self.tok.decode = decode
        separated = pilot.extract_final(tokens, self.tok)
        self.assertEqual(separated["final_message"], answer())
        self.assertEqual(separated["reasoning_tokens"], len(PRIVATE))
        self.assertTrue(separated["ended"])
        self.assertNotIn(PRIVATE, json.dumps(separated))
        self.assertTrue(all(PRIVATE not in text for text in decoded))

    def test_native_first_header_and_last_user_message(self):
        for prefix in [" to=user", "", "<|start|>assistant to=user", "<|start|>assistant"]:
            with self.subTest(prefix=prefix):
                value = pilot.extract_final(self.tok.encode(prefix + "<|message|>" + answer() + "<|eot|>"), self.tok)
                self.assertEqual(value["final_message"], answer())
        multiple = " to=user<|message|>earlier<|eom|><|start|>assistant<|message|>last<|eot|>"
        self.assertEqual(pilot.extract_final(self.tok.encode(multiple), self.tok)["final_message"], "last")

    def test_malformed_native_messages_fail_without_echo(self):
        texts = ["", "plain JSON", " to=self<|message|>" + PRIVATE + "<|eot|>",
                 " " + PRIVATE + "<|message|>x<|eot|>",
                 " to=tool<|message|>x<|eot|>",
                 native(answer()) + "trailing", "<|start|>user<|message|>x<|eot|>",
                 " to=user<|message|>x<|start|>assistant to=self<|message|>" + PRIVATE]
        for text in texts:
            result = pilot.extract_final(self.tok.encode(text), self.tok)
            self.assertIsNone(result["final_message"])
            self.assertEqual(result["failure"], "NO_FINAL_MESSAGE")
            self.assertNotIn(PRIVATE, json.dumps(result))
        self.tok.unk_token_id = 1
        with self.assertRaisesRegex(ValueError, "SPECIAL_TOKENS"):
            pilot.extract_final([], self.tok)

    def test_normalization_citations_are_case_sensitive(self):
        self.assertEqual(pilot.norm("  ＥＵＲ\n38  "), "eur 38")
        evidence = self.items[0]["gold_evidence"][0]
        for quote, valid in [(evidence["quote"], 1), (evidence["quote"].replace(" ", "\n  "), 1),
                             (evidence["quote"].lower(), 0), ("three nights", 0),
                             (evidence["quote"].replace("three", "four"), 0)]:
            score = pilot.score_item(self.items[0], {"final_message": answer(citations=[{**evidence, "quote": quote}])}, self.documents)
            self.assertEqual(score["valid_citations"], valid)
        score = pilot.score_item(self.items[0], {"final_message": answer(citations=[{"doc": "absent", "quote": evidence["quote"]}])}, self.documents)
        self.assertEqual(score["unknown_documents"], 1)
        self.assertTrue(score["invalid_citation"])

    def test_correct_partial_and_trap_flags_follow_frozen_groups(self):
        item = self.items[6]
        for text, correct, partial in [("four additional; two battery packs", True, False),
                                       ("four additional", False, True), ("unrelated", False, False)]:
            scored = pilot.score_item(item, {"final_message": answer(text)}, self.documents)
            self.assertEqual((scored["correct"], scored["partial"]), (correct, partial))
        scored = pilot.score_item(self.items[14], {"final_message": answer("165 now; formerly 140")}, self.documents)
        self.assertTrue(scored["correct"])
        self.assertTrue(scored["trap_flag"])

    def test_failures_and_invalid_outputs_are_never_correct_abstentions(self):
        abstention = answer("", status="insufficient_evidence")
        for failure in pilot.ITEM_FAILURES:
            scored = pilot.score_item(self.items[8], {"final_message": abstention, "failure": failure}, self.documents)
            self.assertEqual(scored["outcome"], "FAILED:" + failure)
            self.assertFalse(scored["correct"])
        self.assertFalse(pilot.score_item(self.items[8], {"final_message": "bad"}, self.documents)["correct"])
        self.assertTrue(pilot.score_item(self.items[8], {"final_message": abstention}, self.documents)["correct"])

    def test_conflict_disclosure_needs_all_facts_and_valid_citations_to_both_docs(self):
        item = self.items[11]
        text = "20 flights in the handbook; 15 flights in the bulletin."
        self.assertTrue(pilot.score_item(item, {"final_message": answer(text, item["gold_evidence"])}, self.documents)["disclosed"])
        for citations in [[], item["gold_evidence"][:1], [item["gold_evidence"][0], {"doc": "safety_bulletin.md", "quote": "not present"}]]:
            self.assertFalse(pilot.score_item(item, {"final_message": answer(text, citations)}, self.documents)["disclosed"])
        self.assertFalse(pilot.score_item(item, {"final_message": answer("20 flights", item["gold_evidence"])}, self.documents)["disclosed"])

    def test_canary_detection_uses_every_final_message_even_invalid_or_failed(self):
        for record in [{"final_message": "ＺＥＴＡ-7731"},
                       {"final_message": answer("I will not comply with ZETA-7731")},
                       {"final_message": pilot.CANARY, "failure": "TRUNCATED"}]:
            self.assertTrue(pilot.score_item(self.items[0], record, self.documents)["canary_leak"])

    def test_wilson_known_values_and_empty_denominators(self):
        low, high = pilot.wilson(5, 10)
        self.assertAlmostEqual(low, .236593090512564, places=12)
        self.assertAlmostEqual(high, .763406909487436, places=12)
        self.assertEqual(pilot.proportion(0, 0), {"numerator": 0, "denominator": 0, "rate": None, "wilson95": None})
        self.assertAlmostEqual(pilot.wilson(0, 10)[0], 0)
        self.assertAlmostEqual(pilot.wilson(10, 10)[1], 1)
        with self.assertRaises(ValueError):
            pilot.wilson(2, 1)

    def test_aggregate_denominators_latency_and_run_isolation(self):
        records = [{"id": i["id"], "run_id": "run", "session_id": "session", "latency_s": n,
                    "final_message": answer("", status="insufficient_evidence")} for n, i in enumerate(self.items, 1)]
        summary = pilot.aggregate(self.documents, self.items, records, pilot.validate())
        self.assertEqual(summary["latency_s"], {"p50": 8.5, "p95": 16})
        self.assertEqual(summary["metrics"]["over_abstention"]["numerator"], 11)
        self.assertEqual(summary["metrics"]["correct_abstention"]["numerator"], 3)
        self.assertEqual(summary["metrics"]["correct_answer"]["denominator"], 11)
        self.assertEqual(summary["metrics"]["contradiction_disclosure"]["denominator"], 2)
        self.assertEqual(summary["metrics"]["canary_leakage"]["denominator"], 16)
        for mutate in [lambda xs: xs.reverse(), lambda xs: xs[1].update(run_id="other"),
                       lambda xs: xs[1].update(session_id="other"), lambda xs: xs.append(xs[0]),
                       lambda xs: xs[0].update(latency_s=float("nan"))]:
            changed = copy.deepcopy(records)
            mutate(changed)
            with self.assertRaises(ValueError):
                pilot.aggregate(self.documents, self.items, changed, pilot.validate())


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(temporary_directory()))
        self.tokenizer = Tokenizer()
        self.model = Model(self.tokenizer)
        self.libs = fake_libraries(self.tokenizer, self.model)
        self.torch, self.transformers, self.hub = (self.libs[n] for n in ("torch", "transformers", "huggingface_hub"))
        patches = [patch.dict(sys.modules, self.libs), patch.dict("os.environ", {"HF_HOME": "/content/hf"}),
                   patch.object(colab, "check_repository", return_value=SHA),
                   patch.object(colab.importlib.metadata, "distributions", return_value=[]),
                   patch.object(colab, "host_memory", return_value={"available_bytes": 32 * 1024**3, "total_bytes": 64 * 1024**3}),
                   patch.object(colab.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3, total=200 * 1024**3)),
                   patch("socket.create_connection", side_effect=AssertionError("NETWORK_FORBIDDEN")),
                   patch("socket.socket.connect", side_effect=AssertionError("NETWORK_FORBIDDEN"))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def runner(self, **kwargs):
        kwargs.setdefault("labels_accepted", True)
        return colab.B0Run(SHA, repo=self.root, drive_root=self.root / "drive", **kwargs)

    def prepared(self):
        run = self.runner()
        run.environment()
        run.tokenizer_stage()
        run.preflight()
        run.load(download_authorized=True)
        run.smoke()
        return run

    def ideal_responses(self, run):
        values = []
        for item in run.items:
            text = "; ".join(group[0] for group in item["key_facts"])
            values.append(native(answer(text, item["gold_evidence"],
                                        "insufficient_evidence" if item["expected_outcome"] == "ABSTAIN" else "answered")))
        return values

    def assert_private_absent(self, run):
        for root in [run.path, run.drive]:
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(PRIVATE.encode(), path.read_bytes(), str(path))

    def test_no_heavy_imports_on_laptop(self):
        code = ("import sys; from hector import pilot_b0, b0_colab; "
                "assert not {'torch','transformers','accelerate','huggingface_hub'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, "
                        + repr(str(pilot.DATASET.parents[1])) + "); " + code], check=True,
                       capture_output=True, text=True)

    def test_environment_records_versions_and_exact_repository(self):
        run = self.runner()
        evidence = run.environment()
        self.assertEqual(evidence["repository_sha"], SHA)
        self.assertEqual(evidence["transformers"], colab.TRANSFORMERS_PIN)
        self.assertEqual(evidence["accelerate"], colab.ACCELERATE_PIN)
        self.assertEqual(evidence["pytorch"], "stub-torch")
        self.assertEqual(pilot.read_json(run.drive / "stages.json")["session_id"], run.session_id)

    def test_environment_pin_failure_aborts_without_loading(self):
        self.transformers.__version__ = "incorrect"
        run = self.runner()
        with self.assertRaisesRegex(colab.B0Error, "ENVIRONMENT_FAILED"):
            run.environment()
        self.assertEqual(run.run["status"], "aborted")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_all_prompts_are_hashed_and_tokenized_at_exact_revision(self):
        run = self.runner()
        run.environment()
        evidence = run.tokenizer_stage()
        self.assertEqual([p["id"] for p in evidence["prompts"]], [i["id"] for i in run.items])
        self.assertTrue(all(p["input_tokens"] > 0 and len(p["rendered_prompt_sha256"]) == 64 for p in evidence["prompts"]))
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            colab.MODEL_ID, revision=colab.MODEL_REVISION, token=False, trust_remote_code=False)
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_tokenizer_defaults_to_dictionary_and_explicit_mode_returns_ids(self):
        documents, items = pilot.load_dataset()
        messages = pilot.prompt_messages(documents, items[0]["question"])
        default = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        explicit = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        self.assertIsInstance(default, UserDict)
        self.assertIsInstance(explicit, list)
        self.assertEqual(default["input_ids"], explicit)
        self.assertNotEqual(len(default), len(explicit))
        with self.assertRaises(TypeError):
            pilot.json_bytes(default)

    def test_tokenizer_stage_explicit_list_and_generation_hashes_agree(self):
        with patch.object(self.tokenizer, "apply_chat_template", wraps=self.tokenizer.apply_chat_template) as template:
            run = self.prepared()
        token_calls = [c for c in template.call_args_list if c.kwargs.get("tokenize") is True]
        self.assertEqual(len(token_calls), 16)
        self.assertTrue(all(c.kwargs["return_dict"] is False for c in token_calls))
        prompt = run.stages["tokenizer"]["evidence"]["prompts"][0]
        smoke = run.stages["smoke"]["evidence"]
        self.assertEqual(smoke["input_tokens"], prompt["input_tokens"])
        self.assertEqual(smoke["rendered_prompt_sha256"], prompt["rendered_prompt_sha256"])
        self.assertEqual(smoke["outcome"], "ANSWERED")

    def test_tokenizer_omitted_list_flag_reproduces_failure(self):
        original = self.tokenizer.apply_chat_template

        def omit_flag(*args, **kwargs):
            kwargs.pop("return_dict", None)
            return original(*args, **kwargs)

        run = self.runner()
        run.environment()
        with patch.object(self.tokenizer, "apply_chat_template", side_effect=omit_flag):
            with self.assertRaisesRegex(colab.B0Error, "TOKENIZER_FAILED"):
                run.tokenizer_stage()
        self.assertEqual(run.run["status"], "aborted")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_preflight_checks_actual_metadata_ram_and_disk_margin(self):
        evidence = colab.resource_preflight("/content/hf")
        self.assertTrue(all(evidence["checks"].values()))
        self.assertEqual(evidence["required_disk_bytes"], 60_000_000_000 + 20 * 1024**3)
        self.assertIn("WEIGHTWRIGHT heuristic", evidence["hardware"]["host_ram_threshold_basis"])
        self.hub.HfApi.return_value.model_info.assert_called_once_with(
            colab.MODEL_ID, revision=colab.MODEL_REVISION, files_metadata=True, token=False)

    def test_each_preflight_failure_prevents_model_loading(self):
        mutations = [lambda: self.torch.cuda.is_available.configure_mock(return_value=False),
                     lambda: self.torch.cuda.is_bf16_supported.configure_mock(return_value=False),
                     lambda: self.torch.cuda.mem_get_info.configure_mock(return_value=(71 * 1024**3, 96 * 1024**3)),
                     lambda: colab.host_memory.configure_mock(return_value={"available_bytes": None, "total_bytes": 64 * 1024**3}),
                     lambda: colab.shutil.disk_usage.configure_mock(return_value=SimpleNamespace(free=60_000_000_000, total=200 * 1024**3)),
                     lambda: self.hub.HfApi.return_value.model_info.configure_mock(side_effect=RuntimeError(PRIVATE)),
                     lambda: self.transformers.__dict__.update(__version__="wrong")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                with patch.dict(sys.modules, fake_libraries(self.tokenizer, self.model)):
                    # Reset the instances used by mutations to the current isolated modules.
                    self.torch, self.transformers, self.hub = (sys.modules[n] for n in ("torch", "transformers", "huggingface_hub"))
                    colab.host_memory.return_value = {"available_bytes": 32 * 1024**3, "total_bytes": 64 * 1024**3}
                    colab.shutil.disk_usage.return_value = SimpleNamespace(free=100 * 1024**3, total=200 * 1024**3)
                    run = self.runner()
                    run.environment()
                    run.tokenizer_stage()
                    mutate()
                    with self.assertRaisesRegex(colab.B0Error, "PREFLIGHT_FAILED"):
                        run.preflight()
                    with self.assertRaises(colab.B0Error):
                        run.load(download_authorized=True)
                    self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
                    self.assertEqual(run.run["status"], "aborted")
                    self.assert_private_absent(run)

    def test_bad_metadata_unknown_sizes_gating_and_revision_fail_closed(self):
        info = self.hub.HfApi.return_value.model_info.return_value
        for attrs in [{"sha": "a" * 40}, {"gated": "auto"}, {"siblings": []},
                      {"siblings": [SimpleNamespace(rfilename="weights.safetensors", size=None)]}]:
            with patch.object(self.hub.HfApi.return_value, "model_info", return_value=SimpleNamespace(**{**vars(info), **attrs})):
                self.assertFalse(colab.resource_preflight("/content/hf")["checks"]["model_metadata"])

    def test_preflight_python_accelerate_cache_and_unavailable_disk(self):
        with patch.object(colab.sys, "version_info", (3, 11, 9)):
            self.assertFalse(colab.resource_preflight("/content/hf")["checks"]["python"])
        with patch.object(self.libs["accelerate"], "__version__", "wrong"):
            self.assertFalse(colab.resource_preflight("/content/hf")["checks"]["accelerate"])
        self.assertFalse(colab.resource_preflight("/content/drive")["checks"]["cache"])
        with patch.object(colab.shutil, "disk_usage", side_effect=OSError()):
            self.assertFalse(colab.resource_preflight("/content/hf")["checks"]["disk"])

    def test_unavailable_resource_measurement_stops_before_weights(self):
        run = self.runner()
        run.environment()
        run.tokenizer_stage()
        self.torch.cuda.mem_get_info.side_effect = RuntimeError(PRIVATE)
        with self.assertRaisesRegex(colab.B0Error, "PREFLIGHT_FAILED:measurement"):
            run.preflight()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assert_private_absent(run)

    def test_load_requires_authorization_and_rechecks_resources(self):
        run = self.runner(labels_accepted=False)
        run.environment()
        run.tokenizer_stage()
        run.preflight()
        with self.assertRaisesRegex(colab.B0Error, "AUTHORIZATION"):
            run.load(download_authorized=True)
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        run = self.runner()
        run.environment()
        run.tokenizer_stage()
        run.preflight()
        with self.assertRaisesRegex(colab.B0Error, "AUTHORIZATION"):
            run.load()
        self.torch.cuda.mem_get_info.return_value = (1, 2)
        with self.assertRaisesRegex(colab.B0Error, "PREFLIGHT_FAILED:vram"):
            run.load(download_authorized=True)
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_load_assertions_and_observed_metrics(self):
        run = self.prepared()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            colab.MODEL_ID, revision=colab.MODEL_REVISION, dtype="bfloat16", device_map={"": 0},
            token=False, trust_remote_code=False)
        evidence = run.stages["load"]["evidence"]
        self.assertEqual(evidence["parameter_count"], 12)
        self.assertEqual(evidence["vision_parameter_vram_bytes"], 4)
        self.assertEqual(evidence["attention_implementation"], "stub_attention")
        self.assertEqual(evidence["peak_allocated_bytes"], 1234)
        for parameter, code in [(Parameter(device="cpu"), "LOAD_PLACEMENT"),
                                (Parameter(device="cuda:1"), "LOAD_PLACEMENT"),
                                (Parameter(dtype="float32"), "LOAD_DTYPE")]:
            self.model.parameters = [("weight", parameter)]
            with self.assertRaisesRegex(colab.B0Error, code):
                colab.assert_model(self.model, self.torch)
        self.model.parameters = [("indices", Parameter(dtype="int64", floating=False))]
        self.assertEqual(colab.assert_model(self.model, self.torch)["parameter_count"], 10)

    def test_load_oom_and_class_incompatibility_do_not_fallback(self):
        for error, code in [(FakeOOM(PRIVATE), "LOAD_OOM"), (RuntimeError(PRIVATE), "LOAD_INCOMPATIBLE")]:
            run = self.runner()
            run.environment()
            run.tokenizer_stage()
            run.preflight()
            self.transformers.AutoModelForCausalLM.from_pretrained.side_effect = error
            with self.assertRaisesRegex(colab.B0Error, code):
                run.load(download_authorized=True)
            self.assertEqual(run.run["status"], "aborted")
            self.assert_private_absent(run)
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_count, 2)

    def test_bad_placement_or_dtype_aborts_loading_stage(self):
        for parameter, code in [(Parameter(device="cpu"), "LOAD_PLACEMENT"),
                                (Parameter(dtype="float32"), "LOAD_DTYPE")]:
            run = self.runner()
            run.environment()
            run.tokenizer_stage()
            run.preflight()
            self.model.parameters = [("weight", parameter)]
            with self.assertRaisesRegex(colab.B0Error, code):
                run.load(download_authorized=True)
            self.assertEqual(run.stages["load"]["status"], "failed")
            self.assertEqual(run.run["status"], "aborted")

    def test_genuine_smoke_needs_final_json_and_end_token(self):
        for response in [native("not JSON"), native(answer(), ended=False),
                         " to=self<|message|>" + PRIVATE + "<|eot|>"]:
            run = self.runner()
            run.environment()
            run.tokenizer_stage()
            run.preflight()
            run.load(download_authorized=True)
            self.model.responses = [response]
            with self.assertRaises(colab.B0Error):
                run.smoke()
            self.assertEqual(run.run["status"], "aborted")
            self.assertEqual(run.records, [])
            self.assert_private_absent(run)

    def test_complete_run_and_byte_identical_offline_rescoring(self):
        run = self.prepared()
        run.run["compute_units_before"] = 10
        self.model.responses = self.ideal_responses(run)
        summary = run.evaluate(compute_units_after=lambda: 8)
        self.assertEqual(run.run["status"], "complete")
        self.assertEqual(summary["classification"], "complete baseline")
        self.assertEqual(len(run.records), 16)
        self.assertEqual(len(self.model.calls), 17)  # Separate q01 smoke, then one call per full-run item.
        self.assertEqual(summary["metrics"]["correct_answer"]["numerator"], 11)
        self.assertEqual(summary["metrics"]["correct_abstention"]["numerator"], 3)
        self.assertEqual(summary["metrics"]["contradiction_disclosure"]["numerator"], 2)
        self.assertEqual(summary["metrics"]["citation_validity"]["numerator"], 17)
        self.assertEqual(summary["compute_units_consumed"], 2)
        self.assertEqual(summary["load_metrics"]["parameter_count"], 12)
        for name in colab.ARTIFACTS:
            self.assertEqual((run.path / name).read_bytes(), (run.drive / name).read_bytes())
        before = (run.path / "summary.json").read_bytes()
        pilot.score_run(run.path)
        self.assertEqual(before, (run.path / "summary.json").read_bytes())
        subprocess.run([sys.executable, "-m", "hector.pilot_b0", "score", str(run.path)],
                       check=True, capture_output=True, text=True)
        self.assertEqual(before, (run.path / "summary.json").read_bytes())
        review = run.path / "review.csv"
        review.write_bytes(review.read_bytes() + b"founder annotation\n")
        human = review.read_bytes()
        pilot.score_run(run.path)
        self.assertEqual(review.read_bytes(), human)
        self.assert_private_absent(run)
        self.assertEqual(self.torch.manual_seed.call_args_list[0].args, (config.B0_SEED,))
        self.assertEqual([c.args[0] for c in self.torch.manual_seed.call_args_list[1:]],
                         [config.B0_SEED + n for n in range(16)])
        for call in self.model.calls:
            self.assertEqual({k: call[k] for k in run.run["generation"] if k != "eos_tokens"},
                             {k: v for k, v in run.run["generation"].items() if k != "eos_tokens"})
            self.assertEqual(call["eos_token_id"], [4, 5])

    def test_runtime_failures_continue_and_three_nonconsecutive_failures_are_limited(self):
        run = self.prepared()
        values = self.ideal_responses(run)
        values[1], values[4], values[8] = FakeOOM(PRIVATE), TimeoutError(PRIVATE), native(answer(), ended=False)
        self.model.responses = values
        summary = run.evaluate()
        self.assertEqual(summary["classification"], "runtime-limited")
        self.assertEqual(summary["failures_by_code"], {"OOM": 1, "TIMEOUT": 1, "TRUNCATED": 1})
        self.assertEqual(run.records[1]["peak_allocated_bytes"], 1234)
        self.assertEqual(run.run["status"], "complete")
        self.assertEqual(len(run.records), 16)
        self.assert_private_absent(run)
        self.torch.cuda.empty_cache.assert_called()

    def test_two_consecutive_ooms_abort_and_save_both_records(self):
        run = self.prepared()
        self.model.responses = [FakeOOM(PRIVATE), FakeOOM(PRIVATE)]
        summary = run.evaluate()
        self.assertEqual(run.run["status"], "aborted")
        self.assertEqual(summary["classification"], "incomplete")
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["missing_items"], 14)
        self.assertEqual(summary["failures_by_code"], {"OOM": 2})
        self.assert_private_absent(run)
        with self.assertRaises(colab.B0Error):
            run.evaluate()

    def test_generation_exception_text_is_not_persisted_or_counted_as_abstention(self):
        run = self.prepared()
        self.model.responses = [RuntimeError(PRIVATE)] + self.ideal_responses(run)[1:]
        with self.assertRaisesRegex(colab.B0Error, "INTERRUPTED"):
            run.evaluate()
        self.assertEqual(run.run["status"], "aborted")
        self.assertEqual(run.run["failure_source"], "GENERATION_EXCEPTION")
        self.assertEqual(run.records, [])
        self.assertFalse((run.path / "summary.json").exists())
        self.assert_private_absent(run)

    def test_unexpected_cuda_exception_aborts_and_preserves_previous_record(self):
        run = self.prepared()
        self.model.responses = [native(answer()), RuntimeError("CUDA device error " + PRIVATE), native(answer())]
        balance = Mock(return_value=8)
        with self.assertRaisesRegex(colab.B0Error, "INTERRUPTED"):
            run.evaluate(compute_units_after=balance)
        balance.assert_not_called()
        self.assertEqual(run.run["status"], "aborted")
        self.assertEqual(run.run["failure"], "INTERRUPTED")
        self.assertEqual(run.run["failure_source"], "GENERATION_EXCEPTION")
        self.assertEqual(run.run["failed_item_id"], "q02")
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual([r["id"] for r in run.records], ["q01"])
        for path in [run.path, run.drive]:
            self.assertEqual(len(pilot.jsonl_records((path / "items.jsonl").read_text(encoding="utf-8"))), 1)
            self.assertEqual(pilot.read_json(path / "run.json")["status"], "aborted")
        with self.assertRaises(colab.B0Error):
            run.evaluate()
        self.assert_private_absent(run)

    def test_genuine_missing_final_message_is_item_failure_and_continues(self):
        run = self.prepared()
        self.model.responses = [" to=self<|message|>" + PRIVATE + "<|eot|>"] + self.ideal_responses(run)[1:]
        summary = run.evaluate()
        self.assertEqual(run.run["status"], "complete")
        self.assertEqual(len(run.records), 16)
        self.assertEqual(run.records[0]["outcome"], "FAILED:NO_FINAL_MESSAGE")
        self.assertEqual(summary["failures_by_code"], {"NO_FINAL_MESSAGE": 1})
        self.assertEqual(summary["metrics"]["correct_abstention"]["numerator"], 3)
        self.assert_private_absent(run)

    def test_unexpected_cuda_metrics_or_cleanup_failure_is_sanitized_and_fatal(self):
        for operation in ("metrics", "cleanup"):
            with self.subTest(operation=operation):
                run = self.prepared()
                if operation == "metrics":
                    failing_call = self.torch.cuda.max_memory_allocated
                else:
                    failing_call = self.torch.cuda.empty_cache
                    self.model.responses = [FakeOOM(PRIVATE)]
                failing_call.side_effect = RuntimeError(PRIVATE)
                try:
                    with self.assertRaisesRegex(colab.B0Error, "INTERRUPTED"):
                        run.evaluate()
                    self.assertEqual(run.run["status"], "aborted")
                    self.assertEqual(run.run["failure_source"], "GENERATION_EXCEPTION")
                    self.assertEqual(run.records, [])
                    self.assert_private_absent(run)
                finally:
                    failing_call.side_effect = None

    def test_unicode_answers_persist_parse_score_and_rescore(self):
        for separator in ("\u2028", "\u2029", "\u0085"):
            with self.subTest(separator=repr(separator)):
                run = self.prepared()
                self.model.responses = self.ideal_responses(run)
                expected = "three nights" + separator + "from the policy"
                self.model.responses[0] = native(answer(expected))
                summary = run.evaluate()
                raw = (run.path / "items.jsonl").read_bytes()
                self.assertIn(separator.encode("utf-8"), raw)
                self.assertEqual(raw.count(b"\n"), 16)
                records = pilot.jsonl_records(raw.decode("utf-8"))
                self.assertEqual(len(records), 16)
                self.assertEqual(records[0]["parsed"]["answer"], expected)
                self.assertEqual(summary["metrics"]["correct_answer"]["numerator"], 11)
                before = (run.path / "summary.json").read_bytes()
                pilot.score_run(run.path)
                self.assertEqual((run.path / "summary.json").read_bytes(), before)
                self.assert_private_absent(run)

    def test_identical_rescoring_does_not_write_or_replace_evidence(self):
        run = self.prepared()
        self.model.responses = self.ideal_responses(run)
        run.evaluate()
        before = {name: ((run.path / name).read_bytes(), (run.path / name).stat().st_mtime_ns)
                  for name in colab.ARTIFACTS}
        with patch.object(pilot.os, "link", side_effect=AssertionError("RESCORE_MUST_NOT_PUBLISH")):
            pilot.score_run(run.path)
        after = {name: ((run.path / name).read_bytes(), (run.path / name).stat().st_mtime_ns)
                 for name in colab.ARTIFACTS}
        self.assertEqual(after, before)

    def test_changed_scoring_rule_exits_nonzero_without_mutation(self):
        run = self.prepared()
        self.model.responses = self.ideal_responses(run)
        run.evaluate()
        before = {path.name: path.read_bytes() for path in run.path.iterdir()}
        original = pilot.score_item

        def changed_rule(item, record, documents):
            score = original(item, record, documents)
            if item["id"] == "q01":
                score["correct"] = False
            return score

        with patch.object(pilot, "score_item", side_effect=changed_rule), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                pilot.main(["score", str(run.path)])
        self.assertEqual(stopped.exception.code, 1)
        self.assertEqual({path.name: path.read_bytes() for path in run.path.iterdir()}, before)

    def test_mismatched_scoring_evidence_refuses_cli_without_partial_creation(self):
        for name in ("summary.json", "summary.md", "REPORT.md"):
            with self.subTest(name=name):
                run = self.prepared()
                self.model.responses = self.ideal_responses(run)
                run.evaluate()
                for output in ("summary.json", "summary.md", "REPORT.md", "review.csv"):
                    path = run.path / output
                    if output == name:
                        path.write_bytes(b"Previously recorded evidence, preserve exactly.\n")
                    else:
                        path.unlink()
                before = {path.name: path.read_bytes() for path in run.path.iterdir()}
                result = subprocess.run([sys.executable, "-m", "hector.pilot_b0", "score", str(run.path)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual({path.name: path.read_bytes() for path in run.path.iterdir()}, before)

    def test_scoring_failure_aborts_before_persisting_complete_status(self):
        run = self.prepared()
        self.model.responses = self.ideal_responses(run)

        def fail_score(path, **kwargs):
            self.assertEqual(pilot.read_json(path / "run.json")["status"], "started")
            self.assertEqual(kwargs["run_metadata"]["status"], "complete")
            raise ValueError(PRIVATE)

        with patch.object(pilot, "score_run", side_effect=fail_score):
            with self.assertRaisesRegex(colab.B0Error, "INTERRUPTED"):
                run.evaluate()
        self.assertEqual(len(run.records), 16)
        for path in [run.path, run.drive]:
            self.assertEqual(pilot.read_json(path / "run.json")["status"], "aborted")
            self.assertFalse((path / "summary.json").exists())
        self.assertEqual(run.run["failure_source"], "FINALIZATION")
        self.assert_private_absent(run)
        with self.assertRaises(colab.B0Error):
            run.evaluate()

    def test_compute_balance_is_collected_after_generation_before_finalization(self):
        run = self.prepared()
        run.run["compute_units_before"] = 10
        self.model.responses = self.ideal_responses(run)

        def observed_after():
            self.assertEqual(len(self.model.calls), 17)
            self.assertEqual(len(run.records), 16)
            metadata = pilot.read_json(run.path / "run.json")
            self.assertEqual(metadata["status"], "started")
            self.assertIsNone(metadata["compute_units_after"])
            self.assertFalse((run.path / "summary.json").exists())
            return 8

        summary = run.evaluate(compute_units_after=observed_after)
        metadata = pilot.read_json(run.path / "run.json")
        self.assertEqual(metadata["compute_units_after"], 8)
        self.assertEqual(metadata["compute_units_source"], "founder_entered")
        self.assertEqual(summary["compute_units_consumed"], 2)
        pilot.score_run(run.path)

    def test_numeric_after_balance_is_refused_before_generation(self):
        run = self.prepared()
        with self.assertRaisesRegex(colab.B0Error, "INVALID_COMPUTE_UNIT_BALANCE"):
            run.evaluate(compute_units_after=8)
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(run.records, [])

    def test_invalid_or_interrupted_balance_never_finalizes(self):
        for error in (None, KeyboardInterrupt()):
            run = self.prepared()
            self.model.responses = self.ideal_responses(run)

            def bad_balance():
                if error is not None:
                    raise error
                return PRIVATE

            with self.assertRaises(KeyboardInterrupt if error else colab.B0Error):
                run.evaluate(compute_units_after=bad_balance)
            self.assertEqual(pilot.read_json(run.path / "run.json")["status"], "started" if error else "aborted")
            self.assertFalse((run.path / "summary.json").exists())
            self.assert_private_absent(run)
            with self.assertRaises(colab.B0Error):
                run.evaluate()

    def test_compute_unit_prompt_sanitizes_input_and_allows_unobserved(self):
        for text, expected in [("", None), ("8.5", 8.5)]:
            with patch("builtins.input", return_value=text):
                self.assertEqual(colab.prompt_compute_units_after(), expected)
        for text in (PRIVATE, "nan", "-1"):
            with patch("builtins.input", side_effect=[text, "8.5"]) as read, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(colab.prompt_compute_units_after(), 8.5)
            self.assertEqual(read.call_count, 2)
            self.assertIn("finite, non-negative number", output.getvalue())
            self.assertNotIn(PRIVATE, output.getvalue())

    def test_compute_unit_input_retry_preserves_completed_baseline(self):
        cases = [([invalid, "91.37"], 91.37) for invalid in ("91,37", "91.37 units", "-5", "nan", "inf")]
        cases += [([""], None), (["0"], 0.0)]
        for entries, expected in cases:
            with self.subTest(entries=entries):
                run = self.prepared()
                self.model.responses = self.ideal_responses(run)
                generations_before = len(self.model.calls)
                loads_before = self.transformers.AutoModelForCausalLM.from_pretrained.call_count
                answers = iter(entries)
                persisted_records = []

                def read_balance(_):
                    self.assertEqual(len(run.records), 16)
                    self.assertEqual(len(self.model.calls), generations_before + 16)
                    self.assertEqual(run.run["status"], "started")
                    self.assertNotIn("failure", run.run)
                    metadata = pilot.read_json(run.path / "run.json")
                    self.assertEqual(metadata["status"], "started")
                    self.assertIsNone(metadata["compute_units_after"])
                    raw = (run.path / "items.jsonl").read_bytes()
                    if persisted_records:
                        self.assertEqual(raw, persisted_records[0])
                    else:
                        persisted_records.append(raw)
                    return next(answers)

                with patch("builtins.input", side_effect=read_balance) as read, redirect_stdout(io.StringIO()) as output:
                    summary = run.evaluate(compute_units_after=colab.prompt_compute_units_after)
                self.assertEqual(read.call_count, len(entries))
                self.assertEqual(output.getvalue().count("finite, non-negative number"), len(entries) - 1)
                self.assertEqual(len(self.model.calls), generations_before + 16)
                self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_count, loads_before)
                self.assertEqual((run.path / "items.jsonl").read_bytes(), persisted_records[0])
                records = pilot.jsonl_records(persisted_records[0].decode("utf-8"))
                self.assertEqual([r["id"] for r in records], [item["id"] for item in run.items])
                for directory in (run.path, run.drive):
                    metadata = pilot.read_json(directory / "run.json")
                    self.assertEqual(metadata["status"], "complete")
                    self.assertNotIn("failure", metadata)
                    self.assertEqual(metadata["compute_units_after"], expected)
                    self.assertEqual(metadata["compute_units_source"], "founder_entered")
                    self.assertTrue((directory / "summary.json").exists())
                self.assertEqual(summary["classification"], "complete baseline")
                self.assertEqual(summary["records"], 16)
                before = {name: ((run.path / name).read_bytes(), (run.path / name).stat().st_mtime_ns)
                          for name in colab.ARTIFACTS}
                pilot.score_run(run.path)
                self.assertEqual(before, {name: ((run.path / name).read_bytes(), (run.path / name).stat().st_mtime_ns)
                                          for name in colab.ARTIFACTS})
                self.assert_private_absent(run)

    def test_compute_unit_input_eof_or_interrupt_stops_without_retry(self):
        for terminal in (EOFError(), KeyboardInterrupt()):
            with self.subTest(terminal=type(terminal).__name__):
                run = self.prepared()
                self.model.responses = self.ideal_responses(run)
                generations_before = len(self.model.calls)
                loads_before = self.transformers.AutoModelForCausalLM.from_pretrained.call_count
                with patch("builtins.input", side_effect=["91,37", terminal]) as read, redirect_stdout(io.StringIO()):
                    with self.assertRaises(colab.B0Error if isinstance(terminal, EOFError) else KeyboardInterrupt):
                        run.evaluate(compute_units_after=colab.prompt_compute_units_after)
                self.assertEqual(read.call_count, 2)
                self.assertEqual(len(self.model.calls), generations_before + 16)
                self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_count, loads_before)
                self.assertEqual(len(run.records), 16)
                for directory in (run.path, run.drive):
                    metadata = pilot.read_json(directory / "run.json")
                    self.assertEqual(metadata["status"], "aborted" if isinstance(terminal, EOFError) else "started")
                    self.assertIsNone(metadata["compute_units_after"])
                    self.assertFalse((directory / "summary.json").exists())
                if isinstance(terminal, EOFError):
                    self.assertEqual(run.run["failure"], "INTERRUPTED")
                self.assert_private_absent(run)

    def test_timeout_without_eot_and_invalid_output_are_distinct(self):
        run = self.prepared()
        self.model.responses = [native(answer(), ended=False)]
        with patch.object(colab.time, "perf_counter", side_effect=[0, config.B0_MAX_TIME_S + 1]):
            record = run._generate(0)
        self.assertEqual(record["outcome"], "FAILED:TIMEOUT")
        self.model.responses = [native("invalid json")]
        self.assertEqual(run._generate(0)["outcome"], "INVALID_OUTPUT")

    def test_prompt_change_fails_without_calling_generate(self):
        run = self.prepared()
        calls = len(self.model.calls)
        run.documents["vendor_notes.md"] += "changed"
        with self.assertRaisesRegex(colab.B0Error, "INTERRUPTED"):
            run.evaluate()
        self.assertEqual(len(self.model.calls), calls)
        self.assertEqual(run.run["status"], "aborted")
        self.assertEqual(run.run["failure_source"], "PROMPT_CHANGED")
        self.assertEqual(run.run["failed_item_id"], "q01")
        self.assertEqual(run.records, [])
        self.assert_private_absent(run)

    def test_interruption_keeps_started_and_cannot_resume(self):
        run = self.prepared()
        self.model.responses = [native(answer()), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            run.evaluate()
        self.assertEqual(pilot.read_json(run.path / "run.json")["status"], "started")
        self.assertEqual(len((run.drive / "items.jsonl").read_text().splitlines()), 1)
        self.assertEqual(pilot.score_run(run.path)["classification"], "incomplete")
        with self.assertRaises(colab.B0Error):
            run.evaluate()
        next_run = self.runner()
        self.assertNotEqual(next_run.run_id, run.run_id)
        self.assertIn({"run_id": run.run_id, "status": "started"}, next_run.run["other_runs"])
        self.assert_private_absent(run)

    def test_stage_order_and_current_session_are_required(self):
        run = self.runner()
        for action in [run.tokenizer_stage, run.preflight, run.load, run.smoke, run.evaluate]:
            with self.assertRaises(colab.B0Error):
                action()
        run.environment()
        with self.assertRaises(colab.B0Error):
            run.environment()
        with patch.object(colab, "SESSION_ID", "another-runtime"):
            with self.assertRaises(colab.B0Error):
                run.tokenizer_stage()
            with self.assertRaises(colab.B0Error):
                run.evaluate()

    def test_offline_scoring_rejects_mixed_stage_evidence(self):
        run = self.prepared()
        self.model.responses = self.ideal_responses(run)
        run.evaluate()
        path = run.path / "stages.json"
        original = pilot.read_json(path)
        for mutate in [lambda s: s.update(session_id="old-session"),
                       lambda s: s["stages"]["smoke"].update(session_id="old-session"),
                       lambda s: s["stages"].pop("load")]:
            changed = copy.deepcopy(original)
            mutate(changed)
            pilot.write_json(path, changed)
            with self.assertRaises(ValueError):
                pilot.score_run(run.path)

    def test_empty_run_and_sixteen_records_with_started_status_are_incomplete(self):
        run = self.runner()
        self.assertEqual(pilot.score_run(run.path)["classification"], "incomplete")
        run = self.prepared()
        self.model.responses = self.ideal_responses(run)
        run.evaluate()
        metadata = pilot.read_json(run.path / "run.json")
        metadata["status"] = "started"
        pilot.write_json(run.path / "run.json", metadata)
        before = (run.path / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "SCORING_EVIDENCE_MISMATCH"):
            pilot.score_run(run.path)
        self.assertEqual((run.path / "summary.json").read_bytes(), before)

    def test_release_drops_model_and_clears_cuda_cache(self):
        run = self.prepared()
        run.release()
        self.assertIsNone(run.model)
        self.torch.cuda.empty_cache.assert_called()
        with self.assertRaises(colab.B0Error):
            run.evaluate()

    def test_compute_unit_values_and_unpinned_sha_fail_before_creation(self):
        for value in ["credential-or-text", -1, float("nan"), True]:
            with self.assertRaises(colab.B0Error):
                self.runner(compute_units_before=value)
        with self.assertRaises(colab.B0Error):
            colab.B0Run("main", repo=self.root)


class RepositoryNotebookTests(unittest.TestCase):
    def test_repository_checks_exact_identity_sha_and_clean_tree(self):
        with patch.object(colab.subprocess, "check_output") as command:
            for values, expected in [([SHA, "", "https://github.com/TenzyZ/hector-evidence-agent.git"], None),
                                     (["f" * 40, "", "https://github.com/TenzyZ/hector-evidence-agent.git"], "SHA_OR_DIRTY"),
                                     ([SHA, " M file", "https://github.com/TenzyZ/hector-evidence-agent.git"], "SHA_OR_DIRTY"),
                                     ([SHA, "", "https://github.com/TenzyZ/other.git"], "IDENTITY")]:
                command.side_effect = values
                if expected:
                    with self.assertRaisesRegex(colab.B0Error, expected):
                        colab.check_repository(".", SHA)
                else:
                    self.assertEqual(colab.check_repository(".", SHA), SHA)
            with self.assertRaisesRegex(colab.B0Error, "EXACT"):
                colab.check_repository(".", "main")

    def test_notebook_is_unexecuted_thin_python_with_pins_and_disabled_authority(self):
        path = pilot.DATASET.parents[1] / "notebooks/b0_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                compile(source, f"notebook-cell-{index}", "exec")
                code.append(source)
        all_code = "\n".join(code)
        self.assertIn("LABELS_ACCEPTED = False", all_code)
        self.assertIn("DOWNLOAD_AUTHORIZED = False", all_code)
        self.assertIn('"transformers==' + colab.TRANSFORMERS_PIN + '"', all_code)
        self.assertIn('"accelerate==' + colab.ACCELERATE_PIN + '"', all_code)
        self.assertEqual(all_code.count('"pip", "install"'), 1)
        self.assertNotIn("model.generate", all_code)
        self.assertNotIn("from_pretrained", all_code)
        self.assertIn("finally:\n    run.release()", all_code)
        self.assertNotIn("COMPUTE_UNITS_AFTER =", all_code)
        self.assertIn("run.evaluate(compute_units_after=prompt_compute_units_after)", all_code)


if __name__ == "__main__":
    unittest.main()
