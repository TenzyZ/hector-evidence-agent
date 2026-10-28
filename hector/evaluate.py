"""Dataset validation and deterministic Configuration A evaluation; no model calls."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import uuid

from . import config
from .ingest import ingest
from .pilot_b0 import json_bytes, proportion, sha256, strict_json
from .store import open_store, query_terms


class EvaluationError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass
class Dataset:
    documents: dict
    chunks: dict
    items: list
    identities: dict


def validate(corpus, questions) -> Dataset:
    corpus = Path(corpus)
    if not corpus.is_dir():
        raise EvaluationError("INVALID_CORPUS", "Corpus must be an existing directory.")
    documents, chunks, files = {}, {}, {}
    with open_store() as store:
        for path in sorted(corpus.iterdir()):
            if not path.is_file():
                raise EvaluationError("INVALID_CORPUS", "Corpus entries must be document files.")
            with path.open("rb") as stream:
                data = stream.read(config.MAX_FILE_BYTES + 1)
            document, passages = ingest(path.name, data)
            store.add(document, passages)
            documents[path.name] = document
            chunks[path.name] = passages
            files[path.name] = sha256(data)
    if not documents:
        raise EvaluationError("EMPTY_CORPUS", "Corpus must contain documents.")
    data = Path(questions).read_bytes()
    try:
        lines = data.decode("utf-8-sig").split("\n")
    except UnicodeDecodeError:
        raise EvaluationError("INVALID_DATASET", "Questions must be UTF-8 JSONL.") from None
    if lines[-1] == "":
        lines.pop()
    if not lines:
        raise EvaluationError("INVALID_DATASET", "Questions must contain items.")
    items, seen = [], set()
    for number, line in enumerate(lines, 1):
        try:
            item = strict_json(line)
        except ValueError:
            raise EvaluationError("INVALID_DATASET", f"Invalid JSON on line {number}.") from None
        _validate_item(item, number, documents, chunks)
        if item["id"] in seen:
            raise EvaluationError("DUPLICATE_ID", f"Duplicate item ID on line {number}.")
        seen.add(item["id"])
        items.append(item)
    identities = {
        "corpus_files_sha256": files,
        "corpus_sha256": sha256(json_bytes(files)),
        "questions_sha256": sha256(data),
        "hash_method": "Raw file bytes; combined hashes use sorted UTF-8 JSON with indent=2 and trailing LF.",
        "normalized_documents_sha256": {name: doc.sha256 for name, doc in documents.items()},
    }
    identities["dataset_sha256"] = sha256(json_bytes({
        "corpus": files, "questions_sha256": identities["questions_sha256"]}))
    return Dataset(documents, chunks, items, identities)


def _validate_item(item, number, documents, chunks):
    required = {"id", "split", "category", "question", "expected_outcome", "gold_evidence",
                "key_facts", "trap_fact", "canary", "notes"}
    if not isinstance(item, dict) or item.keys() != required:
        raise EvaluationError("INVALID_ITEM", f"Missing or unexpected schema fields on line {number}.")
    if any(not isinstance(item[key], str) or not item[key].strip() for key in ("id", "question")):
        raise EvaluationError("INVALID_ITEM", f"Invalid ID or question on line {number}.")
    if not config.QUESTION_CHARS[0] <= len(item["question"]) <= config.QUESTION_CHARS[1]:
        raise EvaluationError("INVALID_ITEM", f"Question length outside QUESTION_CHARS on line {number}.")
    if not query_terms(item["question"]):
        raise EvaluationError("INVALID_ITEM", f"Question has no searchable word tokens on line {number}.")
    enums = {
        "split": ("dev", "test"),
        "category": ("answerable_single", "answerable_multi", "unanswerable", "contradictory",
                     "misleading_quote", "injection", "fabrication_bait"),
        "expected_outcome": ("ANSWER", "ABSTAIN", "DISCLOSE_CONFLICT"),
    }
    if any(item[key] not in values for key, values in enums.items()):
        raise EvaluationError("INVALID_ITEM", f"Invalid enum on line {number}.")
    if (not isinstance(item["notes"], str)
            or any(item[key] is not None and not isinstance(item[key], str) for key in ("trap_fact", "canary"))
            or not isinstance(item["key_facts"], list)
            or any(not isinstance(fact, str) or not fact.strip() for fact in item["key_facts"])
            or not isinstance(item["gold_evidence"], list)):
        raise EvaluationError("INVALID_ITEM", f"Invalid schema field types on line {number}.")
    for evidence in item["gold_evidence"]:
        if (not isinstance(evidence, dict)
                or not isinstance(evidence.get("doc"), str)
                or not isinstance(evidence.get("quote"), str)
                or not evidence["quote"].strip()):
            raise EvaluationError("INVALID_GOLD", f"Invalid gold evidence on line {number}.")
        name, quote = evidence["doc"], evidence["quote"]
        if name not in documents or quote not in documents[name].text:
            raise EvaluationError("INVALID_GOLD", f"Gold quote absent from its document on line {number}.")
        if not any(quote in passage.text for passage in chunks[name]):
            raise EvaluationError("GOLD_CROSSES_CHUNKS", f"Gold quote crosses chunk boundaries on line {number}.")


def _git_state():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--ignore-submodules=all"],
                                cwd=root, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        raise EvaluationError("PROVENANCE_UNAVAILABLE", "Cannot record local Git provenance.") from None
    return {"git_commit": commit, "dirty": bool(status)}


def run(corpus, questions, *, split="dev", configuration="A", output_root="eval/results"):
    if configuration != "A" or split != "dev":
        raise EvaluationError("UNAUTHORIZED_EVALUATION", "Phase 02 runs Configuration A on dev fixtures only.")
    dataset = validate(corpus, questions)
    items = [item for item in dataset.items if item["split"] == split]
    if not items:
        raise EvaluationError("EMPTY_SPLIT", "No items in the selected split.")
    provenance = _git_state()
    records = []
    gold_found = gold_total = abstained = abstain_total = 0
    reciprocal_ranks = []
    now = datetime.now(timezone.utc)
    with open_store() as store:
        for name, document in dataset.documents.items():
            store.add(document, dataset.chunks[name])
        for item in items:
            started = time.perf_counter()
            hits = store.search(item["question"], config.PRESEED_K)
            gold = item["gold_evidence"]

            def contains_gold(hit, evidence):
                return (hit.doc_id == dataset.documents[evidence["doc"]].doc_id
                        and evidence["quote"] in hit.text)

            found = sum(any(contains_gold(hit, evidence) for hit in hits) for evidence in gold)
            first = next((rank for rank, hit in enumerate(hits, 1)
                          if any(contains_gold(hit, evidence) for evidence in gold)), None)
            reciprocal = 1 / first if first else 0.0
            if gold:
                reciprocal_ranks.append(reciprocal)
            gold_found += found
            gold_total += len(gold)
            if item["expected_outcome"] == "ABSTAIN":
                abstain_total += 1
                abstained += not hits
            records.append({
                "run_id": str(uuid.uuid4()), "item_id": item["id"], "run": 1, "config": "A",
                "split": split, "category": item["category"], "question": item["question"],
                "outcome": "ANSWERED" if hits else "ABSTAINED", "reason": None,
                "contradiction_detected": False, "claims": [],
                "usage": {"model_calls": 0, "judge_calls": 0, "input_tokens": 0,
                          "output_tokens": 0, "cost_usd": 0.0,
                          "latency_ms": (time.perf_counter() - started) * 1000},
                "versions": {"agent_prompt": None, "judge_prompt": None, "model": None,
                             "base_url_host": None, "git_commit": provenance["git_commit"]},
                "steps": [{"step": 1, "tool": "search_passages",
                           "arguments": {"query": item["question"], "k": config.PRESEED_K},
                           "result_chunk_ids": [hit.chunk_id for hit in hits], "error": None,
                           "input_tokens": 0, "output_tokens": 0,
                           "latency_ms": (time.perf_counter() - started) * 1000}],
                "hits": [asdict(hit) for hit in hits], "gold_found": found,
                "gold_total": len(gold), "reciprocal_rank": reciprocal,
            })
    metrics = {"R1": proportion(gold_found, gold_total),
               "R2": {"value": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
                      "denominator": len(reciprocal_ranks),
                      "reciprocal_rank_sum": sum(reciprocal_ranks)},
               "O1": proportion(abstained, abstain_total)}
    for metric in metrics.values():
        value = metric.get("rate", metric.get("value"))
        metric.update(mean=value, min=value, max=value)
    summary = {
        "config": "A", "split": split, "run_count": 1, "item_count": len(items),
        "utc_date": now.isoformat(), **provenance, **dataset.identities,
        "agent_prompt_version": None, "judge_prompt_version": None, "model": None,
        "model_repository": None, "model_revision": None, "hardware": None,
        "base_url_host": None, "sampling": None,
        "limits": {key: getattr(config, key) for key in vars(config) if key.isupper()},
        "total_tokens": 0, "total_cost_usd": 0.0, "metrics": metrics,
        "metric_definitions": {
            "R1": "Gold quotes in their document's top-5 hits / all gold quotes (micro recall).",
            "R2": "Mean reciprocal first gold-hit rank over items with gold evidence; a miss contributes zero; no eligible items yields null.",
            "O1": "Zero-hit ABSTAIN items / all ABSTAIN items; other outcomes never count as abstention.",
        },
        "limitations": [
            "Small synthetic dev fixtures only; no production evaluation, G1 result or test-split result.",
            "Lexical retrieval and English-only items; document support does not establish document truth.",
            "Test-split isolation is a process rule; validate reads labels but run accepts dev only in Phase 02.",
            "Human ratings, a same-family judge and model precision comparisons do not apply to A.",
            "One deterministic retrieval run; mean and min-max coincide. Wilson 95% intervals for proportions.",
        ],
    }
    output = Path(output_root) / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "summary.json").write_bytes(json_bytes(summary))
    markdown = "# Configuration A — retrieval only\n\n"
    markdown += f"UTC: {summary['utc_date']} · Split: {split} · Items: {len(items)} · Runs: 1\n\n"
    markdown += "| Metric | Value | Numerator / denominator | Wilson 95% |\n|---|---|---|---|\n"
    for name, metric in metrics.items():
        markdown += (f"| {name} | {metric['mean']} | "
                     f"{metric.get('numerator', metric.get('reciprocal_rank_sum'))} / {metric['denominator']} | "
                     f"{metric.get('wilson95')} |\n")
    markdown += "\n" + "\n".join("- " + text for text in summary["metric_definitions"].values()) + "\n\n"
    markdown += "\n".join("- " + text for text in summary["limitations"]) + "\n\n"
    markdown += "Provenance, file identities and all limits: `summary.json`.\n"
    (output / "summary.md").write_bytes(markdown.encode("utf-8"))
    return {"output_dir": str(output), "summary": summary}
