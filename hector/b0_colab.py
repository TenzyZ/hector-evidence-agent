"""Founder-run Colab stages. Importing this module never imports ML libraries.

No resume, offloading, alternate model class, precision fallback or inference retry.
"""

from datetime import datetime, timezone
import gc
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

from . import config
from . import pilot_b0 as pilot

MODEL_ID = "meta-models/Muse-Glimmer-30B"
# Anonymous model metadata verified 2026-10-28:
# https://huggingface.co/api/models/meta-models/Muse-Glimmer-30B
MODEL_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
TRANSFORMERS_PIN = "5.20.2"
ACCELERATE_PIN = "1.15.0"
# PyPI metadata verified 2026-10-28; both releases are at least seven days old.
# https://pypi.org/pypi/transformers/5.20.2/json (2026-10-19)
# https://pypi.org/pypi/accelerate/1.15.0/json (2026-09-09)
RELEASE_DATES = {"transformers": "2026-10-19", "accelerate": "2026-09-09"}
SESSION_ID = uuid.uuid4().hex
STAGES = ("environment", "tokenizer", "preflight", "load", "smoke")
ARTIFACTS = ("stages.json", "run.json", "items.jsonl", "summary.json", "summary.md", "review.csv", "REPORT.md")


class B0Error(RuntimeError):
    """A fixed reason code, never an upstream exception or model text."""


def check_repository(repo, expected_sha):
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise B0Error("EXACT_REPOSITORY_SHA_REQUIRED")
    if Path(repo).resolve() != Path(__file__).resolve().parents[1]:
        raise B0Error("EXECUTING_CODE_REPOSITORY_MISMATCH")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=repo, text=True)
        origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise B0Error("REPOSITORY_CHECK_FAILED") from None
    if origin != "https://github.com/TenzyZ/hector-evidence-agent.git":
        raise B0Error("REPOSITORY_IDENTITY")
    if head != expected_sha or dirty:
        raise B0Error("REPOSITORY_SHA_OR_DIRTY_TREE")
    return head


def host_memory():
    """Linux MemAvailable is measured, not inferred from free-only RAM."""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(value.split()[0]) * 1024
        return {"available_bytes": values.get("MemAvailable"), "total_bytes": values.get("MemTotal")}
    except (OSError, ValueError):
        return {"available_bytes": None, "total_bytes": None}


def environment_snapshot(repo, expected_sha):
    import torch
    import transformers
    import accelerate

    head = check_repository(repo, expected_sha)
    if sys.version_info < (3, 12):
        raise B0Error("PREFLIGHT_FAILED:python")
    if transformers.__version__ != TRANSFORMERS_PIN or accelerate.__version__ != ACCELERATE_PIN:
        raise B0Error("PINNED_PACKAGE_MISMATCH")
    if os.environ.get("HF_HOME") != "/content/hf":
        raise B0Error("VM_LOCAL_CACHE_REQUIRED")
    for date in RELEASE_DATES.values():
        if (datetime.now(timezone.utc).date() - datetime.fromisoformat(date).date()).days < 7:
            raise B0Error("PACKAGE_RELEASE_TOO_RECENT")
    return {"repository_sha": head, "python": platform.python_version(),
            "platform": platform.platform(), "pytorch": torch.__version__,
            "cuda_build": torch.version.cuda, "transformers": transformers.__version__,
            "accelerate": accelerate.__version__, "hf_home": "/content/hf",
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                         if d.metadata["Name"]}, "release_dates": RELEASE_DATES}


def checkpoint_metadata():
    from huggingface_hub import HfApi

    info = HfApi(token=False).model_info(MODEL_ID, revision=MODEL_REVISION,
                                        files_metadata=True, token=False)
    shards = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
    if (info.sha != MODEL_REVISION or info.gated is not False or not shards
            or any(not isinstance(f.size, int) or f.size <= 0 for f in shards)):
        raise B0Error("MODEL_METADATA_INVALID")
    return {"revision": info.sha, "checkpoint_bytes": sum(f.size for f in shards),
            "shards": [{"filename": f.rfilename, "bytes": f.size} for f in shards]}


def resource_preflight(cache_dir):
    import torch
    import transformers
    import accelerate

    checks = {"cuda": torch.cuda.is_available(), "python": sys.version_info >= (3, 12),
              "transformers": transformers.__version__ == TRANSFORMERS_PIN,
              "accelerate": accelerate.__version__ == ACCELERATE_PIN,
              "cache": str(cache_dir) == "/content/hf"}
    hardware = {"gpu_name": None, "compute_capability": None, "free_vram_bytes": None,
                "total_vram_bytes": None, "host_ram": host_memory(),
                "host_ram_threshold_basis": "WEIGHTWRIGHT heuristic; not a Meta-certified minimum",
                "python": platform.python_version(), "pytorch": torch.__version__,
                "cuda_build": torch.version.cuda, "transformers": transformers.__version__,
                "accelerate": accelerate.__version__}
    checks["bf16"] = False
    if checks["cuda"]:
        checks["bf16"] = torch.cuda.is_bf16_supported()
        free, total = torch.cuda.mem_get_info(0)
        hardware.update(gpu_name=torch.cuda.get_device_name(0),
                        compute_capability=list(torch.cuda.get_device_capability(0)),
                        free_vram_bytes=free, total_vram_bytes=total)
    checks["vram"] = (hardware["free_vram_bytes"] or 0) >= config.B0_MIN_FREE_VRAM_GIB * 1024**3
    checks["host_ram"] = (hardware["host_ram"]["available_bytes"] or 0) >= config.B0_MIN_HOST_RAM_GIB * 1024**3
    try:
        metadata = checkpoint_metadata()
        checks["model_metadata"] = True
    except Exception:
        metadata = None
        checks["model_metadata"] = False
    try:
        disk = shutil.disk_usage(cache_dir)
        hardware["disk_free_bytes"] = disk.free
        hardware["disk_total_bytes"] = disk.total
    except OSError:
        hardware["disk_free_bytes"] = None
        hardware["disk_total_bytes"] = None
    required = metadata["checkpoint_bytes"] + config.B0_DISK_MARGIN_GIB * 1024**3 if metadata else None
    checks["disk"] = required is not None and (hardware["disk_free_bytes"] or 0) >= required
    return {"checks": checks, "hardware": hardware, "model_metadata": metadata,
            "required_disk_bytes": required}


def assert_model(model, torch):
    count = vision_count = vision_bytes = 0
    for name, parameter in model.named_parameters():
        if str(parameter.device) != "cuda:0":
            raise B0Error("LOAD_PLACEMENT")
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            raise B0Error("LOAD_DTYPE")
        count += parameter.numel()
        if "vision_tower" in name.split("."):
            vision_count += parameter.numel()
            vision_bytes += parameter.numel() * parameter.element_size()
    if not count:
        raise B0Error("LOAD_EMPTY_MODEL")
    return {"parameter_count": count, "vision_encoder_loaded": bool(vision_count),
            "vision_parameter_count": vision_count, "vision_parameter_vram_bytes": vision_bytes,
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "text_attention_implementation": getattr(getattr(model.config, "text_config", None),
                                                     "_attn_implementation", None)}


def gpu_metrics(torch):
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def check_compute_units(value):
    import math

    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                              or not math.isfinite(value) or value < 0):
        raise B0Error("INVALID_COMPUTE_UNIT_BALANCE")


def prompt_compute_units_after():
    """Called after generation; only a founder-entered numeric balance can escape."""
    while True:
        text = input("Colab compute-unit balance now (blank if unobserved): ").strip()
        try:
            value = float(text) if text else None
            check_compute_units(value)
        except (ValueError, B0Error):
            print("Enter a finite, non-negative number or leave blank.")
            continue
        return value


class B0Run:
    """One in-memory runtime session and one immutable run identity."""

    def __init__(self, expected_sha, *, repo=None, drive_root=None, compute_units_before=None,
                 labels_accepted=False):
        if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
            raise B0Error("EXACT_REPOSITORY_SHA_REQUIRED")
        check_compute_units(compute_units_before)
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[1]
        self.expected_sha = expected_sha
        self.manifest = pilot.validate()
        self.documents, self.items = pilot.load_dataset()
        self.session_id = SESSION_ID
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"b0-{stamp}-{uuid.uuid4().hex[:6]}"
        self.path = self.repo / "eval" / "results" / self.run_id
        self.drive = Path(drive_root) / self.run_id if drive_root else None
        self.stages = {}
        self.tokenizer = self.model = None
        self.records = []
        self._evaluation_started = False
        self.gpu_started = None
        self.run = {"run_id": self.run_id, "session_id": self.session_id, "status": "started",
                    "repository_sha": expected_sha, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "freeze": self.manifest, "prompt_version": pilot.PROMPT_VERSION,
                    "prompt_sha256": self.manifest["prompt_sha256"], "seed_base": config.B0_SEED,
                    "generation": {"do_sample": True, "temperature": config.B0_TEMPERATURE,
                                   "top_p": config.B0_TOP_P, "top_k": config.B0_TOP_K,
                                   "max_new_tokens": config.B0_MAX_NEW_TOKENS,
                                   "max_time": config.B0_MAX_TIME_S,
                                   "eos_tokens": ["<|eot|>", "<|end_of_text|>"]},
                    "labels_accepted_by_founder": labels_accepted is True,
                    "compute_units_before": compute_units_before, "compute_units_after": None,
                    "compute_units_source": "founder_entered",
                    "gpu_wall_clock_s": None,
                    "gpu_wall_clock_basis": "model_loading_start_to_evaluation_end",
                    "model_api_charges_usd": 0,
                    "colab_subscription_cost": None}
        self.path.mkdir(parents=True, exist_ok=False)
        pilot.write_json(self.path / "run.json", self.run)
        (self.path / "items.jsonl").touch(exist_ok=False)
        self._persist()

    def _persist(self):
        observed = {}
        for root in {self.path.parent, self.drive.parent if self.drive else self.path.parent}:
            for path in sorted(root.glob("b0-*/run.json")):
                if path.parent.name != self.run_id:
                    other = pilot.read_json(path)
                    observed[other["run_id"]] = {"run_id": other["run_id"], "status": other["status"]}
        self.run["other_runs"] = [observed[key] for key in sorted(observed)]
        pilot.write_json(self.path / "run.json", self.run)
        pilot.write_json(self.path / "stages.json", {"run_id": self.run_id,
                                                     "session_id": self.session_id, "stages": self.stages})
        if self.drive:
            self.drive.mkdir(parents=True, exist_ok=True)
            for name in ARTIFACTS:
                source = self.path / name
                if source.exists():
                    target = self.drive / (name + ".tmp")
                    shutil.copyfile(source, target)
                    target.replace(self.drive / name)

    def _require(self, stage):
        if self.session_id != SESSION_ID or self.run["status"] != "started":
            raise B0Error("RUN_SESSION_OR_STATUS")
        position = STAGES.index(stage)
        if stage in self.stages or any(self.stages.get(s, {}).get("status") != "passed" for s in STAGES[:position]):
            raise B0Error("STAGE_ORDER")

    def _stage(self, name, evidence, *, passed=True):
        self.stages[name] = {"status": "passed" if passed else "failed", "run_id": self.run_id,
                             "session_id": self.session_id, "evidence": evidence}
        self._persist()

    def _abort(self, reason):
        self.run.update(status="aborted", failure=reason)
        if self.gpu_started is not None:
            self.run["gpu_wall_clock_s"] = time.monotonic() - self.gpu_started
        self._persist()
        raise B0Error(reason) from None

    def environment(self):
        self._require("environment")
        try:
            evidence = environment_snapshot(self.repo, self.expected_sha)
        except Exception:
            self._stage("environment", {"failure": "ENVIRONMENT_FAILED"}, passed=False)
            self._abort("ENVIRONMENT_FAILED")
        self.run["environment"] = evidence
        self._stage("environment", evidence)
        return evidence

    def tokenizer_stage(self):
        self._require("tokenizer")
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                           token=False, trust_remote_code=False)
            pilot.special_ids(self.tokenizer)
            prompts = []
            for item in self.items:
                messages = pilot.prompt_messages(self.documents, item["question"])
                rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                             return_dict=False)
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise B0Error("TOKENIZER_FAILED")
                if not isinstance(rendered, str) or not rendered.endswith("<|start|>assistant"):
                    raise B0Error("UNEXPECTED_GENERATION_PREFIX")
                prompts.append({"id": item["id"], "input_tokens": len(tokens),
                                "rendered_prompt_sha256": pilot.sha256(rendered.encode("utf-8")),
                                "token_ids_sha256": pilot.sha256(pilot.json_bytes(tokens))})
        except Exception:
            self.tokenizer = None
            self._stage("tokenizer", {"failure": "TOKENIZER_FAILED"}, passed=False)
            self._abort("TOKENIZER_FAILED")
        evidence = {"model_revision": MODEL_REVISION, "prompts": prompts,
                    "chat_template_sha256": pilot.sha256(str(self.tokenizer.chat_template).encode("utf-8"))}
        self._stage("tokenizer", evidence)
        return evidence

    def preflight(self):
        self._require("preflight")
        try:
            evidence = resource_preflight("/content/hf")
        except Exception:
            self._stage("preflight", {"failure": "PREFLIGHT_FAILED:measurement"}, passed=False)
            self._abort("PREFLIGHT_FAILED:measurement")
        passed = all(evidence["checks"].values())
        self._stage("preflight", evidence, passed=passed)
        if not passed:
            failed = next(name for name, passed in evidence["checks"].items() if not passed)
            self._abort(f"PREFLIGHT_FAILED:{failed}")
        self.run["hardware"] = evidence["hardware"]
        self._persist()
        return evidence

    def load(self, *, download_authorized=False):
        self._require("load")
        if download_authorized is not True or not self.run["labels_accepted_by_founder"]:
            raise B0Error("FOUNDER_LABEL_AND_DOWNLOAD_AUTHORIZATION_REQUIRED")
        # Recheck current resources immediately before the first weight-loading call.
        self.stages.pop("preflight")
        self.preflight()
        import torch
        from transformers import AutoModelForCausalLM

        self.gpu_started = time.monotonic()
        started = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats(0)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map={"": 0},
                token=False, trust_remote_code=False)
            evidence = assert_model(self.model, torch)
            self.model.eval()
            torch.cuda.synchronize(0)
            evidence.update(load_time_s=time.perf_counter() - started, **gpu_metrics(torch),
                            allocated_after_load_bytes=torch.cuda.memory_allocated(0),
                            reserved_after_load_bytes=torch.cuda.memory_reserved(0))
        except Exception as error:
            code = ("LOAD_OOM" if isinstance(error, torch.OutOfMemoryError)
                    else str(error) if isinstance(error, B0Error) and str(error) in
                    {"LOAD_PLACEMENT", "LOAD_DTYPE", "LOAD_EMPTY_MODEL"} else "LOAD_INCOMPATIBLE")
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            self._stage("load", {"failure": code}, passed=False)
            self._abort(code)
        self._stage("load", evidence)
        return evidence

    def _generate(self, index):
        import torch

        item = self.items[index]
        record = {"id": item["id"], "run_id": self.run_id, "session_id": self.session_id,
                  "final_message": None, "parsed": None, "messages": [], "reasoning_tokens": 0,
                  "input_tokens": None, "output_tokens": None, "latency_s": None,
                  "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                  "rendered_prompt_sha256": None, "failure": None}
        started = None
        encoded = output = None
        failure_source = "GENERATION_EXCEPTION"
        try:
            messages = pilot.prompt_messages(self.documents, item["question"])
            encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                          return_tensors="pt", return_dict=True)
            ids = encoded["input_ids"][0].tolist()
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_hash = pilot.sha256(rendered.encode("utf-8"))
            frozen_prompt = self.stages["tokenizer"]["evidence"]["prompts"][index]
            if (prompt_hash != frozen_prompt["rendered_prompt_sha256"]
                    or pilot.sha256(pilot.json_bytes(ids)) != frozen_prompt["token_ids_sha256"]):
                failure_source = "PROMPT_CHANGED"
                raise B0Error("PROMPT_CHANGED")
            record.update(input_tokens=len(ids), rendered_prompt_sha256=prompt_hash)
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            torch.manual_seed(config.B0_SEED + index)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            eos = pilot.special_ids(self.tokenizer)
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded, do_sample=True, temperature=config.B0_TEMPERATURE, top_p=config.B0_TOP_P,
                    top_k=config.B0_TOP_K, max_new_tokens=config.B0_MAX_NEW_TOKENS,
                    max_time=config.B0_MAX_TIME_S,
                    eos_token_id=[eos["<|eot|>"], eos["<|end_of_text|>"]])
            torch.cuda.synchronize(0)
            record["latency_s"] = time.perf_counter() - started
            generated = output[0].tolist()[len(ids):]
            record["output_tokens"] = len(generated)
            separated = pilot.extract_final(generated, self.tokenizer)
            for key in ("final_message", "messages", "reasoning_tokens"):
                record[key] = separated[key]
            record["failure"] = separated["failure"]
            if not separated["ended"]:
                record["failure"] = "TIMEOUT" if record["latency_s"] >= config.B0_MAX_TIME_S else "TRUNCATED"
            parsed = pilot.parse_final(record["final_message"])
            record.update(parsed=parsed["parsed"], code_fence_stripped=parsed["code_fence_stripped"])
        except Exception as error:
            if not isinstance(error, (torch.OutOfMemoryError, TimeoutError)):
                self.run.update(failure_source=failure_source, failed_item_id=item["id"])
                self._abort("INTERRUPTED")
            record["failure"] = ("OOM" if isinstance(error, torch.OutOfMemoryError)
                                 else "TIMEOUT")
            record["failure_source"] = "GENERATION_EXCEPTION"
            # Never copy an exception message: tokenizer/model exceptions may contain reasoning.
            if started is not None:
                record["latency_s"] = time.perf_counter() - started
        if started is not None:
            try:
                record.update(gpu_metrics(torch))
            except (torch.OutOfMemoryError, TimeoutError):
                record["resource_metrics_unavailable"] = True
            except Exception:
                self.run.update(failure_source="GENERATION_EXCEPTION", failed_item_id=item["id"])
                self._abort("INTERRUPTED")
        if record["failure"] == "OOM":
            encoded = output = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                self.run.update(failure_source="GENERATION_EXCEPTION", failed_item_id=item["id"])
                self._abort("INTERRUPTED")
        record["outcome"] = pilot.score_item(item, record, self.documents)["outcome"]
        return record

    def smoke(self):
        self._require("smoke")
        record = self._generate(0)
        passed = record["failure"] is None and record["parsed"] is not None
        self._stage("smoke", record, passed=passed)
        if not passed:
            self._abort(record["failure"] or "INVALID_OUTPUT")
        return record

    def evaluate(self, *, compute_units_after=None):
        """Collect the optional balance callback after generation, then finalize once."""
        if compute_units_after is not None and not callable(compute_units_after):
            raise B0Error("INVALID_COMPUTE_UNIT_BALANCE")
        if (self.session_id != SESSION_ID or self.run["status"] != "started" or self._evaluation_started
                or self.model is None or self.tokenizer is None
                or any(self.stages.get(s, {}).get("status") != "passed" for s in STAGES)):
            raise B0Error("FULL_RUN_STAGE_OR_SESSION")
        self._evaluation_started = True
        consecutive_ooms = 0
        terminal_status = "complete"
        for index in range(len(self.items)):
            record = self._generate(index)
            with (self.path / "items.jsonl").open("ab") as handle:
                handle.write(pilot.json_bytes(record).replace(b"\n", b" ").rstrip() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records.append(record)
            self._persist()
            consecutive_ooms = consecutive_ooms + 1 if record["failure"] == "OOM" else 0
            if consecutive_ooms == config.B0_MAX_CONSECUTIVE_OOMS:
                self.run["failure"] = "OOM"
                terminal_status = "aborted"
                break
        try:
            self.run["gpu_wall_clock_s"] = time.monotonic() - self.gpu_started
            observed_balance = compute_units_after() if compute_units_after is not None else None
            check_compute_units(observed_balance)
            self.run.update(compute_units_after=observed_balance, status=terminal_status)
            summary = pilot.score_run(self.path, run_metadata=self.run)
            self._persist()
        except Exception:
            self.run["failure_source"] = "FINALIZATION"
            self._abort("INTERRUPTED")
        return summary

    def release(self):
        import torch

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
