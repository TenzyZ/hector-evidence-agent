# Phase Plan

Deadline: **project submission due 2026-11-04, 9:00 PM ET** (https://dev.meta.ai/events/global-hackathon, verified 2026-10-27).
Workflow: **one phase → one branch → one pull request.** Each phase starts only after the founder accepts the previous one.

| Role | Responsibility |
|---|---|
| Claude (planning sessions) | architecture, specifications, evaluation data, report writing |
| Codex Sol | implementation phases |
| Claude Opus, in a separate session | independent review of each PR. An author's self-review is never reported as independent |
| Founder | scope authorization, paid-usage and GPU authorization, Colab runs, human labels, acceptance, all Git delivery (commit/push/PR/merge unless delegated in writing for a phase) |

## Schedule

| Phase | Name | Owner | Target date | Depends on | Paid usage |
|---|---|---|---|---|---|
| 01 | Architecture and specifications | Claude | Oct 27 | — | none |
| 01b | B0 pilot: Muse Glimmer 30B BF16 on Google Colab, gate G0 | Codex Sol (code), founder (Colab runs) | Oct 27–28 | 01 | no API charges; Colab compute units, recorded |
| 02 | Ingestion, retrieval, CLI, eval harness (config A) | Codex Sol | Oct 28 | 01 | none |
| 03 | Evaluation dataset + retrieval gate G1 | Claude | Oct 28–29 | 01 (authoring), 02 (running) | none |
| 04 | Model adapter + agent loop (config B) | Codex Sol | Oct 29–30 | 02, 03 | per chosen backend: Colab units, or hosted ≤ $5 |
| 05 | Evidence verifier + decision policy (config C), dev evaluation | Codex Sol | Oct 31–Nov 1 | 04 | same backend; hosted ≤ $8 |
| 06 | Web UI + demo script | Codex Sol | Nov 1–2 | 05 (contract from 04) | same backend; hosted ≤ $2 |
| 07 | Final evaluation, report, submission package | Claude + founder | Nov 3 | 05, 06 | same backend; hosted ≤ $10 |
| — | Buffer; founder submits | Founder | Nov 4 (before 9 PM ET) | 07 | — |

- **Feature freeze:** end of Nov 2. After the final test-split run (Phase 07), any code change requires re-running it.
- **Phase 01b and Phase 02 can run in parallel after Phase 01**, with coordinated changes to three shared files: `.gitignore`, `hector/__init__.py`, and `hector/config.py`. Whichever phase lands first creates all three; the later phase reuses `hector/__init__.py` and extends `.gitignore` and `hector/config.py`, preserving the first phase's ignore entries and constants.
- **Hosted inference cap:** $25 in total, and only if a hosted backend is approved. This matches [PROJECT.md §8](../PROJECT.md#8-constraints). Colab compute units are recorded per run.

**Cut order if behind schedule.** Cut top first:

1. public deployment (demo locally in the video);
2. final runs 3 → 1;
3. the `read_passage` tool;
4. the Web UI, replaced by a formatted CLI demo.

**Never cut:** the verifier, abstention, the forgery suite, the honest report.

---

## Phase 01 — Architecture and specifications (this phase)

- **Branch:** `feat/meta-architecture-phase-01`
- **Files:** `PROJECT.md`, `ARCHITECTURE.md`, `AGENTS.md`, `README.md`, `docs/EVALUATION_CONTRACT.md`, `docs/PHASE_PLAN.md`, `docs/PROVENANCE.md`.
- **PR:** #1, a single PR. Revision 2 made Colab BF16 the primary B0 path; Together AI became a fallback that needs separate approval. Revision 3 repairs the documented review findings.
- **Acceptance:** all seven files exist and are internally consistent; model, runtime and provider claims carry sources (ARCHITECTURE §13); no code; no secrets; no private material.
- **Founder gate:**
  - An independent Opus review of the published PR head, separate from the author's self-review.
  - Approve the architecture and answer the three decisions in [PROJECT.md §9](../PROJECT.md#9-open-founder-decisions).

## Phase 01b — B0 pilot on Google Colab

- **Branch:** `feat/b0-colab-pilot-phase-01b`, created from `main` after Phase 01 merges. Not created yet.
- **Objective:** the stdlib B0 harness, a Colab-only loading and generation module, and a thin notebook. Then the staged B0 run (B0-0 to B0-4) by the founder, producing gate G0's evidence.
- **Specification:** ARCHITECTURE §5.8, §7 (B0 rows), §10 (B0 row); EVALUATION_CONTRACT §2a and Appendix A.

**Authorized files** (create; if Phase 02 created the three shared files listed above, reuse `hector/__init__.py` and extend `.gitignore` and `hector/config.py` without discarding Phase 02 content):

```text
.gitignore                 # .env, .env.*, __pycache__/, *.db, eval/results/
hector/__init__.py
hector/config.py           # B0 constants from ARCHITECTURE §7 only
hector/pilot_b0.py         # stdlib: validate/freeze, prompt, parse (to=self / to=user split), score, summary, review.csv; CLI
hector/b0_colab.py         # Colab only: preflight P, load, generate, item records (imports torch/transformers inside functions)
notebooks/b0_colab.ipynb   # thin driver: commit check; one install cell pinning transformers (≥ 5.15) and accelerate to exact releases ≥ 7 days old; Drive mount; calls into hector/
tests/test_pilot_b0.py     # stdlib: dataset, parser, scoring, Wilson, byte-identical re-score; stub model/tokenizer for preflight decisions, placement/dtype asserts, OOM path, no reasoning text written
eval/pilot-b0/corpus/*.md  # Appendix A, byte-exact
eval/pilot-b0/questions.jsonl
eval/pilot-b0/FREEZE.json
docs/eval/pilot-b0/        # after the run: the reviewed run directory and REPORT.md
```

**Founder prerequisites:**
- Accept the Appendix A labels.
- Authorize Colab GPU use and the ~60 GB checkpoint download into the Colab VM.
- Have a Colab runtime with a GPU that passes preflight P.

No API key or Hugging Face token is needed.

**Execution** (founder, in Colab, in order; stop at the first failure):
1. B0-0 environment;
2. B0-1 tokenizer;
3. resource preflight P;
4. weight download;
5. B0-2 load smoke;
6. B0-3 generation smoke on q01;
7. B0-4 full run of 16 items;
8. offline `score` on the laptop;
9. copy the reviewed run into `docs/eval/pilot-b0/`.

**Acceptance:**
1. `python -m unittest discover -s tests -v` passes on a laptop without torch, transformers or a GPU.
2. `validate` passes on the Appendix A files, and `FREEZE.json` matches them.
3. Tests prove:
   - no reasoning text is ever written;
   - `FAILED` and `INVALID_OUTPUT` never count as abstentions;
   - a failing preflight downloads nothing;
   - a non-GPU placement or non-BF16 parameter fails the load.
4. The run's `stages.json` holds B0-0, B0-1, P, B0-2 and B0-3 from the same runtime session as its B0-4 run; evidence is never mixed across runs.
5. `score` reproduces `summary.json` byte for byte.
6. `REPORT.md` takes every number from `summary.json` and states the §2a limitations. It labels the run as a complete baseline or runtime-limited.
7. The diff touches only authorized files and contains no credentials or reasoning text.

**Stop and ask:**
- preflight P fails;
- BF16 loading fails or does not fit;
- `AutoModelForCausalLM` cannot load the checkpoint;
- a fallback (quantized or hosted) seems necessary. A fallback is a separate proposal and is never reported as B0.

**Founder gate:** an independent Opus review of the published PR. The founder then reads `REPORT.md` against `summary.json` and fills in or spot-checks `review.csv`.

## Phase 02 — Ingestion, retrieval, CLI, eval harness (config A)

- **Branch:** `feat/ingest-retrieval-phase-02`
- **Objective:** a stdlib-only Python package that ingests `.txt`/`.md`, stores chunks with exact offsets in SQLite FTS5, searches safely, and scores retrieval-only baseline A on a labeled dataset.
- **Specification:** ARCHITECTURE §5.1, §5.2, §5.7 (`ingest`, `search`, `eval` subcommands), §7, §11; EVALUATION_CONTRACT §3.1 (item schema), §4 (R1, R2, O1 for A), §6.

**Authorized files** (create; if Phase 01b created the three shared files, reuse `hector/__init__.py` and extend `.gitignore` and `hector/config.py` without discarding Phase 01b content):

```text
pyproject.toml            # name "hector", requires-python ">=3.12", dependencies = []
.gitignore                # .env, *.db, __pycache__/, eval/results/
hector/__init__.py
hector/__main__.py        # argparse CLI: ingest, search, eval validate, eval run --config A
hector/config.py          # ARCHITECTURE §7 constants (ingest/retrieval subset)
hector/ingest.py          # M1
hector/store.py           # M2
hector/evaluate.py        # validate dataset; run config A; write runs.jsonl, summary.json, summary.md
tests/test_ingest.py
tests/test_store.py
tests/test_evaluate.py
tests/test_cli.py
tests/fixtures/corpus/*.md, *.txt   # ≤ 5 tiny fictional docs + malformed samples
tests/fixtures/questions.jsonl      # ≤ 8 items covering every field of the item schema
```

Also authorized: updating the status note at the top of `README.md`. No other file.

**Dependencies:** none. No network access in code or tests. No model calls. `eval` subcommands default to `--corpus eval/corpus --questions eval/questions.jsonl`, which do not exist until Phase 03, so the fixtures are passed explicitly here.

**Commands that must work:**

```text
python -m unittest discover -s tests -v
python -m hector ingest --db h.db tests/fixtures/corpus/*.md     # *.db is git-ignored
python -m hector search --db h.db -k 5 "verify"
python -m hector eval validate --corpus tests/fixtures/corpus --questions tests/fixtures/questions.jsonl
python -m hector eval run --config A --corpus tests/fixtures/corpus --questions tests/fixtures/questions.jsonl --split dev
```

**Acceptance criteria.** Each one is proven by a named test or a command transcript in the PR.

1. Every ingest error code in ARCHITECTURE §10 is triggered by a fixture and returns that code.
2. Chunk invariants (ARCHITECTURE §5.1 acceptance) hold on every fixture.
3. Re-ingesting identical content returns the same `doc_id` and adds no rows.
4. CRLF, BOM and NFD inputs normalize as specified, and offsets refer to the normalized text.
5. The FTS5 availability check raises `FTS5_UNAVAILABLE` (simulated) with a clear message.
6. `search` returns `verified` for query `verify`. Queries `"`, `NEAR(`, `*`, `-x`, `text:foo`, `^a`, `a AND`, `()` do not raise. An all-punctuation query gives `EMPTY_QUERY`. `k` outside 1–8 gives `INVALID_K`. Results are ordered by bm25.
7. `eval validate` rejects:
   - a duplicate id;
   - a bad enum value;
   - a missing field;
   - a gold quote absent from its document;
   - a gold quote that only spans a chunk boundary.

   It accepts the valid fixture set.
8. `eval run --config A` writes `runs.jsonl`, `summary.json` (with the provenance fields from EVALUATION_CONTRACT §6 that apply to A), and `summary.md`. R1, R2 and O1 match hand-computed values on the fixtures.
9. CLI commands print JSON and exit non-zero on error. `ingest` expands glob patterns itself, because Windows shells pass `*.md` through unexpanded.
10. `pyproject.toml` declares no dependencies. The diff touches only authorized files.

**Founder gate:** run the unittest command and one CLI ingest/search on fixtures; confirm the diff scope; Claude Opus review is clean or its findings are resolved.

## Phase 03 — Evaluation dataset and retrieval gate G1

- **Branch:** `feat/eval-dataset-phase-03`
- **Objective:** author the frozen corpus and 60 labeled items (EVALUATION_CONTRACT §3.1), validate them with Phase 02 tooling, and measure G1.
- **Authorized files:** `eval/corpus/*.md`, `eval/questions.jsonl`, `eval/FREEZE.md` (SHA-256 list, item counts, author, date).
- **Acceptance:**
  - `eval validate` passes.
  - Category counts and the 30/30 stratified split match §3.1.
  - The corpus contains no real personal data and no private material.
  - `eval run --config A --split dev` reports R1. **G1: R1 ≥ 0.80.**
- **Founder gate:** review a 10-item sample for label quality; approve the freeze. If G1 fails, decide between chunking/query changes and an embedding retriever, as a new phase, before Phase 04.

## Phase 04 — Model adapter and agent loop (config B)

- **Branch:** `feat/agent-loop-phase-04`
- **Objective:** M3 and M4 as specified; `ask --no-verify`; `eval run --config BC` producing B results (C fields empty until Phase 05).
- **Specification:** ARCHITECTURE §5.3, §5.4, §6, §7, §8.
- **Authorized files:**
  - create `hector/model.py`, `hector/agent.py`, `tests/test_model.py` (local fake HTTP server for HTTP mode; stubbed generation for in-process mode) and `tests/test_agent.py` (scripted fake model);
  - edit `hector/__main__.py`, `hector/config.py`, `hector/evaluate.py`, `tests/test_cli.py`.
- **Founder prerequisites:**
  - choose the agent backend (ARCHITECTURE §5.3: an OpenAI-compatible server on the Colab GPU, in-process Transformers, or an approved hosted fallback), informed by B0's results;
  - for a hosted backend, also: an account, `HECTOR_MODEL_API_KEY` set locally (never committed), and authorization of ≤ $5 for this phase;
  - for Colab, also: authorization of GPU use.
- **Smoke test (≤ 10 calls):** one plain request and one tool-calling round trip against the chosen backend. Record in the PR:
  - `top_k` acceptance;
  - the reasoning field name;
  - whether `usage` is present and includes reasoning tokens;
  - `chat_template_kwargs.reasoning_strength` acceptance;
  - observed latency.
- **Acceptance:**
  1. All applicable adapter errors and protocol expectations in ARCHITECTURE §5.3 are covered: fake-server tests for HTTP mode, stubbed-generation tests for in-process mode, including structured tool-call parsing, schema validation, malformed inputs and retry/stop behavior. Both modes use the same normalized response contract and M4 validation.
  2. The key never appears in logs, exceptions, or traces (test).
  3. All loop branches in §5.4 are covered by scripted-fake tests.
  4. A tool outside the allowlist can never execute (test).
  5. `ask --no-verify` returns a valid `AnswerResult` on fixtures with the fake model.
  6. A dev-split B run completes with a cost report. O3 and P3 are reported.
- **Founder gate:** review smoke-test findings; confirm spend; approve the thresholds in EVALUATION_CONTRACT §5 before Phase 05 starts.

## Phase 05 — Evidence verifier and decision policy (config C)

- **Branch:** `feat/verifier-phase-05`
- **Objective:** M5 and M6; the forgery suite; C outcomes; `eval score` with human labels; dev-split B/C evaluation.
- **Specification:** ARCHITECTURE §5.5, §5.6; EVALUATION_CONTRACT §3.2, §4, §5.
- **Authorized files:**
  - create `hector/verify.py`, `tests/test_verify.py`, `tests/test_verify_forgery.py`;
  - edit `hector/agent.py` (integration only), `hector/evaluate.py`, `hector/__main__.py`, `hector/config.py`.
- **Paid:** ≤ $8 for dev runs only if the optional hosted backend is approved (schedule above); this is a spending cap, not an actual-cost guarantee or a Colab BF16 limit. Colab compute units are recorded separately; GPU use still requires founder authorization.
- **Acceptance:**
  - H1 passes.
  - Offset round-trip holds on all genuine forgery-suite cases.
  - The judge is never called for deterministic failures (test with a call-counting fake).
  - The decision table (§5.6) is covered row by row.
  - A dev-split BC run reports every deterministic metric.
  - `claims_to_label.csv` is blinded (no config, verdict or status columns).
- **Founder gate:** spot-check 10 C results against their quotes; confirm no displayed invalid citation (H2).

## Phase 06 — Web UI and demo script

- **Branch:** `feat/web-ui-phase-06`
- **Objective:** the Gradio UI of ARCHITECTURE §5.7. The demo script uses the evaluation corpus and `dev`-split questions only.
- **Authorized files:**
  - create `app.py`, `docs/DEMO_SCRIPT.md`, `tests/test_render.py`;
  - edit `pyproject.toml` (add `gradio`, pinned to a release ≥ 7 days old) and `README.md` (run instructions).
- **Founder decision:** public deployment yes/no and host. If yes, the deployment must satisfy ARCHITECTURE §9: secret store, spend cap, rate limit, no traces, data notice.
- **Acceptance:**
  - Rendering escapes HTML (test).
  - Per-session corpora are isolated: two sessions cannot see each other's documents (test or scripted check).
  - The demo script shows the four PROJECT.md §7.3 scenes end to end, run manually with the steps recorded.
  - Upload limits are enforced in the UI.

## Phase 07 — Final evaluation, report, submission package

- **Branch:** `docs/final-report-phase-07`
- **Objective:** the frozen test-split run (`--runs 3 --final`), human labels, `docs/EVAL_REPORT.md` (EVALUATION_CONTRACT §7), a README update with actual numbers, and a video outline.
- **Owners:** founder for labels, video, and submission; Claude for report and README text.
- **Authorized files:** `docs/EVAL_REPORT.md`, `docs/eval/summary.json` and `docs/eval/claims_labeled.csv` (copied from the git-ignored `eval/results/<final run>/`), `README.md`, `docs/PROVENANCE.md`, `docs/DEMO_SCRIPT.md`, and `LICENSE` if the founder chooses one.
- **Paid:** ≤ $10 only if the optional hosted backend is approved (schedule above); this is a spending cap, not an actual-cost guarantee or a Colab BF16 limit. Colab compute units are recorded separately; GPU use still requires founder authorization.
- **Acceptance:**
  - Every number in the README and report traces to `summary.json` or the label CSV.
  - The limitations section is present.
  - Provenance is updated with the final reuse register.
- **Founder gate:** choose a repository license (a `LICENSE` file is added in this phase only if the founder chooses one); confirm the theme; final read-through; submission before Nov 4, 9:00 PM ET.
