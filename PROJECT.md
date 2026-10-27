# Hector Evidence Agent — Project Definition

Organization: **WEIGHTWRIGHT Research** · Event: **Meta Global AI Developer Hackathon 2026** · Track: **Startup** · Proposed theme: **Push the Frontier**
Status: **Phase 01 specification, revision 3.** No application code exists yet. See [README.md](README.md) for current status.
Identity: WEIGHTWRIGHT's first product and the canonical Hector repository, continuing earlier private Hector research (§1a, [PROVENANCE §2](docs/PROVENANCE.md#2-historical-private-hector-research)).

## 1. Thesis

Teams that answer questions from documents need two things at once: a useful answer, and a way to check it. Chatbots with citations give the first and only look like the second. A citation can point at text that exists but does not say what the answer claims.

Hector's bet: an agentic open-weight model (Meta **Muse Glimmer 30B**) can investigate a document set and propose answers, **if** every proposed claim must pass an independent evidence check before a user sees it, and the system abstains when the check fails.

The differentiator is **auditable evidence handling**, not the chat interface.

### 1a. Relation to earlier Hector research

Between 2026-10-17 and 2026-10-27, WEIGHTWRIGHT researched Hector privately as offline, local extraction from PDF, DOCX and HTML. That work had two stages:
- a deterministic baseline (R0), which the founder accepted;
- a planned small extractive question-answering model (R1, DeBERTa), whose evaluation never ran.

This project continues that line, and the founder designated it the canonical Hector on 2026-10-27.

| Kept as principles | Changed in this project |
|---|---|
| Verifiable answer provenance at exact source offsets | Model: a small local extractive model → Muse Glimmer 30B, an agentic open-weight model |
| Deterministic verification before anything model-judged | Task: single-field extraction → atomic claims with verbatim quotes |
| Fail-closed behaviour and explicit abstention | Formats: PDF/DOCX/HTML → `.txt`/`.md` (MVP) |
| Frozen evaluation inputs, reproducible scoring, honest reporting | Retrieval: per-document BM25 → corpus-wide SQLite FTS5 (not equivalent; see [ARCHITECTURE §14](ARCHITECTURE.md#14-continuity-with-historical-hector)) |
| Model output kept separate from evidence | Execution: offline CPU → a Colab GPU for B0 |

No code, data or reports were imported. Earlier results and gates, including R0 acceptance and R1's unfinished Gate 6, do not transfer. This project earns its own evidence.

## 2. Research question

> Can an agentic language model investigate documents, generate useful answers, and support its conclusions with independently checked source evidence while abstaining from unsupported claims?

Operationalized in [docs/EVALUATION_CONTRACT.md](docs/EVALUATION_CONTRACT.md) as three comparisons on one labeled dataset:

- **A.** Retrieval only.
- **B.** Muse Glimmer with retrieved evidence, no verification.
- **C.** Muse Glimmer plus evidence verification, i.e. the full Hector.

Hypothesis: C shows fewer unsupported claims than B, and abstains correctly on unanswerable questions more often than B, at an acceptable cost in over-abstention, latency, and money. The result is reported whichever way it comes out.

The smaller **B0 pilot** comes first ([EVALUATION_CONTRACT §2a](docs/EVALUATION_CONTRACT.md#2a-b0-pilot-muse-glimmer-alone-phase-01b)). It measures Muse Glimmer alone, in BF16 on a Google Colab GPU, on 16 labeled questions. B0 is a reference point for the model, not a substitute for B or C.

## 3. Target users

| User | Job | Why evidence matters |
|---|---|---|
| Compliance / audit analyst | "Does policy X permit Y?" across policy and contract documents | Answers must be defensible to a reviewer |
| Research / due-diligence analyst | Extract facts from reports and filings | Wrong numbers are costly; sources must be shown |
| Operations / support lead | Answer questions from internal runbooks | Must say "not covered" instead of inventing procedure |

Hackathon demo users: judges, using a provided sample corpus or their own `.txt`/`.md` files.

## 4. What "supported" means here

| Check | Kind | What it proves | What it does not prove |
|---|---|---|---|
| Source identity | deterministic | the cited passage exists in *this* corpus and was seen by the agent | anything about meaning |
| Quote span | deterministic | the quoted text occurs verbatim (after whitespace/quote normalization) at recorded offsets | that the claim follows from it |
| Numbers | deterministic | numbers in the claim appear in the cited passages | that they are used correctly |
| Support judge | model-judged | a separate Muse Glimmer call, in a fresh context, rated the claim as entailed by the cited passages | logical certainty; the judge can be wrong, and it shares a model family with the agent |

Hector never claims to eliminate hallucination, and never claims that documents are true.

## 5. MVP features

1. Upload `.txt` / `.md` files (limits in [ARCHITECTURE.md §7](ARCHITECTURE.md#7-limits-and-budgets)) into a private per-session corpus.
2. Lexical retrieval (SQLite FTS5, bm25) with exact source offsets.
3. A bounded Muse Glimmer agent with three read-only tools (`search_passages`, `read_passage`, `submit_answer`), at most 6 model calls.
4. Answers expressed only as atomic claims, each with 1–3 verbatim quotes.
5. Four-layer verifier and a deterministic outcome: `SUPPORTED`, `QUALIFIED`, `ABSTAINED`, `FAILED`.
6. Evidence display: the quote highlighted within its passage, with filename and IDs. Withheld claims are shown with reasons in an audit section.
7. CLI for ingest, search, ask, and eval. A Web UI for the demo.
8. Reproducible A/B/C evaluation with a published report.

## 6. Exclusions (MVP)

- PDF, DOCX, HTML, scanned documents/OCR, images (founder decision on PDF: §9).
- Embeddings, vector databases, rerankers (revisited only if the retrieval gate fails).
- Fine-tuning or training any model.
- Quantized checkpoints for B0. They are a separately approved fallback only, and their results are never reported as BF16.
- Committing or redistributing model weights. B0 downloads the BF16 checkpoint only into the temporary Colab VM.
- Multi-agent orchestration, agent frameworks, verifier-driven revision loops.
- Accounts, persistence across sessions, sharing, team features.
- Web search or any tool that reaches beyond the uploaded corpus.
- Non-English evaluation (the model is multilingual, but our dataset is English only).

## 7. Success criteria

1. **Evaluation:** the thresholds in [EVALUATION_CONTRACT §5](docs/EVALUATION_CONTRACT.md#5-pass-criteria) are met, or honestly reported as missed, on the frozen test split.
2. **Invariant:** zero displayed citations that fail deterministic checks in config C.
3. **Demo:** one uninterrupted session that shows a `SUPPORTED` answer, a `QUALIFIED` answer, an `ABSTAINED` unanswerable question, and a prompt-injection document that fails to change the answer.
4. **Truthfulness:** the README and submission claim only what the evaluation report shows.

## 8. Constraints

| Constraint | Value | Source |
|---|---|---|
| Submission deadline | **2026-11-04, 9:00 PM ET** (judging Nov 4–8; finals Nov 9) | https://dev.meta.ai/events/global-hackathon (verified 2026-10-27) |
| Build window remaining | ~8 days from 2026-10-27 | — |
| Agent model | Muse Glimmer 30B only (Apache 2.0) | ARCHITECTURE §8 |
| B0 inference | BF16 checkpoint `meta-models/Muse-Glimmer-30B` on a Google Colab GPU, in-process with Hugging Face Transformers. No API key and no per-token charges | ARCHITECTURE §5.8, §8 |
| Agent inference (Phase 04+) | Backend chosen in Phase 04 (ARCHITECTURE §5.3). A hosted provider such as Together AI is a fallback that needs separate approval. Hackathon Meta Model API credits do not cover Glimmer | ARCHITECTURE §8 |
| Compute budget | B0: the founder's Colab plan and compute units, recorded per run. Colab GPUs are not guaranteed (ARCHITECTURE S17). Hosted APIs: a proposed $25 cap for Phases 04–07 applies only if a hosted backend is approved | founder |
| Paid calls / GPU use / downloads | none without explicit founder authorization per phase | [AGENTS.md](AGENTS.md) |
| Workflow | one phase → one branch → one PR; founder accepts each phase | [docs/PHASE_PLAN.md](docs/PHASE_PLAN.md) |
| Code origin | New code by default. Private Hector material enters only through [PROVENANCE §4](docs/PROVENANCE.md#4-reuse-policy)'s exception process. Nothing has been imported | [docs/PROVENANCE.md](docs/PROVENANCE.md) |

## 9. Open founder decisions

Required before Phase 02:

1. **Approve this architecture** (Python ≥ 3.12, stdlib-only core, SQLite FTS5, Colab BF16 for B0, claims-only answers).
2. **Format scope:** `.txt`/`.md` only (recommended), or add PDF text-layer extraction (adds the `pypdf` dependency and a phase).
3. **Rules check:** read the participant Official Rules (not publicly retrievable on 2026-10-27) and confirm that nothing in them conflicts with the provenance policy. See [docs/PROVENANCE.md](docs/PROVENANCE.md).

Required before Phase 01b runs on Colab:
- accept the B0 labels ([EVALUATION_CONTRACT Appendix A](docs/EVALUATION_CONTRACT.md#appendix-a-b0-dataset-and-prompt-proposed));
- authorize Colab GPU use and the ~60 GB checkpoint download into the Colab VM.

Required later (listed in [docs/PHASE_PLAN.md](docs/PHASE_PLAN.md) at the phase that needs them):
- the agent backend, plus key and spend cap if it is hosted (Phase 04);
- eval thresholds (before Phase 05);
- public deployment yes/no (Phase 06);
- repository license and final theme (Phase 07).

## 10. Submission objectives

The exact required deliverables are **unconfirmed** until the Official Rules are read. We plan to produce:

- The public repository with working code, tests, and these specifications.
- The evaluation report: A/B/C results, limitations, and cost.
- A short demo video following §7.3.
- A project description consistent with the README.
