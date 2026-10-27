# Hector Evidence Agent

Evidence-checked question answering over your own documents, built on Meta's open-weight **Muse Glimmer 30B**.
By **WEIGHTWRIGHT Research** for the Meta Global AI Developer Hackathon 2026 (Startup track).

> **Status (2026-10-27): Phase 01, architecture and specifications only.**
> There is no application code yet. Everything below describes the *planned* system. Nothing here has been implemented, run, or evaluated.

## The idea

Many document chatbots attach citations. Few check them. An exact quote proves that the quoted text exists. It does not prove the answer follows from it.

Hector is designed so that:

1. The model answers only in **atomic claims**, each with verbatim quotes from passages it actually retrieved.
2. Every claim passes an **evidence verifier** before the user sees it:
   - **Deterministic checks:** the source exists, the quote matches at exact offsets, and the numbers appear in the cited passages.
   - **A separate model judgment** of whether the passage supports the claim, clearly labelled as model-judged.
3. Claims that fail are withheld and shown in an audit section with the reason. If nothing survives, Hector **abstains** instead of guessing.

Outcomes: `SUPPORTED`, `QUALIFIED`, `ABSTAINED`, `FAILED`. Hector checks support **by your documents**. It does not check whether the documents are true.

## Planned design (summary)

- **Input:** `.txt` and `.md` files in a private, per-session corpus.
- **Retrieval:** SQLite FTS5 with bm25 ranking. Python standard library; no vector database.
- **Agent:** Muse Glimmer 30B via Together AI's OpenAI-compatible API. Three read-only tools, at most 6 model calls, one tool call per turn.
- **Interfaces:** a CLI for ingest, search, ask, and evaluation; a small web UI for the demo.

Full design: [ARCHITECTURE.md](ARCHITECTURE.md).

## How it will be evaluated

Three configurations on 60 labeled questions about fictional documents:

- **A:** retrieval only.
- **B:** agent without verification.
- **C:** full Hector.

The questions include unanswerable, contradictory, misleading-quote, prompt-injection, and citation-bait items. Metrics, thresholds and limitations: [docs/EVALUATION_CONTRACT.md](docs/EVALUATION_CONTRACT.md). No results exist yet.

## Documents

| File | Contents |
|---|---|
| [PROJECT.md](PROJECT.md) | thesis, users, MVP scope, exclusions, constraints, open decisions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | components, data flow, trust boundaries, serving decision, sources |
| [docs/EVALUATION_CONTRACT.md](docs/EVALUATION_CONTRACT.md) | baselines, dataset, metrics, pass criteria |
| [docs/PHASE_PLAN.md](docs/PHASE_PLAN.md) | deadline-aware build phases and acceptance gates |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | what this repository contains, prior private research, eligibility questions |
| [AGENTS.md](AGENTS.md) | rules for coding agents working here |

## Roadmap

Phase 01 specifications (now) → 02 ingestion and retrieval → 03 evaluation dataset → 04 agent loop → 05 verifier → 06 web UI → 07 final evaluation and report. Submission is due 2026-11-04. See [docs/PHASE_PLAN.md](docs/PHASE_PLAN.md).

## License and provenance

No license has been chosen yet, so default copyright applies until a `LICENSE` file is added. All content is new to this repository; see [docs/PROVENANCE.md](docs/PROVENANCE.md). Muse Glimmer is released by Meta under Apache 2.0.
