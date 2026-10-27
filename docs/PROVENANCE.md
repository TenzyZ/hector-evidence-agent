# Provenance

Purpose: make it unambiguous what this public repository contains, where it came from, and what is still uncertain about hackathon eligibility.

## 1. This repository

| Fact | Value | How verified |
|---|---|---|
| Repository | https://github.com/TenzyZ/hector-evidence-agent (public) | GitHub API, 2026-10-27 |
| Created | 2026-10-27T15:06:22Z | GitHub API `createdAt` |
| Initial commit | `58e5512` "Initial commit". Contains only a 2-line `README.md` | `git log`, `git ls-tree` |
| Hackathon window | Oct 26, 12:00 PM ET to Nov 4, 9:00 PM ET | https://cerebralvalley.ai/e/meta-ai-developer-hackathon (verified 2026-10-27) |
| Canonical status | Founder decision, 2026-10-27: this is the canonical Hector repository and WEIGHTWRIGHT's first product. The private `TenzyZ/Hector` repository is a read-only research source | founder |
| Phase 01 content | the seven specification files, written from scratch on 2026-10-27. Revision 1 is commit `273d69d` (PR #1); revision 2 (Colab BF16 for B0) was made the same day; revision 3 (review-finding repairs) was made on 2026-10-27 | `git log`, this phase's diff |

The repository was created after the hackathon began. Every file in it is written for this project. A new repository does not make earlier work newly authored: earlier Hector research is described in §2 as earlier work.

## 2. Historical private Hector research

WEIGHTWRIGHT Research has earlier, **private** research under the name "Hector", in the repository `TenzyZ/Hector`, dated 2026-10-17 to 2026-10-27. It studied offline extraction from PDF, DOCX and HTML with verifiable evidence:
- a deterministic baseline (R0), accepted by the founder;
- a planned extractive question-answering model (R1, DeBERTa), whose evaluation never ran. Its environment gate (Gate 6) is unfinished.

This repository is the canonical continuation of that research (PROJECT §1a).

- **Nothing from it is in this repository.** No source code, prompts, data, evaluation sets or results, or internal reports were copied. PROJECT §1a and ARCHITECTURE §14 give a short factual description for transparency.
- **Earlier results and gates do not transfer.** R0's measurements and acceptance, and R1's Gate 6, are not evidence for this repository, which earns its own (EVALUATION_CONTRACT §5).
- **How the earlier repository was accessed:**
  - Revision 1's access history has not been independently verified.
  - On 2026-10-27, Claude Opus 5.5's migration audit read historical source files, requirements, `pytest.ini`, `data/manifest.json`, protocol documents and `docs/results/*.md` reports. Recorded operations included Serena `get_symbols_overview` and `find_symbol`, Claude Code `Read` and `Grep`, shell reads (`ls`, `cat`, `head`, `grep`, `sed`), a Python JSON reader and non-mutating Git commands.
  - **Serena safeguard:** the external `Hector-R1-ENV` project configuration has `read_only: true`. The archived server log records startup at 20:53:48 UAE with ten active read tools and no active editing tools; all 32 recorded Serena calls (1 `initial_instructions`, 14 `get_symbols_overview`, 17 `find_symbol`) followed, from 20:54:05 to 21:34:41. The external configuration and archived server/audit records were checked during this repair. The active-tool restriction is specific to Serena.
  - The audit approved **zero historical source files for import**. Any future migration still requires §4's exception and founder-authorization process; no historical source has been transferred into this repository.
  - The recorded audit checks found the historical working tree and history unchanged.

## 3. Eligibility: open questions

The participant **Official Rules were not publicly retrievable on 2026-10-27**. The event page states that they "will be published before the Hackathon begins on October 26, 2026 and must be accepted by the selected participants".

Until the founder reads them (likely via the participant portal or acceptance email), the following are **unknown**:

1. Whether code, models, or research created before the event may be used, and whether that use must be disclosed.
2. Whether a project must be started during the event window.
3. Ownership, licensing, and open-source requirements for submissions.
4. Required submission artifacts (repository, video, description, deployed demo).
5. Model-use requirements. The public pages say participants may "run Meta open weights models", which Muse Glimmer is, but the exact wording of the rules is unverified.
6. Rules on team composition and organizational entries (Startup track).

The public eligibility text excludes sanctioned regions and employees, contractors, and interns of Meta and Cerebral Valley, along with their families. The founder must confirm personal eligibility.

**Conservative default until confirmed:** all code is new code written in this repository during the event. A newly created repository is **not** by itself evidence that older Hector work is eligible.

## 4. Reuse policy

1. **Default: no reuse** of private Hector code, prompts, data, reports, or evaluation sets.
2. **Exception process.** Every step is required:
   - the founder confirms the Official Rules permit it;
   - the reuse is entered in the register (§6) **before** use;
   - it is implemented in a dedicated, founder-authorized phase.
3. **Third-party code** enters only as declared dependencies with compatible licenses, pinned to releases at least 7 days old. Copied snippets need attribution and a license check, recorded in §6.
4. **Evaluation data** is authored fresh for this project (fictional organizations; no real personal data). See [EVALUATION_CONTRACT.md §3](EVALUATION_CONTRACT.md#3-data). The B0 pilot data (EVALUATION_CONTRACT Appendix A: 6 documents, 16 items, prompt `b0-v1`) was written by Claude Opus 5.5 on 2026-10-27 and awaits founder acceptance. It uses nothing from the private repository.
5. **Public safety check before every PR:** no secrets, API keys, `.env` files, private repository content, or internal report text in the diff.

## 5. Attribution and AI assistance

**Third-party components:**

| Component | Use | License / terms |
|---|---|---|
| Meta Muse Glimmer 30B | B0 model (BF16 checkpoint run on Google Colab); later the agent and judge model. Weights are downloaded only into the temporary Colab VM and never committed or redistributed | Apache 2.0 per Meta (ARCHITECTURE S2, S16); Meta usage policy applies (S1) |
| Hugging Face Transformers, Accelerate (planned, Phase 01b; Colab only) | model loading and generation for B0 | to be verified when pinned |
| PyTorch (Colab's preinstalled build) | GPU runtime for B0 | to be verified and recorded per run |
| Google Colab | B0 execution environment | Google's service terms (founder's account) |
| Together AI (fallback only, needs separate approval) | hosted inference | provider terms |
| SQLite (via Python `sqlite3`) | storage, FTS5 retrieval | public domain |
| Python standard library | runtime | PSF License |
| Gradio (planned, Phase 06) | Web UI | to be verified when added |

**AI assistance:**

- **Phase 01 specifications:** revision 1 is recorded at `273d69d`; its AI author and runtime have not been independently verified. Revision 2 (Colab BF16 for B0, continuity sections, B0 dataset) was authored by Claude Opus 5.5 in T3 Code through the Claude Agent SDK. Revision 3's targeted documentation repairs were made by Codex Sol in T3 Code through the Codex harness. The author's self-review is not an independent review.
- **Revision 4:** the current local correction pass by Codex Sol on the Revision 3 specification, pending review. The base specification's revision labels remain at 3; this pass has not been committed or published.
- **Earlier interrupted T3 session (separate from Revision 2):** this phase continues an assignment Claude Opus 5.5 began in T3 Code, which stopped at a usage limit. No files from that session were recovered: its local git checkpoints contain no file changes relative to `58e5512`. Its visible research notes were treated as leads and re-verified.
- **Later phases:** implementation is planned by Codex Sol, with review by Claude Opus. The founder accepts every phase.

**Design guidance:** general engineering reference notes held by the founder (prompt engineering, agent harness engineering) informed the design. They are not reproduced here.

## 5a. Colab preflight observations (founder-run)

On 2026-10-27 the founder ran `Hector_Muse_30B_Preflight.ipynb` in Google Colab. The notebook is not in this repository. The values below are the founder's reported observations; they have not been re-measured.

| Runtime | Observed |
|---|---|
| CPU preflight | Transformers 5.20.2. Muse Glimmer configuration recognized; tokenizer loaded; chat template rendered; `MuseGlimmerProcessor` loaded; the baseline prompt and two test inputs tokenized |
| GPU preflight (a separate runtime) | CUDA available; NVIDIA RTX PRO 6000 Blackwell Server Edition; 94.43 GiB free of 94.97 GiB VRAM; 193.17 GiB free disk; PyTorch 2.11.0+cu130 |

**What this shows:** preparation compatibility, and that one Colab session offered a GPU large enough on paper for BF16 (ARCHITECTURE §8).

**What it does not show:**
- that the model weights load;
- that the model generates text;
- any benchmark output, accuracy or latency;
- that the same GPU will be allocated again (ARCHITECTURE S17).

It is not a model benchmark, and it passes none of the B0 stages (EVALUATION_CONTRACT §2a).

## 6. Reuse register

| # | Component | Origin (repo, commit, date) | Owner / license | Change made | Rules check | Approved by / date |
|---|---|---|---|---|---|---|
| — | none | — | — | — | — | — |
