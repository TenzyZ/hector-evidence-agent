# Provenance

Purpose: make it unambiguous what this public repository contains, where it came from, and what is still uncertain about hackathon eligibility.

## 1. This repository

| Fact | Value | How verified |
|---|---|---|
| Repository | https://github.com/TenzyZ/hector-evidence-agent (public) | GitHub API, 2026-10-27 |
| Created | 2026-10-27T15:06:22Z | GitHub API `createdAt` |
| Initial commit | `58e5512` "Initial commit". Contains only a 2-line `README.md` | `git log`, `git ls-tree` |
| Hackathon window | Oct 26, 12:00 PM ET to Nov 4, 9:00 PM ET | https://cerebralvalley.ai/e/meta-ai-developer-hackathon (verified 2026-10-27) |
| Phase 01 content | the seven specification files, written from scratch on 2026-10-27 | this phase's diff |

The repository was created after the hackathon began. Every file in it is written for this project.

## 2. Historical private Hector research

WEIGHTWRIGHT Research has earlier, **private** research on evidence-grounded agents under the name "Hector", in a separate private repository.

- **Nothing from it is in this repository:** no source code, prompts, data, evaluation results, or internal reports were copied, downloaded, or paraphrased.
- **Conceptual continuity is disclosed here:** this project carries over the name "Hector" and the general research interest in evidence-grounded answering. These specifications were written from the founder's brief and the public sources listed in [ARCHITECTURE.md §13](../ARCHITECTURE.md#13-sources-verified-2026-10-27).
- **Untouched:** the Devin session that wrote these files did not access the private repository, its pull requests, or its review status.

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
4. **Evaluation data** is authored fresh for this project (fictional organizations; no real personal data). See [EVALUATION_CONTRACT.md §3](EVALUATION_CONTRACT.md#3-data).
5. **Public safety check before every PR:** no secrets, API keys, `.env` files, private repository content, or internal report text in the diff.

## 5. Attribution and AI assistance

**Third-party components:**

| Component | Use | License / terms |
|---|---|---|
| Meta Muse Glimmer 30B | agent and judge model, called through Together AI. Weights are not downloaded or redistributed | Apache 2.0 per Meta (ARCHITECTURE S2); Meta usage policy applies (S1) |
| Together AI | hosted inference | provider terms (founder's account) |
| SQLite (via Python `sqlite3`) | storage, FTS5 retrieval | public domain |
| Python standard library | runtime | PSF License |
| Gradio (planned, Phase 06) | Web UI | to be verified when added |

**AI assistance:**

- **Phase 01 specifications:** written by Claude Opus 5.5 running in Devin.
- **Earlier T3 session:** this phase continues an assignment Claude Opus 5.5 began in T3 Code, which stopped at a usage limit. No files from that session were recovered: its local git checkpoints contain no file changes relative to `58e5512`. Its visible research notes were treated as leads and re-verified.
- **Later phases:** implementation is planned by Codex Sol, with review by Claude Opus. The founder accepts every phase.

**Design guidance:** general engineering reference notes held by the founder (prompt engineering, agent harness engineering) informed the design. They are not reproduced here.

## 6. Reuse register

| # | Component | Origin (repo, commit, date) | Owner / license | Change made | Rules check | Approved by / date |
|---|---|---|---|---|---|---|
| — | none | — | — | — | — | — |
