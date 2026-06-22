# Clinical Trial Matching Copilot — Multi-Agent Demo

A small but complete **multi-agent AI architecture** built with LangGraph + Groq.
Three agents collaborate to screen a patient against a clinical trial — and,
crucially, one agent can **catch another's overconfidence and force a correction**,
with no human in the loop.

> This is an *architecture* demo, not a clinical decision system. Real deployment
> needs validated pipelines, audit trails, and clinician oversight.

© Richard Abishai · richardabish.ai

---

## What it shows

A single LLM can read a patient and a trial and guess a match — but it hides where
*reading* ends and *judging* begins, and it fills missing data with confident guesses.
This demo splits that one opaque step into three honest ones:

| Agent | Job | Never does |
|-------|-----|-----------|
| **Researcher** | Extract structured facts, flag gaps | Decide eligibility |
| **Evaluator** | Judge from evidence; may say *needs_more_evidence* | Read raw notes / invent facts |
| **Orchestrator** | Route, handle the bounce-back, cap retries, finalize | Touch medicine |

The teaching moment: the evaluator returns `needs_more_evidence`, the orchestrator
routes the work **back** to the researcher with the evaluator's question attached,
and the system self-corrects, then finalizes conservatively.

Flip `EVALUATOR_ENABLED = False` and the same patient gets a confident, caveat-free
"likely eligible" — proving the architecture *is* the safety.

---

## The code, block by block

`clinical_trial_demo.py` is organized into 8 numbered blocks so it reads top-to-bottom:

1. **Imports & the demo switch** — `EVALUATOR_ENABLED` toggles the climax.
2. **Shared State** — one `TypedDict` whiteboard; "state in, partial state out."
3. **The model + `safe_json`** — lazy Groq client; crash-proof JSON parsing.
4. **Agent prompts** — the job description (and hard rules) for each agent.
5. **Agent nodes** — Researcher, Evaluator, Finalizer functions.
6. **The Orchestrator** — routing + the bounce-back loop (the brain).
7. **Build the graph** — add nodes, wire conditional edges, compile.
8. **Run the demo** — one patient, one trial, the ambiguous liver criterion.

Files:
- `clinical_trial_demo.py` — the full, runnable demo
- `clinical_trial_demo_SKELETON.py` — node bodies blanked for live-coding
- `RUN_OF_SHOW.md` — the presenter script
- `Clinical_Trial_Copilot_Deck.pptx` — the slides

---

## Installation

### Prerequisites
- **Python 3.10+** (developed on 3.12). Check: `python --version`
- **pip** (bundled with Python). Check: `pip --version`
- A **Groq API key** — free at <https://console.groq.com> → API Keys.

### Step 1 — (recommended) create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### Step 2 — install the three packages
```bash
pip install -U langgraph langchain langchain-groq
```
This pulls in LangGraph, LangChain core, and the Groq integration, plus their
shared dependencies (pydantic, httpx, orjson, etc.).

Pinned versions this demo was verified against:
```
langgraph==1.2.5
langchain==1.3.9
langchain-groq==1.1.3
```

### Step 3 — set your API key
```bash
# macOS / Linux:
export GROQ_API_KEY="your_key_here"
# Windows (PowerShell):
$env:GROQ_API_KEY="your_key_here"
```

### Step 4 — run it
```bash
python clinical_trial_demo.py
```

---

## ⏱ Estimated install times

Measured on a **mid-range laptop**: 8 GB RAM, SSD, ~50 Mbps connection, fresh
Python 3.12. Your numbers will vary with network speed and cache state.

| Step | What happens | First-time (cold) | Cached / repeat |
|------|--------------|-------------------|-----------------|
| Install Python (if needed) | Download + installer | 4–7 min | — |
| Create virtualenv | `python -m venv` | 10–20 sec | 10–20 sec |
| `pip install` the 3 packages | Download ~40–60 MB of wheels + deps, then install | **60–120 sec** | 15–30 sec |
| Set API key | One command | instant | instant |
| First `python` run | Imports + first Groq call (network round-trip) | 5–12 sec | 3–6 sec |
| **Total, from a working Python** | | **~2–3.5 min** | **<1 min** |

Notes:
- Most of the install time is **network download**, not compilation — these are
  pure-Python wheels, so there's no C build step on a normal setup.
- On a **slow connection (~10 Mbps)** the `pip install` step can stretch to
  **4–6 minutes**; on **fast/NVMe + 200 Mbps** it's often **30–45 seconds**.
- If `pip` rebuilds anything from source (rare, older Python), add 1–3 minutes.
- Behind a corporate proxy, set `HTTPS_PROXY` before installing.

---

## Troubleshooting

- **`GROQ_API_KEY` not found** — the key isn't set in the current shell. Re-run the
  `export` / `$env:` command in the *same* terminal you run Python from.
- **`ModuleNotFoundError: langgraph`** — the virtualenv isn't active, or you
  installed into a different Python. Re-activate `.venv` and reinstall.
- **JSON / parse errors at runtime** — handled: `safe_json()` strips markdown
  fences and falls back gracefully, so a stray model formatting won't crash the run.
- **Rate limit / 429 from Groq** — free tier has limits; wait a moment and re-run.
- **Different verdict than expected** — `temperature=0` makes this close to
  deterministic, but models still vary slightly. If the evaluator holds
  `needs_more_evidence` on both passes, the retry cap ends the loop safely —
  that's intended behavior, not a bug.

---

## Adapts to other domains

The same graph — extract → evaluate → govern — is:
- **Fintech**: loan / KYC diligence
- **Edtech**: student-intervention triage
- **Scientific R&D**: research-paper screening

Architecture travels. Fork it, swap the domain, keep the shape.

---

## References

- LangGraph docs — docs.langchain.com/oss/python/langgraph
- LangGraph StateGraph API — reference.langchain.com/python/langgraph
- Multi-agent guidance — docs.langchain.com/oss/python/langchain/multi-agent
- Groq + LangChain — console.groq.com/docs/langchain
- Groq models & limits — console.groq.com/docs/models