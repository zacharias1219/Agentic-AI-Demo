# Clinical Trial Matching Copilot — Multi-Agent Demo

A teaching demo of **multi-agent architecture** with LangGraph + Groq. Three agents
share state to screen a patient against a clinical trial — and, crucially, **one agent
can catch another agent's mistake and force a correction.**

> This is an *architecture* demo, not a clinical decision system. Real deployment
> requires validated pipelines, safety review, auditability, and domain oversight.

## The idea

A single LLM can summarize a patient and a trial's criteria. But it's weak at
*separating gathering evidence from making a judgment*, and it tends to be
confidently wrong when data is missing. So we split the work:

| Agent | Job | Does NOT do |
|-------|-----|-------------|
| **Researcher** | Extract structured evidence, flag gaps | Decide eligibility |
| **Evaluator** | Judge eligibility; act as a **gatekeeper** | Invent facts |
| **Orchestrator** | Route the flow, handle the bounce-back, finalize | Read messy notes itself |

## The architecture

```
intake → orchestrator → research → orchestrator → evaluate → orchestrator → finalize
                            ▲                          │
                            └──────── bounce-back ◄─────┘
                         (evaluator: "needs_more_evidence")
```

The teaching moment: on the first pass the researcher is overconfident. The evaluator
returns `needs_more_evidence` with a pointed question. The orchestrator routes **back**
to the researcher with that question. Second pass, the researcher is honest about the
unresolved hepatic exclusion, the evaluator returns `unclear`, and the system finalizes
conservatively. No human in the loop — the system self-corrects.

## Run it

```bash
pip install -U langgraph langchain langchain-groq
export GROQ_API_KEY="your_key"
python clinical_trial_demo.py
```

## See WHY the architecture exists

Open the file and flip the switch at the top:

```python
EVALUATOR_ENABLED = False
```

Re-run. With no gatekeeper, the same patient gets a confident "LIKELY ELIGIBLE" —
no caveat, no question, despite an unresolved exclusion criterion. That's the danger
the architecture removes.

## Key engineering lessons

1. **Separate extraction from judgment.** Different tasks, different agents.
2. **Let agents refuse.** `needs_more_evidence` is the evaluator declining to guess.
3. **Make retries explicit.** The orchestrator routes the loop — it isn't buried in a prompt.
4. **Never trust raw LLM JSON.** `safe_json()` strips fences, extracts the object, and
   has a fallback that never crashes mid-run.
5. **temperature=0 for demos.** Deterministic behavior on stage.

## Adapts to other domains

The same graph is:
- **Fintech** — loan / KYC diligence (extract → assess risk → route exceptions)
- **Edtech** — student intervention triage (gather signals → evaluate → escalate)
- **Scientific R&D** — paper screening (extract claims → verify → synthesize)

Architecture travels. Fork it.

## Files
- `clinical_trial_demo.py` — the full, runnable demo
- `clinical_trial_demo_SKELETON.py` — node functions blanked out for live-coding
- `RUN_OF_SHOW.md` — presenter timeline and talk track
