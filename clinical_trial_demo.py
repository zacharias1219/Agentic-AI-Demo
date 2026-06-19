"""
Clinical Trial Matching Copilot — Multi-Agent Demo (LangGraph + Groq)
=====================================================================

A teaching demo of multi-agent architecture. Three agents share state:
  - Researcher : extracts structured evidence (does NOT judge eligibility)
  - Evaluator  : judges eligibility AND can bounce work back if evidence is weak
  - Orchestrator: routes the flow, handles the bounce-back, finalizes

THE TEACHING MOMENT:
  The Evaluator is a *gatekeeper*. On the first pass the Researcher is
  overconfident about the patient's liver status. The Evaluator catches it,
  returns 'needs_more_evidence' with a pointed question, and the Orchestrator
  routes BACK to the Researcher. Second pass, the Researcher is honest about
  the unresolved hepatic severity. Now the Evaluator returns 'unclear' and we
  finalize conservatively. The audience watches one agent correct another.

THE CLIMAX:
  Flip EVALUATOR_ENABLED = False and re-run. With no gatekeeper, the system
  confidently finalizes with no caveat — proving live WHY the architecture
  exists.

Run:
  pip install -U langgraph langchain langchain-groq
  export GROQ_API_KEY="your_key"
  python clinical_trial_demo.py
"""

import os
import re
import json
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =====================================================================
# DEMO SWITCH — flip this to False for the climax ("delete the gatekeeper")
# =====================================================================
EVALUATOR_ENABLED = False


# =====================================================================
# 0. Pretty live logging so the audience SEES the graph execute
# =====================================================================
STEP = 0
def log(node: str, msg: str) -> None:
    global STEP
    STEP += 1
    print(f"\n[{STEP:02d}] -- {node.upper():<14} -- {msg}")


# =====================================================================
# 1. Shared State
# =====================================================================
class TrialState(TypedDict, total=False):
    patient_case: str
    trial_criteria: str
    evidence_pack: Dict[str, Any]
    evidence_gaps: List[str]
    evaluator_report: Dict[str, Any]
    final_recommendation: str
    next_step: str
    retry_count: int
    bounce_reason: str        # the evaluator's question, fed back to researcher


# =====================================================================
# 2. LLM — temperature 0 for a deterministic live demo
# =====================================================================
_llm = None
def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    return _llm


# =====================================================================
# 3. CRASH-PROOF JSON PARSING  (the thing that saves you on stage)
#    LLMs sometimes wrap JSON in ```fences``` or add a preamble line.
#    Never call json.loads() on raw model output in a live demo.
# =====================================================================
def safe_json(raw: Any) -> Dict[str, Any]:
    # Groq returns a string; handle list-of-parts defensively too.
    if isinstance(raw, list):
        raw = "".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw)
    text = str(raw).strip()

    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    # Grab the outermost {...} in case of preamble/postamble
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last-resort fallback so the DEMO NEVER DIES mid-talk
        return {
            "_parse_error": True,
            "_raw": str(raw)[:500],
            "evidence_gaps": ["Parser fallback engaged"],
        }


# =====================================================================
# 4. Prompts
# =====================================================================
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a clinical research EXTRACTION agent.
Extract structured facts from the patient case and trial criteria.
You do NOT decide eligibility — that is another agent's job.

Return valid JSON only with keys:
  patient_facts, trial_requirements, evidence_gaps, matching_notes

Rules:
- If data is missing or ambiguous, list it in evidence_gaps. Do NOT paper over it.
- Never invent lab values, severities, or diagnoses.
{focus}"""),
    ("human",
     "PATIENT CASE:\n{patient_case}\n\nTRIAL CRITERIA:\n{trial_criteria}\n\nReturn JSON only."),
])

evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a clinical trial EVALUATOR and gatekeeper.
You receive only structured evidence from another agent.

Return valid JSON only with keys:
  verdict, confidence, included_matches, exclusion_risks, reasoning, questions_for_coordinator

verdict must be exactly one of:
  eligible, unclear, unlikely_eligible, needs_more_evidence

Rules:
- Be conservative. Patient safety over optimism.
- If a KEY EXCLUSION criterion cannot be cleared from the evidence,
  AND this is the first review, return 'needs_more_evidence' and put a
  precise question in questions_for_coordinator.
- If evidence has already been re-reviewed and the gap remains genuinely
  unresolvable, return 'unclear' (not needs_more_evidence).
- confidence is between 0 and 1.
This is review pass #{review_pass}."""),
    ("human",
     "STRUCTURED EVIDENCE:\n{evidence_pack}\n\nReturn JSON only."),
])


# =====================================================================
# 5. NODE FUNCTIONS  ◄◄◄ THESE ARE WHAT YOU TYPE LIVE
# =====================================================================
def intake_node(state: TrialState) -> Dict[str, Any]:
    log("intake", "new patient case received, resetting counters")
    return {"retry_count": 0, "next_step": "research"}


def researcher_node(state: TrialState) -> Dict[str, Any]:
    bounce = state.get("bounce_reason")
    if bounce:
        log("researcher", f"RE-EXAMINING. Evaluator asked: {bounce}")
        focus = ("FOCUS: An evaluator flagged unresolved exclusion criteria below. "
                 "Re-examine honestly. If severity is not explicit in the source, "
                 "mark it UNRESOLVED rather than inferring.\n"
                 f"Evaluator's question: {bounce}")
    else:
        log("researcher", "first-pass extraction of patient + trial facts")
        focus = ""

    chain = researcher_prompt | get_llm()
    raw = chain.invoke({
        "patient_case": state["patient_case"],
        "trial_criteria": state["trial_criteria"],
        "focus": focus,
    }).content
    parsed = safe_json(raw)

    gaps = parsed.get("evidence_gaps", [])
    log("researcher", f"extracted. evidence_gaps = {gaps}")
    return {"evidence_pack": parsed, "evidence_gaps": gaps}


def evaluator_node(state: TrialState) -> Dict[str, Any]:
    review_pass = state.get("retry_count", 0) + 1
    log("evaluator", f"judging eligibility (review pass #{review_pass})")

    chain = evaluator_prompt | get_llm()
    raw = chain.invoke({
        "evidence_pack": json.dumps(state["evidence_pack"], indent=2),
        "review_pass": review_pass,
    }).content
    parsed = safe_json(raw)

    verdict = parsed.get("verdict", "unclear")
    conf = parsed.get("confidence", "?")
    log("evaluator", f"verdict = {verdict.upper()}  (confidence {conf})")
    return {"evaluator_report": parsed}


def finalizer_node(state: TrialState) -> Dict[str, Any]:
    report = state.get("evaluator_report", {})
    verdict = report.get("verdict", "unclear")
    questions = report.get("questions_for_coordinator", [])

    if verdict == "eligible":
        txt = "Preliminary result: LIKELY ELIGIBLE. Next: coordinator review before screening."
    elif verdict == "unlikely_eligible":
        txt = "Preliminary result: UNLIKELY ELIGIBLE. Next: review exclusions, consider alternate trials."
    else:
        txt = ("Preliminary result: POTENTIALLY ELIGIBLE, NOT SAFE TO FINALIZE. "
               "Next: obtain clarification on unresolved exclusion criteria.")
    if questions:
        txt += "\n  Open questions: " + "; ".join(questions)

    log("finalize", "synthesizing coordinator note")
    return {"final_recommendation": txt, "next_step": "done"}


# =====================================================================
# 6. ORCHESTRATOR — the brain that routes (incl. the bounce-back)
# =====================================================================
def orchestrator_node(state: TrialState) -> Dict[str, Any]:
    # No evidence yet -> research
    if not state.get("evidence_pack"):
        log("orchestrator", "no evidence yet -> route to researcher")
        return {"next_step": "research"}

    # Climax mode: no gatekeeper. Jump straight to finalize after research.
    if not EVALUATOR_ENABLED:
        if not state.get("evaluator_report"):
            log("orchestrator", "⚠ EVALUATOR DISABLED -> finalizing on raw research")
            # fabricate an optimistic report to show the danger
            return {"next_step": "finalize",
                    "evaluator_report": {"verdict": "eligible"}}
        return {"next_step": "finalize"}

    # Normal mode: we have evidence but no judgment yet -> evaluate
    if not state.get("evaluator_report"):
        log("orchestrator", "evidence ready -> route to evaluator")
        return {"next_step": "evaluate"}

    # The GATEKEEPER bounce-back
    verdict = state["evaluator_report"].get("verdict")
    if verdict == "needs_more_evidence" and state.get("retry_count", 0) < 1:
        q = (state["evaluator_report"].get("questions_for_coordinator") or ["(no question)"])[0]
        log("orchestrator", f"evaluator BOUNCED IT BACK -> re-research. Q: {q}")
        return {
            "next_step": "research",
            "bounce_reason": q,
            "retry_count": state.get("retry_count", 0) + 1,
            "evaluator_report": None,   # clear so evaluator runs again
        }

    log("orchestrator", "judgment accepted -> finalize")
    return {"next_step": "finalize"}


def router(state: TrialState) -> str:
    return state["next_step"]


# =====================================================================
# 7. BUILD GRAPH  (pre-written — you walk through this, don't type it)
# =====================================================================
builder = StateGraph(TrialState)
builder.add_node("intake", intake_node)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("research", researcher_node)
builder.add_node("evaluate", evaluator_node)
builder.add_node("finalize", finalizer_node)

builder.set_entry_point("intake")
builder.add_edge("intake", "orchestrator")
builder.add_conditional_edges(
    "orchestrator", router,
    {"research": "research", "evaluate": "evaluate", "finalize": "finalize"},
)
builder.add_edge("research", "orchestrator")
builder.add_edge("evaluate", "orchestrator")
builder.add_edge("finalize", END)

graph = builder.compile()


# =====================================================================
# 8. DEMO INPUT
# =====================================================================
if __name__ == "__main__":
    initial_state: TrialState = {
        "patient_case": """
58-year-old female with stage II ER-positive breast cancer.
ECOG performance status 1.
Completed prior chemotherapy 8 months ago.
Mild liver enzyme elevation noted in recent review (no LFT values on file).
No documented brain metastases.
No clear note about active infection.
""",
        "trial_criteria": """
Inclusion:
- Age 18 to 70
- Histologically confirmed breast cancer
- ECOG 0 or 1
- Prior standard therapy allowed
Exclusion:
- Severe hepatic impairment
- Active CNS metastases
- Uncontrolled infection
""",
    }

    print("=" * 70)
    print(f"  CLINICAL TRIAL MATCHING COPILOT   |  EVALUATOR_ENABLED = {EVALUATOR_ENABLED}")
    print("=" * 70)

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)
    print(result["final_recommendation"])