"""
==============================================================================
  CLINICAL TRIAL MATCHING COPILOT  —  A Multi-Agent Architecture Demo
  LangGraph + Groq (Llama-3.3-70B)
------------------------------------------------------------------------------
  The code is organised into 8 numbered BLOCKS so it can be taught in order:

    BLOCK 1  Imports & the demo switch
    BLOCK 2  Shared State        (the whiteboard all agents read & write)
    BLOCK 3  The model (Groq)     + crash-proof JSON parsing
    BLOCK 4  Agent prompts        (the job description for each agent)
    BLOCK 5  The agent NODES      (Researcher, Evaluator, Finaliser)
    BLOCK 6  The ORCHESTRATOR     (the brain: routing + bounce-back)
    BLOCK 7  Build the GRAPH      (wire the nodes together)
    BLOCK 8  Run the demo

  (c) Richard Abishai - richardabish.ai
==============================================================================
"""

# =============================================================================
# BLOCK 1 - IMPORTS & THE DEMO SWITCH
# -----------------------------------------------------------------------------
# Flip EVALUATOR_ENABLED to False to run the "delete the gatekeeper" climax:
# the system loses its ability to doubt and becomes confidently wrong.
# =============================================================================
import os
import re
import json
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

EVALUATOR_ENABLED = True          # <- the one line we toggle live


# =============================================================================
# BLOCK 2 - SHARED STATE
# -----------------------------------------------------------------------------
# Think of this as a shared whiteboard. Every agent reads from it and writes
# back to it. In LangGraph each node takes the State and returns a partial
# update to it - "State in, partial State out." That is the whole mental model.
# =============================================================================
class TrialState(TypedDict, total=False):
    patient_case: str               # raw input: the patient's notes
    trial_criteria: str             # raw input: the trial's rules
    evidence_pack: Dict[str, Any]   # researcher's structured extraction
    evidence_gaps: List[str]        # what the researcher couldn't resolve
    evaluator_report: Dict[str, Any]# evaluator's verdict + reasoning
    final_recommendation: str       # the coordinator-facing output
    next_step: str                  # the orchestrator's routing decision
    retry_count: int                # how many times we've re-investigated
    bounce_reason: str              # the evaluator's question, fed back


# =============================================================================
# BLOCK 3 - THE MODEL (GROQ) + CRASH-PROOF JSON PARSING
# -----------------------------------------------------------------------------
# get_llm() is lazy so the file imports even without a key set.
# temperature=0 -> the most deterministic behaviour a model gives (good onstage)
#
# safe_json() is the unsung hero of any live LLM demo. Models sometimes wrap
# JSON in ```fences``` or add a chatty sentence. Calling json.loads() on raw
# output WILL crash you mid-talk. This strips the junk and never throws.
# =============================================================================
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


def safe_json(raw: Any) -> Dict[str, Any]:
    # Groq returns a string; handle list-of-parts defensively too.
    if isinstance(raw, list):
        raw = "".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw)
    text = str(raw).strip()

    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    # Grab the outermost { ... } in case of preamble / postamble
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last-resort fallback so the DEMO NEVER DIES mid-talk
        return {"_parse_error": True, "_raw": str(raw)[:400],
                "evidence_gaps": ["Parser fallback engaged"]}


# =============================================================================
# BLOCK 4 - AGENT PROMPTS  (each agent's job description)
# -----------------------------------------------------------------------------
# Two specialists. Note the hard rules baked into each:
#   - Researcher: extract only, NEVER decide. Be honest about gaps.
#   - Evaluator : judge from evidence only, and it is ALLOWED to say
#                 "needs_more_evidence" - i.e. permission to be uncertain.
# =============================================================================
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a clinical research EXTRACTION agent.
Extract structured facts from the patient case and trial criteria.
You do NOT decide eligibility - that is another agent's job.

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
- If a KEY EXCLUSION criterion cannot be cleared from the evidence, AND this is
  the first review, return 'needs_more_evidence' with a precise question.
- If the gap remains genuinely unresolvable after re-review, return 'unclear'.
- confidence is between 0 and 1.
This is review pass #{review_pass}."""),
    ("human",
     "STRUCTURED EVIDENCE:\n{evidence_pack}\n\nReturn JSON only."),
])


# =============================================================================
# BLOCK 5 - THE AGENT NODES
# -----------------------------------------------------------------------------
# Each node is a plain function: takes State, returns a partial State update.
# Notice the researcher's "bounce" branch - on a re-run it is steered by the
# evaluator's exact question. The second pass is a TARGETED re-investigation,
# not a blind retry.
# =============================================================================
def intake_node(state: TrialState) -> Dict[str, Any]:
    print(f"\n[INTAKE]       new case received")
    return {"retry_count": 0, "next_step": "research"}


def researcher_node(state: TrialState) -> Dict[str, Any]:
    bounce = state.get("bounce_reason")
    if bounce:
        print(f"[RESEARCHER]   re-examining - evaluator asked: {bounce}")
        focus = ("FOCUS: An evaluator flagged unresolved exclusion criteria. "
                 "Re-examine honestly. If severity is not explicit in the source, "
                 "mark it UNRESOLVED rather than inferring.\n"
                 f"Evaluator's question: {bounce}")
    else:
        print(f"[RESEARCHER]   first-pass extraction")
        focus = ""

    chain = researcher_prompt | get_llm()
    raw = chain.invoke({
        "patient_case": state["patient_case"],
        "trial_criteria": state["trial_criteria"],
        "focus": focus,
    }).content
    parsed = safe_json(raw)
    gaps = parsed.get("evidence_gaps", [])
    print(f"[RESEARCHER]   evidence_gaps = {gaps}")
    return {"evidence_pack": parsed, "evidence_gaps": gaps}


def evaluator_node(state: TrialState) -> Dict[str, Any]:
    review_pass = state.get("retry_count", 0) + 1
    print(f"[EVALUATOR]    judging (review pass #{review_pass})")

    chain = evaluator_prompt | get_llm()
    raw = chain.invoke({
        "evidence_pack": json.dumps(state["evidence_pack"], indent=2),
        "review_pass": review_pass,
    }).content
    parsed = safe_json(raw)
    print(f"[EVALUATOR]    verdict = {parsed.get('verdict','?').upper()} "
          f"(confidence {parsed.get('confidence','?')})")
    return {"evaluator_report": parsed}


def finalizer_node(state: TrialState) -> Dict[str, Any]:
    report = state.get("evaluator_report", {})
    verdict = report.get("verdict", "unclear")
    questions = report.get("questions_for_coordinator", [])

    if verdict == "eligible":
        txt = "LIKELY ELIGIBLE. Next: coordinator review before screening."
    elif verdict == "unlikely_eligible":
        txt = "UNLIKELY ELIGIBLE. Next: review exclusions, consider alternate trials."
    else:
        txt = ("POTENTIALLY ELIGIBLE, NOT SAFE TO FINALIZE. "
               "Next: obtain clarification on unresolved exclusion criteria.")
    if questions:
        txt += "\n  Open questions: " + "; ".join(questions)

    print(f"[FINALIZE]     done")
    return {"final_recommendation": txt, "next_step": "done"}


# =============================================================================
# BLOCK 6 - THE ORCHESTRATOR  (the brain)
# -----------------------------------------------------------------------------
# The only node that touches no medicine. It routes. Read the branches top to
# bottom - the highlighted one is the whole demo:
#   evaluator said "needs_more_evidence" -> DON'T finalize -> send work BACK to
#   the researcher with the evaluator's question, and bump the retry counter.
# In THIS 3-agent demo the routing is simple. In a real system (see the deck)
# the orchestrator manages many agents, tools, retries, budgets and escalation
# - that is where it earns its keep.
# =============================================================================
def orchestrator_node(state: TrialState) -> Dict[str, Any]:
    # 1) No evidence yet -> research
    if not state.get("evidence_pack"):
        print("[ORCHESTRATOR] no evidence yet -> researcher")
        return {"next_step": "research"}

    # 2) Climax mode: gatekeeper disabled -> finalise on raw research (unsafe)
    if not EVALUATOR_ENABLED:
        if not state.get("evaluator_report"):
            print("[ORCHESTRATOR] !! EVALUATOR DISABLED -> finalising on raw research")
            return {"next_step": "finalize", "evaluator_report": {"verdict": "eligible"}}
        return {"next_step": "finalize"}

    # 3) Have evidence, no verdict yet -> evaluate
    if not state.get("evaluator_report"):
        print("[ORCHESTRATOR] evidence ready -> evaluator")
        return {"next_step": "evaluate"}

    # 4) * THE BOUNCE-BACK * - evaluator refused, and we can still retry
    verdict = state["evaluator_report"].get("verdict")
    if verdict == "needs_more_evidence" and state.get("retry_count", 0) < 1:
        q = (state["evaluator_report"].get("questions_for_coordinator") or ["(no question)"])[0]
        print(f"[ORCHESTRATOR] evaluator BOUNCED IT BACK -> re-research. Q: {q}")
        return {
            "next_step": "research",
            "bounce_reason": q,
            "retry_count": state.get("retry_count", 0) + 1,
            "evaluator_report": None,   # clear so evaluator runs fresh
        }

    # 5) Verdict accepted (or retry budget spent) -> finalise
    print("[ORCHESTRATOR] judgment accepted -> finalize")
    return {"next_step": "finalize"}


def router(state: TrialState) -> str:
    return state["next_step"]


# =============================================================================
# BLOCK 7 - BUILD THE GRAPH  (wire the nodes together)
# -----------------------------------------------------------------------------
# Add nodes, set the entry point, then add edges. The conditional edges out of
# the orchestrator are what make routing (and the bounce-back loop) possible.
# graph.compile() validates the wiring before anything runs.
# =============================================================================
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


# =============================================================================
# BLOCK 8 - RUN THE DEMO
# -----------------------------------------------------------------------------
# One realistic patient, one oncology trial. Most criteria clearly met - but
# the liver/hepatic exclusion is ambiguous, which is what triggers the loop.
# =============================================================================
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

    print("=" * 68)
    print(f"  CLINICAL TRIAL MATCHING COPILOT   |  EVALUATOR_ENABLED = {EVALUATOR_ENABLED}")
    print("=" * 68)

    result = graph.invoke(initial_state)

    print("\n" + "=" * 68)
    print("  FINAL RECOMMENDATION")
    print("=" * 68)
    print(result["final_recommendation"])