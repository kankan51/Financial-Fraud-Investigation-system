"""
LLM Fraud Investigation Agent — LangGraph version
==================================================

Same investigation as investigation_agent.py (same data, same tools, same
system prompt), rebuilt as an explicit LangGraph StateGraph instead of a
hand-rolled loop. Reuses TransactionData, CaseTools and SYSTEM_PROMPT from
investigation_agent.py so the two scripts can't drift apart on logic.

    evidence_cases.json --> LangGraph (agent node <-> tools node) --> investigation_results.json

SETUP:
  pip install langgraph pandas
  # plus ONE provider package, matching whichever you use (see LLM_PROVIDER below):
  pip install langchain-anthropic anthropic        # provider=anthropic (paid, no free tier)
  pip install langchain-google-genai               # provider=gemini    (free, no card required)
  pip install langchain-groq                        # provider=groq      (free, no card required)

PROVIDER (set before running — defaults to gemini, the free option):
  export LLM_PROVIDER=gemini     # or: anthropic, groq
  export GOOGLE_API_KEY=your_key_here      # if provider=gemini  (console: aistudio.google.com/apikey)
  export GROQ_API_KEY=your_key_here        # if provider=groq    (console: console.groq.com)
  export ANTHROPIC_API_KEY=your_key_here   # if provider=anthropic

  Optional: INVESTIGATOR_MODEL overrides the default model for whichever
  provider is selected (defaults: gemini-3.8-flash / openai/gpt-oss-20b
  / claude-sonnet-5).

  Model catalogs on free-tier providers change often (Groq especially has
  deprecated/retired several models in 2026). If you hit a 404 "model not
  found" error, check the provider's current live list before assuming
  the script is broken:
    Groq:   curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY" | python3 -m json.tool
    Gemini: curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" | python3 -m json.tool
  Then override with: export INVESTIGATOR_MODEL=<whatever is actually live>

USAGE:
  1. python ieee_cis_pipeline.py --data_dir ./data     (creates evidence_cases.json)
  2. python investigation_agent_langgraph.py --data_dir ./data --max_cases 3

NOTE ON FREE-TIER MODELS
  Gemini Flash and Groq's Llama models are smaller/faster than Claude and
  can be less careful about sticking to the system prompt's rules (e.g.
  treating device/email strings as data, not instructions; not inventing
  V-column meanings). Spot-check a few verdicts against the evidence
  packages before trusting this at any real scale, more so than you'd
  need to with a frontier model.

WHY A GRAPH, NOT JUST A LOOP
  The plain version (investigation_agent.py) is a while-loop: call the model,
  if it wants a tool, run it, repeat. That's the whole agent. LangGraph makes
  the same thing explicit as a state machine with two nodes:

      agent (calls Claude) --tool requested--> tools (runs it) --> agent
      agent (no tool call, or verdict submitted) --> END

  For a single-tool-loop agent like this, that's genuinely more ceremony
  than the plain version for no real benefit. It starts paying off once you
  add things a plain loop makes awkward: extra nodes for a separate "senior
  analyst" review step, a conditional branch that routes low-confidence
  verdicts back for a second pass, or persistence/checkpointing so a stalled
  investigation can resume. This file is written to make adding that kind
  of node straightforward later.
"""

import argparse
import json
import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from investigation_agent import TransactionData, CaseTools, SYSTEM_PROMPT

RECURSION_LIMIT = 25  # caps agent<->tools round trips per case
MAX_RETRIES = 5        # transient server errors (503/429) per model call
RETRY_BASE_DELAY = 5   # seconds; doubles each retry (5, 10, 20, 40, 80)

TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                     "rate limit", "overloaded", "deadline exceeded")
# A 429 can mean "wait a few seconds" (retry helps) or "you're out of
# requests until tomorrow" (retry is pointless). Both trip TRANSIENT_MARKERS.
# These extra markers catch the second case so it fails fast instead of
# burning MAX_RETRIES on a quota that won't reset for hours.
DAILY_QUOTA_MARKERS = ("perday", "per day", "requests per day", "daily limit")


# Smaller/free-tier models occasionally emit a submit_verdict call that's
# missing required fields or is cut off mid-generation - the provider's own
# schema validation then rejects it with a 400, before it ever reaches our
# code. This is usually a one-off generation glitch, not a real malformed
# request, so it's worth a retry rather than failing the whole case.
MALFORMED_TOOL_CALL_MARKERS = ("tool_use_failed", "did not match schema",
                               "missing properties", "tool call validation failed")


def _is_transient(err: Exception) -> bool:
    text = str(err).lower()
    if any(marker.lower() in text for marker in TRANSIENT_MARKERS):
        return True
    return any(marker.lower() in text for marker in MALFORMED_TOOL_CALL_MARKERS)


def _is_daily_quota_exhausted(err: Exception) -> bool:
    text = str(err).lower()
    if any(marker in text for marker in DAILY_QUOTA_MARKERS):
        return True
    # Fallback: a retryDelay of several minutes or more also signals "come
    # back later", not "retry in a few seconds" - not worth looping on here.
    import re
    m = re.search(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+)s", text)
    return bool(m) and int(m.group(1)) > 120


def invoke_with_retry(llm, messages):
    """
    Free-tier model APIs (Gemini and Groq especially) return transient
    503/429 errors under load. Retries with exponential backoff before
    giving up, so a temporary "high demand" blip doesn't kill a whole run.
    A 429 caused by a DAILY quota being used up is different - no amount
    of retrying in this process will fix it, so that fails immediately
    with a clear message instead of wasting MAX_RETRIES. Non-transient
    errors (bad API key, invalid request) also raise immediately.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if _is_daily_quota_exhausted(e):
                raise RuntimeError(
                    "Daily free-tier request quota exhausted for this model. "
                    "Retrying won't help until it resets - either wait, switch "
                    "LLM_PROVIDER (e.g. groq, which has a much higher daily cap "
                    "on the free tier), or set INVESTIGATOR_MODEL to a different "
                    f"model.\nOriginal error: {e}"
                ) from e
            if not _is_transient(e) or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    transient error ({type(e).__name__}), retrying in {delay}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(delay)

PROVIDER_DEFAULTS = {
    "gemini": "gemini-3.8-flash",
    "groq": "openai/gpt-oss-20b",
    "anthropic": "claude-sonnet-5",
}
PROVIDER_KEY_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_llm(tools: list):
    """
    Builds the chat model for whichever provider LLM_PROVIDER names
    (default: gemini, the free no-credit-card option). All three speak
    the same LangChain .bind_tools()/.invoke() interface, so nothing
    else in the graph needs to know which one is running.
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider not in PROVIDER_DEFAULTS:
        raise SystemExit(f"Unknown LLM_PROVIDER '{provider}'. Use: gemini, groq, or anthropic.")

    key_env = PROVIDER_KEY_ENV[provider]
    if not os.environ.get(key_env):
        raise SystemExit(f"LLM_PROVIDER={provider} needs {key_env} set (export {key_env}=...).")

    model = os.environ.get("INVESTIGATOR_MODEL", PROVIDER_DEFAULTS[provider])

    # Without an explicit timeout, a hung connection can block forever -
    # no exception ever fires, so invoke_with_retry never gets a chance
    # to retry or fail. 60s is generous for one model turn but finite.
    REQUEST_TIMEOUT = 60

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # max_output_tokens gives the model room to finish the submit_verdict
        # call fully. Too low a cap is a common cause of a tool call that
        # gets cut off mid-generation and fails schema validation downstream.
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_output_tokens=3000, timeout=REQUEST_TIMEOUT)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=0, max_tokens=3000, timeout=REQUEST_TIMEOUT)
    else:  # anthropic
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, max_tokens=3000, timeout=REQUEST_TIMEOUT)

    return llm.bind_tools(tools)


# ---------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangGraph appends to this automatically
    verdict: dict | None
    trace: list  # [{"tool", "input", "output"}, ...] - full replay log, not just tool names


# ---------------------------------------------------------------------
# Tools — thin LangChain wrappers around CaseTools, bound to one case.
# submit_verdict is a "tool" only so the model can emit a structured,
# schema-checked result through the same tool-calling channel; the graph
# intercepts it (see tool_node) rather than treating it as more evidence.
# ---------------------------------------------------------------------
def build_tools(case_tools: CaseTools):
    @tool
    def get_transaction_details() -> dict:
        """Raw details of the flagged transaction: amount, product, card and address
        fields, email domains, device and browser info, time of day. Does not
        include any fraud label."""
        return case_tools.get_transaction_details()

    @tool
    def get_account_history(limit: int = 15) -> dict:
        """Earlier transactions from the same proxy account (card1 + addr1), before
        the flagged one: counts, amount statistics, device and email variety,
        days since last activity, and recent transactions.

        Args:
            limit: recent transactions to list (max 25, default 15).
        """
        return case_tools.get_account_history(limit=limit)

    @tool
    def find_linked_transactions(link_type: str) -> dict:
        """Earlier transactions from OTHER accounts sharing an attribute with the
        flagged one, with counts and confirmed-fraud counts among old-enough
        labels.

        Args:
            link_type: one of "device_info" (same device string),
                "card_other_address" (same card, different billing address),
                "address_other_cards" (same billing address, different card).
        """
        return case_tools.find_linked_transactions(link_type)

    @tool
    def submit_verdict(
        verdict: str,
        confidence: float,
        fraud_pattern: str,
        key_evidence: list[str],
        recommended_action: str,
        analyst_summary: str,
        limitations: str,
    ) -> dict:
        """Submit the final structured verdict. Call exactly once, when the
        investigation is finished.

        Args:
            verdict: "fraud", "legitimate", or "needs_human_review".
            confidence: 0 to 1.
            fraud_pattern: e.g. account_takeover, stolen_card, card_testing,
                synthetic_identity, none, unclear.
            key_evidence: findings, each traceable to a tool result or the SHAP package.
            recommended_action: "approve", "hold_and_verify_identity",
                "decline_and_flag", or "escalate_to_analyst".
            analyst_summary: three sentences an analyst can act on.
            limitations: what the evidence could not show.
        """
        return {
            "verdict": verdict, "confidence": confidence, "fraud_pattern": fraud_pattern,
            "key_evidence": key_evidence, "recommended_action": recommended_action,
            "analyst_summary": analyst_summary, "limitations": limitations,
        }

    return [get_transaction_details, get_account_history, find_linked_transactions, submit_verdict]


# ---------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------
def make_agent_node(llm_with_tools):
    def agent_node(state: AgentState):
        response = invoke_with_retry(llm_with_tools, state["messages"])
        return {"messages": [response]}
    return agent_node


def make_tool_node(tools_by_name: dict):
    def tool_node(state: AgentState):
        last = state["messages"][-1]
        outputs, verdict, new_trace_entries = [], None, []

        for call in last.tool_calls:
            tool_fn = tools_by_name.get(call["name"])
            try:
                result = tool_fn.invoke(call["args"]) if tool_fn else {"error": f"unknown tool {call['name']}"}
            except Exception as e:  # tool errors go back to the model, not up the stack
                result = {"error": f"{type(e).__name__}: {e}"}

            if call["name"] == "submit_verdict" and "error" not in result:
                verdict = result
            else:
                # Full input+output, not just the call name - this is what
                # lets a later viewer (or a published-page "live replay")
                # see exactly what evidence the agent actually gathered.
                new_trace_entries.append({"tool": call["name"], "input": call["args"], "output": result})
            outputs.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

        update = {"messages": outputs, "trace": state.get("trace", []) + new_trace_entries}
        if verdict is not None:
            update["verdict"] = verdict
        return update
    return tool_node


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def route_after_tools(state: AgentState):
    return END if state.get("verdict") is not None else "agent"


# ---------------------------------------------------------------------
# Build the graph for one case
# ---------------------------------------------------------------------
def build_graph(case_tools: CaseTools):
    tools = build_tools(case_tools)
    tools_by_name = {t.name: t for t in tools}
    llm = get_llm(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm))
    graph.add_node("tools", make_tool_node(tools_by_name))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return graph.compile()


FALLBACK_VERDICT = {
    "verdict": "needs_human_review", "confidence": 0.0, "fraud_pattern": "unclear",
    "key_evidence": [], "recommended_action": "escalate_to_analyst",
    "analyst_summary": "The agent did not submit a verdict within its turn limit.",
    "limitations": "Investigation incomplete.",
}


def investigate(data: TransactionData, case: dict) -> dict:
    case_tools = CaseTools(data, case["transaction_id"])
    app = build_graph(case_tools)
    case_for_model = {k: v for k, v in case.items() if k != "band"}  # band is bookkeeping

    init_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="Investigate this flagged transaction.\n\n" + json.dumps(case_for_model, indent=2)),
        ],
        "verdict": None,
        "trace": [],
    }
    final_state = app.invoke(init_state, config={"recursion_limit": RECURSION_LIMIT})
    verdict = final_state.get("verdict") or FALLBACK_VERDICT
    return {**verdict, "tool_calls": final_state.get("trace", [])}


def main(data_dir: str, cases_path: str, max_cases: int, out_path: str):
    # get_llm() does the real key check (provider-specific) the first time
    # it's called, inside build_graph() for the first case. This early
    # check just avoids loading the dataset first and failing right after.
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    key_env = PROVIDER_KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env):
        raise SystemExit(f"LLM_PROVIDER={provider} needs {key_env} set (export {key_env}=...).")

    with open(cases_path) as f:
        cases = json.load(f)[:max_cases]

    print(f"Loading transaction data from {data_dir}...")
    data = TransactionData(data_dir)

    # Resume support: a prior run's successes are never redone. Without
    # this, stopping (Ctrl+C, a crash, the Mac sleeping overnight) means
    # every case - including ones already paid for in API calls - gets
    # investigated again from scratch on the next run.
    results = []
    done_ids = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                results = json.load(f)
            done_ids = {r["transaction_id"] for r in results if "error" not in r}
        except (json.JSONDecodeError, KeyError):
            results = []
        if done_ids:
            print(f"Resuming: {len(done_ids)} case(s) already completed in {out_path}, skipping those.")

    for i, case in enumerate(cases):
        tid = case["transaction_id"]
        if tid in done_ids:
            continue
        results = [r for r in results if r["transaction_id"] != tid]  # drop any stale failed attempt
        print(f"\nInvestigating transaction {tid} (band: {case.get('band')}, ML score {case['ml_risk_score']:.3f})...")
        try:
            verdict = investigate(data, case)
            error = None
        except Exception as e:
            # One case failing (model error, bad schema, provider outage)
            # shouldn't throw away every case that already succeeded. Log
            # it, record the failure on this case, and keep going.
            print(f"  FAILED: {type(e).__name__}: {e}")
            verdict = {**FALLBACK_VERDICT, "tool_calls": []}
            error = f"{type(e).__name__}: {e}"

        record = {
            "transaction_id": tid,
            "band": case.get("band"),
            "ml_risk_score": case["ml_risk_score"],
            **verdict,
            "ground_truth_is_fraud": data.ground_truth(tid),  # evaluation only
        }
        if error:
            record["error"] = error
        results.append(record)

        if error is None:
            print(f"  verdict: {verdict['verdict']} (confidence {verdict['confidence']:.2f}), "
                  f"action: {verdict['recommended_action']}, actual fraud: {record['ground_truth_is_fraud']}")
            print(f"  {verdict['analyst_summary']}")

        # Write after every case, not just at the end - so a crash on
        # case 3 doesn't lose the results already earned from cases 1-2.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    succeeded = sum(1 for r in results if "error" not in r)
    print(f"\nSaved {len(results)} investigations to {out_path} "
          f"({succeeded} succeeded, {len(results) - succeeded} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--cases", default="evidence_cases.json")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--out", default="investigation_results.json")
    args = parser.parse_args()
    main(args.data_dir, args.cases, args.max_cases, args.out)
