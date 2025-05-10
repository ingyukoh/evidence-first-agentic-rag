from __future__ import annotations

from time import perf_counter

from langgraph.graph import END, START, StateGraph

from .generation import Generator
from .models import AgentState, TraceEvent
from .retrieval import BM25Retriever
from .security import detect_prompt_injection
from .tools import analyze_skill_gap, inspect_evidence_quality, select_tools


def _event(node: str, started: float, **metadata: object) -> TraceEvent:
    return TraceEvent(node=node, duration_ms=(perf_counter() - started) * 1_000, metadata=metadata)


def build_graph(retriever: BM25Retriever, generator: Generator):
    def guard_input(state: AgentState) -> AgentState:
        started = perf_counter()
        reason = detect_prompt_injection(state["query"])
        return {
            "blocked": reason is not None,
            "block_reason": reason,
            "trace": [
                *state.get("trace", []),
                _event("guard_input", started, blocked=bool(reason)),
            ],
        }

    def retrieve(state: AgentState) -> AgentState:
        started = perf_counter()
        hits = retriever.search(state["query"], state.get("top_k", 3))
        return {
            "hits": hits,
            "trace": [
                *state.get("trace", []),
                _event("retrieve", started, hit_ids=[hit.document.id for hit in hits]),
            ],
        }

    def run_tools(state: AgentState) -> AgentState:
        started = perf_counter()
        observations: list[str] = []
        for tool in select_tools(state["query"]):
            if tool == "skill_gap":
                observations.append(analyze_skill_gap(state["query"], state["hits"]))
            elif tool == "evidence_quality":
                observations.append(inspect_evidence_quality(state["hits"]))
        return {
            "tool_observations": observations,
            "trace": [
                *state.get("trace", []),
                _event("run_tools", started, tools=select_tools(state["query"])),
            ],
        }

    def synthesize(state: AgentState) -> AgentState:
        started = perf_counter()
        answer = generator.generate(
            state["query"], state.get("hits", []), state.get("tool_observations", [])
        )
        return {
            "answer": answer,
            "trace": [*state.get("trace", []), _event("synthesize", started)],
        }

    def blocked_response(state: AgentState) -> AgentState:
        started = perf_counter()
        return {
            "hits": [],
            "tool_observations": [],
            "answer": "Request blocked by the input safety policy.",
            "trace": [*state.get("trace", []), _event("blocked_response", started)],
        }

    def route_after_guard(state: AgentState) -> str:
        return "blocked_response" if state.get("blocked") else "retrieve"

    graph = StateGraph(AgentState)
    graph.add_node("guard_input", guard_input)
    graph.add_node("retrieve", retrieve)
    graph.add_node("run_tools", run_tools)
    graph.add_node("synthesize", synthesize)
    graph.add_node("blocked_response", blocked_response)
    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges(
        "guard_input",
        route_after_guard,
        {"blocked_response": "blocked_response", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "run_tools")
    graph.add_edge("run_tools", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("blocked_response", END)
    return graph.compile()
