from typing import TypedDict


class AgentState(TypedDict):
    query: str
    session_id: str
    risk_appetite: str
    intent: str
    retrieved_docs: list[str]
    sources: list[dict]
    retriever_output: str
    analysis_output: str
    portfolio_output: str
    risk_output: str
    final_answer: str
    evaluator_verdict: str
    evaluator_score: int
    evaluator_feedback: str
    confidence: float
    attempts: int
    agents_invoked: list[str]
    latency: float
    error: str
