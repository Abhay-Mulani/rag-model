import time
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.state import AgentState
from src.config import Settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.corpus import load_corpus
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.keyword_search import BM25Retriever
from src.retrieval.semantic_search import SemanticRetriever
from src.tools.market_data_tool import calculate_portfolio_risk, suggest_allocation
from src.tools.rag_tool import setup_rag_tool


class FinanceWorkflow:
    def __init__(self, generator, settings: Settings) -> None:
        self.settings = settings
        documents = chunk_documents(
            load_corpus(settings.data_path), settings.chunk_size, settings.chunk_overlap
        )
        self.keyword = BM25Retriever(documents)
        self.semantic = SemanticRetriever(
            documents, settings.faiss_index_path, settings.embedding_model
        )
        semantic_backend = self.semantic if settings.enable_semantic_retrieval else self.keyword
        self.retriever = HybridRetriever(
            semantic_backend,
            self.keyword,
            settings.semantic_weight,
            settings.bm25_weight,
            settings.retrieval_top_k,
        )
        self.generator = generator
        setup_rag_tool(self.retriever, generator)
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner)
        graph.add_node("retriever", self._retriever)
        graph.add_node("analysis_agent", self._analysis)
        graph.add_node("portfolio_agent", self._portfolio)
        graph.add_node("risk_agent", self._risk)
        graph.add_node("validator", self._validator)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retriever")
        graph.add_conditional_edges(
            "retriever",
            self._route_specialist,
            {
                "analysis_agent": "analysis_agent",
                "portfolio_agent": "portfolio_agent",
                "risk_agent": "risk_agent",
                "validator": "validator",
            },
        )
        graph.add_edge("analysis_agent", "validator")
        graph.add_edge("portfolio_agent", "validator")
        graph.add_edge("risk_agent", "validator")
        graph.add_conditional_edges(
            "validator", self._after_validation, {"retry": "retriever", "end": END}
        )
        self.graph = graph.compile(checkpointer=MemorySaver())

    def run(
        self, query: str, session_id: str | None = None, risk_appetite: str = "moderate"
    ) -> AgentState:
        initial = {
            "query": query,
            "session_id": session_id or str(uuid4()),
            "risk_appetite": risk_appetite,
            "intent": "",
            "retrieved_docs": [],
            "sources": [],
            "retriever_output": "",
            "analysis_output": "",
            "portfolio_output": "",
            "risk_output": "",
            "final_answer": "",
            "evaluator_verdict": "",
            "evaluator_score": 0,
            "evaluator_feedback": "",
            "confidence": 0.0,
            "attempts": 0,
            "agents_invoked": [],
            "latency": 0.0,
            "error": "",
        }
        result = self.graph.invoke(
            initial, config={"configurable": {"thread_id": initial["session_id"]}}
        )
        return result

    def _planner(self, state: AgentState) -> dict:
        query = state["query"].lower()
        intent = (
            "risk" if any(word in query for word in ("risk", "volatility", "drawdown", "beta"))
            else "portfolio" if any(word in query for word in ("portfolio", "allocation", "sip"))
            else "analysis" if any(word in query for word in ("analysis", "compare", "outlook"))
            else "market_info"
        )
        return {
            "intent": intent,
            "agents_invoked": state["agents_invoked"] + ["planner"],
        }

    def _retriever(self, state: AgentState) -> dict:
        started = time.perf_counter()
        documents = self.retriever.invoke(state["query"])
        return {
            "retrieved_docs": [document.page_content for document in documents],
            "sources": [document.metadata for document in documents],
            "retriever_output": f"Retrieved {len(documents)} grounded documents.",
            "agents_invoked": state["agents_invoked"] + ["retriever"],
            "latency": state["latency"] + (time.perf_counter() - started),
        }

    def _analysis(self, state: AgentState) -> dict:
        documents = self._documents_from_state(state)
        answer, confidence = self.generator.generate(
            f"Analyze this financial question: {state['query']}", documents
        )
        return {
            "analysis_output": answer,
            "final_answer": answer,
            "confidence": confidence,
            "agents_invoked": state["agents_invoked"] + ["analysis_agent"],
        }

    def _portfolio(self, state: AgentState) -> dict:
        guidance = suggest_allocation(state["risk_appetite"])
        answer, confidence = self.generator.generate(
            f"Explain portfolio considerations for: {state['query']}",
            self._documents_from_state(state),
        )
        return {
            "portfolio_output": f"{answer}\n\nAllocation framework: {guidance}",
            "final_answer": f"{answer}\n\nAllocation framework: {guidance}",
            "confidence": confidence,
            "agents_invoked": state["agents_invoked"] + ["portfolio_agent"],
        }

    def _risk(self, state: AgentState) -> dict:
        answer, confidence = self.generator.generate(
            f"Assess investment risk for: {state['query']}",
            self._documents_from_state(state),
        )
        risk = calculate_portfolio_risk([])
        return {
            "risk_output": f"{answer}\n\nRisk framework: {risk}",
            "final_answer": f"{answer}\n\nRisk framework: {risk}",
            "confidence": confidence,
            "agents_invoked": state["agents_invoked"] + ["risk_agent"],
        }

    def _validator(self, state: AgentState) -> dict:
        passed = bool(state["retrieved_docs"]) and state["confidence"] >= self.settings.relevance_threshold
        return {
            "evaluator_verdict": "approved" if passed else "needs_retry",
            "evaluator_score": round(state["confidence"] * 5),
            "evaluator_feedback": "Grounded evidence found." if passed else "More evidence required.",
            "agents_invoked": state["agents_invoked"] + ["validator"],
            "attempts": state["attempts"] + 1,
        }

    def _route_specialist(self, state: AgentState) -> str:
        return {
            "analysis": "analysis_agent",
            "portfolio": "portfolio_agent",
            "risk": "risk_agent",
        }.get(state["intent"], "validator")

    def _after_validation(self, state: AgentState) -> str:
        if (
            state["evaluator_verdict"] == "needs_retry"
            and state["attempts"] < self.settings.max_validation_retries
        ):
            return "retry"
        return "end"

    @staticmethod
    def _documents_from_state(state: AgentState):
        from langchain_core.documents import Document

        return [
            Document(page_content=text, metadata=source)
            for text, source in zip(state["retrieved_docs"], state["sources"])
        ]


def build_graph(generator, settings: Settings):
    return FinanceWorkflow(generator, settings).graph
