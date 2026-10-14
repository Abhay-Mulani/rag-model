import time

from langchain_core.tools import tool

_retriever = None
_generator = None


def setup_rag_tool(retriever, generator) -> None:
    global _retriever, _generator
    _retriever = retriever
    _generator = generator


@tool
def rag_query(question: str) -> dict:
    """Retrieve cited financial evidence and generate a grounded answer."""
    if _retriever is None or _generator is None:
        return {"error": "RAG tool is not initialized."}
    started = time.perf_counter()
    documents = _retriever.invoke(question)
    answer, confidence = _generator.generate(question, documents)
    return {
        "answer": answer,
        "retrieved_docs": [document.page_content[:500] for document in documents],
        "sources": [document.metadata for document in documents],
        "confidence": confidence,
        "latency": round(time.perf_counter() - started, 3),
    }
