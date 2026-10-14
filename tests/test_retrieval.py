from src.models import Document
from src.retrieval.hybrid import HybridRetriever


def test_hybrid_retrieval_prioritizes_matching_document() -> None:
    documents = [
        Document("refund", "Refund policy", "Refunds are available within 30 days.", "billing", "refund.md"),
        Document("password", "Password reset", "Reset links expire after 30 minutes.", "account", "password.md"),
    ]
    results = HybridRetriever(documents).search("How long do I have to request a refund?")
    assert results
    assert results[0].document.document_id == "refund"
