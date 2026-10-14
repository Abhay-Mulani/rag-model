import re
from collections import Counter

from src.models import Document, RetrievedDocument


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:


    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self.document_tokens = [_tokens(f"{d.title} {d.content}") for d in documents]
        self.document_frequency = Counter(
            token for tokens in self.document_tokens for token in set(tokens)
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        results: list[RetrievedDocument] = []
        total_documents = len(self.documents)
        for document, tokens in zip(self.documents, self.document_tokens):
            token_counts = Counter(tokens)
            overlap = query_set.intersection(token_counts)
            keyword_score = len(overlap) / len(query_set)
            weighted_overlap = sum(
                1 + (total_documents / (1 + self.document_frequency[token]))
                for token in overlap
            )
            semantic_score = weighted_overlap / (
                len(query_set) * (total_documents + 1)
            )
            score = (0.65 * keyword_score) + (0.35 * min(semantic_score, 1.0))
            if score > 0:
                results.append(
                    RetrievedDocument(
                        document=document,
                        score=score,
                        keyword_score=keyword_score,
                        semantic_score=min(semantic_score, 1.0),
                    )
                )
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
