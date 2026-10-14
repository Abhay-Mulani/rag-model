from langchain_core.documents import Document as LCDocument


class HybridRetriever:
    def __init__(
        self,
        semantic_retriever,
        keyword_retriever,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        k: int = 8,
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.k = k

    def invoke(self, query: str) -> list[LCDocument]:
        scores: dict[str, float] = {}
        documents: dict[str, LCDocument] = {}
        for retriever, weight in (
            (self.semantic_retriever, self.semantic_weight),
            (self.keyword_retriever, self.keyword_weight),
        ):
            for rank, document in enumerate(retriever.invoke(query, self.k), start=1):
                key = document.page_content.strip()
                documents[key] = document
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [documents[key] for key in ranked[: self.k]]
