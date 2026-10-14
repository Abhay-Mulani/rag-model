from langchain_core.documents import Document as LCDocument


class BM25Retriever:
    def __init__(self, documents: list[LCDocument]) -> None:
        self.documents = documents
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise RuntimeError("BM25 retrieval requires rank-bm25.") from error
        self._engine = BM25Okapi(
            [document.page_content.lower().split() for document in documents]
        )

    def invoke(self, query: str, k: int = 8) -> list[LCDocument]:
        scores = self._engine.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [self.documents[index] for index in ranked[:k] if scores[index] > 0]
