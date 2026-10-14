from pathlib import Path

from langchain_core.documents import Document as LCDocument


class SemanticRetriever:
    def __init__(self, documents: list[LCDocument], index_path: str, model_name: str) -> None:
        self.documents = documents
        self.index_path = Path(index_path)
        self.model_name = model_name
        self._store = None

    def _ensure_store(self):
        if self._store is not None:
            return self._store
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
        except ImportError as error:
            raise RuntimeError(
                "Semantic retrieval requires langchain-community, faiss-cpu, "
                "and sentence-transformers."
            ) from error
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        if self.index_path.exists():
            self._store = FAISS.load_local(
                str(self.index_path), embeddings, allow_dangerous_deserialization=True
            )
        else:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._store = FAISS.from_documents(self.documents, embeddings)
            self._store.save_local(str(self.index_path))
        return self._store

    def invoke(self, query: str, k: int = 8) -> list[LCDocument]:
        return self._ensure_store().max_marginal_relevance_search(query, k=k)
