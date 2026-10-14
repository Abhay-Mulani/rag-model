from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "local")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    max_validation_retries: int = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))
    gemini_max_retries: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    gemini_retry_backoff: float = float(os.getenv("GEMINI_RETRY_BACKOFF", "2"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/processed/faiss_index")
    semantic_search_k: int = int(os.getenv("SEMANTIC_SEARCH_K", "8"))
    bm25_search_k: int = int(os.getenv("BM25_SEARCH_K", "8"))
    semantic_weight: float = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.6"))
    bm25_weight: float = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./finance_rag.db")
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "data/raw")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    enable_semantic_retrieval: bool = os.getenv(
        "ENABLE_SEMANTIC_RETRIEVAL", "true"
    ).lower() == "true"
    data_path: str = os.getenv("DATA_PATH", "data/knowledge_base.json")
    feedback_path: str = os.getenv("FEEDBACK_PATH", "data/feedback.jsonl")


settings = Settings()
