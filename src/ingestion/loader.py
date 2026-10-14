import json
from pathlib import Path

from src.models import Document


def load_documents(path: str | Path) -> list[Document]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")
    records = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Knowledge base must contain a JSON list")
    return [
        Document(
            document_id=str(record["id"]),
            title=str(record["title"]),
            content=str(record["content"]),
            category=str(record.get("category", "general")),
            source=str(record.get("source", record["id"])),
        )
        for record in records
    ]
