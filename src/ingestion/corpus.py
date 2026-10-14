import json
from pathlib import Path

from langchain_core.documents import Document


def load_corpus(path: str = "data/knowledge_base.json") -> list[Document]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(
            page_content=str(record["content"]),
            metadata={
                "id": str(record["id"]),
                "title": str(record["title"]),
                "category": str(record.get("category", "general")),
                "source": str(record.get("source", record["id"])),
            },
        )
        for record in records
    ]
