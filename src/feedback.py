import json
from pathlib import Path
from threading import Lock

from src.models import FeedbackRequest


class FeedbackStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def save(self, feedback: FeedbackRequest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = feedback.model_dump()
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
