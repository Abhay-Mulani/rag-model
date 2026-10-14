import time

import requests


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini cannot serve a request after transient retries."""


class LocalGenerator:
    """Deterministic generator used for offline development and CI."""

    def generate(
        self, query: str, retrieved: list
    ) -> tuple[str, float]:
        if not retrieved:
            return (
                (
                    "I could not find relevant financial research. "
                    "Please verify the question and try again."
                ),
                0.15,
            )
        best = retrieved[0]
        if hasattr(best, "page_content"):
            content = best.page_content
            title = best.metadata.get("title", "Research")
        else:
            content = best.document.content
            title = best.document.title
        answer = (
            f"Based on {title}: {content} "
            "This is educational information, not financial advice."
        )
        confidence = min(
            0.95, max(0.35, 0.55 + (0.1 if len(retrieved) > 1 else 0))
        )
        return answer, confidence


class GeminiGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
        retry_backoff: float = 2,
    ) -> None:
        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        self.api_key = api_key
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0, retry_backoff)

    def generate(
        self, query: str, retrieved: list
    ) -> tuple[str, float]:
        if not retrieved:
            return (
                (
                    "I could not find relevant financial research. "
                    "Please verify the question and try again."
                ),
                0.15,
            )
        context = "\n\n".join(
            f"Source: {item.metadata.get('source', 'unknown')}\n{item.page_content}"
            for item in retrieved
        )
        prompt = (
            "You are a financial-market intelligence assistant. Answer only from "
            "the supplied context. If the context does not answer the question, "
            "say that the available evidence is insufficient. Cite the source "
            "name and include a brief educational disclaimer.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        payload = self._request(prompt)
        return self._parse_response(payload, 0.55)

    def _request(self, prompt: str) -> dict:
        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 180, "temperature": 0.2},
        }
        transient_statuses = {429, 500, 502, 503, 504}
        last_error = "unknown Gemini availability error"
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    params={"key": self.api_key},
                    json=request_body,
                    timeout=60,
                )
            except requests.RequestException as error:
                last_error = str(error)
                if attempt >= self.max_retries:
                    raise GeminiUnavailableError(
                        "Gemini could not be reached after retries. "
                        "Check your internet connection and try again."
                    ) from error
            else:
                if response.status_code < 400:
                    return response.json()
                last_error = response.text[:500]
                if response.status_code not in transient_statuses:
                    raise RuntimeError(
                        f"Gemini inference failed ({response.status_code}): "
                        f"{last_error}"
                    )
                if attempt >= self.max_retries:
                    raise GeminiUnavailableError(
                        f"Gemini is temporarily unavailable ({response.status_code}) "
                        f"after {self.max_retries + 1} attempts: {last_error}"
                    )
            time.sleep(self.retry_backoff * (2**attempt))
        raise GeminiUnavailableError(f"Gemini unavailable: {last_error}")

    @staticmethod
    def _parse_response(
        payload: dict, retrieval_score: float
    ) -> tuple[str, float]:
        try:
            answer = str(
                payload["candidates"][0]["content"]["parts"][0]["text"]
            ).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Gemini returned an unexpected response format"
            ) from error
        if not answer:
            raise RuntimeError("Gemini returned an empty answer")
        confidence = min(0.95, max(0.35, retrieval_score + 0.2))
        return answer, confidence
