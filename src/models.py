from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2_000)
    session_id: str | None = Field(default=None, max_length=100)
    risk_appetite: str = Field(default="moderate", pattern="^(conservative|moderate|aggressive)$")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_context: list[str]
    agents_invoked: list[str]
    confidence: float = Field(ge=0, le=1)
    session_id: str
    intent: str = "market_info"
    analysis: str = ""
    risk_assessment: str = ""


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2_000)
    rating: int = Field(ge=-1, le=1)
    correction: str | None = Field(default=None, max_length=2_000)
    session_id: str | None = Field(default=None, max_length=100)


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    content: str
    category: str
    source: str


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    keyword_score: float
    semantic_score: float


@dataclass
class WorkflowState:
    query: str
    session_id: str
    risk_appetite: str = "moderate"
    intent: str = "market_info"
    rewritten_query: str = ""
    retrieved: list[RetrievedDocument] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    validation_passed: bool = False
    retry_count: int = 0
    feedback: dict[str, Any] = field(default_factory=dict)
    agents_invoked: list[str] = field(default_factory=list)
    analysis: str = ""
    portfolio_guidance: str = ""
    risk_assessment: str = ""
