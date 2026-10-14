from fastapi import FastAPI, HTTPException

from src.config import settings
from src.feedback import FeedbackStore
from src.generation import GeminiGenerator, GeminiUnavailableError, LocalGenerator
from src.models import FeedbackRequest, QueryRequest, QueryResponse
from src.graph.workflow import FinanceWorkflow


def create_app() -> FastAPI:
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
            )
        generator = GeminiGenerator(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_max_retries,
            settings.gemini_retry_backoff,
        )
    elif settings.llm_provider == "local":
        generator = LocalGenerator()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    workflow = FinanceWorkflow(generator=generator, settings=settings)
    feedback_store = FeedbackStore(settings.feedback_path)
    app = FastAPI(
        title="Agentic Customer Support RAG",
        version="0.1.0",
        description="Multi-agent customer-support assistant grounded in a local knowledge base.",
    )
    app.state.workflow = workflow
    app.state.feedback_store = feedback_store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            result = workflow.run(
                request.query, request.session_id, request.risk_appetite
            )
        except GeminiUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return QueryResponse(
            answer=result["final_answer"],
            sources=[
                source.get("source", "unknown") for source in result["sources"]
            ],
            retrieved_context=result["retrieved_docs"],
            agents_invoked=result["agents_invoked"],
            confidence=result["confidence"],
            session_id=result["session_id"],
            intent=result["intent"],
            analysis=result["analysis_output"],
            risk_assessment=result["risk_output"],
        )

    @app.post("/feedback", status_code=204)
    def feedback(request: FeedbackRequest) -> None:
        feedback_store.save(request)

    return app


app = create_app()
