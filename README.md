# AlphaFin Financial Market Intelligence

This capstone implements a financial-market intelligence assistant inspired by
the reference architecture: a LangGraph multi-agent workflow, hybrid
retrieval, Gemini generation, FastAPI, Streamlit, Docker, and GitHub Actions.
The specialist agents are Planner, Retriever, Analysis, Portfolio, Risk, and
Validator.

## Local quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

Open `http://localhost:8000/docs` for Swagger UI. In another terminal, run:

```powershell
streamlit run ui/main.py
```

The curated demo financial research knowledge base is in
`data/knowledge_base.json`, and the expanded stock snapshot is in
`data/raw/stock_data.csv`. These are static educational/demo inputs, not live
market data; refresh them from a licensed or public source before production
use. For the LLM-backed path, set
`LLM_PROVIDER=gemini` and provide a Google AI Studio API key in
`GEMINI_API_KEY`. The model can be changed with `GEMINI_MODEL`.
The local provider remains available for offline tests and CI.

## API

`POST /query` accepts a question, optional `session_id`, and risk appetite,
returning the answer, intent, source references, retrieved context, confidence,
specialist risk assessment, and agent trace.
`POST /feedback` stores thumbs-up/down feedback and optional corrections in
`data/feedback.jsonl`. `GET /health` is used by deployment checks.

## Docker

```powershell
docker compose up --build
```

The API is available on port 8000 and the Streamlit UI on port 8501.
The Streamlit UI waits up to 300 seconds for a query by default. Set
`API_TIMEOUT` to override this when running the UI locally or in Docker.

## CI/CD

The workflow in `.github/workflows/ci.yml` runs Ruff, Pytest, and a Docker
build on pushes and pull requests. A push to `main` publishes the image to
GitHub Container Registry. Configure deployment infrastructure separately if a
hosted endpoint is required.

## External items needed

The local provider requires no external service. The Gemini provider requires
a Google AI Studio API key and network access. To publish or deploy
the image, the repository must exist on GitHub and the target hosting
provider's credentials and endpoint are needed.
