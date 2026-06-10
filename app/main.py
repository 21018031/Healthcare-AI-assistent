from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app.agent import HealthcareAssistantAgent
from app.config import get_settings


class IngestRequest(BaseModel):
    directory: str | None = Field(default=None)
    xml_path: str | None = Field(default=None)


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    collection: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=10)
    chat_history: list[dict] | None = Field(default=None)


settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Knowledge Base Assistant",
    version="0.1.0",
    description="Grounded RAG API for healthcare internal documents using Hugging Face Cloud Inference.",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("request started method=%s path=%s", request.method, request.url.path)
    response = await call_next(request)
    logger.info(
        "request completed method=%s path=%s status=%s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def get_agent() -> HealthcareAssistantAgent:
    try:
        return HealthcareAssistantAgent(settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Healthcare Knowledge Base Assistant API",
        "collection": settings.collection_name,
    }


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "collection": settings.collection_name}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest | None = None) -> IngestResponse:
    agent = get_agent()

    directory_path = Path(request.directory) if request and request.directory else None
    xml_path = Path(request.xml_path) if request and request.xml_path else None
    if directory_path is None and xml_path is None:
        directory_path = settings.knowledge_base_dir

    try:
        documents_loaded, chunks_indexed = agent.ingest(
            directory=directory_path,
            xml_path=xml_path,
        )
        logger.info(f"Ingested {documents_loaded} documents and indexed {chunks_indexed} chunks")
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error(f"Ingestion error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        documents_loaded=documents_loaded,
        chunks_indexed=chunks_indexed,
        collection=agent.collection_name,
    )


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    agent = get_agent()
    try:
        return agent.answer_question(
            question=request.question,
            top_k=request.top_k or settings.top_k,
            chat_history=request.chat_history,
        )
    except Exception as exc:
        logger.exception("Error processing question")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def stats() -> dict:
    try:
        agent = get_agent()
        total_chunks = agent.vector_store.count()

        return {
            "total_chunks": total_chunks,
            "collection": agent.collection_name,
            "mock_mode": settings.use_mock,
            "knowledge_base_dir": str(settings.knowledge_base_dir),
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {"total_chunks": 0, "collection": "None", "error": str(e)}