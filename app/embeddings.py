from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings
from app.rag import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        Path(settings.vector_db_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
        
        # Determine whether to use cloud enterprise OpenAI or local serverless execution
        if not settings.use_mock and settings.openai_api_key:
            logger.info("Initializing vector library using OpenAI embedding models...")
            from openai import OpenAI
            
            class OpenAiEmbeddingFunction:
                def __init__(self, api_key: str, model: str) -> None:
                    self._client = OpenAI(api_key=api_key)
                    self._model = model
                    
                def __call__(self, input: list[str]) -> list[list[float]]:
                    response = self._client.embeddings.create(model=self._model, input=input)
                    return [item.embedding for item in response.data]
            
            embedding_fn = OpenAiEmbeddingFunction(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
        else:
            logger.info("Using local ONNX MiniLM embedding model...")
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
            embedding_fn = ONNXMiniLM_L6_V2()

        self._collection: Collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": str(chunk.source), "chunk_id": chunk.id} for chunk in chunks],
        )
        logger.info(f"Upserted {len(chunks)} chunks into collection")
        return len(chunks)

    def query(self, question: str, top_k: int) -> list[dict]:
        if self._collection.count() == 0:
            logger.warning("Vector store is empty. No documents ingested yet.")
            return []

        result = self._collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "chunk_id": str(metadata.get("chunk_id", "unknown")),
                "source": str(metadata.get("source", "unknown")),
                "text": text,
                "distance": distance,
            }
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]