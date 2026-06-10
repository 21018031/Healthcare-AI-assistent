from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from app.config import Settings
from app.llm import LlmClient
from app.rag import Chunk, Document, chunk_documents, clean_text, load_documents, read_documents_from_file

logger = logging.getLogger(__name__)


class SimpleEmbeddingFunction:
    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> list[float]:
        counts = [0.0] * self.dimension
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % self.dimension
            counts[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in counts))
        if norm > 0:
            counts = [value / norm for value in counts]
        return counts


class HealthcareAssistantAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.collection_name
        self.use_mock = settings.use_mock
        self.persist_dir = Path(settings.vector_db_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_fn = SimpleEmbeddingFunction()
        self.vector_store = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.llm_client = LlmClient(settings) if not self.use_mock else None

    def ingest(
        self,
        directory: Path | None = None,
        xml_path: Path | None = None,
    ) -> Tuple[int, int]:
        if xml_path is not None:
            if not xml_path.exists() or not xml_path.is_file():
                raise FileNotFoundError(f"XML path not found: {xml_path}")
            documents = read_documents_from_file(xml_path)
        else:
            kb_dir = Path(directory) if directory is not None else Path(self.settings.knowledge_base_dir)
            documents = load_documents(kb_dir)

        if not documents:
            return 0, 0

        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        if self.vector_store.count() > 0:
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.vector_store = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        self.vector_store.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": chunk.source, "chunk_id": chunk.id} for chunk in chunks],
        )

        return len(documents), len(chunks)

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        steps_executed: list[dict[str, Any]] = []

        if self._is_booking_request(question):
            details = self._extract_booking_details(question)
            answer = self._mock_booking_answer(question, details)
            steps_executed.append({"action": "retrieve_healthcare_context", "query": question})
            steps_executed.append({"action": "check_available_slots", "query": question, "details": details})
            return {
                "answer": answer,
                "sources": [],
                "confidence": "high",
                "steps_executed": steps_executed,
            }

        if self.vector_store.count() == 0:
            logger.info("Vector store empty, ingesting default knowledge base before answering.")
            try:
                self.ingest()
            except Exception as exc:
                logger.warning("Automatic ingestion failed: %s", exc)

        def query_vector_store() -> tuple[list[str], list[dict[str, Any]], list[float]]:
            results = self.vector_store.query(
                query_texts=[question],
                n_results=top_k or self.settings.top_k,
                include=["documents", "metadatas", "distances"],
            )
            return (
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            )

        documents, metadatas, distances = query_vector_store()
        steps_executed.append({"action": "retrieve_healthcare_context", "query": question, "results": len(documents)})

        if (not documents or not any(documents) or not self._is_relevant(question, documents)) and not getattr(self, "_kb_refreshed", False):
            self._kb_refreshed = True
            logger.info("No relevant results, refreshing knowledge base and retrying query.")
            try:
                self.ingest()
            except Exception as exc:
                logger.warning("Retry ingestion failed: %s", exc)
            documents, metadatas, distances = query_vector_store()
            steps_executed.append({"action": "retrieve_healthcare_context", "query": question, "results": len(documents), "retry": True})

        if not documents or not any(documents) or not self._is_relevant(question, documents):
            return {
                "answer": "I could not find this information in the provided documents.",
                "sources": [],
                "confidence": "none",
                "steps_executed": steps_executed,
            }

        chunks = [
            {
                "id": metadata.get("chunk_id", "unknown"),
                "source": metadata.get("source", "unknown"),
                "text": doc,
            }
            for doc, metadata in zip(documents, metadatas)
        ]

        if self.llm_client is not None and self.llm_client.is_available():
            llm_answer = self.llm_client.answer(question, chunks)
            if llm_answer and llm_answer != "I could not find this information in the provided documents.":
                answer = llm_answer
            else:
                answer = self._build_answer_from_context(question, documents)
        else:
            answer = self._build_answer_from_context(question, documents)

        confidence = self._estimate_confidence(distances)
        if confidence == "none":
            return {
                "answer": "I could not find this information in the provided documents.",
                "sources": [],
                "confidence": "none",
                "steps_executed": steps_executed,
            }

        sources = [
            {
                "document": metadata.get("source", "unknown"),
                "chunk": metadata.get("chunk_id", "unknown"),
            }
            for metadata in metadatas
        ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "steps_executed": steps_executed,
        }

    def _is_booking_request(self, question: str) -> bool:
        return bool(re.search(r"\b(book|schedule|appointment|reserve|slot|visit)\b", question, re.IGNORECASE))

    def _extract_booking_details(self, question: str) -> dict[str, Any]:
        specialties = [
            "cardiology",
            "pediatrics",
            "neurology",
            "dermatology",
            "oncology",
            "orthopedics",
            "general medicine",
            "orthopedic",
            "pediatric",
        ]
        specialty = "General Medicine"
        for item in specialties:
            if item in question.lower():
                specialty = item.title()
                break
        # Parse weekdays and simple relative dates (today/tomorrow/next)
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        q = question.lower()
        day = None
        for wd in weekdays:
            if f"next {wd.lower()}" in q:
                day = wd
                break
        if not day:
            for wd in weekdays:
                if wd.lower() in q:
                    day = wd
                    break

        from datetime import datetime, timedelta

        if not day:
            if "tomorrow" in q:
                day = (datetime.now() + timedelta(days=1)).strftime("%A")
            elif "today" in q:
                day = datetime.now().strftime("%A")
            elif "this week" in q or "next week" in q:
                day = (datetime.now() + timedelta(days=2)).strftime("%A")
            else:
                day = (datetime.now() + timedelta(days=1)).strftime("%A")

        return {
            "specialty": specialty,
            "day": day,
        }

    def _mock_booking_answer(self, question: str, details: dict[str, Any]) -> str:
        # Provide two mock time slots and a friendly confirmation hint
        from datetime import datetime, timedelta
        try:
            target_day = details.get("day", "soon")
            today = datetime.now()
            weekday_map = {d: i for i, d in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}
            if target_day in weekday_map:
                days_ahead = (weekday_map[target_day] - today.weekday() + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
                date_str = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            else:
                date_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            date_str = "soon"

        slots = ["10:00 AM", "11:45 AM", "2:30 PM"]
        # choose two slots based on specialty hash so responses vary a bit
        specialty_hash = abs(hash(details['specialty']))
        slot1 = slots[specialty_hash % len(slots)]
        slot2 = slots[(specialty_hash + 1) % len(slots)]
        return (
            f"I found mock appointment availability for {details['specialty']} on {details['day']} ({date_str}) at {slot1} and {slot2}. "
            "This is a mock appointment suggestion; please contact the hospital scheduling desk to confirm availability."
        )

    def _build_answer_from_context(self, question: str, documents: list[str]) -> str:
        best_answer = documents[0].strip()
        if best_answer:
            return f"Based on the provided documents, {best_answer}"
        return "I could not find this information in the provided documents."

    def _is_relevant(self, question: str, documents: list[str]) -> bool:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "he",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "was",
            "were",
            "with",
            "what",
            "which",
            "who",
            "whom",
            "where",
            "when",
            "why",
            "how",
        }
        question_tokens = set(re.findall(r"\w+", question.lower())) - stopwords
        document_tokens = set(re.findall(r"\w+", " ".join(documents).lower())) - stopwords
        return bool(question_tokens & document_tokens)

    def _estimate_confidence(self, distances: list[float]) -> str:
        if not distances:
            return "none"
        best_distance = min(distances)
        if best_distance < 0.35:
            return "high"
        if best_distance < 0.6:
            return "medium"
        if best_distance < 0.95:
            return "low"
        return "none"
