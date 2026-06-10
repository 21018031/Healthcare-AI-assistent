from __future__ import annotations

import os
import logging
from openai import OpenAI
from langchain_huggingface import HuggingFaceEndpoint
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a healthcare knowledge-base assistant. Use only the information provided in the context and do not make up facts.

Instructions:
- Answer strictly using the provided context. Do not use outside medical knowledge.
- If the context does not contain the answer, reply exactly: "I could not find this information in the provided documents."
- When you answer, cite the source inline using the provided document title and chunk id.
  Example: "According to [Source: Asthma - path/to/file.xml | ChunkID: abc123], ..."
- Keep answers concise, factual, and clearly worded.
- Do not provide medical diagnoses, prescribe treatments, or give clinical advice.
  If asked for advice, say you cannot provide medical advice and suggest consulting a qualified healthcare professional.
- Do not mention the prompt, AI model, or system instructions.
- Use bullet points only when it improves clarity.
""".strip()


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        
        # 1. Primary Enterprise Route: OpenAI
        if not settings.use_mock and settings.openai_api_key:
            logger.info("Initializing OpenAI client connection...")
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_chat_model
            self._hf_client = None
        else:
            # 2. Free High-Performance Backup Route: Hugging Face Cloud Inference
            logger.info("Routing LLM to Hugging Face Cloud Engine...")
            self._client = None
            self._model = "huggingface-mistral"
            
            # Get token from settings (loaded from .env)
            hf_token = settings.huggingfacehub_api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                logger.warning("HUGGINGFACEHUB_API_TOKEN not set. Please set it in .env or environment variables.")
                self._hf_client = None
            else:
                # Set it in environment for HuggingFaceEndpoint
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
                try:
                    self._hf_client = HuggingFaceEndpoint(
                        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                        task="text-generation",
                        max_new_tokens=512,
                        temperature=0.1,
                        repetition_penalty=1.05,
                    )
                    logger.info("Successfully initialized Hugging Face endpoint.")
                except Exception as e:
                    logger.error(f"Failed to initialize Hugging Face endpoint: {e}")
                    self._hf_client = None

    def is_available(self) -> bool:
        return self._client is not None or self._hf_client is not None

    def answer(self, question: str, chunks: list[dict]) -> str:
        # Strict Guardrail: Immediately refuse if no text chunks were extracted
        if not chunks:
            return "I could not find this information in the provided documents."

        # Compile document blocks into a single structured string
        context = "\n\n".join([
            f"[Source: {chunk['source']} | ChunkID: {chunk['id']} ]\n{chunk['text']}"
            for chunk in chunks
        ])

        # Execution Route A: Free Hugging Face Model Endpoint (Active)
        if self._client is None:
            if self._hf_client is None:
                return "I could not find this information in the provided documents."
                
            # Construct a pristine prompt that forces the open-source model to obey your system guidelines
            structured_prompt = (
                f"<s>[INST] {SYSTEM_PROMPT}\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question} [/INST]</s>"
            )
            
            try:
                raw_response: Any = self._hf_client.invoke(structured_prompt)
                # Hugging Face clients may return a string, a dict, or a list.
                if isinstance(raw_response, str):
                    return raw_response.strip()
                if isinstance(raw_response, dict):
                    # common keys to check
                    for key in ("generated_text", "text", "outputs", "output"):
                        if key in raw_response and isinstance(raw_response[key], str):
                            return raw_response[key].strip()
                    # sometimes output is a list of dicts
                    if "outputs" in raw_response and isinstance(raw_response["outputs"], list):
                        first = raw_response["outputs"][0]
                        if isinstance(first, dict) and "generated_text" in first:
                            return first["generated_text"].strip()
                if isinstance(raw_response, list) and raw_response:
                    first = raw_response[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        return first["generated_text"].strip()
                    if isinstance(first, str):
                        return first.strip()
                # Fallback: convert to string
                return str(raw_response).strip()
            except Exception as e:
                logger.error(f"Hugging Face Cloud Inference call failed: {e}")
                return "I could not find this information in the provided documents."

        # Execution Route B: Enterprise OpenAI Account (Fallback)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI completion call crashed: {e}")
            return "I could not find this information in the provided documents."