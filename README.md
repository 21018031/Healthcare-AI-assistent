# Healthcare AI Assistant

Prototype RAG API for healthcare clients who need AI assistants that answer only from
internal clinical, operational, or compliance documents.

## Objective

Build a small healthcare-focused AI assistant that answers user questions from a given
set of healthcare documents using a Retrieval-Augmented Generation pipeline.

This project demonstrates:

- RAG pipeline design
- LLM integration
- Prompt engineering for grounded answers
- Vector database based semantic search
- Basic agentic/tool-based workflow
- API development with FastAPI
- Docker-based deployment
- Clean coding and documentation

## Features

- Ingests `.txt`, `.md`, and `.pdf` documents from `data/`
- Stores embeddings in persistent ChromaDB under `vector_store/`
- Retrieves relevant context for a user question
- Generates grounded answers with an OpenAI chat model
- Returns source citations and retrieved chunks
- Refuses to answer when the information is not in the documents
- Exposes functionality through a FastAPI API
- Includes basic logging and error handling

## Architecture

```text
Healthcare documents
        |
        v
Document loader and XML parser
        |
        v
Chunking
        |
        v
OpenAI embeddings
        |
        v
ChromaDB vector store
        |
        v
Retriever
        |
        v
HealthcareAssistantAgent
        |
        v
Grounded LLM answer with citations
        |
        v
FastAPI response
```

## Workflow

1. Place healthcare documents in `data/`.
2. Call `/ingest`.
3. The app loads documents, chunks text, creates embeddings, and stores them in ChromaDB.
4. Call `/ask` with a user question.
5. The agent retrieves relevant chunks from the vector database.
6. The LLM receives only the retrieved context and a grounding prompt.
7. The API returns an answer, citations, and retrieved source chunks.

The assistant is instructed to avoid hallucination. If the retrieved documents do not
contain enough information, it should say that it does not have enough information in
the provided documents.

## Recommended Dataset

For this prototype, use **MedlinePlus Health Topics XML** as the main knowledge base.
It is public, healthcare-focused, and does not contain real patient data or PHI.

Download this file into `data/`:

https://medlineplus.gov/xml/mplus_topics_compressed_2026-06-06.zip

Then extract it in the same folder so you have:

```text
data/
  mplus_topics_2026-06-06.xml
```

The app supports `.txt`, `.md`, `.pdf`, and MedlinePlus `.xml` files. During ingestion,
each MedlinePlus health topic is treated as a separate document, and the topic title plus
MedlinePlus URL are used as the citation source.

Optional datasets:

- MedQuAD for QA evaluation: https://github.com/abachaa/MedQuAD
- WHO Fact Sheets for additional public-health content: https://www.who.int/news-room/fact-sheets

Avoid real patient notes, identifiable patient records, or confidential client documents.

## Setup

Python 3.11 is recommended on Windows because ChromaDB depends on a native package
called `chroma-hnswlib`. Python 3.12 on Windows may try to build that dependency from
source and fail with `Failed to build chroma-hnswlib`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_DIR=./vector_store
KNOWLEDGE_BASE_DIR=./data
LOG_LEVEL=INFO
```

## Run

Run the FastAPI backend server:

```powershell
uvicorn app.main:app --reload
```

Run the Streamlit user interface in another terminal:

```powershell
streamlit run app/ui.py
```

## API

Health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Ingest documents:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ingest
```

Ask a question:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"question":"When should a medication error be escalated?"}'
```

## Docker

```powershell
docker compose up --build
```

## Prompt Engineering

We have designed a system prompt that enforces strict medical grounding, factual citation, and safety limits:

```text
You are a healthcare knowledge-base assistant for internal clinical, operational,
and compliance documents.

Rules:
- Answer only with information supported by the provided context.
- Do not use outside medical knowledge.
- If the context does not contain the answer, say exactly: "I could not find this information in the provided documents."
- Cite sources inline using the provided chunk labels, for example [source:abc123].
- Be concise, factual, and keep responses clear and professional.
- Avoid providing direct medical diagnosis, prescribing treatments, or giving clinical advice. If asked, state clearly that you cannot provide medical advice and suggest consulting a healthcare professional.
```

## Agentic Routing Workflow

The assistant implements a custom router to direct queries:
1. **Appointment Scheduler Tool**: If the user's question is related to appointment scheduling, dates, slots, or bookings, the agent routes the query to a mock slot-checker tool `check_available_slots(department, date)`.
2. **Grounded Document Search (RAG)**: For general healthcare and knowledge queries, the agent queries ChromaDB, retrieves context, and uses the LLM to formulate a grounded response.

