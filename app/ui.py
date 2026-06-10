"""
Healthcare AI Assistant - Streamlit UI
A violet-and-white interface for the healthcare RAG assistant.
"""

import streamlit as st
import requests
import time
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background: #faf7ff;
    }

    .main-container {
        background: linear-gradient(180deg, #f5f0ff 0%, #ffffff 100%);
        padding: 32px;
        border-radius: 28px;
    }

    .headline {
        color: #4f2a7a;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }

    .subtitle {
        color: #6a4f9b;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 28px;
    }

    .card {
        background: #ffffff;
        border: 1px solid rgba(103, 46, 187, 0.16);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 18px 45px rgba(103, 46, 187, 0.08);
        margin-bottom: 20px;
    }

    .source-item {
        color: #5a3b8a;
        font-size: 0.95rem;
        margin: 4px 0;
    }

    .stButton>button {
        background: #5f3dc4;
        color: white;
        border: none;
    }

    .stTextInput>div>div>input {
        border-radius: 16px;
        border: 1px solid #d7c3ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"
if "xml_path" not in st.session_state:
    st.session_state.xml_path = "data/mplus_topics_2026-06-06.xml"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0

def check_backend_health(api_url: str) -> bool:
    try:
        
        response = requests.get(api_url, timeout=2)
        return response.status_code == 200
    except Exception as exc:
        logger.warning("Backend health check failed: %s", exc)
        return False


def ingest_documents(api_url: str, xml_path: str) -> Optional[Dict[str, Any]]:
    try:
        payload = {"xml_path": xml_path.strip()}
        response = requests.post(f"{api_url}/ingest", json=payload, timeout=120)
        return response.json() if response.status_code == 200 else None
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return None


def ask_question(api_url: str, question: str, top_k: int) -> Optional[Dict[str, Any]]:
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "chat_history": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
                if "content" in msg
            ],
        }
        response = requests.post(f"{api_url}/ask", json=payload, timeout=30)
        return response.json() if response.status_code == 200 else None
    except Exception as exc:
        logger.error("Question failed: %s", exc)
        return None

with st.sidebar:
    st.markdown("## Assistant settings")
    api_url = st.text_input("Backend URL", key="api_url")
    st.session_state.top_k = st.slider("Top results", 1, 10, st.session_state.top_k)
    st.text_input("XML path to ingest", key="xml_path")
    st.markdown("---")


    try:
       response = requests.get("http://127.0.0.1:8000/health", timeout=5)
       st.write("DEBUG:", response.status_code, response.text)
       backend_active = response.status_code == 200
    except Exception as e:
        st.error(f"DEBUG ERROR: {e}")
        backend_active = False
    if backend_active:
        st.success("Backend is connected")
    else:
        st.error("Backend offline")
        st.write("Start the server with `uvicorn app.main:app --reload`.")

    if st.button("Refresh status", use_container_width=True):
        st.rerun()

    if backend_active and st.button("Index documents", use_container_width=True):
        result = ingest_documents(api_url, st.session_state.xml_path)
        if result:
            st.success(
                f"Indexed {result.get('chunks_indexed', 0)} chunks from {result.get('documents_loaded', 0)} documents."
            )
        else:
            st.error("Document ingestion failed.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.last_latency = 0.0
        st.rerun()

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='headline'>Healthcare AI Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Ask healthcare questions and get grounded responses from your documents.</div>",
    unsafe_allow_html=True,
)

if not backend_active:
    st.warning("Backend is not available. Please start the API server and refresh.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

st.markdown("<div class='card'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {msg['content']}")
    if msg.get("meta") and msg["meta"].get("sources"):
        with st.expander("Sources", expanded=False):
            for src in msg["meta"]["sources"]:
                document = src.get("document", src.get("source", "Unknown"))
                chunk = src.get("chunk", src.get("chunk_id", ""))
                st.markdown(f"<div class='source-item'>• {document} {chunk}</div>", unsafe_allow_html=True)
    st.markdown("---")

question = st.text_input("Ask a question", placeholder="Example: What does the policy say about infection control?", key="user_question")
send = st.button("Send question")

if send and question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Fetching answer..."):
        start = time.time()
        response = ask_question(st.session_state.api_url, question, st.session_state.top_k)
        elapsed = time.time() - start
    if response:
        answer = response.get("answer", "No response returned.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": {"sources": response.get("sources", []), "latency": elapsed},
        })
        st.session_state.total_queries += 1
        st.session_state.last_latency = elapsed

        st.rerun()
    
    else:
        st.error("Failed to get a response from the backend.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div style='margin-top:24px; color:#5a3b8a;'>",
    unsafe_allow_html=True,
)
st.write(f"Total queries: {st.session_state.total_queries}")
st.write(f"Last response time: {st.session_state.last_latency:.2f}s")
st.markdown("</div>", unsafe_allow_html=True)
