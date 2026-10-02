"""
Streamlit Web UI for Agentic AI eBook RAG Chatbot.
Run with: streamlit run streamlit_app.py
"""

import sys
import streamlit as st
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import settings
from src.rag_graph import get_rag_pipeline
from src.vectorstore import get_vector_store

st.set_page_config(
    page_title="Agentic AI eBook - RAG Chatbot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .chunk-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #6366f1;
    }
    .badge-high {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: #854d0e;
        color: #facc15;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=70)
    st.title("Knowledge Base")
    st.info("📖 **Target Document:** `Ebook-Agentic-AI.pdf`\n60 pages covering Autonomous Systems, Multi-Agent Orchestration, and Readiness.")
    
    st.divider()
    st.subheader("⚙️ System Status")
    try:
        store = get_vector_store()
        vec_count = store.count()
        st.success(f"**Vector Store:** {settings.get_effective_vector_db().upper()}")
        st.write(f"📊 **Vectors Indexed:** `{vec_count}` chunks")
    except Exception as e:
        st.error(f"Vector Store Error: {e}")

    st.write(f"🧠 **Embedding Model:** `{settings.EMBEDDING_PROVIDER}`")
    st.write(f"🤖 **LLM Engine:** `{settings.get_effective_llm_provider()}`")

    st.divider()
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "What is Agentic AI and how does it differ from traditional LLMs?",
        "What are the core pillars of an Agentic AI system from perception to execution?",
        "What is a Multi-Agent System (MAS) and how does it operate in a supply chain crisis?",
        "What are the key challenges and mitigation strategies of Multi-Agent Systems?",
        "What are the essential steps an organization should take to assess readiness?",
        "How is Agentic AI practically applied in manufacturing and retail?"
    ]

    selected_sample = None
    for q in sample_queries:
        if st.button(f"👉 {q[:42]}...", key=q, help=q):
            selected_sample = q

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main View
st.markdown('<div class="main-header">📖 Agentic AI eBook RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph Pipeline • Strict Grounding • Verification & Confidence Scoring</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander(f"📚 Retrieved Context Chunks ({len(msg['chunks'])}) | Confidence: {msg.get('confidence', 'N/A')}"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk #{i} — Page {c.get('page')} | Relevance Score: `{c.get('relevance_score')}`**")
                    st.caption(f"Section: {c.get('chapter')}")
                    st.text(c.get("text"))
                    st.divider()

# Handle Prompt Input
user_query = st.chat_input("Ask a question about the Agentic AI eBook...")
if selected_sample:
    user_query = selected_sample

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Invoke LangGraph RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant excerpts & verifying groundedness..."):
            pipeline = get_rag_pipeline()
            result = pipeline.run(user_query)

            answer = result["final_answer"]
            chunks = result["retrieved_chunks"]
            conf = result["confidence_score"]
            level = result["confidence_level"]
            is_grounded = result["is_grounded"]

            # Display confidence badge
            badge_class = "badge-high" if level == "HIGH" else ("badge-med" if level == "MEDIUM" else "badge-low")
            badge_html = f'<span class="{badge_class}">Confidence: {conf} ({level})</span>'
            grounded_html = '<span style="color:#10b981; margin-left:10px;">🛡️ Strictly Grounded</span>' if is_grounded else '<span style="color:#ef4444; margin-left:10px;">⚠️ Unverified</span>'
            
            st.markdown(f"{badge_html} {grounded_html}", unsafe_allow_html=True)
            st.markdown(answer)

            # Retrieved Context Expandable Drawer
            if chunks:
                with st.expander(f"📚 Retrieved Context Chunks ({len(chunks)})"):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Chunk #{i} — Page {c.get('page')} | Relevance Score: `{c.get('relevance_score')}`**")
                        st.caption(f"Chapter: {c.get('chapter')}")
                        st.text(c.get("text"))
                        st.divider()

            # Record in session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "confidence": f"{conf} ({level})",
                "chunks": chunks
            })
