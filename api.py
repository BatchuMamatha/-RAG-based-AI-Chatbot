"""
FastAPI Server for Agentic AI eBook RAG Chatbot.
Exposes RESTful endpoints for chat, sample queries, ingestion, and system health,
and serves the modern Web UI.
"""

import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.rag_graph import get_rag_pipeline
from src.vectorstore import get_vector_store
from ingest import run_ingestion
import json

# App Initialization
app = FastAPI(
    title="Agentic AI eBook - RAG Chatbot API",
    description="Production-ready RAG pipeline powered by LangGraph, Pinecone / ChromaDB, and strict grounding verification.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class ChatRequest(BaseModel):
    question: str = Field(..., example="What are the core pillars of an Agentic AI system?")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=[], example=[])

class ContextChunk(BaseModel):
    chunk_id: Optional[str] = None
    text: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    relevance_score: Optional[float] = None
    source: Optional[str] = "Ebook-Agentic-AI.pdf"

class SourceItem(BaseModel):
    page: int
    chapter: str
    score: float
    chunk_id: str

class ChatResponse(BaseModel):
    question: str
    final_answer: str
    retrieved_chunks: List[ContextChunk]
    confidence_score: float
    confidence_level: str
    is_grounded: bool
    sources: List[SourceItem]

class IngestResponse(BaseModel):
    status: str
    vector_count: Optional[int] = None
    chunks_count: Optional[int] = None
    message: str

# API Endpoints
@app.get("/api/health")
def health_check():
    """Returns system status, active vector store, and indexed vector count."""
    try:
        store = get_vector_store()
        count = store.count()
        return {
            "status": "healthy",
            "vector_store": settings.get_effective_vector_db(),
            "indexed_vectors": count,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "llm_provider": settings.get_effective_llm_provider(),
            "pdf_source": settings.PDF_PATH
        }
    except Exception as e:
        return {
            "status": "warning",
            "error": str(e)
        }

@app.get("/api/sample-queries")
def get_sample_queries():
    """Returns curated sample questions based on the Agentic AI eBook."""
    sample_file = Path(settings.BASE_DIR) / "sample_queries.json"
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Executes the LangGraph RAG pipeline on the user's question.
    Returns:
    - final_answer (strictly grounded in eBook)
    - retrieved_chunks (with text, page numbers, chapters, relevance scores)
    - confidence_score (0.0 to 1.0) & level (HIGH, MEDIUM, LOW)
    """
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        pipeline = get_rag_pipeline()
        result = pipeline.run(
            question=payload.question.strip(),
            chat_history=payload.chat_history
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/api/ingest", response_model=IngestResponse)
def trigger_ingest(force: bool = False):
    """Triggers PDF ingestion and vector indexing."""
    try:
        res = run_ingestion(force=force)
        return IngestResponse(
            status=res["status"],
            vector_count=res.get("vectors_stored") or res.get("vector_count"),
            chunks_count=res.get("chunks_count"),
            message="Ingestion process completed successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# Mount static web directory
web_dir = Path(__file__).resolve().parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(web_dir / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=settings.HOST, port=settings.PORT, reload=False)
