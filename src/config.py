"""
Configuration management for RAG-based AI Chatbot.
Loads settings from environment variables or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

class Settings:
    BASE_DIR: Path = BASE_DIR
    # Knowledge Base
    PDF_PATH: str = os.getenv("PDF_PATH", str(BASE_DIR / "Ebook-Agentic-AI.pdf"))
    
    # Vector DB settings
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "auto").lower()  # 'pinecone', 'chroma', or 'auto'
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

    # Embedding settings
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()  # 'huggingface', 'openai', 'gemini'
    HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto").lower()  # 'openai', 'gemini', 'groq', 'auto'
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Chunking & Retrieval Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
    
    # API Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    @classmethod
    def get_effective_vector_db(cls) -> str:
        if cls.VECTOR_DB_TYPE in ("pinecone", "chroma"):
            return cls.VECTOR_DB_TYPE
        # Auto mode: if Pinecone API key is provided and not empty, use pinecone, else chroma
        if cls.PINECONE_API_KEY and len(cls.PINECONE_API_KEY.strip()) > 5:
            return "pinecone"
        return "chroma"

    @classmethod
    def get_effective_llm_provider(cls) -> str:
        if cls.LLM_PROVIDER in ("openai", "gemini", "groq"):
            return cls.LLM_PROVIDER
        if cls.OPENAI_API_KEY and len(cls.OPENAI_API_KEY.strip()) > 5:
            return "openai"
        if cls.GEMINI_API_KEY and len(cls.GEMINI_API_KEY.strip()) > 5:
            return "gemini"
        if cls.GROQ_API_KEY and len(cls.GROQ_API_KEY.strip()) > 5:
            return "groq"
        return "local_fallback"

settings = Settings()
