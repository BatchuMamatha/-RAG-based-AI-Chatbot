"""
Embedding factory supporting HuggingFace (local), OpenAI, and Google Gemini.
Produces normalized float vector embeddings.
"""

from typing import List, Optional
import os
from src.config import settings

class BaseEmbeddingProvider:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.HF_EMBEDDING_MODEL
        from sentence_transformers import SentenceTransformer
        # Disable oneDNN warning noise if desired
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        self.model = SentenceTransformer(self.model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return emb[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dim

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name or settings.OPENAI_EMBEDDING_MODEL
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self._dim = 1536

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dim

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_EMBEDDING_MODEL
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key
        )
        self._dim = 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dim

def get_embedding_provider() -> BaseEmbeddingProvider:
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbeddingProvider()
    elif provider == "gemini" and settings.GEMINI_API_KEY:
        return GeminiEmbeddingProvider()
    # Default to local HuggingFace for reliable offline/free execution
    return HuggingFaceEmbeddingProvider()
