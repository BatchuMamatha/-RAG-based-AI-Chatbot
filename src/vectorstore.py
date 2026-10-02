"""
Unified Vector Store interface and implementations for Pinecone and ChromaDB.
Handles index creation, batch upserting, similarity search, and score normalization.
"""

from typing import List, Dict, Any, Optional
import time
import math
from src.config import settings
from src.embeddings import BaseEmbeddingProvider, get_embedding_provider

class BaseVectorStore:
    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        raise NotImplementedError

    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def is_indexed(self) -> bool:
        return self.count() > 0

class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone cloud vector database implementation using Pinecone v7 SDK.
    """
    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.api_key = settings.PINECONE_API_KEY
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is not set. Please provide it in your .env file.")
        
        from pinecone import Pinecone, ServerlessSpec
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.cloud = settings.PINECONE_CLOUD
        self.region = settings.PINECONE_REGION
        self.dimension = self.embedding_provider.dimension
        self._ensure_index()

    def _ensure_index(self):
        from pinecone import ServerlessSpec
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"[Pinecone] Creating index '{self.index_name}' (dim={self.dimension}, metric=cosine)...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            # Wait until ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print(f"[Pinecone] Index '{self.index_name}' is ready.")
        self.index = self.pc.Index(self.index_name)

    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 50) -> int:
        total = len(documents)
        upserted = 0
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            
            vectors = []
            for doc, emb in zip(batch_docs, batch_embs):
                vectors.append({
                    "id": doc["id"],
                    "values": emb,
                    "metadata": {
                        "text": doc["text"],
                        "page": doc.get("page", 1),
                        "chapter": doc.get("chapter", ""),
                        "source": doc.get("source", "Ebook-Agentic-AI.pdf"),
                        "char_count": doc.get("char_count", len(doc["text"]))
                    }
                })
            self.index.upsert(vectors=vectors)
            upserted += len(vectors)
            print(f"[Pinecone] Upserted {upserted}/{total} chunks.")
        return upserted

    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        top_k = top_k or settings.TOP_K
        res = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        results = []
        for match in res.get("matches", []):
            score = float(match.get("score", 0.0))
            # Pinecone cosine similarity ranges roughly between 0.0 and 1.0 for normalized vectors
            metadata = match.get("metadata", {})
            results.append({
                "id": match.get("id"),
                "score": round(max(0.0, min(1.0, score)), 4),
                "text": metadata.get("text", ""),
                "page": metadata.get("page", 1),
                "chapter": metadata.get("chapter", ""),
                "source": metadata.get("source", "Ebook-Agentic-AI.pdf")
            })
        return results

    def count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0

class ChromaVectorStore(BaseVectorStore):
    """
    Local ChromaDB vector database implementation.
    Persistent, fast, and does not require cloud credentials.
    """
    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        import chromadb
        from pathlib import Path
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "agentic_ai_ebook"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 100) -> int:
        total = len(documents)
        upserted = 0
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            
            ids = [d["id"] for d in batch_docs]
            texts = [d["text"] for d in batch_docs]
            metadatas = [{
                "page": int(d.get("page", 1)),
                "chapter": str(d.get("chapter", "")),
                "source": str(d.get("source", "Ebook-Agentic-AI.pdf")),
                "char_count": int(d.get("char_count", len(d["text"])))
            } for d in batch_docs]
            
            self.collection.upsert(
                ids=ids,
                embeddings=batch_embs,
                documents=texts,
                metadatas=metadatas
            )
            upserted += len(ids)
            print(f"[ChromaDB] Upserted {upserted}/{total} chunks.")
        return upserted

    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        top_k = top_k or settings.TOP_K
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        results = []
        if not res or not res["ids"] or not res["ids"][0]:
            return results

        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma cosine distance = 1 - cosine_similarity
            # Thus similarity = 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            results.append({
                "id": chunk_id,
                "score": round(similarity, 4),
                "text": doc_text,
                "page": meta.get("page", 1),
                "chapter": meta.get("chapter", ""),
                "source": meta.get("source", "Ebook-Agentic-AI.pdf")
            })
        return results

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

def get_vector_store(embedding_provider: Optional[BaseEmbeddingProvider] = None) -> BaseVectorStore:
    """
    Factory method to initialize either Pinecone or ChromaDB.
    """
    embedding_provider = embedding_provider or get_embedding_provider()
    db_type = settings.get_effective_vector_db()

    if db_type == "pinecone":
        try:
            print("[VectorStore] Initializing Pinecone Vector Store...")
            return PineconeVectorStore(embedding_provider=embedding_provider)
        except Exception as e:
            print(f"[VectorStore WARNING] Failed to connect to Pinecone ({e}). Falling back to local ChromaDB.")
            return ChromaVectorStore(embedding_provider=embedding_provider)
    else:
        print("[VectorStore] Initializing local ChromaDB Vector Store...")
        return ChromaVectorStore(embedding_provider=embedding_provider)
