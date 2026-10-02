"""
Data ingestion pipeline for Agentic AI eBook.
Extracts text from PDF -> splits into structured chunks -> generates embeddings -> stores in Pinecone / Chroma.
"""

import sys
import argparse
import time
from pathlib import Path

# Fix Windows console utf-8 printing
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import settings
from src.pdf_loader import PDFDocumentLoader
from src.embeddings import get_embedding_provider
from src.vectorstore import get_vector_store

def run_ingestion(pdf_path: str = None, force: bool = False) -> dict:
    """
    Ingests the PDF into the configured vector store (Pinecone or ChromaDB).
    """
    pdf_path = pdf_path or settings.PDF_PATH
    print(f"\n{'='*60}")
    print(f"📖 Starting Agentic AI eBook Ingestion Pipeline")
    print(f"Target PDF: {pdf_path}")
    print(f"Vector DB: {settings.get_effective_vector_db().upper()}")
    print(f"Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Step 1: Load and chunk PDF
    print("[1/4] Parsing PDF & generating semantic chunks...")
    loader = PDFDocumentLoader(pdf_path=pdf_path)
    chunks = loader.load_and_chunk()
    print(f"✓ Extracted {len(chunks)} chunks across {chunks[-1]['page']} pages.")

    # Step 2: Initialize Embedding Provider
    print(f"\n[2/4] Initializing embeddings ({settings.EMBEDDING_PROVIDER})...")
    embedding_provider = get_embedding_provider()
    print(f"✓ Embedding model ready. Vector dimension: {embedding_provider.dimension}")

    # Step 3: Initialize Vector Store
    print(f"\n[3/4] Connecting to Vector Store...")
    vector_store = get_vector_store(embedding_provider=embedding_provider)
    
    current_count = vector_store.count()
    print(f"Current vectors in database: {current_count}")
    if current_count > 0 and not force:
        print(f"ℹ Database already contains {current_count} vectors. Use --force to re-index.")
        test_search = vector_store.similarity_search(
            embedding_provider.embed_query("What is Agentic AI?"),
            top_k=2
        )
        return {
            "status": "already_indexed",
            "vector_count": current_count,
            "chunks_count": len(chunks),
            "test_results": test_search
        }

    # Step 4: Generate Embeddings and Upsert
    print(f"\n[4/4] Generating embeddings & storing vectors in {settings.get_effective_vector_db().upper()}...")
    texts = [c["text"] for c in chunks]
    
    # Generate embeddings in batches
    batch_size = 32
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        emb_batch = embedding_provider.embed_documents(batch)
        embeddings.extend(emb_batch)
        print(f"  Embedded {len(embeddings)}/{total} chunks...")

    # Upsert to vector store
    upserted_count = vector_store.upsert_documents(chunks, embeddings)
    elapsed = round(time.time() - start_time, 2)

    print(f"\n🎉 Ingestion Completed Successfully in {elapsed}s!")
    print(f"✓ Total Chunks Processed: {len(chunks)}")
    print(f"✓ Vectors Stored: {upserted_count}")

    # Verification query
    print("\n🔍 Running verification search: 'What is Agentic AI?'...")
    test_query_vector = embedding_provider.embed_query("What is Agentic AI?")
    test_results = vector_store.similarity_search(test_query_vector, top_k=2)
    for i, res in enumerate(test_results, 1):
        print(f"  Top Match #{i} (Page {res.get('page')}, Score: {res.get('score')}):")
        snippet = res.get('text', '').replace('\n', ' ')[:120]
        print(f"    \"{snippet}...\"\n")

    return {
        "status": "success",
        "chunks_count": len(chunks),
        "vectors_stored": upserted_count,
        "elapsed_seconds": elapsed,
        "test_results": test_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Agentic AI eBook into Pinecone or ChromaDB")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if already populated")
    args = parser.parse_args()

    run_ingestion(pdf_path=args.pdf, force=args.force)
