"""
LangGraph RAG pipeline for Agentic AI eBook.
Implements retrieval, relevance grading, grounded answer generation,
groundedness evaluation, and confidence scoring.
"""

from typing import TypedDict, List, Dict, Any, Optional
import math
import re
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.embeddings import get_embedding_provider
from src.vectorstore import get_vector_store
from src.llm import get_llm

class RAGState(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]
    query_vector: List[float]
    retrieved_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]
    answer: str
    confidence_score: float
    confidence_level: str
    is_grounded: bool
    sources: List[Dict[str, Any]]

class LangGraphRAGPipeline:
    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        self.vector_store = get_vector_store(self.embedding_provider)
        self.llm = get_llm()
        self.graph = self._build_graph()

    def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Embeds question and retrieves top matching chunks from vector store."""
        question = state["question"]
        query_vector = self.embedding_provider.embed_query(question)
        docs = self.vector_store.similarity_search(query_vector, top_k=settings.TOP_K)
        return {
            "query_vector": query_vector,
            "retrieved_docs": docs
        }

    def _grade_documents_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Grades retrieved chunks by relevance score and computes confidence score.
        Filters out low-relevance noise.
        """
        retrieved = state.get("retrieved_docs", [])
        threshold = settings.MIN_RELEVANCE_SCORE

        graded = [doc for doc in retrieved if doc.get("score", 0.0) >= threshold]

        # Calculate confidence score
        if not graded:
            confidence = 0.0
            level = "LOW"
        else:
            top_score = graded[0]["score"]
            avg_top3 = sum(d["score"] for d in graded[:3]) / min(len(graded), 3)
            # Weighted confidence: 60% top score, 40% top-3 average
            confidence = round(0.6 * top_score + 0.4 * avg_top3, 3)

            if confidence >= 0.70:
                level = "HIGH"
            elif confidence >= 0.45:
                level = "MEDIUM"
            else:
                level = "LOW"

        return {
            "graded_docs": graded,
            "confidence_score": confidence,
            "confidence_level": level
        }

    def _router_decision(self, state: RAGState) -> str:
        """Determines whether to generate answer or report insufficient knowledge."""
        graded = state.get("graded_docs", [])
        if not graded:
            return "handle_no_context"
        return "generate"

    def _handle_no_context_node(self, state: RAGState) -> Dict[str, Any]:
        """Handles questions where no sufficiently relevant context exists in the eBook."""
        return {
            "answer": "I am sorry, but the provided Agentic AI eBook does not contain sufficient information to answer this question.",
            "confidence_score": 0.0,
            "confidence_level": "LOW",
            "is_grounded": True,
            "sources": []
        }

    def _generate_node(self, state: RAGState) -> Dict[str, Any]:
        """Synthesizes answer strictly grounded in retrieved chunks."""
        question = state["question"]
        docs = state.get("graded_docs", [])

        # Build context string with citations
        context_parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.get("page", "N/A")
            chapter = doc.get("chapter", "General")
            text = doc.get("text", "").strip()
            context_parts.append(f"[Excerpt {i} | Page {page} | Section: {chapter}]\n{text}")

        context_block = "\n\n".join(context_parts)

        system_prompt = (
            "You are an expert AI assistant dedicated to answering questions strictly and accurately "
            "based ONLY on the provided excerpts from the 'Agentic AI' eBook.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer strictly using ONLY the information given in the Context excerpts below.\n"
            "2. Do NOT extrapolate, invent facts, or rely on outside world knowledge not found in the context.\n"
            "3. If the context does not explicitly provide the answer, say: "
            "'I am sorry, but the provided Agentic AI eBook does not contain sufficient information to answer this question.'\n"
            "4. Include page number references (e.g., [Page X]) in your explanation when mentioning specific facts.\n"
            "5. Maintain a professional, executive-ready, and concise tone."
        )

        user_content = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Strictly Grounded Answer:"
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]
            response = self.llm.invoke(messages)
            raw_answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            # Fallback if external LLM encounters API error
            print(f"[LLM Error] {e}. Falling back to deterministic local extractive generator.")
            from src.llm import LocalExtractiveLLM
            fallback = LocalExtractiveLLM()
            response = fallback.invoke(user_content)
            raw_answer = response.content

        return {"answer": raw_answer.strip()}

    def _groundedness_checker_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Validates whether the generated response is grounded in the retrieved chunks
        and flags potential unsupported claims.
        """
        answer = state.get("answer", "")
        graded_docs = state.get("graded_docs", [])
        
        # If the answer already acknowledges no context, it is grounded
        if "does not contain sufficient information" in answer.lower():
            return {"is_grounded": True}

        # Combine text of all context chunks
        combined_context = " ".join([d.get("text", "") for d in graded_docs]).lower()
        
        # Extract meaningful terms from answer (words > 4 chars, excluding common stopwords)
        stopwords = {
            "about", "above", "after", "again", "against", "all", "and", "any", "are", 
            "because", "been", "before", "being", "below", "between", "both", "but", 
            "cannot", "could", "did", "does", "doing", "down", "during", "each", "few", 
            "for", "from", "further", "had", "has", "have", "having", "her", "here", 
            "hers", "herself", "him", "himself", "his", "how", "into", "its", "itself", 
            "just", "more", "most", "other", "our", "ours", "ourselves", "out", "over", 
            "own", "same", "she", "should", "some", "such", "than", "that", "the", 
            "their", "theirs", "them", "themselves", "then", "there", "these", "they", 
            "this", "those", "through", "too", "under", "until", "very", "was", "were", 
            "what", "when", "where", "which", "while", "who", "whom", "why", "with", 
            "would", "agentic", "ebook", "based", "according"
        }
        
        answer_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', answer) if w.lower() not in stopwords]
        
        if not answer_words:
            return {"is_grounded": True}

        matched_words = [w for w in answer_words if w in combined_context]
        groundedness_ratio = len(matched_words) / len(answer_words)

        is_grounded = groundedness_ratio >= 0.55
        
        # Adjust confidence if groundedness is weak
        confidence = state.get("confidence_score", 0.5)
        if not is_grounded:
            confidence = round(confidence * 0.7, 3)

        return {
            "is_grounded": is_grounded,
            "confidence_score": confidence
        }

    def _format_response_node(self, state: RAGState) -> Dict[str, Any]:
        """Formats sources, citations, and metadata."""
        docs = state.get("graded_docs", []) or state.get("retrieved_docs", [])
        sources = []
        seen = set()

        for doc in docs:
            page = doc.get("page", 1)
            key = (page, doc.get("chapter", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "page": page,
                    "chapter": doc.get("chapter", "Agentic AI"),
                    "score": doc.get("score", 0.0),
                    "chunk_id": doc.get("id", "")
                })

        return {"sources": sources}

    def _build_graph(self):
        """Builds and compiles the LangGraph StateGraph."""
        builder = StateGraph(RAGState)

        # Add Nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade_documents", self._grade_documents_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("handle_no_context", self._handle_no_context_node)
        builder.add_node("groundedness_checker", self._groundedness_checker_node)
        builder.add_node("format_response", self._format_response_node)

        # Add Edges
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade_documents")

        # Conditional Edge after grading
        builder.add_conditional_edges(
            "grade_documents",
            self._router_decision,
            {
                "generate": "generate",
                "handle_no_context": "handle_no_context"
            }
        )

        builder.add_edge("generate", "groundedness_checker")
        builder.add_edge("groundedness_checker", "format_response")
        builder.add_edge("handle_no_context", "format_response")
        builder.add_edge("format_response", END)

        return builder.compile()

    def run(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Executes the LangGraph pipeline for a given user query.
        Returns final answer, retrieved context chunks, and confidence score.
        """
        initial_state: RAGState = {
            "question": question,
            "chat_history": chat_history or [],
            "query_vector": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "answer": "",
            "confidence_score": 0.0,
            "confidence_level": "LOW",
            "is_grounded": False,
            "sources": []
        }

        final_state = self.graph.invoke(initial_state)

        # Format retrieved context chunks for user return
        retrieved_chunks_output = [
            {
                "chunk_id": doc.get("id"),
                "text": doc.get("text"),
                "page": doc.get("page"),
                "chapter": doc.get("chapter"),
                "relevance_score": doc.get("score"),
                "source": doc.get("source", "Ebook-Agentic-AI.pdf")
            }
            for doc in final_state.get("graded_docs", []) or final_state.get("retrieved_docs", [])
        ]

        return {
            "final_answer": final_state.get("answer", ""),
            "retrieved_chunks": retrieved_chunks_output,
            "confidence_score": final_state.get("confidence_score", 0.0),
            "confidence_level": final_state.get("confidence_level", "LOW"),
            "is_grounded": final_state.get("is_grounded", True),
            "sources": final_state.get("sources", []),
            "question": question
        }

# Singleton instance
_rag_pipeline_instance: Optional[LangGraphRAGPipeline] = None

def get_rag_pipeline() -> LangGraphRAGPipeline:
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = LangGraphRAGPipeline()
    return _rag_pipeline_instance
