"""
LLM factory supporting OpenAI, Google Gemini, Groq, and a deterministic local fallback.
Enforces low temperature (0.0) for strict factuality and groundedness.
"""

from typing import List, Dict, Any, Optional
import re
from src.config import settings

class LocalExtractiveLLM:
    """
    Deterministic grounded response generator used when no external LLM API key
    is configured. Directly synthesizes answers from retrieved chunks without hallucination.
    """
    def invoke(self, prompt: Any) -> Any:
        if isinstance(prompt, list):
            prompt_str = "\n".join([m.content if hasattr(m, 'content') else str(m) for m in prompt])
        elif hasattr(prompt, 'content'):
            prompt_str = prompt.content
        else:
            prompt_str = str(prompt)

        # Parse context block and question block from prompt
        context = ""
        question = ""
        
        ctx_match = re.search(r"Context:\s*([\s\S]+?)\s*(?:Question:|$)", prompt_str, re.IGNORECASE)
        if ctx_match:
            context = ctx_match.group(1).strip()
            
        q_match = re.search(r"Question:\s*([^\n\r]+)", prompt_str, re.IGNORECASE)
        if q_match:
            question = q_match.group(1).strip()
            
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        if not context or "No relevant context found" in context:
            return MockMessage(
                "I am sorry, but the provided Agentic AI eBook does not contain sufficient information to answer this question."
            )
            
        # Clean excerpt headers from context for synthesis
        cleaned_context = re.sub(r'\[Excerpt \d+ \| Page \d+ \| Section: [^\]]+\]', '', context)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_context) if len(s.strip()) > 25]
        
        # Keyword match scoring
        q_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', question.lower()))
        scored_sentences = []
        for s in sentences:
            s_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', s.lower()))
            overlap = len(q_words.intersection(s_words))
            scored_sentences.append((overlap, s))
            
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored_sentences[:5] if score > 0]
        
        if not top_sentences:
            top_sentences = sentences[:3]
            
        # Deduplicate while preserving order
        seen_s = set()
        deduped = []
        for s in top_sentences:
            cleaned_s = re.sub(r'\s+', ' ', s).strip()
            if cleaned_s not in seen_s:
                seen_s.add(cleaned_s)
                deduped.append(cleaned_s)

        answer_text = "Based on the Agentic AI eBook:\n\n" + " ".join(deduped)
        return MockMessage(answer_text)

def get_llm():
    """
    Returns an initialized LLM based on configuration and available API keys.
    """
    provider = settings.get_effective_llm_provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.0,
            google_api_key=settings.GEMINI_API_KEY
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0.0,
            api_key=settings.GROQ_API_KEY
        )
    else:
        print("[LLM] No API key detected (OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY). Using Local Extractive Engine.")
        return LocalExtractiveLLM()
