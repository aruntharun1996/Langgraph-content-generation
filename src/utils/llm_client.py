import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@lru_cache(maxsize=None)
def get_generation_llm() -> ChatGoogleGenerativeAI:
    """Returns a cached Gemini LLM instance for content generation."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GENERATION_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.8,       # Higher creativity for generation
        max_output_tokens=2048,
    )


@lru_cache(maxsize=None)
def get_evaluation_llm() -> ChatGoogleGenerativeAI:
    """Returns a cached Gemini LLM instance for content evaluation."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("EVALUATION_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,       # Low temperature for consistent, objective scoring
        max_output_tokens=1024,
    )
