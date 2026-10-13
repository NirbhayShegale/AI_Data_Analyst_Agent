
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=4)
def get_llm(api_key: str, model: str) -> ChatOpenAI:
    """Return a reusable LLM client."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        max_tokens=1000,
    )
