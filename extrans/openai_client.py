import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI


def get_openai_client(base_url: Optional[str] = None) -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    api_base = base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    if api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    return OpenAI(api_key=api_key)

