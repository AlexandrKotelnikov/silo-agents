from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LiveSettings:
    qdrant_url: str
    qdrant_collection: str
    qdrant_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    embedding_base_url: str
    embedding_model: str
    retrieval_limit: int = 3

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env") -> "LiveSettings":
        load_dotenv(dotenv_path)
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "silo_records"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            llm_model=os.getenv("LLM_MODEL", "qwen3:4b-instruct"),
            llm_api_key=os.getenv("LLM_API_KEY", "ollama"),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
            retrieval_limit=int(os.getenv("RETRIEVAL_LIMIT", "3")),
        )
