"""Shared application settings loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# START: Resolve a configured path relative to the project root when needed.
def _project_path(environment_name: str, default: str) -> str:
    configured_path = Path(os.getenv(environment_name, default))
    if configured_path.is_absolute():
        return str(configured_path)
    return str(PROJECT_ROOT / configured_path)
# END: Resolve a configured path relative to the project root when needed.

DATA_DIR = _project_path("DATA_DIR", "data")
VECTORSTORE_DIR = _project_path("VECTORSTORE_DIR", "vectorstore")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
