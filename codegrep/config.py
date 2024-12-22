import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


IGNORE_PATHS = [
    "*.git",
    "*.venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
]
