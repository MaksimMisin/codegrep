import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.cwd() / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# llm-based search
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GPT4O_MINI_MODEL = os.getenv("GPT4O_MINI_MODEL", "gpt-4o-mini")


IGNORE_PATHS = [
    "*.git/",
    "*.venv/",
    "*__pycache__/*",
    "*.mypy_cache/*",
    "*.pytest_cache/*",
    "*.codegrep/*",
    "node_modules/*",
    "dist/*",
    "build/*",
    "target/*",
    "*.egg-info/*",
    "*.ipynb_checkpoints/*",
    ".claude/*",
    ".yarn/*",
    ".vscode/*",
    ".vim/*",
]

# File extensions that should be ignored (non-textual or irrelevant for code search)
IGNORE_EXTENSIONS = {
    # Binary and compiled files
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".pkl",
    ".pklz",
    # Image files
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".ico",
    ".svg",
    ".webp",
    # Audio/Video files
    ".mp3",
    ".wav",
    ".mp4",
    ".avi",
    ".mov",
    # Archive files
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    # Other binary formats
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".db",
    ".sqlite",
    ".sqlite3",
    # Cache and temporary files
    ".cache",
    ".log",
    ".tmp",
    ".temp",
}
