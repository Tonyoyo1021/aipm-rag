import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parents[1]
DEFAULT_DOCUMENTS = (
    WORKSPACE_DIR / "AI产品经理能力全景指南.md",
    WORKSPACE_DIR / "秋招 AI 面试八股准备手册.md",
)


@dataclass(frozen=True)
class Settings:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    document_paths: tuple[Path, ...]
    index_dir: Path
    api_key: str | None
    base_url: str | None
    embedding_model: str
    chat_model: str
    request_timeout: int
    min_score: float


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    paths_value = os.getenv("RAG_DOCUMENT_PATHS")
    if paths_value:
        document_paths = tuple(
            (PROJECT_DIR / item.strip()).resolve()
            for item in paths_value.split(",")
            if item.strip()
        )
    else:
        document_paths = DEFAULT_DOCUMENTS

    settings = Settings(
        chunk_size=_env_int("RAG_CHUNK_SIZE", 800),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 120),
        top_k=_env_int("RAG_TOP_K", 4),
        document_paths=document_paths,
        index_dir=(PROJECT_DIR / os.getenv("RAG_INDEX_DIR", ".index")).resolve(),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "BAAI/bge-m3"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "XingChenAGI/Xing4.0-29B"),
        request_timeout=_env_int("RAG_REQUEST_TIMEOUT", 120),
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.45")),
    )
    if settings.chunk_size <= 0:
        raise ValueError("RAG_CHUNK_SIZE 必须大于 0")
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP 必须大于等于 0 且小于 RAG_CHUNK_SIZE")
    if settings.top_k <= 0:
        raise ValueError("RAG_TOP_K 必须大于 0")
    if settings.request_timeout <= 0:
        raise ValueError("RAG_REQUEST_TIMEOUT 必须大于 0")
    if not -1 <= settings.min_score <= 1:
        raise ValueError("RAG_MIN_SCORE 必须在 -1 到 1 之间")
    return settings
