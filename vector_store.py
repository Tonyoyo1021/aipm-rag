import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import Settings
from documents import Chunk, load_and_split
from embeddings import Embedder

INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class LocalVectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunk 数量与向量数量不一致")
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding 返回了零向量")
        self.vectors = vectors / norms

    @classmethod
    def load(cls, index_dir: Path) -> "LocalVectorIndex":
        chunks = [Chunk(**item) for item in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(index_dir / "vectors.npy", allow_pickle=False)
        return cls(chunks, vectors)

    def save(self, index_dir: Path, manifest: dict) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors, allow_pickle=False)
        chunk_data = [
            {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in self.chunks
        ]
        (index_dir / "chunks.json").write_text(
            json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def search(self, query_vector: list[float], top_k: int = 4) -> list[SearchResult]:
        query = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if query.shape != (self.vectors.shape[1],):
            raise ValueError("查询向量维度与索引不一致，请使用相同的 Embedding 模型重建索引")
        if norm == 0:
            raise ValueError("查询向量是零向量")
        scores = self.vectors @ (query / norm)
        indices = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [SearchResult(self.chunks[index], float(scores[index])) for index in indices]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_manifest(settings: Settings) -> dict:
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "base_url": settings.base_url or "https://api.openai.com/v1",
        "documents": [
            {"source": path.name, "sha256": _file_digest(path)}
            for path in settings.document_paths
        ],
    }


def _is_current(index_dir: Path, manifest: dict) -> bool:
    manifest_path = index_dir / "manifest.json"
    if not all((index_dir / name).is_file() for name in ("manifest.json", "chunks.json", "vectors.npy")):
        return False
    return json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def build_index(settings: Settings, force: bool = False) -> tuple[LocalVectorIndex, bool]:
    manifest = index_manifest(settings)
    if not force and _is_current(settings.index_dir, manifest):
        return LocalVectorIndex.load(settings.index_dir), False

    chunks = load_and_split(settings.document_paths, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("文档切分结果为空，无法建立索引")
    vectors = np.asarray(Embedder(settings).embed_documents(chunks), dtype=np.float32)
    index = LocalVectorIndex(chunks, vectors)
    index.save(settings.index_dir, manifest)
    return index, True
