from openai import OpenAI

from config import Settings
from documents import Chunk


def embedding_text(chunk: Chunk) -> str:
    return (
        f"来源文件：{chunk.metadata['source']}\n"
        f"标题路径：{chunk.metadata['heading_path']}\n"
        f"正文：\n{chunk.text}"
    )


class Embedder:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError(
                "缺少 OPENAI_API_KEY。请复制 .env.example 为 .env，"
                "配置支持 Embedding 的 API Key、Base URL 和模型名后再运行。"
            )
        self.model = settings.embedding_model
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=1,
        )

    def embed_documents(self, chunks: list[Chunk], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(self._embed([embedding_text(chunk) for chunk in batch]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
