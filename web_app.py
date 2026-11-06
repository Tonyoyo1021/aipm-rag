import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from openai import APITimeoutError, OpenAIError

from config import PROJECT_DIR, load_settings
from rag import RAGAssistant

WEB_DIR = PROJECT_DIR / "web"
app = Flask(__name__, static_folder=str(WEB_DIR / "static"), static_url_path="/static")


def _index_status(settings) -> dict:
    chunks_path = settings.index_dir / "chunks.json"
    index_ready = all(
        (settings.index_dir / name).is_file()
        for name in ("manifest.json", "chunks.json", "vectors.npy")
    )
    chunk_count = 0
    if chunks_path.is_file():
        chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8")))
    return {
        "ready": index_ready,
        "chunk_count": chunk_count,
        "document_count": len(settings.document_paths),
        "documents": [path.name for path in settings.document_paths],
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "configured": bool(settings.api_key),
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "request_timeout": settings.request_timeout,
    }


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/status")
def status():
    return jsonify(_index_status(load_settings()))


@app.post("/api/ask")
def ask():
    settings = load_settings()
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    top_k = payload.get("top_k", settings.top_k)
    if not question:
        return jsonify({"error": "请输入一个问题"}), 400
    if len(question) > 2000:
        return jsonify({"error": "问题不能超过 2000 个字符"}), 400
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 12:
        return jsonify({"error": "Top-K 必须是 1 到 12 之间的整数"}), 400

    settings = replace(settings, top_k=top_k)
    started = time.perf_counter()
    try:
        assistant = RAGAssistant(settings)
        answer = assistant.ask(question)
    except APITimeoutError:
        return jsonify({"error": "模型服务响应超时，请稍后重试或调大 RAG_REQUEST_TIMEOUT"}), 504
    except OpenAIError:
        return jsonify({"error": "模型服务调用失败，请检查 API Key、模型名、平台额度或稍后重试"}), 502
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    results = [
        {
            "id": result.chunk.id,
            "score": result.score,
            "source": result.chunk.metadata["source"],
            "heading_path": result.chunk.metadata["heading_path"],
            "chunk_index": result.chunk.metadata["chunk_index"],
            "text": result.chunk.text,
        }
        for result in answer.results
    ]
    return jsonify(
        {
            "answer": answer.model_text,
            "results": results,
            "meta": {
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "index_created": assistant.index_created,
                "chunk_count": len(assistant.index.chunks),
                "embedding_model": settings.embedding_model,
                "chat_model": settings.chat_model,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI PM 学习资料问答助手 Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机可访问")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
