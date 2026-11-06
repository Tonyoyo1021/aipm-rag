import argparse
from dataclasses import replace

from openai import OpenAIError

from config import load_settings
from rag import RAGAssistant


def print_results(results) -> None:
    print("\n=== Top-K 检索结果 ===")
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        excerpt = " ".join(chunk.text.split())[:300]
        print(f"\n[{number}] 相似度：{result.score:.4f}")
        print(f"来源：{chunk.metadata['source']}")
        print(f"标题：{chunk.metadata['heading_path'] or '无'}")
        print(f"片段：{excerpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI PM 学习资料问答助手")
    parser.add_argument("question", help="要提问的问题")
    parser.add_argument("--top-k", type=int, help="覆盖本次检索的 Top-K，默认 4")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引后再提问")
    parser.add_argument("--retrieve-only", action="store_true", help="只检索，不调用 LLM 回答")
    args = parser.parse_args()

    settings = load_settings()
    if args.top_k is not None:
        if args.top_k <= 0:
            parser.error("--top-k 必须大于 0")
        settings = replace(settings, top_k=args.top_k)

    try:
        assistant = RAGAssistant(settings, force_rebuild=args.rebuild)
        print("已构建新索引。" if assistant.index_created else "文档和配置未变化，已复用本地索引。")
        if args.retrieve_only:
            results = assistant.retrieve(args.question)
            print_results(results)
            return
        answer = assistant.ask(args.question)
        print_results(answer.results)
        print("\n=== 最终回答 ===")
        print(answer.text)
    except (FileNotFoundError, RuntimeError, ValueError, OpenAIError) as error:
        raise SystemExit(f"运行失败：{error}") from error


if __name__ == "__main__":
    main()
