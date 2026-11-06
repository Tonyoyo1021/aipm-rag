import argparse

from config import load_settings
from vector_store import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 AI PM 学习资料的本地向量索引")
    parser.add_argument("--force", action="store_true", help="忽略缓存并强制重建索引")
    args = parser.parse_args()
    settings = load_settings()
    try:
        index, created = build_index(settings, force=args.force)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"构建失败：{error}") from error
    action = "已构建" if created else "配置和文档均未变化，复用"
    print(f"{action}本地索引：{settings.index_dir}")
    print(f"文档数：{len(settings.document_paths)}；chunk 数：{len(index.chunks)}；向量维度：{index.vectors.shape[1]}")
    print(f"chunk_size={settings.chunk_size}，chunk_overlap={settings.chunk_overlap}，top_k={settings.top_k}")


if __name__ == "__main__":
    main()
