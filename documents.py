import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ")


@dataclass(frozen=True)
class Document:
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def load_markdown(path: Path) -> Document:
    if not path.is_file():
        raise FileNotFoundError(f"知识库文件不存在：{path}")
    return Document(text=path.read_text(encoding="utf-8"), metadata={"source": path.name})


def _without_front_matter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return lines[index + 1 :]
    return lines


def _split_long_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            minimum_end = start + max(overlap + 1, chunk_size // 2)
            for separator in SEPARATORS:
                position = text.rfind(separator, minimum_end, end)
                if position != -1:
                    end = position + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
        while start < end and text[start].isspace():
            start += 1


def split_markdown(document: Document, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("切分参数必须满足 chunk_size > chunk_overlap >= 0")

    sections: list[tuple[tuple[str, ...], str]] = []
    headings: dict[int, str] = {}
    current_path: tuple[str, ...] = ()
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        current_lines = []
        if text and any(line.strip() and not HEADING_RE.match(line) for line in text.splitlines()):
            sections.append((current_path, text))

    for line in _without_front_matter(document.text.splitlines()):
        match = HEADING_RE.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            headings[level] = match.group(2).strip()
            for deeper_level in range(level + 1, 7):
                headings.pop(deeper_level, None)
            current_path = tuple(headings[key] for key in sorted(headings))
        current_lines.append(line)
    flush_section()

    chunks: list[Chunk] = []
    source = document.metadata["source"]
    for heading_path, section_text in sections:
        title_path = " > ".join(heading_path)
        for piece in _split_long_text(section_text, chunk_size, chunk_overlap):
            chunk_index = len(chunks)
            chunk_id = hashlib.sha256(
                f"{source}\n{title_path}\n{chunk_index}\n{piece}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={
                        "source": source,
                        "heading_path": title_path,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return chunks


def load_and_split(paths: Iterable[Path], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(split_markdown(load_markdown(path), chunk_size, chunk_overlap))
    return chunks
