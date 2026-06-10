from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".xml"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Knowledge base path is not a directory: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        documents.extend(read_documents_from_file(path))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        normalized = " ".join(document.text.split())
        start = 0
        chunk_index = 0
        while start < len(normalized):
            end = start + chunk_size
            if end >= len(normalized):
                end = len(normalized)
            else:
                # Find the last space inside the window to avoid cutting words
                last_space = normalized.rfind(" ", start, end)
                if last_space != -1 and last_space > start + int(chunk_size * 0.7):
                    end = last_space

            text = normalized[start:end].strip()
            if text:
                chunks.append(
                    Chunk(
                        id=chunk_id(document.source, chunk_index, text),
                        source=document.source,
                        text=text,
                    )
                )
            chunk_index += 1
            if end >= len(normalized):
                break
            
            next_start = end - chunk_overlap
            if next_start >= end or next_start <= start:
                start = end
            else:
                start = next_start
    return chunks


def read_documents_from_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(source=str(path), text=text)] if text.strip() else []
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [Document(source=str(path), text=text)] if text.strip() else []
    if suffix == ".xml":
        return read_medlineplus_xml(path)
    raise ValueError(f"Unsupported document type: {path}")


def read_medlineplus_xml(path: Path) -> list[Document]:
    documents: list[Document] = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "health-topic":
            continue

        title = element.attrib.get("title", "Untitled health topic")
        url = element.attrib.get("url", str(path))
        summary = element.findtext("full-summary", default="")
        also_called = [
            child.text or "" for child in element.findall("./also-called") if child.text
        ]

        text_parts = [f"Title: {title}"]
        if also_called:
            text_parts.append(f"Also called: {', '.join(also_called)}")
        if summary:
            text_parts.append(clean_text(summary))

        text = "\n".join(text_parts).strip()
        if text:
            documents.append(Document(source=f"{title} - {url}", text=text))

        element.clear()
    return documents


def chunk_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(without_tags.split())