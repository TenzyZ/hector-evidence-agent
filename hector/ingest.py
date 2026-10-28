"""Normalize UTF-8 documents and chunk them without changing source spans."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata

from . import config


class IngestError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    media_type: str
    sha256: str
    char_count: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    start: int
    end: int
    text: str


def ingest(filename: str, data: bytes) -> tuple[Document, list[Chunk]]:
    extension = Path(filename).suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise IngestError("UNSUPPORTED_TYPE", "Only .txt and .md files are accepted.")
    if len(data) > config.MAX_FILE_BYTES:
        raise IngestError("FILE_TOO_LARGE", "File exceeds MAX_FILE_BYTES.")
    if b"\0" in data:
        raise IngestError("BINARY_CONTENT", "NUL bytes are not accepted.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise IngestError("INVALID_UTF8", "File must be valid UTF-8.") from None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if ord(c) >= 32 or c in "\t\n")
    text = unicodedata.normalize("NFC", text)
    if not text.strip():
        raise IngestError("EMPTY_DOCUMENT", "Normalized document is empty.")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document = Document("d_" + digest[:16], Path(filename).name,
                        "text/markdown" if extension == ".md" else "text/plain",
                        digest, len(text), text)
    return document, chunk_document(document)


def chunk_document(document: Document) -> list[Chunk]:
    text = document.text
    limit = config.CHUNK_MAX_CHARS
    spans = []
    start = 0
    for separator in re.finditer(r"\n{2,}", text):
        spans.append((start, separator.start()))
        start = separator.end()
    spans.append((start, len(text)))

    chunks = []
    packed_start = packed_end = None

    def _trim(start, end):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def emit(start, end):
        start, end = _trim(start, end)
        if start < end:
            index = len(chunks)
            chunks.append(Chunk(f"{document.doc_id}:{index:04d}", document.doc_id,
                                index, start, end, text[start:end]))

    for start, end in spans:
        start, end = _trim(start, end)
        if start == end:
            continue
        if packed_start is not None:
            if end - packed_start <= limit:
                packed_end = end
                continue
            emit(packed_start, packed_end)
            packed_start = packed_end = None
        while end - start > limit:
            cut = start + limit
            for position in range(cut - 1, start, -1):
                if text[position].isspace():
                    cut = position
                    break
            emit(start, cut)
            start = cut
            while start < end and text[start].isspace():
                start += 1
        if start < end:
            packed_start, packed_end = start, end
    if packed_start is not None:
        emit(packed_start, packed_end)
    return chunks
