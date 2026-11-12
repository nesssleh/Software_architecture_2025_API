"""Разбор входных документов — Adapter для разных форматов (TXT, PDF, DOCX) + выделение заголовка."""
from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod

_TITLE = re.compile(r"^\s*(?:Резюме\.\s*)?(?:Желаемая должность|Вакансия|Должность|Position)\s*:\s*([^.\n]+)[.\n]\s*", re.I)


class DocumentParser(ABC):
    @abstractmethod
    def to_text(self, data: bytes) -> str: ...


class TextParser(DocumentParser):
    def to_text(self, data: bytes) -> str:
        return data.decode("utf-8", errors="replace")


class PdfParser(DocumentParser):
    def to_text(self, data: bytes) -> str:
        from pypdf import PdfReader  # type: ignore

        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


class DocxParser(DocumentParser):
    def to_text(self, data: bytes) -> str:
        import docx  # python-docx

        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)


PARSERS: dict[str, DocumentParser] = {"txt": TextParser(), "pdf": PdfParser(), "docx": DocxParser()}


def parse(data: bytes, filename: str = "doc.txt") -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in PARSERS:
        raise ValueError(f"unsupported format: {ext}")
    return re.sub(r"[ \t]+", " ", PARSERS[ext].to_text(data)).strip()


def split_title(text: str) -> tuple[str | None, str]:
    """Отделяет заголовок («Желаемая должность: …», «Вакансия: …») от тела: название роли — не навык."""
    m = _TITLE.match(text)
    return (m.group(1).strip(), text[m.end():]) if m else (None, text)
