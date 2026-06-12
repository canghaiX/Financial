from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ChunkType = Literal["text", "table", "mixed"]


class RagChunk(BaseModel):
    """A clean, source-traceable unit for indexing and retrieval."""

    chunk_id: str
    document_id: str
    document_title: str
    source_file: str
    chapter_title: str
    chapter_index: int
    page_start: int
    page_end: int
    chunk_type: ChunkType
    text: str
    table_html: str = ""
    table_markdown: str = ""
    table_caption: str = ""
    context_before: str = ""
    context_after: str = ""
    source_blocks: list[str] = Field(default_factory=list)
    bbox: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("page_start", "page_end")
    @classmethod
    def page_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page numbers must be one-based positive integers")
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be empty")
        return value


class MinerUBlock(BaseModel):
    """Normalized block extracted from MinerU outputs."""

    block_id: str
    block_type: Literal["text", "title", "table"]
    text: str = ""
    page_start: int
    page_end: int
    bbox: list[Any] = Field(default_factory=list)
    table_html: str = ""
    table_markdown: str = ""
    table_caption: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)
