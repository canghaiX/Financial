from __future__ import annotations

import csv
import hashlib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from financial_agentic_rag.config import resolve_project_path
from financial_agentic_rag.schemas.indexing import MinerUBlock, RagChunk
from financial_agentic_rag.utils.io import read_json


CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇0-9]+[编章节条])")
PAGE_NUMBER_RE = re.compile(r"^\s*(?:第\s*)?\d+\s*(?:页)?\s*$")
TOC_DOTS_RE = re.compile(r"\.{4,}|…{2,}|·{4,}")
PUNCT_ONLY_RE = re.compile(r"^[\s，。、“”‘’；：！？（）()《》【】\[\]\-—_]+$")


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"}:
            self._current_cell = []
            self._in_cell = True

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(clean_text(" ".join(self._current_cell)))
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


@dataclass(frozen=True)
class BuildStats:
    documents: int
    chunks: int
    rejected: int


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    by_filename = {row.get("filename", ""): row for row in rows}
    by_stem = {Path(row.get("filename", "")).stem: row for row in rows if row.get("filename")}
    by_bbbs = {row.get("bbbs", ""): row for row in rows if row.get("bbbs")}
    return {**by_filename, **by_stem, **by_bbbs}


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return True
    if PAGE_NUMBER_RE.match(cleaned):
        return True
    if PUNCT_ONLY_RE.match(cleaned):
        return True
    if TOC_DOTS_RE.search(cleaned) and len(cleaned) < 120:
        return True
    return False


def detect_chapter(text: str) -> str | None:
    cleaned = clean_text(text)
    if len(cleaned) > 80:
        return None
    match = CHAPTER_RE.match(cleaned)
    return cleaned if match else None


def stable_chunk_id(parts: list[Any]) -> str:
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def table_html_to_markdown(table_html: str) -> str:
    if not table_html.strip():
        return ""
    parser = _TableHTMLParser()
    parser.feed(table_html)
    rows = parser.rows
    if not rows:
        return clean_text(table_html)
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * width
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown_has_header(markdown: str) -> bool:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    return len(lines) >= 2 and bool(re.match(r"^\|?\s*:?-{3,}:?\s*(\||$)", lines[1]))


def extract_markdown_header(markdown: str) -> tuple[str, str] | None:
    lines = [line for line in markdown.splitlines() if line.strip()]
    if len(lines) >= 2 and markdown_has_header(markdown):
        return lines[0], lines[1]
    return None


def prepend_markdown_header(markdown: str, header: tuple[str, str] | None) -> str:
    if not header or markdown_has_header(markdown):
        return markdown
    return "\n".join([header[0], header[1], markdown])


def find_mineru_output_dir(root: Path, pdf_path: Path) -> Path | None:
    candidates = [
        root / pdf_path.stem,
        root / pdf_path.name,
    ]
    candidates.extend(path for path in root.glob(f"**/{pdf_path.stem}") if path.is_dir())
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _page_from_item(item: dict[str, Any]) -> tuple[int, int] | None:
    for key in ("page_idx", "page", "page_no", "page_num"):
        if key in item and item[key] is not None:
            try:
                page = int(item[key])
                if key == "page_idx":
                    page += 1
                return max(page, 1), max(page, 1)
            except (TypeError, ValueError):
                pass
    if "page_start" in item or "page_end" in item:
        try:
            start = int(item.get("page_start") or item.get("page_end"))
            end = int(item.get("page_end") or start)
            return max(start, 1), max(end, start)
        except (TypeError, ValueError):
            return None
    return None


def _block_type(item: dict[str, Any]) -> str:
    raw = str(
        item.get("type")
        or item.get("category")
        or item.get("block_type")
        or item.get("layout_type")
        or ""
    ).lower()
    if "table" in raw or item.get("table_body") or item.get("html"):
        return "table"
    if "title" in raw:
        return "title"
    return "text"


def _item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "md_content", "value"):
        if item.get(key):
            return clean_text(str(item[key]))
    return ""


def _item_table_html(item: dict[str, Any]) -> str:
    for key in ("table_html", "html", "table_body"):
        if item.get(key):
            return str(item[key]).strip()
    return ""


def normalize_content_list(data: Any, source_name: str) -> list[MinerUBlock]:
    items = data if isinstance(data, list) else data.get("content_list", data.get("items", []))
    blocks: list[MinerUBlock] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        pages = _page_from_item(item)
        if not pages:
            continue
        block_type = _block_type(item)
        text = _item_text(item)
        table_html = _item_table_html(item)
        table_markdown = item.get("table_markdown") or table_html_to_markdown(table_html)
        if block_type == "table" and not text:
            text = clean_text(table_markdown or table_html)
        if not text and not table_html:
            continue
        blocks.append(
            MinerUBlock(
                block_id=str(item.get("id") or item.get("block_id") or f"{source_name}:{index}"),
                block_type=block_type,  # type: ignore[arg-type]
                text=text,
                page_start=pages[0],
                page_end=pages[1],
                bbox=item.get("bbox") or item.get("poly") or [],
                table_html=table_html,
                table_markdown=table_markdown,
                table_caption=clean_text(str(item.get("caption") or item.get("table_caption") or "")),
                raw=item,
            )
        )
    return blocks


def normalize_middle_json(data: Any, source_name: str) -> list[MinerUBlock]:
    pdf_info = data.get("pdf_info", []) if isinstance(data, dict) else []
    blocks: list[MinerUBlock] = []
    for page_index, page in enumerate(pdf_info):
        page_no = int(page.get("page_idx", page_index)) + 1
        page_blocks = page.get("para_blocks") or page.get("blocks") or []
        for block_index, block in enumerate(page_blocks):
            if not isinstance(block, dict):
                continue
            lines = block.get("lines", [])
            texts: list[str] = []
            for line in lines:
                spans = line.get("spans", []) if isinstance(line, dict) else []
                texts.extend(str(span.get("content", "")) for span in spans if isinstance(span, dict))
            text = clean_text(block.get("text") or "\n".join(texts))
            block_type = _block_type(block)
            table_html = _item_table_html(block)
            table_markdown = block.get("table_markdown") or table_html_to_markdown(table_html)
            if block_type == "table" and not text:
                text = clean_text(table_markdown or table_html)
            if not text and not table_html:
                continue
            blocks.append(
                MinerUBlock(
                    block_id=str(block.get("id") or f"{source_name}:p{page_no}:b{block_index}"),
                    block_type=block_type,  # type: ignore[arg-type]
                    text=text,
                    page_start=page_no,
                    page_end=page_no,
                    bbox=block.get("bbox") or [],
                    table_html=table_html,
                    table_markdown=table_markdown,
                    table_caption=clean_text(str(block.get("caption") or "")),
                    raw=block,
                )
            )
    return blocks


def parse_markdown_fallback(path: Path) -> list[MinerUBlock]:
    if not path.exists():
        return []
    blocks: list[MinerUBlock] = []
    buffer: list[str] = []
    page = 1
    block_index = 0

    def flush() -> None:
        nonlocal buffer, block_index
        text = clean_text("\n".join(buffer))
        if text:
            blocks.append(
                MinerUBlock(
                    block_id=f"{path.name}:md:{block_index}",
                    block_type="text",
                    text=text,
                    page_start=page,
                    page_end=page,
                )
            )
            block_index += 1
        buffer = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"page[_ -]*(\d+)", line, flags=re.I)
        if match:
            flush()
            page = int(match.group(1))
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks


def load_mineru_blocks(mineru_dir: Path, pdf_path: Path) -> list[MinerUBlock]:
    output_dir = find_mineru_output_dir(mineru_dir, pdf_path)
    if not output_dir:
        return []
    middle_files = sorted(output_dir.rglob("*middle*.json"))
    for path in middle_files:
        return normalize_middle_json(read_json(path), path.name)
    json_files = sorted(output_dir.rglob("*content_list*.json"))
    for path in json_files:
        blocks = normalize_content_list(read_json(path), path.name)
        if blocks:
            return blocks
    md_files = sorted(output_dir.rglob("*.md"))
    return parse_markdown_fallback(md_files[0]) if md_files else []


def build_chunks_for_document(
    pdf_path: Path,
    blocks: list[MinerUBlock],
    manifest_row: dict[str, str],
    chunk_size: int = 1000,
    min_text_length: int = 20,
    context_before: int = 2,
    context_after: int = 2,
) -> tuple[list[RagChunk], list[dict[str, Any]]]:
    chunks: list[RagChunk] = []
    rejected: list[dict[str, Any]] = []
    document_id = manifest_row.get("bbbs") or pdf_path.stem
    document_title = manifest_row.get("title") or pdf_path.stem
    current_chapter = "未识别章节"
    current_chapter_index = 0
    last_table_header: tuple[str, str] | None = None
    text_buffer: list[MinerUBlock] = []

    def metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "gbrq": manifest_row.get("gbrq", ""),
            "sxrq": manifest_row.get("sxrq", ""),
            "flxz": manifest_row.get("flxz", ""),
            "zdjgName": manifest_row.get("zdjgName", ""),
        }
        if extra:
            base.update(extra)
        return base

    def emit_text_buffer(force: bool = False) -> None:
        nonlocal text_buffer
        if not text_buffer:
            return
        text = "\n".join(block.text for block in text_buffer if not is_noise(block.text))
        text = clean_text(text)
        if not force and len(text) < chunk_size:
            return
        if len(text) < min_text_length:
            text_buffer = []
            return
        pages = [page for block in text_buffer for page in (block.page_start, block.page_end)]
        chunk = RagChunk(
            chunk_id=stable_chunk_id([document_id, current_chapter, min(pages), max(pages), text]),
            document_id=document_id,
            document_title=document_title,
            source_file=str(pdf_path),
            chapter_title=current_chapter,
            chapter_index=current_chapter_index,
            page_start=min(pages),
            page_end=max(pages),
            chunk_type="text",
            text=text,
            source_blocks=[block.block_id for block in text_buffer],
            bbox=[block.bbox for block in text_buffer if block.bbox],
            metadata=metadata(),
        )
        chunks.append(chunk)
        text_buffer = []

    for index, block in enumerate(blocks):
        if block.page_start < 1 or block.page_end < 1:
            rejected.append({"source_file": str(pdf_path), "reason": "missing_page", "block": block.model_dump()})
            continue
        chapter = detect_chapter(block.text)
        if chapter:
            current_chapter = chapter
            current_chapter_index += 1

        if block.block_type == "table":
            emit_text_buffer(force=True)
            before_blocks = [
                prev for prev in blocks[max(0, index - context_before) : index]
                if prev.block_type != "table" and not is_noise(prev.text)
            ]
            after_blocks = [
                nxt for nxt in blocks[index + 1 : index + 1 + context_after]
                if nxt.block_type != "table" and not is_noise(nxt.text)
            ]
            before = clean_text("\n".join(item.text for item in before_blocks))
            after = clean_text("\n".join(item.text for item in after_blocks))
            table_markdown = block.table_markdown or table_html_to_markdown(block.table_html)
            header_missing = not markdown_has_header(table_markdown)
            if header_missing and last_table_header:
                table_markdown = prepend_markdown_header(table_markdown, last_table_header)
                header_missing = False
            header = extract_markdown_header(table_markdown)
            if header:
                last_table_header = header
            table_text = clean_text("\n\n".join(part for part in [before, block.table_caption, table_markdown, after] if part))
            if len(table_text) < min_text_length:
                rejected.append({"source_file": str(pdf_path), "reason": "table_too_short", "block": block.model_dump()})
                continue
            chunk = RagChunk(
                chunk_id=stable_chunk_id([document_id, block.block_id, block.page_start, block.page_end, table_text]),
                document_id=document_id,
                document_title=document_title,
                source_file=str(pdf_path),
                chapter_title=current_chapter,
                chapter_index=current_chapter_index,
                page_start=block.page_start,
                page_end=block.page_end,
                chunk_type="table" if not before and not after else "mixed",
                text=table_text,
                table_html=block.table_html,
                table_markdown=table_markdown,
                table_caption=block.table_caption,
                context_before=before,
                context_after=after,
                source_blocks=[*(item.block_id for item in before_blocks), block.block_id, *(item.block_id for item in after_blocks)],
                bbox=block.bbox,
                metadata=metadata({"table_header_missing": header_missing}),
            )
            chunks.append(chunk)
            continue

        if is_noise(block.text):
            rejected.append({"source_file": str(pdf_path), "reason": "noise", "block_id": block.block_id, "text": block.text})
            continue
        text_buffer.append(block)
        buffered_text = "\n".join(item.text for item in text_buffer)
        if len(buffered_text) >= chunk_size:
            emit_text_buffer(force=True)

    emit_text_buffer(force=True)
    return chunks, rejected


def build_all_chunks(config: dict[str, Any]) -> tuple[list[RagChunk], list[dict[str, Any]], BuildStats]:
    mineru_config = config.get("mineru", {})
    chunk_config = config.get("chunking", {})
    output_config = config.get("output_paths", {})
    pdf_dir = resolve_project_path(mineru_config.get("source_pdf_dir", "pdf"))
    mineru_dir = resolve_project_path(mineru_config.get("output_dir", "data/processed/mineru"))
    manifest = load_manifest(resolve_project_path(mineru_config.get("manifest_path", "pdf/manifest.csv")))
    all_chunks: list[RagChunk] = []
    rejected: list[dict[str, Any]] = []
    markdown_dir = resolve_project_path(output_config.get("markdown_dir", "data/processed/markdown"))
    markdown_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        manifest_row = manifest.get(pdf_path.name) or manifest.get(pdf_path.stem) or {}
        blocks = load_mineru_blocks(mineru_dir, pdf_path)
        if not blocks:
            rejected.append({"source_file": str(pdf_path), "reason": "missing_mineru_output"})
            continue
        chunks, doc_rejected = build_chunks_for_document(
            pdf_path=pdf_path,
            blocks=blocks,
            manifest_row=manifest_row,
            chunk_size=int(chunk_config.get("chunk_size", 1000)),
            min_text_length=int(chunk_config.get("min_text_length", 20)),
            context_before=int(chunk_config.get("context_blocks_before_table", 2)),
            context_after=int(chunk_config.get("context_blocks_after_table", 2)),
        )
        all_chunks.extend(chunks)
        rejected.extend(doc_rejected)
        md_lines = [f"# {manifest_row.get('title') or pdf_path.stem}", ""]
        for chunk in chunks:
            md_lines.extend(
                [
                    f"## {chunk.chapter_title} | p.{chunk.page_start}-{chunk.page_end} | {chunk.chunk_type}",
                    "",
                    chunk.text,
                    "",
                ]
            )
        (markdown_dir / f"{pdf_path.stem}.md").write_text("\n".join(md_lines), encoding="utf-8")

    stats = BuildStats(
        documents=len(list(pdf_dir.glob("*.pdf"))),
        chunks=len(all_chunks),
        rejected=len(rejected),
    )
    return all_chunks, rejected, stats
