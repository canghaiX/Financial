from pathlib import Path

from financial_agentic_rag.indexing.mineru_chunks import build_chunks_for_document, is_noise
from financial_agentic_rag.schemas.indexing import MinerUBlock


def test_noise_cleaning_keeps_legal_article_numbers() -> None:
    assert is_noise("12")
    assert is_noise("................ 23")
    assert not is_noise("第一条 为了保护民事主体的合法权益，制定本法。")


def test_table_chunk_contains_context_and_header() -> None:
    blocks = [
        MinerUBlock(block_id="b1", block_type="title", text="第一章 基本规定", page_start=1, page_end=1),
        MinerUBlock(block_id="b2", block_type="text", text="下表列明处罚标准。", page_start=1, page_end=1),
        MinerUBlock(
            block_id="t1",
            block_type="table",
            text="",
            table_html="<table><tr><th>行为</th><th>处罚</th></tr><tr><td>A</td><td>B</td></tr></table>",
            page_start=1,
            page_end=1,
        ),
        MinerUBlock(block_id="b3", block_type="text", text="以上标准应依法适用。", page_start=1, page_end=1),
    ]
    chunks, rejected = build_chunks_for_document(
        pdf_path=Path("pdf/test.pdf"),
        blocks=blocks,
        manifest_row={"bbbs": "doc1", "title": "测试法"},
    )
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "mixed"]
    assert not rejected
    assert table_chunks
    table = table_chunks[0]
    assert "下表列明处罚标准" in table.context_before
    assert "以上标准应依法适用" in table.context_after
    assert "| 行为 | 处罚 |" in table.table_markdown
    assert table.document_title == "测试法"
    assert table.chapter_title == "第一章 基本规定"
    assert table.page_start == 1


def test_cross_page_table_preserves_page_and_reuses_header() -> None:
    blocks = [
        MinerUBlock(block_id="h", block_type="title", text="第二章 表格", page_start=2, page_end=2),
        MinerUBlock(
            block_id="t1",
            block_type="table",
            table_markdown="| 项目 | 金额 |\n| --- | --- |\n| A | 1 |",
            text="项目 金额 A 1",
            page_start=2,
            page_end=2,
        ),
        MinerUBlock(
            block_id="t2",
            block_type="table",
            table_markdown="| B | 2 |",
            text="B 2",
            page_start=3,
            page_end=3,
        ),
    ]
    chunks, _ = build_chunks_for_document(
        pdf_path=Path("pdf/test.pdf"),
        blocks=blocks,
        manifest_row={"bbbs": "doc1", "title": "测试法"},
    )
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type in {"table", "mixed"}]
    assert len(table_chunks) == 2
    assert table_chunks[1].page_start == 3
    assert "| 项目 | 金额 |" in table_chunks[1].table_markdown
    assert not table_chunks[1].metadata["table_header_missing"]
