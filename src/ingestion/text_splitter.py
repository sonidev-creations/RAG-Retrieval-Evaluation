import re
from typing import TypedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkData(TypedDict):
    id: str
    text: str
    document: str
    page: int
    chunk_index: int


SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving paragraph structure."""
    paragraphs = PARAGRAPH_BREAK.split(text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = SENTENCE_ENDINGS.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def create_chunks(
    pages: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[ChunkData]:
    """Create semantic sentence-aware chunks from extracted pages."""
    all_chunks: list[ChunkData] = []
    global_index = 0

    for page_data in pages:
        text = page_data["text"]
        document = page_data["document"]
        page = page_data["page"]

        sentences = _split_into_sentences(text)
        if not sentences:
            continue

        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                all_chunks.append({
                    "id": f"{document}::p{page}::c{global_index}",
                    "text": chunk_text,
                    "document": document,
                    "page": page,
                    "chunk_index": global_index,
                })
                global_index += 1

                overlap_chars = 0
                overlap_sentences: list[str] = []
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_chars += len(s)

                current_chunk = overlap_sentences
                current_length = overlap_chars

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            all_chunks.append({
                "id": f"{document}::p{page}::c{global_index}",
                "text": chunk_text,
                "document": document,
                "page": page,
                "chunk_index": global_index,
            })
            global_index += 1

    logger.info("Created %d chunks from %d pages", len(all_chunks), len(pages))
    return all_chunks
