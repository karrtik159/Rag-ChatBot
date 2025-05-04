# src/rag_chatbot/chunker.py

from typing import List
import tiktoken

from models import RawEntry, Chunk, db_settings

def chunk_text(entries: List[RawEntry]) -> List[Chunk]:
    """
    Splits each RawEntry into smaller, overlapping token chunks.

    Args:
        entries (List[RawEntry]): List of raw text blocks extracted from documents.

    Returns:
        List[Chunk]: List of tokenized sub-chunks with metadata.
    """
    # Load tokenizer and settings
    encoder = tiktoken.get_encoding("cl100k_base")
    max_tokens = db_settings.MAX_TOKENS
    overlap = db_settings.OVERLAP

    chunks: List[Chunk] = []

    for entry in entries:
        # Tokenize the full text of this entry
        tokens = encoder.encode(entry.text)
        start = 0
        sub_idx = 0

        # Slide window over tokens
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            text_chunk = encoder.decode(chunk_tokens)

            # Build Chunk model
            chunk = Chunk(
                document_name=entry.document_name,
                page=entry.page,
                text=text_chunk,
                is_ocr=entry.is_ocr,
                source=entry.source,
                chunk_index=entry.chunk_index,
                sub_chunk_index=sub_idx,
            )
            chunks.append(chunk)

            # Advance window
            sub_idx += 1
            start += max_tokens - overlap

    return chunks
