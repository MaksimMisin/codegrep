from openai import OpenAI
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from codegrep.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class ChunkEmbedding:
    """Container for chunk content and its embedding."""

    content: str
    embedding: np.ndarray


def process_text(text: str) -> Optional[List[ChunkEmbedding]]:
    """Process text by chunking and generating embeddings for each chunk.

    Args:
        text: Input text to process

    Returns:
        List of ChunkEmbedding objects containing chunk content and embeddings,
        or None if processing fails
    """
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(text)

        if not chunks:
            print("Warning: No chunks generated from input text")
            return None

        # Generate embeddings for all chunks
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=chunks,
        )

        # Create ChunkEmbedding objects combining content and embeddings
        chunk_embeddings = [
            ChunkEmbedding(
                content=chunk, embedding=np.array(data.embedding, dtype=np.float32)
            )
            for chunk, data in zip(chunks, response.data)
        ]

        return chunk_embeddings

    except Exception as e:
        print(f"Error processing text: {e}")
        return None


def generate_query_embedding(query: str) -> Optional[np.ndarray]:
    """Generate embedding for a search query.

    Args:
        query: Search query text

    Returns:
        numpy.ndarray: Query embedding vector, or None if generation fails
    """
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=[query],  # Single query, no chunking needed
        )
        return np.array([response.data[0].embedding], dtype=np.float32)

    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None
