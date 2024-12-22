from typing import Optional, Dict, List
import faiss
import numpy as np
from pathlib import Path
import time
import json
from codegrep.config import EMBEDDING_DIM
from codegrep.embeddings import process_text


class FAISSIndex:
    def __init__(self, index_dir: Optional[Path] = None):
        """Initialize FAISSIndex with a configurable index directory."""

        self.metadata: Dict[int, Dict] = {}
        self.file_timestamps: Dict[str, float] = {}

        # Use provided directory or current working directory
        self.index_path = index_dir if index_dir else Path.cwd()
        self.index_path = self.index_path / ".codegrep"
        self.index_file = self.index_path / "faiss.index"
        self.metadata_file = self.index_path / "metadata.json"
        self.timestamps_file = self.index_path / "timestamps.json"

        # Create data directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize or load existing index
        if self.index_file.exists() and self.metadata_file.exists():
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)

    def save_index(self):
        """Save index, metadata, and timestamps to disk."""
        faiss.write_index(self.index, str(self.index_file))

        # Save metadata and timestamps as JSON
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)
        with open(self.timestamps_file, "w") as f:
            json.dump(self.file_timestamps, f)

    def load_index(self):
        """Load index, metadata, and timestamps from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found at {self.index_file}")

        self.index = faiss.read_index(str(self.index_file))

        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
                # Convert string keys back to integers
                self.metadata = {int(k): v for k, v in self.metadata.items()}

        if self.timestamps_file.exists():
            with open(self.timestamps_file, "r") as f:
                self.file_timestamps = json.load(f)
        else:
            self.file_timestamps = {}

    def file_needs_update(self, filepath: str) -> bool:
        """Check if a file needs to be reindexed based on its modification time."""
        try:
            current_mtime = Path(filepath).stat().st_mtime
            last_indexed_time = self.file_timestamps.get(filepath, 0)
            return current_mtime > last_indexed_time
        except Exception as e:
            print(f"Error checking file modification time for {filepath}: {e}")
            return True  # If there's any error, assume file needs update

    def remove_file_from_index(self, filepath: str) -> None:
        """Remove a file's vectors and metadata from the index.

        Args:
            filepath: Full path to the file to remove from the index
        """
        if self.index.ntotal == 0:
            return

        # Convert filepath to relative path to match metadata
        relative_filepath = str(Path(filepath).relative_to(self.index_path.parent))

        # Find indices of vectors to remove
        indices_to_remove = []
        new_metadata = {}
        current_index = 0

        # Identify vectors to remove and build new metadata
        for idx, meta in sorted(self.metadata.items()):
            if meta["filepath"] != relative_filepath:
                # Keep this entry, but update its index
                new_metadata[current_index] = meta
                current_index += 1
            else:
                indices_to_remove.append(idx)

        if not indices_to_remove:
            return  # File not found in index

        # Create a boolean mask for vectors to keep
        total_vectors = self.index.ntotal
        keep_mask = np.ones(total_vectors, dtype=bool)
        keep_mask[indices_to_remove] = False

        # Extract the vectors we want to keep
        vectors = faiss.vector_to_array(self.index.get_xb()).reshape(-1, self.index.d)
        kept_vectors = vectors[keep_mask]

        # Clear the current index
        self.index.reset()

        # Add back the vectors we want to keep
        if len(kept_vectors) > 0:
            self.index.add(kept_vectors)

        # Update metadata
        self.metadata = new_metadata

        # Remove from timestamps
        if filepath in self.file_timestamps:
            del self.file_timestamps[filepath]

    def add_to_index(self, full_content: str, filename: str, filepath: str) -> bool:
        """Process and add file content to the index.

        Args:
            full_content: Complete file content
            filename: Name of the file
            filepath: Path to the file (relative to repo root)

        Returns:
            bool: True if successful, False otherwise
        """
        # Process text to get chunks and embeddings
        chunk_embeddings = process_text(full_content)
        if not chunk_embeddings:
            return False

        # Extract embeddings array for FAISS
        embeddings = np.vstack([ce.embedding for ce in chunk_embeddings])

        if embeddings.shape[1] != self.index.d:
            print(
                f"Error: Embedding dimension {embeddings.shape[1]} does not match index dimension {self.index.d}"
            )
            return False

        # Add embeddings to FAISS index
        self.index.add(embeddings)
        start_idx = len(self.metadata)

        # Store metadata for each chunk
        for i, ce in enumerate(chunk_embeddings):
            self.metadata[start_idx + i] = {
                "content": ce.content,
                "chunk_index": i,
                "total_chunks": len(chunk_embeddings),
                "filename": filename,
                "filepath": filepath,
            }

        # Update timestamp
        abs_path = str(Path(self.index_path).parent / filepath)
        self.file_timestamps[abs_path] = time.time()
        return True

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search the index with the given query embedding.

        Args:
            query_embedding: Embedding vector for the search query
            k: Number of results to return

        Returns:
            List of dictionaries containing search results
        """
        if self.index.ntotal == 0:
            return []

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Convert distances to similarity scores and combine with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                similarity = 1 / (1 + distances[0][i])

                results.append(
                    {
                        "filepath": metadata["filepath"],
                        "filename": metadata["filename"],
                        "content": metadata["content"],
                        "similarity": similarity,
                    }
                )

        return results
