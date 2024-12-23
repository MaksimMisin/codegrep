from dataclasses import dataclass
from pathlib import Path
import faiss
import json
import time
from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from codegrep.embeddings import embedding_model
from codegrep.config import EMBEDDING_DIM


@dataclass
class SearchResult:
    filepath: str
    filename: str
    content: str
    relevance: float


class FAISSIndex:
    SAVED_INDEX_NAME = "index"

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        embedding_model: Embeddings = embedding_model,
    ):
        """Initialize FAISSIndex with a configurable index directory.

        index_dir: Path
            Absolute path to the directory where the index will be stored.
        embedding_model: Embeddings
            Langchain Embeddings object used to encode text.
        """
        self.file_timestamps: Dict[str, float] = {}
        self.filepath_to_ids: Dict[str, List[str]] = {}

        # Use provided directory or current working directory
        self.index_dir = index_dir if index_dir else Path.cwd()
        self.index_path = self.index_dir / ".codegrep"
        self.timestamps_file = self.index_path / "timestamps.json"
        self.filepaths_file = self.index_path / "filepaths.json"

        # Create data directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.embedding_model = embedding_model

        # Initialize or load existing index
        if (self.index_path / f"{self.SAVED_INDEX_NAME}.faiss").exists():
            self.load_index()
        else:
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.index = FAISS(
                embedding_model,
                index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self._save_metadata()

    def _save_metadata(self):
        """Save timestamps and filepath mappings."""
        with open(self.timestamps_file, "w") as f:
            json.dump(self.file_timestamps, f)
        with open(self.filepaths_file, "w") as f:
            json.dump(self.filepath_to_ids, f)

    def _load_metadata(self):
        """Load timestamps and filepath mappings."""
        if self.timestamps_file.exists():
            with open(self.timestamps_file, "r") as f:
                self.file_timestamps = json.load(f)
        if self.filepaths_file.exists():
            with open(self.filepaths_file, "r") as f:
                self.filepath_to_ids = json.load(f)

    def save_index(self):
        """Save index and metadata to disk."""
        self.index.save_local(str(self.index_path), self.SAVED_INDEX_NAME)
        self._save_metadata()

    def load_index(self):
        """Load index and metadata from disk."""
        self.index = FAISS.load_local(
            str(self.index_path),
            self.embedding_model,
            index_name=self.SAVED_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        self._load_metadata()

    def indexed_files(self, absolute_paths: bool = False) -> List[str]:
        """Get a list of files that have been indexed."""
        if absolute_paths:
            return [str(self.index_dir / filepath) for filepath in self.filepath_to_ids]
        return list(self.filepath_to_ids.keys())

    def file_needs_update(self, filepath: str) -> bool:
        """Check if a file needs to be reindexed based on its modification time.
        Takes path relative to the index directory.
        """
        try:
            current_mtime = (self.index_dir / filepath).stat().st_mtime
            last_indexed_time = self.file_timestamps.get(filepath, 0)
            return current_mtime > last_indexed_time
        except Exception as e:
            print(f"Error checking file modification time for {filepath}: {e}")
            return True

    def remove_file_from_index(self, filepath: str) -> None:
        """Remove a file's vectors and metadata from the index."""
        if filepath in self.filepath_to_ids:
            ids_to_remove = self.filepath_to_ids[filepath]
            self.index.delete(ids_to_remove)
            del self.filepath_to_ids[filepath]
            if filepath in self.file_timestamps:
                del self.file_timestamps[filepath]

    def add_to_index(self, full_content: str, filepath: str) -> bool:
        """Process and add file content to the index."""
        try:
            # Remove existing entries for this file
            self.remove_file_from_index(filepath)

            # Create Document objects with metadata
            raw_document = Document(
                page_content=full_content,
                metadata={"filepath": filepath, "filename": Path(filepath).name},
            )
            documents = self.text_splitter.split_documents([raw_document])

            # Add to index and get IDs
            ids = self.index.add_documents(documents)

            # Update mappings
            self.filepath_to_ids[filepath] = ids
            self.file_timestamps[filepath] = time.time()

            return True

        except Exception as e:
            print(f"Error adding file to index: {e}")
            return False

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search the index with the given query embedding.

        Returns top k unique files, using the highest relevance score when
        multiple chunks from the same file match.
        """
        # Request more results initially to account for duplicates
        results = self.index.similarity_search_with_score(query, k=10 * k)

        # Group results by filepath and keep highest relevance score
        file_results = {}
        for doc, score in results:
            filepath = doc.metadata["filepath"]
            relevance = 1 / (1 + score)

            if (
                filepath not in file_results
                or relevance > file_results[filepath]["relevance"]
            ):
                file_results[filepath] = {
                    "filename": doc.metadata["filename"],
                    "content": doc.page_content,
                    "relevance": relevance,
                }

        # Convert to list of SearchResults, sorted by relevance
        formatted_results = [
            SearchResult(
                filepath=filepath,
                filename=data["filename"],
                content=data["content"],
                relevance=data["relevance"],
            )
            for filepath, data in file_results.items()
        ]

        # Sort by relevance and limit to k results
        formatted_results.sort(key=lambda x: x.relevance, reverse=True)
        return formatted_results[:k]
