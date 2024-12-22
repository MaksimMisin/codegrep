import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import subprocess
from openai.types import Embedding, CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from codegrep.embeddings import process_text, generate_query_embedding, ChunkEmbedding
from codegrep.index import FAISSIndex
from codegrep.cli import (
    collect_repository_files,
    process_single_file,
    update_index,
    search_repository,
)

# Test data
SAMPLE_CODE = """
def hello_world():
    print("Hello, World!")
    return True

class Calculator:
    def add(self, a, b):
        return a + b
"""


def init_git_repo(path):
    """Initialize a git repository and add files to it."""
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=str(path), capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=str(path), capture_output=True
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for embeddings."""

    def create_mock_response(*args, **kwargs):
        mock_embedding = [0.1] * 1536  # Matches EMBEDDING_DIM from config
        embedding_obj = Embedding(embedding=mock_embedding, index=0, object="embedding")
        return CreateEmbeddingResponse(
            data=[embedding_obj],
            model="mock-model",
            usage=Usage(prompt_tokens=10, total_tokens=100),
            object="list",
        )

    return create_mock_response


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create a temporary directory for the FAISS index and initialize git."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    # Initialize git repository
    init_git_repo(index_dir)
    return index_dir


class TestEmbeddings:
    @pytest.fixture
    def mock_openai(self):
        with patch("codegrep.embeddings.client") as mock_client:
            mock_embedding = [0.1] * 1536
            mock_data = MagicMock()
            mock_data.embedding = mock_embedding
            mock_response = MagicMock()
            mock_response.data = [mock_data]
            mock_client.embeddings.create.return_value = mock_response
            yield mock_client

    def test_process_text_empty_input(self, mock_openai):
        result = process_text("")
        assert result is None

    def test_process_text_api_error(self, mock_openai):
        mock_openai.embeddings.create.side_effect = Exception("API Error")
        result = process_text("Test content")
        assert result is None

    def test_generate_query_embedding(self, mock_openai):
        test_query = "test query"
        result = generate_query_embedding(test_query)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1536)
        mock_openai.embeddings.create.assert_called_once()
        call_args = mock_openai.embeddings.create.call_args[1]
        assert call_args["input"] == [test_query]

    def test_generate_query_embedding_api_error(self, mock_openai):
        mock_openai.embeddings.create.side_effect = Exception("API Error")
        result = generate_query_embedding("test query")
        assert result is None


class TestFAISSIndex:
    def test_index_initialization(self, temp_index_dir):
        index = FAISSIndex(temp_index_dir)
        assert index.index is not None
        assert index.metadata == {}
        assert index.file_timestamps == {}

    def test_add_and_search(self, temp_index_dir):
        index = FAISSIndex(temp_index_dir)
        test_file = temp_index_dir / "test.py"
        test_file.write_text(SAMPLE_CODE)
        subprocess.run(
            ["git", "add", "test.py"], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add test.py"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        success = index.add_to_index(SAMPLE_CODE, "test.py", "test.py")
        assert success

        query_embedding = np.random.rand(1, 1536).astype("float32")
        results = index.search(query_embedding, k=1)
        assert len(results) == 1
        assert results[0]["filename"] == "test.py"
        assert results[0]["filepath"] == "test.py"

    def test_file_needs_update(self, temp_index_dir):
        index = FAISSIndex(temp_index_dir)
        test_file = temp_index_dir / "test.py"
        test_file.write_text(SAMPLE_CODE)

        # Add to git
        subprocess.run(
            ["git", "add", "test.py"], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add test.py"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        assert index.file_needs_update(str(test_file))
        index.add_to_index(SAMPLE_CODE, "test.py", "test.py")
        index.save_index()
        assert not index.file_needs_update(str(test_file))

        test_file.write_text(SAMPLE_CODE + "\n# Modified")
        assert index.file_needs_update(str(test_file))


class TestCLI:
    def test_collect_repository_files(self, temp_index_dir):
        (temp_index_dir / "src").mkdir()
        (temp_index_dir / "src" / "main.py").write_text(SAMPLE_CODE)
        (temp_index_dir / "src" / "test.py").write_text(SAMPLE_CODE)
        (temp_index_dir / "src" / "ignored.py").write_text(SAMPLE_CODE)
        (temp_index_dir / "src" / ".gitignore").write_text("ignored.py")

        # Add files to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add files"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        files = collect_repository_files(temp_index_dir)
        assert len(files) > 0
        assert any("main.py" in f[0] for f in files)
        assert any("test.py" in f[0] for f in files)
        assert not any("ignored.py" in f[0] for f in files)

    def test_process_single_file(self, temp_index_dir):
        index = FAISSIndex(temp_index_dir)
        test_file = temp_index_dir / "test.py"
        test_file.write_text(SAMPLE_CODE)

        # Add to git
        subprocess.run(
            ["git", "add", "test.py"], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add test.py"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        success = process_single_file(index, str(test_file), "test.py")
        assert success
        assert len(index.metadata) > 0

        success = process_single_file(
            index, str(temp_index_dir / "nonexistent.py"), "nonexistent.py"
        )
        assert not success


class TestIntegration:
    @patch("openai.OpenAI")
    def test_full_workflow(self, mock_client, mock_openai_response, temp_index_dir):
        mock_client.return_value.embeddings.create = Mock(
            side_effect=mock_openai_response
        )

        (temp_index_dir / "src").mkdir()
        (temp_index_dir / "src" / "main.py").write_text(SAMPLE_CODE)

        # Add to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add source files"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        index = FAISSIndex(temp_index_dir)
        update_index(temp_index_dir, index)

        results = search_repository("function to add numbers", hits=1, index=index)
        assert len(results) > 0
        assert "Calculator" in results[0]["content"]
