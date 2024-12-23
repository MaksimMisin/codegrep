import pytest
import numpy as np
from pathlib import Path
import subprocess
from langchain_community.embeddings import FakeEmbeddings
from codegrep.config import EMBEDDING_DIM
from codegrep.index import FAISSIndex
from codegrep.cli import (
    collect_repository_files,
    should_ignore_file,
    update_index,
)
import io
import logging

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


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    fake_embeddings = FakeEmbeddings(size=EMBEDDING_DIM)
    monkeypatch.setattr("codegrep.embeddings.embedding_model", fake_embeddings)


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create a temporary directory for the FAISS index and initialize git."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    # Initialize git repository
    init_git_repo(index_dir)
    return index_dir


@pytest.fixture
def capture_logs():
    """Fixture to capture logs from the codegrep logger."""
    log_capture = io.StringIO()
    logger = logging.getLogger("codegrep")

    # Save original handlers and level
    original_handlers = logger.handlers[:]
    original_level = logger.level

    # Clear existing handlers and add string capture handler
    logger.handlers = []
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    yield log_capture

    # Restore original handlers and level
    logger.handlers = original_handlers
    logger.setLevel(original_level)


class TestFAISSIndex:
    def test_index_initialization(self, temp_index_dir):
        index = FAISSIndex(temp_index_dir)
        assert index.index is not None
        assert index.filepath_to_ids == {}
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

        success = index.add_to_index(SAMPLE_CODE, "test.py")
        assert success

        results = index.search("test", k=1)
        assert len(results) == 1
        assert results[0].filename == "test.py"
        assert results[0].filepath == "test.py"

    def test_file_needs_update(self, temp_index_dir):
        index = FAISSIndex(temp_index_dir)
        rel_path = "test.py"
        test_file = temp_index_dir / rel_path
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
        assert index.file_needs_update(rel_path)
        index.add_to_index(SAMPLE_CODE, "test.py")
        index.save_index()
        assert not index.file_needs_update(rel_path)

        test_file.write_text(SAMPLE_CODE + "\n# Modified")
        assert index.file_needs_update(rel_path)


class TestCLI:
    def test_should_ignore_file_with_custom_paths(self, temp_index_dir):
        """Test that should_ignore_file correctly handles custom ignore paths."""
        test_paths = [
            (temp_index_dir / "src/main.py", ["vendor"], False),
            (temp_index_dir / "src/vendor/lib.py", ["vendor"], True),
            (temp_index_dir / "src/tests/test.py", ["vendor", "tests"], True),
            (temp_index_dir / "src/lib/code.py", ["vendor", "tests"], False),
        ]

        for file_path, ignore_paths, expected in test_paths:
            assert should_ignore_file(str(file_path), ignore_paths) == expected

    def test_collect_repository_files_with_custom_ignore(self, temp_index_dir):
        """Test that collect_repository_files respects custom ignore paths."""
        # Create test directory structure
        (temp_index_dir / "src/main").mkdir(parents=True)
        (temp_index_dir / "src/vendor").mkdir(parents=True)
        (temp_index_dir / "src/tests").mkdir(parents=True)

        # Create test files
        (temp_index_dir / "src/main/app.py").write_text("app code")
        (temp_index_dir / "src/vendor/lib.py").write_text("vendor code")
        (temp_index_dir / "src/tests/test_app.py").write_text("test code")

        # Add files to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add test files"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        # Test without custom ignore paths
        files = collect_repository_files(temp_index_dir)
        filenames = {Path(f[1]).name for f in files}
        assert "app.py" in filenames
        assert "lib.py" in filenames
        assert "test_app.py" in filenames

        # Test with custom ignore paths
        files = collect_repository_files(
            temp_index_dir, custom_ignore_paths=["vendor", "tests"]
        )
        filenames = {Path(f[1]).name for f in files}
        assert "app.py" in filenames
        assert "lib.py" not in filenames
        assert "test_app.py" not in filenames


class TestIntegration:
    def test_search_with_files_only_output(self, temp_index_dir, capsys):
        """Test that search results can be displayed in files-only format."""

        # Create test files
        (temp_index_dir / "src").mkdir()
        test_files = {
            "calc.py": """
def add(a, b):
    return a + b
            """,
            "math.py": """
def multiply(a, b):
    return a * b
            """,
        }

        for filename, content in test_files.items():
            (temp_index_dir / "src" / filename).write_text(content)

        # Add to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add math functions"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        # Initialize and update index
        index = FAISSIndex(temp_index_dir)
        update_index(temp_index_dir, index)

        # Get search results
        results = index.search("add numbers", k=2)
        assert len(results) > 0

        # Check that all results have required fields
        for result in results:
            assert isinstance(result.relevance, np.float32().dtype.type)
            assert isinstance(result.filepath, str)
            assert result.filepath.endswith(".py")

        # Test files-only output format
        output_lines = [result.filepath for result in results]
        assert all(line.endswith(".py") for line in output_lines)

    def test_search_with_custom_ignore_paths(self, temp_index_dir):
        """Test that search respects custom ignore paths."""
        # Create test directory structure with files
        (temp_index_dir / "src/app").mkdir(parents=True)
        (temp_index_dir / "src/vendor").mkdir(parents=True)

        (temp_index_dir / "src/app/calc.py").write_text(
            """
def add(a, b):
    return a + b
        """
        )
        (temp_index_dir / "src/vendor/math.py").write_text(
            """
def multiply(a, b):
    return a * b
        """
        )

        # Add to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add app and vendor files"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        # Initialize and update index with custom ignore paths
        index = FAISSIndex(temp_index_dir)
        update_index(temp_index_dir, index, custom_ignore_paths=["vendor"])

        # Verify only non-ignored files are in metadata
        metadata_paths = list(index.filepath_to_ids.keys())
        assert any("app/calc.py" in path for path in metadata_paths)
        assert not any("vendor/math.py" in path for path in metadata_paths)

        # Search and verify results
        results = index.search("math functions", k=2)
        result_paths = [r.filepath for r in results]

        # Only calc.py should be included in results
        assert len(result_paths) > 0  # At least one result should be found
        assert all("vendor" not in path for path in result_paths)

    def test_selective_reindex(self, temp_index_dir, capture_logs):
        """Test that only modified files are re-indexed during updates."""
        # Create test files
        (temp_index_dir / "src").mkdir()
        test_files = {
            "file1.py": """
    def greet(name):
        return f"Hello, {name}!"
            """,
            "file2.py": """
    def calculate(x, y):
        return x + y
            """,
        }

        # Create initial files
        for filename, content in test_files.items():
            (temp_index_dir / "src" / filename).write_text(content)

        # Add to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        # Initialize index and perform initial indexing
        index = FAISSIndex(temp_index_dir)
        update_index(temp_index_dir, index)

        # Check initial indexing log
        initial_logs = capture_logs.getvalue()
        assert "Updated 2 out of 2 files" in initial_logs

        # Clear logs for next operation
        capture_logs.truncate(0)
        capture_logs.seek(0)

        # Perform initial search
        initial_results = index.search("calculate numbers", k=2)
        assert len(initial_results) > 0
        initial_relevance = {r.filepath: r.relevance for r in initial_results}

        # Modify only file2.py
        modified_content = """
    def calculate(x, y):
        # Added multiplication
        return x * y
        """
        (temp_index_dir / "src" / "file2.py").write_text(modified_content)

        # Update index again
        update_index(temp_index_dir, index)

        # Check reindexing logs
        reindex_logs = capture_logs.getvalue()
        assert "Updated 1 out of 2 files" in reindex_logs

        # Perform search again
        new_results = index.search("calculate numbers", k=2)
        assert len(new_results) > 0
        new_relevance = {r.filepath: r.relevance for r in new_results}

        # Verify that file1.py's relevance score hasn't changed
        for filepath, score in initial_relevance.items():
            if "file1.py" in filepath:
                assert new_relevance.get(filepath) == score

        # Verify file2.py appears in results with different relevance
        file2_paths = [r.filepath for r in new_results if "file2.py" in r.filepath]
        assert len(file2_paths) > 0
        assert initial_relevance["src/file2.py"] != new_relevance["src/file2.py"]

    def test_cli_incremental_indexing(self, temp_index_dir, capture_logs, capsys):
        """Test that running codegrep twice only indexes files on the first run."""
        # Create test files with some searchable content
        (temp_index_dir / "src").mkdir()
        test_files = {
            "search.py": """
    def search_algorithm(query, data):
        '''Implements semantic search functionality'''
        results = []
        for item in data:
            if query.lower() in item.lower():
                results.append(item)
        return results
    """,
            "index.py": """
    class SearchIndex:
        '''Main index class for storing searchable data'''
        def __init__(self):
            self.data = []

        def add_document(self, doc):
            self.data.append(doc)
    """,
            "utils.py": """
    def preprocess_query(query):
        '''Clean and prepare search query'''
        return query.strip().lower()
    """,
        }

        # Create initial files
        for filename, content in test_files.items():
            (temp_index_dir / "src" / filename).write_text(content)

        # Add to git
        subprocess.run(
            ["git", "add", "."], cwd=str(temp_index_dir), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add test files"],
            cwd=str(temp_index_dir),
            capture_output=True,
        )

        # Import main CLI function
        from codegrep.cli import main
        import sys
        from unittest.mock import patch

        # First run - should index everything
        with patch.object(
            sys, "argv", ["codegrep", "-q", "search", "-p", str(temp_index_dir)]
        ):
            main()

        # Get first run logs and results
        first_run_logs = capture_logs.getvalue()
        first_run_stdout = capsys.readouterr().out
        assert "Updated 3 out of 3 files" in first_run_logs

        # Extract search results from logs
        first_run_results = [
            line
            for line in first_run_stdout.split("\n")
            if line.startswith("src/") and ".py" in line
        ]

        # Clear logs for second run
        capture_logs.truncate(0)
        capture_logs.seek(0)

        # Second run - should detect no changes
        with patch.object(
            sys, "argv", ["codegrep", "-q", "search", "-p", str(temp_index_dir)]
        ):
            main()

        # Get second run logs and results
        second_run_logs = capture_logs.getvalue()
        second_run_stdout = capsys.readouterr().out
        assert "Updated 0 out of 3 files" in second_run_logs

        # Extract search results from second run
        second_run_results = [
            line
            for line in second_run_stdout.split("\n")
            if line.startswith("src/") and ".py" in line
        ]

        # Verify search results are identical between runs
        assert first_run_results == second_run_results
        assert any("search.py" in line for line in first_run_results)
        assert len(first_run_results) == 3

        # Clear logs for files-only test
        capture_logs.truncate(0)
        capture_logs.seek(0)

        # Test with --files-only flag
        with patch.object(
            sys,
            "argv",
            ["codegrep", "-q", "search", "-p", str(temp_index_dir), "--files-only"],
        ):
            main()

        # Get files-only run output
        files_only_logs = capture_logs.getvalue()
        files_only_stdout = capsys.readouterr().out
        files_only_results = [
            line
            for line in files_only_stdout.split("\n")
            if line.startswith("src/") and line.strip()
        ]

        # Verify files-only output format
        assert all(line.endswith(".py") for line in files_only_results)
        assert not any(
            "(" in line for line in files_only_results
        )  # No relevance scores
        assert not any("Updated:" in line for line in files_only_logs.split("\n"))
