import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from .index import FAISSIndex
from .embeddings import generate_query_embedding


def run_git_command(repo_path: Path, command: List[str]) -> Optional[str]:
    """
    Run a git command in the specified repository.

    Args:
        repo_path: Repository root path
        command: Git command as a list of strings

    Returns:
        Command output as string if successful, None if failed
    """
    try:
        result = subprocess.run(
            ["git"] + command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error running git command: {e}")
        return None


def collect_repository_files(
    repo_path: Path, dry_run: bool = False
) -> List[Tuple[str, str]]:
    """
    Collect all tracked files from the git repository using git ls-files.

    Args:
        repo_path: Root path of the repository
        dry_run: If True, print information about each file's status

    Returns:
        List of tuples containing (absolute_path, relative_path)
    """
    # Check if repo_path is a git repository
    if not run_git_command(repo_path, ["rev-parse", "--git-dir"]):
        print(f"Error: {repo_path} is not a git repository")
        return []

    # Get list of tracked files using git ls-files
    output = run_git_command(repo_path, ["ls-files"])
    if not output:
        print("No files found in git repository")
        return []

    current_files = []
    files_processed = 0

    if dry_run:
        print("\nDry run - analyzing repository files:")
        print("=====================================")

    # Process each file path
    for rel_path in output.split("\n"):
        if not rel_path.strip():  # Skip empty lines
            continue

        try:
            file_path = repo_path / rel_path
            if file_path.is_file():  # Verify file exists
                abs_path = file_path.resolve()
                current_files.append((str(abs_path), rel_path))
                files_processed += 1

                if dry_run:
                    print(f"Found: {rel_path}")
            else:
                if dry_run:
                    print(f"Warning: Tracked file not found: {rel_path}")

        except Exception as e:
            if dry_run:
                print(f"ERROR: Could not process {rel_path}: {e}")
            continue

    print(f"Found {files_processed} files to process")
    if dry_run:
        print("=====================================")

    return current_files


def remove_deleted_files(
    index: FAISSIndex, current_files: List[Tuple[str, str]]
) -> None:
    """
    Remove files from the index that no longer exist in the repository.

    Args:
        index: FAISSIndex instance
        current_files: List of (absolute_path, relative_path) tuples for current files
    """
    current_abs_paths = {abs_path for abs_path, _ in current_files}
    indexed_files = set(index.file_timestamps.keys())
    removed_files = indexed_files - current_abs_paths

    for filepath in removed_files:
        print(f"Removing deleted file from index: {filepath}")
        index.remove_file_from_index(filepath)


def process_single_file(index: FAISSIndex, abs_path: str, rel_path: str) -> bool:
    """
    Process a single file and add it to the index.

    Args:
        index: FAISSIndex instance
        abs_path: Absolute path to the file
        rel_path: Path relative to repository root

    Returns:
        bool: True if file was successfully processed, False otherwise
    """
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove any existing entries for this file before adding new ones
        index.remove_file_from_index(abs_path)

        # Add the processed content to the index
        if index.add_to_index(content, Path(abs_path).name, str(rel_path)):
            print(f"Updated: {rel_path}")
            return True
        else:
            print(f"Warning: Failed to add {rel_path} to index")
            return False

    except Exception as e:
        print(f"Warning: Error processing {rel_path}: {e}")
        return False


def update_index(repo_path: Path, index: FAISSIndex, dry_run: bool = False) -> None:
    """
    Update the FAISS index for the given repository path.

    Args:
        repo_path: Path to the repository root
        index: FAISSIndex instance to update
        dry_run: If True, only print what would be done without making changes
    """
    if dry_run:
        print("Dry run mode - no changes will be made to the index")
    else:
        print("Checking for files to update...")

    files_updated = 0

    # Ensure we have absolute, normalized paths
    repo_path = repo_path.resolve()

    # Collect all current repository files using git ls-files
    current_files = collect_repository_files(repo_path, dry_run)

    if not dry_run:
        # Remove files that no longer exist
        remove_deleted_files(index, current_files)

        # Process files that need updating
        for abs_path, rel_path in current_files:
            if index.file_needs_update(abs_path):
                if process_single_file(index, abs_path, rel_path):
                    files_updated += 1

        # Save the updated index
        index.save_index()
        print(
            f"Index update complete. Updated {files_updated} out of {len(current_files)} files."
        )


def search_repository(query: str, hits: int, index: FAISSIndex) -> List[dict]:
    """
    Search the repository with the given query and return top N results.

    Args:
        query: Search query string
        hits: Number of results to return
        index: FAISSIndex instance to search in
    """
    query_embedding = generate_query_embedding(query)
    if query_embedding is None:
        print("Failed to generate query embedding.")
        return []

    return index.search(query_embedding, hits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="codegrep - Semantic code search tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    search_group = parser.add_argument_group("search options")
    search_group.add_argument(
        "-q", "--query", help="Search query (required unless --dry-run is specified)"
    )
    search_group.add_argument(
        "-n",
        "--hits",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Repository path (default: current directory)",
    )
    parser.add_argument(
        "-i",
        "--index-dir",
        type=Path,
        help="Directory to store the index (default: inside repository path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be indexed without making any changes",
    )

    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a valid directory")
        sys.exit(1)

    # Check if path is a git repository
    if not run_git_command(args.path, ["rev-parse", "--git-dir"]):
        print(f"Error: {args.path} is not a git repository")
        sys.exit(1)

    # Use index_dir if provided, otherwise use repository path
    index_dir = args.index_dir if args.index_dir else args.path
    faiss_index = FAISSIndex(index_dir)

    if args.dry_run:
        # Only run the index update in dry-run mode
        update_index(args.path, faiss_index, dry_run=True)
        return

    # For search mode, query is required
    if not args.query:
        parser.error(
            "the -q/--query argument is required unless --dry-run is specified"
        )

    # Update the index first
    update_index(args.path, faiss_index)

    # Perform the search
    results = search_repository(args.query, args.hits, faiss_index)

    if not results:
        print("No relevant files found.")
        return

    print(f"\nResults for '{args.query}':")
    for result in results:
        relevance_score = result["similarity"]  # Changed from 'score' to 'similarity'
        path = result["filepath"]
        content_preview = result["content"].replace("\n", " ")[:60]
        if len(result["content"]) > 60:
            content_preview += "..."
        print(f"{path} ({relevance_score:.2f}): {content_preview}")


if __name__ == "__main__":
    main()
