import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import fnmatch

from codegrep.index import FAISSIndex
from codegrep.config import IGNORE_PATHS, IGNORE_EXTENSIONS

# Configure logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.WARNING)


def should_ignore_file(
    file_path: str, custom_ignore_paths: Optional[List[str]] = None
) -> bool:
    """
    Check if a file should be ignored based on its path or extension.

    Args:
        file_path: Path to the file to check
        custom_ignore_paths: Optional list of additional paths to ignore

    Returns:
        bool: True if the file should be ignored, False otherwise
    """
    path = Path(file_path)

    # Check custom ignore paths first
    if custom_ignore_paths:
        for ignore_path in custom_ignore_paths:
            if ignore_path in str(path):
                return True

    # Check file extension
    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True

    # Check against ignore patterns
    patterns = list(IGNORE_PATHS)

    for pattern in patterns:
        if fnmatch.fnmatch(str(path), pattern) or fnmatch.fnmatch(path.name, pattern):
            return True

    return False


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
        logger.error(f"Git command failed: {e}")
        logger.debug(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error running git command: {e}")
        return None


def collect_repository_files(
    repo_path: Path,
    custom_ignore_paths: Optional[List[str]] = None,
    dry_run: bool = False,
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
        logger.error(f"Error: {repo_path} is not a git repository")
        return []

    # Get list of tracked files using git ls-files
    output = run_git_command(repo_path, ["ls-files"])
    if not output:
        logger.warning("No files found in git repository")
        return []

    current_files = []
    files_processed = 0
    files_ignored = 0

    if dry_run:
        logger.info("\nDry run - analyzing repository files:")
        logger.info("=====================================")

    # Process each file path
    for rel_path in output.split("\n"):
        if not rel_path.strip():  # Skip empty lines
            continue

        try:
            file_path = repo_path / rel_path
            if file_path.is_file():  # Verify file exists
                # Check if file should be ignored
                if should_ignore_file(str(file_path), custom_ignore_paths):
                    files_ignored += 1
                    if dry_run:
                        logger.debug(f"Ignoring: {rel_path}")
                    continue

                abs_path = file_path.resolve()
                current_files.append((str(abs_path), rel_path))
                files_processed += 1

                if dry_run:
                    logger.info(f"Found: {rel_path}")
            else:
                if dry_run:
                    logger.warning(f"Warning: Tracked file not found: {rel_path}")

        except Exception as e:
            if dry_run:
                logger.error(f"Could not process {rel_path}: {e}")
            continue

    if dry_run:
        logger.info(
            f"Found {files_processed} files to process ({files_ignored} files ignored)"
        )
        logger.info("=====================================")

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
    indexed_files = set(index.indexed_files(absolute_paths=True))
    removed_files = indexed_files - current_abs_paths

    for filepath in removed_files:
        logger.info(f"Removing deleted file from index: {filepath}")
        index.remove_file_from_index(filepath)


def process_single_file(
    index: FAISSIndex,
    abs_path: str,
    rel_path: str,
    custom_ignore_paths: Optional[List[str]] = None,
) -> bool:
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
        # Skip ignored files
        if should_ignore_file(abs_path, custom_ignore_paths):
            logger.debug(f"Skipping ignored file: {rel_path}")
            return False

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove any existing entries for this file before adding new ones
        index.remove_file_from_index(rel_path)

        if not content:
            logger.warning(f"Empty file - {rel_path}")
            return False

        # Add the processed content to the index
        if index.add_to_index(content, str(rel_path)):
            logger.debug(f"Updated: {rel_path}")
            return True
        else:
            logger.warning(f"Failed to add {rel_path} to index")
            return False

    except Exception as e:
        logger.error(f"Error processing {rel_path}: {e}")
        return False


def update_index(
    repo_path: Path,
    index: FAISSIndex,
    custom_ignore_paths: Optional[List[str]] = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> None:
    """
    Update the FAISS index for the given repository path.

    Args:
        repo_path: Path to the repository root
        index: FAISSIndex instance to update
        dry_run: If True, only print what would be done without making changes
        quiet: If True, suppress non-error output
    """
    if not quiet:
        if dry_run:
            logger.info("Dry run mode - no changes will be made to the index")
        else:
            logger.info("Checking for files to update...")

    files_updated = 0
    repo_path = repo_path.resolve()
    current_files = collect_repository_files(repo_path, custom_ignore_paths, dry_run)

    if not dry_run:
        remove_deleted_files(index, current_files)
        for abs_path, rel_path in current_files:
            if index.file_needs_update(rel_path):
                if process_single_file(index, abs_path, rel_path):
                    files_updated += 1

        index.save_index()
        if not quiet:
            logger.info(
                f"Index update complete. Updated {files_updated} out of {len(current_files)} files."
            )


def main() -> None:
    """Main entry point for the codegrep CLI."""
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
    search_group.add_argument(
        "--files-only",
        action="store_true",
        help="Only display matching filenames, one per line",
    )
    search_group.add_argument(
        "--ignore-path",
        type=str,
        action="append",
        help="Ignore files/directories containing this path (can be specified multiple times)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-error output"
    )

    args = parser.parse_args()

    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if not args.path.is_dir():
        logger.error(f"Error: {args.path} is not a valid directory")
        sys.exit(1)

    # Check if path is a git repository
    if not run_git_command(args.path, ["rev-parse", "--git-dir"]):
        logger.error(f"Error: {args.path} is not a git repository")
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

    # Update the index first (quiet if --files-only is specified)
    update_index(
        args.path,
        faiss_index,
        custom_ignore_paths=args.ignore_path,
        dry_run=args.dry_run,
        quiet=args.files_only,
    )

    # Perform the search
    results = faiss_index.search(args.query, k=args.hits)

    if args.files_only:
        # Only print filenames, no other output
        for result in results:
            print(result.filepath)
    else:
        logger.info(f"\nResults for '{args.query}':")
        for result in results:
            relevance_score = result.relevance
            path = result.filepath
            content_preview = result.content.replace("\n", " ")[:60]
            if len(result.content) > 60:
                content_preview += "..."
            print(f"{path} ({relevance_score:.2f}): {content_preview}")


if __name__ == "__main__":
    main()
