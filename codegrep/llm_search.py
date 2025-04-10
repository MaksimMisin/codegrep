import subprocess
import requests
import tempfile
from typing import List, Optional, Tuple, cast
from pathlib import Path
from json_repair import loads as repair_json
from openai import OpenAI
from codegrep.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GPT4O_MINI_MODEL,
)
from codegrep.logging import get_logger

logger = get_logger()


def validate_file_paths(
    repo_path: Path, file_paths: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Validate if the file paths returned by LLM exist in the repository.

    Args:
        repo_path: Repository root path
        file_paths: List of file paths to validate

    Returns:
        Tuple containing (valid_paths, invalid_paths)
    """
    valid_paths = []
    invalid_paths = []

    for path in file_paths:
        full_path = repo_path / path
        if full_path.is_file():
            valid_paths.append(path)
        else:
            invalid_paths.append(path)
            logger.warning(f"LLM suggested nonexistent file: {path}")

    return valid_paths, invalid_paths


def create_retry_prompt(
    repo_content: str,
    query: str,
    n_files: int,
    valid_paths: List[str],
    invalid_paths: List[str],
    all_valid_files: List[str],
) -> str:
    """Create a prompt for the LLM to retry with corrected file paths."""
    # Sample of valid files to help the LLM
    sample_paths = (
        all_valid_files[:25000] if len(all_valid_files) > 25000 else all_valid_files
    )

    return f"""{repo_content}

CORRECTION NEEDED: Your previous file path suggestions contained errors.

Original query: "{query}"

Valid paths you suggested:
{', '.join(valid_paths) if valid_paths else 'None'}

Invalid paths you suggested (these files DO NOT EXIST):
{', '.join(invalid_paths)}

Here are valid file paths in this repository:
{', '.join(sample_paths)}

Please provide a CORRECTED list of files that users need to read to resolve: "{query}".

List in order of importance, starting from the MOST important file.
Do not forget important dependencies.
IMPORTANT: ONLY include files that actually exist in the repository.

Your output should be a json object with the following structure:
{{
  "reasoning": "think through the user's request and how each file might be related to it",
  "files": ["MOST-CRITICAL-file-path1", "2nd-most-critical-file-path2", ...]
}}
Make sure to include at least {n_files} files if possible, but ONLY include files that actually exist.
"""


def _save_debug_file(repo_path: Path, filename: str, content: str) -> None:
    """Save content to a debug file in the repo root."""
    debug_path = repo_path / filename
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Debug file saved: {debug_path}")
    except Exception as e:
        logger.error(f"Failed to save debug file {debug_path}: {e}")


def create_llm_prompt(repo_files_content: str, query: str, n_files: int) -> str:
    return f"""{repo_files_content}

List all files that users needs to read to resolve: "{query}".

List in order of importance, starting from the MOST important file.
Do not foget important dependencies.
Your output should be a json object with the following structure:
{{
  "reasoning": "think through the users's request and how each file might be related to it",
  "files": ["MOST-CRITICAL-file-path1", "2nd-most-critical-file-path2", ...] //list of all file paths
}}
Make sure to include at least {n_files} files."""


def convert_ignore_path_to_glob(ignore_path: str) -> str:
    """Convert simple ignore paths to glob patterns for Repomix."""
    # If it already looks like a glob pattern, return it as is
    if "*" in ignore_path or "?" in ignore_path:
        return ignore_path

    # If it ends with a slash, treat it as a directory
    if ignore_path.endswith("/"):
        return f"**/{ignore_path}**"

    # If it starts with a dot, treat it as an extension
    if ignore_path.startswith("."):
        return f"**/*{ignore_path}"

    # Otherwise, make it a general path glob
    return f"**/{ignore_path}/**"


def collect_repo_files_content(
    repo_path: Path,
    files: List[Tuple[str, str]],
    ignore_paths: Optional[List[str]] = None,
    debug: bool = False,
    dry_run: bool = False,  # New parameter
) -> str:
    """Collect repository content using Repomix for an AI-friendly format."""
    temp_dir = None

    try:
        # Determine output path based on mode
        if dry_run:
            # In dry-run mode, save to current directory
            output_path = Path.cwd() / "repomix-output.txt"
            logger.info(f"Dry run: Saving repomix output to {output_path}")
        else:
            # In normal mode, create a temporary directory
            temp_dir = tempfile.mkdtemp()
            output_path = Path(temp_dir) / "repomix-output.txt"

        cmd = ["repomix", "--compress", f"--output={output_path}"]

        # Add ignore patterns if provided
        if ignore_paths:
            glob_patterns = [convert_ignore_path_to_glob(path) for path in ignore_paths]
            ignore_arg = ",".join(glob_patterns)
            cmd.append(f"--ignore={ignore_arg}")

        # Run repomix from the repository path
        logger.info("Generating repository content with Repomix...")
        logger.debug("Command: " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Repomix failed with error: {result.stderr}")
            logger.info("Falling back to original content collection method")
            content = collect_files_content_manual(files)
            if debug:
                _save_debug_file(repo_path, "codegrep-manual-content.txt", content)
            return content

        # Read the generated output
        with open(output_path, "r", encoding="utf-8") as f:
            repo_content = f.read()

        # Save debug file if debug mode is enabled
        if debug:
            _save_debug_file(repo_path, "codegrep-repomix-output.txt", repo_content)

        logger.debug("Successfully generated repository content with Repomix")

        if dry_run:
            logger.info(f"Repomix output saved to {output_path}")

        return repo_content

    except Exception as e:
        logger.error(f"Error using Repomix: {e}")
        logger.info("Falling back to original content collection method")
        content = collect_files_content_manual(files)
        if debug:
            _save_debug_file(repo_path, "codegrep-manual-content.txt", content)
        return content

    finally:
        # Clean up temporary directory if it was created and not in dry run mode
        if temp_dir and not dry_run:
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")


def collect_files_content_manual(files: List[Tuple[str, str]]) -> str:
    """Manually collect content from files without using Repomix."""
    logger.info("Collecting file contents manually...")
    result = []

    # Add a header
    result.append("# Repository Files\n")

    for abs_path, rel_path in files:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Add file header and content
            result.append(f"\n## File: {rel_path}\n")
            result.append("```")
            result.append(content)
            result.append("```\n")

        except Exception as e:
            logger.error(f"Error reading file {rel_path}: {e}")

    return "\n".join(result)


def search_with_gemini(
    repo_files_content: str,
    query: str,
    n_files: int,
    repo_path: Optional[Path] = None,
    debug: bool = False,
) -> Optional[List[str]]:
    """Search for relevant files using Google's Gemini API."""
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not found in environment variables")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    prompt = create_llm_prompt(repo_files_content, query, n_files)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
            "responseMimeType": "text/plain",
        },
    }

    try:
        response = requests.post(
            api_url, headers={"Content-Type": "application/json"}, json=payload
        )

        if response.status_code == 200:
            content = response.json()
            text = (
                content.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            if debug and repo_path:
                _save_debug_file(repo_path, "codegrep-gemini-response.json", text)

            try:
                result = cast(dict, repair_json(text))
                return result.get("files", [])
            except Exception as e:
                logger.error(f"Error parsing Gemini response as JSON: {e}")
                logger.debug(f"Raw response: {text}")
                return None
        else:
            logger.error(
                f"Gemini API request failed with status code {response.status_code}"
            )
            return None

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None


def search_with_openai(
    repo_files_content: str,
    query: str,
    n_files: int,
    repo_path: Optional[Path] = None,
    debug: bool = False,
) -> Optional[List[str]]:
    """Search for relevant files using OpenAI's GPT-4o-mini as a fallback."""
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found in environment variables")
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = create_llm_prompt(repo_files_content, query, n_files)
    try:
        response = client.chat.completions.create(
            model=GPT4O_MINI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a code search assistant that finds relevant files in a repository.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=8192,
        )

        text = response.choices[0].message.content
        if debug and repo_path:
            _save_debug_file(
                repo_path, "codegrep-openai-response.json", text or "EMPTY RESPONSE"
            )

        try:
            result = cast(dict, repair_json(text))
            return result.get("files", [])
        except Exception as e:
            logger.error(f"Error parsing OpenAI response as JSON: {e}")
            return None

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None


def search_with_llm_and_validate(
    repo_path: Path,
    query: str,
    n_files: int,
    files: List[Tuple[str, str]],
    ignore_paths: Optional[List[str]] = None,
    debug: bool = False,
    max_retries: int = 5,
) -> List[str]:
    """Search for relevant files using LLM APIs with validation and retry mechanism."""
    # Collect repository content
    repo_content = collect_repo_files_content(repo_path, files, ignore_paths, debug)

    # Get all valid file paths in the repo for validation feedback
    all_valid_files = [rel_path for _, rel_path in files]

    # Initial search with LLM
    results = search_with_gemini(repo_content, query, n_files, repo_path, debug)

    # Fall back to OpenAI if Gemini fails
    if results is None:
        logger.info("Falling back to OpenAI for LLM search")
        results = search_with_openai(repo_content, query, n_files, repo_path, debug)

    # If both APIs fail, return empty list
    if results is None:
        logger.error("LLM search failed with both Gemini and OpenAI")
        return []

    # Validate the returned paths
    valid_paths, invalid_paths = validate_file_paths(repo_path, results)

    if not invalid_paths:
        logger.info("All file paths returned by LLM are valid")
        return valid_paths

    # Retry loop if there are invalid paths
    retries = 0
    while invalid_paths and retries < max_retries:
        retries += 1
        logger.info(
            f"Retry #{retries}: Found {len(invalid_paths)} invalid paths. Asking LLM to fix."
        )

        # Create a retry prompt
        retry_prompt = create_retry_prompt(
            repo_content, query, n_files, valid_paths, invalid_paths, all_valid_files
        )

        # Try with Gemini first
        retry_results = search_with_gemini(retry_prompt, n_files, repo_path, debug)

        # Fall back to OpenAI if Gemini fails
        if retry_results is None:
            logger.info(f"Retry #{retries}: Falling back to OpenAI")
            retry_results = search_with_openai(retry_prompt, n_files, repo_path, debug)

        # If both APIs fail, break and return what we have
        if retry_results is None:
            logger.error(f"LLM retry #{retries} failed with both APIs")
            break

        # Save debug info if enabled
        if debug and repo_path:
            _save_debug_file(
                repo_path, f"codegrep-retry-{retries}-results.json", str(retry_results)
            )

        # Validate the new results
        new_valid_paths, new_invalid_paths = validate_file_paths(
            repo_path, retry_results
        )

        # Update our lists
        valid_paths = list(set(valid_paths + new_valid_paths))
        invalid_paths = new_invalid_paths

        # If no improvement after retry, break to avoid wasting API calls
        if not new_valid_paths:
            logger.warning(
                f"Retry #{retries} produced no valid paths. Using previous results."
            )
            break

        logger.info(
            f"Retry #{retries}: Now have {len(valid_paths)} valid paths and {len(invalid_paths)} invalid paths"
        )

    if retries >= max_retries and invalid_paths:
        logger.warning(
            f"Reached maximum retries ({max_retries}). Returning {len(valid_paths)} valid paths found."
        )

    return valid_paths
