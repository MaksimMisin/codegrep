import subprocess
import requests
import tempfile
import os
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
) -> str:
    """Collect repository content using Repomix for an AI-friendly format."""
    try:
        # Create temporary directory to avoid cluttering user's files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output_path = os.path.join(temp_dir, "repomix-output.txt")

            # cmd = ["repomix", "--compress", f"--output={temp_output_path}"]
            cmd = ["repomix", "--compress"]  # , f"--output={temp_output_path}"]

            # Add ignore patterns if provided
            if ignore_paths:
                glob_patterns = [
                    convert_ignore_path_to_glob(path) for path in ignore_paths
                ]
                ignore_arg = ",".join(glob_patterns)
                cmd.append(f"--ignore={ignore_arg}")

            # Run repomix from the repository path
            logger.info("Generating repository content with Repomix...")
            logger.debug("Command: " + " ".join(cmd))
            result = subprocess.run(
                cmd, cwd=str(repo_path), capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Repomix failed with error: {result.stderr}")
                logger.info("Falling back to original content collection method")
                content = collect_files_content_manual(files)
                if debug:
                    _save_debug_file(repo_path, "codegrep-manual-content.txt", content)
                return content
            # Read the generated output
            with open(temp_output_path, "r", encoding="utf-8") as f:
                repo_content = f.read()

            # Save debug file if debug mode is enabled
            if debug:
                _save_debug_file(repo_path, "codegrep-repomix-output.txt", repo_content)

            logger.debug("Successfully generated repository content with Repomix")
            return repo_content

    except Exception as e:
        raise e
        # logger.error(f"Error using Repomix: {e}")
        # logger.info("Falling back to original content collection method")
        # content = collect_files_content_manual(files)
        # if debug:
        #     _save_debug_file(repo_path, "codegrep-manual-content.txt", content)
        # return content


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


def search_with_llm(
    repo_path: Path,
    query: str,
    n_files: int,
    files: List[Tuple[str, str]],
    ignore_paths: Optional[List[str]] = None,
    debug: bool = False,
) -> List[str]:
    """Search for relevant files using LLM APIs with Gemini as primary and OpenAI as backup."""
    # Collect repository content
    repo_content = collect_repo_files_content(repo_path, files, ignore_paths, debug)

    # Try Gemini first
    results = search_with_gemini(repo_content, query, n_files, repo_path, debug)

    # Fall back to OpenAI if Gemini fails
    if results is None:
        logger.info("Falling back to OpenAI for LLM search")
        results = search_with_openai(repo_content, query, n_files, repo_path, debug)

    # If both APIs fail, return an empty list
    if results is None:
        logger.error("LLM search failed with both Gemini and OpenAI")
        return []

    return results
