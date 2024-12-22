# Code Grep

A semantic code search on top of OpenAI embeddings to find relevant code in a Git repository.

## Installation

```bash
git remote add origin git@github.com:MaksimMisin/codegrep.git
pip install .
```

## Configuration

Before using codegrep, you need to set up your OpenAI API key. Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=your_embedding_model_here
```

## Usage

```bash
# Search for code related to "email sending" and get top 10 results
codegrep -q "email sending" -n 10

# Search in a specific directory
codegrep -q "database connection" -p /path/to/repo -n 5
```
