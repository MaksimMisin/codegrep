# codegrep

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
GEMINI_API_KEY=your_gemini_api_key_here // only if you want to use LLM-based search
```

## LLM-based Search

For improved code search quality, codegrep can use Repomix to package your code in an AI-friendly format.

### Prerequisites
Install Repomix using npm:
```bash
npm install -g repomix
```

## Usage

```bash
# Search for code related to "email sending" and get top 10 results
codegrep -q "email sending" -n 10

# Search in a specific directory
codegrep -q "database connection" -p /path/to/repo -n 5
```
