# CP Chatbot

A production-ready Streamlit web application for competitive programming students. Search Codeforces problems with semantic vector search (FAISS), get Socratic hints powered by Gemini, find related practice problems, and receive AI code reviews.

## Features

- **Pre-computed FAISS index** — embeddings built offline; the app loads from disk instantly
- **Codeforces API ingestion** — metadata, tags, and ratings via official API; HTML scraping for statements
- **Semantic & exact search** — natural language queries or direct problem ID lookup (e.g. `1000A`)
- **Sidebar filters** — filter by tags and difficulty rating (800–2400)
- **Socratic hint engine** — 3-level guidance (intuition → algorithm → full solution)
- **Related problems** — FAISS similarity search with rating-matched recommendations
- **Code scratchpad** — paste C++/Python code for Gemini-powered review

## Project Structure

```
CP_chatbot/
├── app.py                  # Streamlit entry point
├── config.py               # Paths and constants
├── requirements.txt
├── src/
│   ├── indexer.py          # Build & save FAISS index (run offline)
│   ├── scraper.py          # Codeforces API + HTML scraper
│   ├── search.py           # Search engine (loads index from disk)
│   ├── gemini_client.py    # Hints, solutions, code review
│   └── utils.py            # LaTeX formatting, rating extraction
├── data/
│   ├── problems/           # Problem JSON files (gitignored)
│   └── index/              # faiss.index + metadata.pkl (gitignored)
└── .streamlit/
    ├── config.toml         # Theme
    └── secrets.toml        # API keys (gitignored)
```

## Quick Start (Local)

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd CP_chatbot
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API key

Create `.streamlit/secrets.toml`:

```toml
API_KEY = "your-gemini-api-key"
```

Or set an environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Ingest problems (optional if JSON files already exist)

```bash
python -m src.scraper
```

This fetches metadata from the Codeforces API and scrapes statements (1 req/sec rate limit).

### 4. Build the search index (required before first run)

```bash
python -m src.indexer
```

Creates `data/index/faiss.index` and `data/index/metadata.pkl`.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Deploy to Streamlit Community Cloud

1. Push your repo to GitHub (exclude large files — they are in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app.py`.
4. Add secrets in the Cloud dashboard:

   ```toml
   API_KEY = "your-gemini-api-key"
   ```

5. **Important:** The FAISS index must exist in the repo or be built at deploy time. Options:
   - Commit `data/index/faiss.index` and `data/index/metadata.pkl` (remove from `.gitignore` temporarily), or
   - Add a build step in `packages.txt` / custom script, or
   - Include problem JSONs and run `python -m src.indexer` in a pre-deploy hook.

6. For environment-variable fallback, set `GEMINI_API_KEY` in Cloud secrets or env vars.

## Deploy to Render / Hugging Face Spaces

Same steps as above. Set `GEMINI_API_KEY` as an environment variable. Run the indexer locally and commit the index artifacts, or build them in a CI step before deploy.

## Usage Tips

| Action | How |
|--------|-----|
| Find by ID | Switch to **Exact Problem ID** mode, type `1000A` |
| Topic search | **Semantic Search** + query like `segment tree` |
| Filter by tag | Select tags in sidebar (e.g. `dp`, `graphs`) |
| Get hints | Open a problem → choose hint level → **Generate Hint** |
| Code review | Paste code in scratchpad → **Review My Code** |

## Rebuilding the Index

Re-run after adding or updating problem JSON files:

```bash
python -m src.indexer
```

## License

MIT
