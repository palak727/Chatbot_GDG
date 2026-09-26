# CP Chatbot

A simple Streamlit app for exploring Codeforces problems. It helps students search for problems, filter by tags and rating, browse related material, and get guidance or code feedback through Groq.

## What this project does

- Search Codeforces problems by topic, title, or exact problem ID
- Combine FAISS and BM25 search for hybrid retrieval
- Filter by tags and rating range
- Browse matching problems when no search query is entered
- Show related problems and helpful suggestions
- Review C++ or Python code with Groq

## Tech stack

- Python
- Streamlit
- FAISS
- BM25 ranking
- sentence-transformers embeddings
- Groq API
- Codeforces API and scraped statement pages

## Project structure

```text
CP_chatbot/
├── app.py                  # Streamlit app
├── benchmark.py            # Benchmarking script
├── config.py               # Paths and configuration
├── requirements.txt
├── scrape_problems.py      # Scraper wrapper
├── src/
│   ├── indexer.py          # Build the FAISS index
│   ├── scraper.py          # Codeforces metadata and statement scraper
│   ├── search.py           # Hybrid search logic
│   ├── groq_client.py      # Groq guidance and code review
│   └── utils.py            # Rating and rendering helpers
├── data/
│   ├── problems/           # Problem JSON data
│   └── index/              # FAISS index and metadata
├── tests/                  # Offline search and regression tests
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml         # Local Groq API key
└── README.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Add your Groq API key in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

You can also set it in the environment instead:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

Run the app:

```bash
streamlit run app.py
```

The app is usually available at `http://localhost:8501`.

## Data and indexing

The project includes a pre-built FAISS index.

Fetch problem metadata and statements:

```bash
python -m src.scraper
```

This uses the official Codeforces API for metadata and Codeforces HTML pages for problem statements. The current dataset contains 7000+ valid problems.

Rebuild the index after changing the problem JSON files:

```bash
python -m src.indexer
```

The indexer supports `--problems-dir`, `--model-name`, and `--no-progress`. It writes the index to `data/index/faiss.index` and the metadata to `data/index/metadata.pkl`.

## Search behavior

The search box accepts:

- problem ID such as `1000A`
- topic text such as `shortest path`
- title or concept text such as `dynamic programming`

Search behavior is designed to be practical:

- exact problem IDs are detected first
- filters are applied before ranking during a query
- with no query, selecting tags or a rating range browses matching problems
- with no query and no filters, the app shows an empty-state prompt instead of loading the full dataset
- multiple tags use AND logic
- rating bounds are inclusive

## Benchmarking and tests

Run the benchmark to check index size, load time, memory usage, and search latency:

```bash
python benchmark.py
```

Run the offline tests:

```bash
python -m unittest discover -s tests -v
```

## How to use it

1. Enter a problem ID, topic, or problem concept in the search box.
2. Use the sidebar to narrow results by tag or rating.
3. Pick a result to view the problem and related suggestions.
4. Ask for guidance or review code if needed.

## Deployment

For Streamlit Community Cloud, set the app entry file to `app.py` and configure `GROQ_API_KEY` in the app secrets. Keep the tracked FAISS index available in the repository, and rebuild it if the dataset changes.


