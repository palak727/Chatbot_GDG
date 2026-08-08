import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROBLEMS_DIR = os.path.join(DATA_DIR, "problems")
INDEX_DIR = os.path.join(DATA_DIR, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

os.makedirs(PROBLEMS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
