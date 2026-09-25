"""CP Chatbot — Production Streamlit application."""

from __future__ import annotations

import os

import streamlit as st

from config import FAISS_INDEX_PATH, METADATA_PATH
from src.groq_client import generate_hint, get_api_key, review_code
from src.search import SearchEngine, is_problem_id_query
from src.utils import extract_rating, render_problem_markdown, tag_pills_html

# Page configuration
st.set_page_config(
    page_title="CP Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Main Header */
        .title-text {
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .sub-text {
            color: var(--text-color-secondary, #6b7280);
            font-size: 0.98rem;
            margin-bottom: 1.5rem;
        }

        /* Metric Cards */
        .metric-card {
            background-color: var(--secondary-background-color, #f8f9fa);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-color-secondary, #6b7280);
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
        }

        /* Related Problem Items */
        .related-card {
            background-color: var(--secondary-background-color, #f8f9fa);
            border-left: 3px solid #4f46e5;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.92rem;
        }

        /* Hint Output Container */
        .hint-box {
            background-color: var(--secondary-background-color, #f8f9fa);
            border-left: 4px solid #4f46e5;
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
            line-height: 1.6;
        }

        /* Code Input Area */
        .stTextArea textarea {
            font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading search index...")
def get_search_engine() -> SearchEngine:
    """Load pre-built FAISS index from disk."""
    engine = SearchEngine()
    engine.load()
    return engine


def render_metrics(problem: dict) -> None:
    """Display problem time, memory limits, and difficulty rating in styled cards."""
    rating = problem.get("rating") or extract_rating(problem.get("tags", []))
    cols = st.columns(3)

    with cols[0]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Time Limit</div>'
            f'<div class="metric-value">{problem.get("time_limit") or "—"}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Memory Limit</div>'
            f'<div class="metric-value">{problem.get("memory_limit") or "—"}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Difficulty</div>'
            f'<div class="metric-value">{rating or "—"}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_problem(problem: dict, engine: SearchEngine) -> None:
    """Render full problem view with metrics, markdown statement, hints, and code review."""
    pid = problem["id"]  # Unique identifier for dynamic widget keys

    title_link = f"https://codeforces.com/problemset/problem/{problem['contest_id']}/{problem['problem_letter']}"
    st.markdown(f"## [{problem['id']}]({title_link}) — {problem['title']}")

    tags_html = tag_pills_html(problem.get("tags", []))
    st.markdown(tags_html, unsafe_allow_html=True)
    st.markdown("")

    render_metrics(problem)
    st.markdown("")

    st.markdown(render_problem_markdown(problem))

    # Related Problems Section
    st.divider()
    st.subheader("Related Problems")
    try:
        similar = engine.get_similar(problem, k=3)
        if similar:
            for sp in similar:
                sp_rating = sp.get("rating") or extract_rating(sp.get("tags", []))
                rating_str = f" · Rating: {sp_rating}" if sp_rating else ""
                st.markdown(
                    f'<div class="related-card">'
                    f'<strong>{sp["id"]}</strong> — {sp["title"]}{rating_str}'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No similar problems found in the current index.")
    except Exception as exc:
        st.warning(f"Could not load related problems: {exc}")

    #  Hint Engine
    st.divider()
    st.subheader("CP Learning Assistant")

    if not get_api_key():
        st.warning(
            "Groq API key not configured. Add `GROQ_API_KEY` to `.streamlit/secrets.toml` "
            "or set the `GROQ_API_KEY` environment variable."
        )
    else:
        hint_level = st.radio(
            "Choose guidance level:",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Level 1 — Conceptual Intuition",
                2: "Level 2 — Algorithmic Approach",
                3: "Level 3 — Full Solution & Complexity",
            }[x],
            horizontal=True,
            key=f"hint_level_{pid}",  # Fixed key
        )
        if st.button("Generate Guidance", type="primary", key=f"hint_btn_{pid}"):  # Fixed key
            with st.spinner("Analyzing problem..."):
                hint_text = generate_hint(problem, hint_level)
            st.markdown(f'<div class="hint-box">{hint_text}</div>', unsafe_allow_html=True)

    # Code Review Section
    st.divider()
    st.subheader("Code Review Scratchpad")
    lang = st.selectbox("Language", ["cpp", "python"], key=f"code_lang_{pid}")  # Fixed key
    placeholder_code = (
        "#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    // solution\n    return 0;\n}"
        if lang == "cpp"
        else "def solve():\n    pass\n\nif __name__ == '__main__':\n    solve()"
    )

    code = st.text_area(
        "Paste code for automated review:",
        height=220,
        placeholder=placeholder_code,
        key=f"code_input_{pid}",  # Fixed key
    )

    if st.button("Review Code", key=f"review_btn_{pid}"):  # Fixed key
        if not code.strip():
            st.warning("Please enter code before requesting a review.")
        else:
            with st.spinner("Reviewing solution..."):
                review = review_code(problem, code, lang)
            st.markdown(review)


def main() -> None:
    st.markdown('<div class="title-text">Competitive Programming Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">Find the right problem, get guidance, and review your solution.</div>',
        unsafe_allow_html=True,
    )

    # Index availability check
    if not os.path.isfile(FAISS_INDEX_PATH) or not os.path.isfile(METADATA_PATH):
        st.error(
            "Search index files missing. Run indexer script to build the vector index:\n\n"
            "```bash\npython -m src.indexer\n```"
        )
        st.stop()

    try:
        engine = get_search_engine()
    except Exception as exc:
        st.error(f"Failed to load search index: {exc}")
        st.stop()

    # Sidebar Filters
    with st.sidebar:
        st.subheader("Search & Filters")
        all_tags = engine.all_tags
        selected_tags = st.multiselect(
            "Filter by Tag",
            options=all_tags,
            help="A problem must contain every selected tag.",
        )

        rating_range = st.slider(
            "Difficulty Rating Range",
            min_value=800,
            max_value=3500,
            value=(800, 3500),
            step=100,
        )

        st.divider()
        filter_count = engine.filter_count(
            selected_tags or None,
            rating_range[0],
            rating_range[1],
        )
        filters_active = bool(selected_tags) or rating_range != (800, 3500)
        if filters_active:
            st.caption(f"{filter_count} problems match your filters")
        st.caption(f"Index status: {len(engine.problems)} problems loaded")
        if get_api_key():
            st.caption("Groq API: Connected")
        else:
            st.caption("Groq API: Missing key")

    # Main Search Input
    query = st.text_input(
        "Search Problems",
        placeholder="Search by topic, title, or problem ID...",
    )

    if query.strip():
        results = engine.search(
            query,
            tags=selected_tags or None,
            rating_min=rating_range[0],
            rating_max=rating_range[1],
            k=5,
        )

        if not results:
            if is_problem_id_query(query) and engine.find_problem_by_id(query):
                st.warning(f"Problem {query.strip().upper()} exists, but does not match the selected filters.")
            else:
                st.warning("No matching problems found. Try adjusting query or sidebar filters.")
        elif len(results) == 1:
            render_problem(results[0], engine)
        else:
            st.subheader(f"Search Results ({len(results)} matches)")
            for i, prob in enumerate(results):
                rating = prob.get("rating") or extract_rating(prob.get("tags", []))
                rating_label = f" Rating: {rating}" if rating else ""
                with st.expander(f"{prob['id']} — {prob['title']} ({rating_label.strip()})"):
                    render_problem(prob, engine)
    elif filters_active:
        results = engine.search(
            "",
            tags=selected_tags or None,
            rating_min=rating_range[0],
            rating_max=rating_range[1],
            k=5,
        )
        if not results:
            st.info("No problems match your current filters.")
        else:
            st.subheader(f"Browse Results ({filter_count} matches)")
            for prob in results:
                rating = prob.get("rating") or extract_rating(prob.get("tags", []))
                rating_label = f" Rating: {rating}" if rating else ""
                with st.expander(f"{prob['id']} — {prob['title']} ({rating_label.strip()})"):
                    render_problem(prob, engine)
    else:
        st.info("Search for a problem or use filters to browse the collection.")


if __name__ == "__main__":
    main()