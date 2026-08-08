"""Shared utilities for problem parsing and LaTeX formatting."""

from __future__ import annotations

import re
from typing import Any


def extract_rating(tags: list[str]) -> int | None:
    """Extract numeric difficulty rating from Codeforces tags (e.g. '*1200')."""
    for tag in tags:
        match = re.match(r"\*(\d+)", tag.strip())
        if match:
            return int(match.group(1))
    return None


def format_latex(text: str) -> str:
    """Normalize MathJax/LaTeX for KaTeX-friendly Streamlit rendering."""
    if not text:
        return ""

    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = re.sub(r"\\le\b", r"\\leq", text)
    text = re.sub(r"\\ge\b", r"\\geq", text)
    text = re.sub(r"\\neq\b", r"\\neq", text)
    text = re.sub(r"\\times\b", r"\\times", text)

    # Convert display math blocks
    text = re.sub(r"\$\$\$(.*?)\$\$\$", r"$$\1$$", text, flags=re.DOTALL)

    # Wrap bare LaTeX commands in inline math delimiters
    def _wrap_inline(match: re.Match[str]) -> str:
        fragment = match.group(0)
        if fragment.startswith("$"):
            return fragment
        return f"${fragment}$"

    text = re.sub(
        r"(?<!\$)(\\(?:leq|geq|neq|times|cdot|sum|prod|sqrt|frac|log|min|max)\b[^$]*)",
        _wrap_inline,
        text,
    )

    # Clean orphaned backslash commands that aren't math
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", text)

    return text.strip()


def build_embedding_text(problem: dict[str, Any]) -> str:
    """Build rich text for embedding a problem."""
    tags = ", ".join(problem.get("tags", []))
    parts = [
        problem.get("title", ""),
        problem.get("statement", ""),
        problem.get("input", ""),
        problem.get("output", ""),
        tags,
    ]
    return " ".join(p for p in parts if p)


def render_problem_markdown(problem: dict[str, Any]) -> str:
    """Render a problem statement as markdown with LaTeX support."""
    sections = [
        f"### {problem.get('title', 'Untitled')}",
        format_latex(problem.get("statement", "")),
    ]
    if problem.get("input"):
        sections.append(f"**Input**\n\n{format_latex(problem['input'])}")
    if problem.get("output"):
        sections.append(f"**Output**\n\n{format_latex(problem['output'])}")
    return "\n\n".join(sections)


def tag_pills_html(tags: list[str]) -> str:
    """Generate HTML pill tags for topic display."""
    pills = []
    for tag in tags:
        css = "rating-pill" if tag.startswith("*") else "topic-pill"
        pills.append(f'<span class="{css}">{tag}</span>')
    return " ".join(pills)
