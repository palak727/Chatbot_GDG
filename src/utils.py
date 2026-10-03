"""Shared utilities for problem parsing and LaTeX formatting."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


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


def build_embedding_text(
    problem: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 256,
) -> str:
    """Build title/tag-first text capped to the embedding model's token budget."""

    title_value = problem.get("title", "")
    title = str(title_value).strip() if title_value is not None else ""

    raw_tags = problem.get("tags", [])
    if isinstance(raw_tags, list):
        tags = ", ".join(
            str(tag).strip()
            for tag in raw_tags
            if isinstance(tag, str) and not tag.startswith("*")
        )
    else:
        tags = ""

    header_parts: list[str] = []

    if title:
        header_parts.append(f"Title: {title}")

    if tags:
        header_parts.append(f"Tags: {tags}")

    header = " ".join(header_parts)

    header_ids = tokenizer.encode(
        header,
        add_special_tokens=True,
    )

    if len(header_ids) > max_tokens:
        raise ValueError(
            "Problem title and tags exceed the embedding token budget."
        )

    body_parts: list[str] = []

    for label, key in (
        ("Statement", "statement"),
        ("Input", "input"),
        ("Output", "output"),
    ):
        value = problem.get(key, "")

        if value is None:
            continue

        value_str = str(value).strip()

        if value_str:
            body_parts.append(f"{label}: {value_str}")

    body = " ".join(body_parts)

    body_budget = max_tokens - len(header_ids)

    if body and body_budget > 0:
        body_ids = tokenizer.encode(
            body,
            add_special_tokens=False,
            truncation=True,
            max_length=body_budget,
        )
    else:
        body_ids = []

    body_text: str = str(
    tokenizer.decode(
        body_ids,
        skip_special_tokens=True,
    )
)

    text_parts: list[str] = []

    if header:
        text_parts.append(header)

    if body_text:
        text_parts.append(body_text)

    text = " ".join(text_parts)

    # Decoding token IDs can slightly change tokenization/spacing.
    # Make sure the final text still fits the embedding model limit.
    while len(tokenizer.encode(text, add_special_tokens=True)) > max_tokens:
        if not body_ids:
            raise ValueError(
                "Embedding text exceeds the token budget after decoding."
            )

        body_ids = body_ids[:-1]

        body_text = str(
    tokenizer.decode(
        body_ids,
        skip_special_tokens=True,
    )
)

        text_parts = []

        if header:
            text_parts.append(header)

        if body_text:
            text_parts.append(body_text)

        text = " ".join(text_parts)

    return text


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
