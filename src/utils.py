"""Shared utilities for problem parsing and LaTeX formatting."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def extract_rating(tags: list[str]) -> int | None:
    """Extract numeric Codeforces rating from tags."""
    for tag in tags:
        m = re.match(r"^\*(\d+)$", tag.strip())
        if m:
            return int(m.group(1))
    return None


def format_latex(text: str) -> str:
    """Normalize MathJax/LaTeX for KaTeX rendering."""
    if not text:
        return ""
    text = (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )

    # Fix common HTML-encoded LaTeX commands.
    for cmd in ("le", "ge", "neq", "times"):
        text = re.sub(
            rf"\\&#x{ord(cmd[0]):02X};{cmd[1:]}\b",
            rf"\\{cmd}",
            text,
            flags=re.IGNORECASE,
        )

    # Normalize display math.
    text = re.sub(r"\$\$\$(.*?)\$\$\$", r"$$\1$$", text, flags=re.DOTALL)

    # Wrap common bare LaTeX commands in inline math.
    def wrap(match: re.Match[str]) -> str:
        s = match.group(0)
        return s if s.startswith("$") else f"${s}$"

    text = re.sub(
        r"(?<!\$)\\(?:leq|geq|neq|times|cdot|sum|prod|sqrt|frac|log|min|max)\b[^$]*",
        wrap,
        text,
    )

    # Remove simple formatting commands.
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", text)

    return text.strip()


def _clean_text(value: Any) -> str:
    """Convert a value safely to a stripped string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if x is not None).strip()
    return str(value).strip()


def build_embedding_text(
    problem: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Build one or more overlapping embedding chunks."""

    title = _clean_text(problem.get("title", ""))
    raw_tags = problem.get("tags", [])

    if isinstance(raw_tags, list):
        tags_list = [
            t.strip() for t in raw_tags
            if isinstance(t, str) and t.strip() and not t.strip().startswith("*")
        ]
    elif isinstance(raw_tags, str):
        tags_list = [
            t.strip() for t in raw_tags.split(",")
            if t.strip() and not t.strip().startswith("*")
        ]
    else:
        tags_list = []

    tags = ", ".join(tags_list)
    rating = problem.get("rating")

    if rating is None and isinstance(raw_tags, list):
        rating = extract_rating([t for t in raw_tags if isinstance(t, str)])

    header_parts = []
    if title:
        header_parts.append(f"Title: {title}")
    if tags:
        header_parts.append(f"Tags: {tags}")
    if rating is not None:
        header_parts.append(f"Rating: {_clean_text(rating)}")

    header = "\n".join(header_parts)
    header_ids = tokenizer.encode(header, add_special_tokens=True)

    if len(header_ids) >= max_tokens:
        raise ValueError("Problem title, tags and rating exceed token budget.")

    statement = _clean_text(problem.get("statement", ""))
    input_text = _clean_text(problem.get("input", ""))
    output_text = _clean_text(problem.get("output", ""))

    body_parts = []
    if statement:
        body_parts.append(f"Statement:\n{statement}")
    if input_text and input_text != statement:
        body_parts.append(f"Input:\n{input_text}")
    if output_text and output_text != statement:
        body_parts.append(f"Output:\n{output_text}")

    body = "\n\n".join(body_parts)
    body_ids = tokenizer.encode(body, add_special_tokens=False)

    chunk_size = max_tokens - len(header_ids) - 8
    if chunk_size <= 0:
        raise ValueError("Not enough token space for the problem statement.")

    if not body_ids:
        return [{
            "text": header,
            "chunk_id": 0,
            "total_chunks": 1,
        }]

    overlap = min(40, chunk_size // 4)
    chunks = []
    start = 0

    while start < len(body_ids):
        end = min(start + chunk_size, len(body_ids))
        ids = body_ids[start:end]

        while ids:
            chunk_body = tokenizer.decode(ids, skip_special_tokens=True).strip()
            text = f"{header}\n\n{chunk_body}" if header else chunk_body

            if len(tokenizer.encode(text, add_special_tokens=True)) <= max_tokens:
                break

            ids = ids[:-1]

        chunks.append({
            "text": text,
            "chunk_id": len(chunks),
            "total_chunks": 0,
        })

        if end >= len(body_ids):
            break

        start = end - overlap

    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def render_problem_markdown(problem: dict[str, Any]) -> str:
    """Render a problem as Markdown with LaTeX."""
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
    """Generate HTML tag pills."""
    return " ".join(
        f'<span class="{"rating-pill" if tag.startswith("*") else "topic-pill"}">{tag}</span>'
        for tag in tags
    )