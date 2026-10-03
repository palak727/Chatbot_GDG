"""Groq API client for hint generation and code review."""

from __future__ import annotations

import os

import streamlit as st
from groq import Groq

MODEL_NAME = "openai/gpt-oss-120b"


def get_api_key() -> str | None:
    """Retrieve Groq API key from Streamlit secrets or environment variables."""
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]

    return os.getenv("GROQ_API_KEY")


def _get_groq_client() -> Groq | None:
    """Create and return a Groq client if an API key is available."""
    api_key = get_api_key()

    if not api_key:
        return None

    return Groq(api_key=api_key)


def generate_hint(problem: dict, hint_level: int) -> str:
    """Generate hints based on the selected guidance level."""
    client = _get_groq_client()

    if not client:
        return "Groq API key not configured."

    level_instructions = {
        1: (
            "Give a small nudge by pointing out the key observation. "
            "Do not name the algorithm or give code."
        ),
        2: (
            "Explain the main algorithm and data structures, then outline "
            "the steps. Do not give a full implementation."
        ),
        3: (
            "Explain the approach step by step, include pseudocode or a "
            "complete implementation, and give time and space complexity."
        ),
    }

    instruction = level_instructions.get(
        hint_level,
        level_instructions[1],
    )

    prompt = f"""Help me understand this competitive-programming problem.

Title: {problem.get('title', 'Unknown')}
Statement:
{problem.get('statement', problem.get('description', ''))}

Guidance level {hint_level}:
{instruction}

Be clear and concise. Focus on the key reasoning, avoid repeating the statement,
and do not invent constraints or details that are not provided."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful competitive-programming tutor. "
                        "Give accurate, clear guidance in concise markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        return (
            response.choices[0].message.content
            or "No guidance generated."
        )

    except Exception as exc:
        return f"Groq Error: {exc}"


def review_code(problem: dict, code: str, language: str) -> str:
    """Review submitted code for bugs, edge cases, and complexity bottlenecks."""
    client = _get_groq_client()

    if not client:
        return "Groq API key not configured."

    prompt = f"""Review this solution for the problem below.

Problem: {problem.get('title', 'Unknown')}
Statement:
{problem.get('statement', problem.get('description', ''))}

Language: {language}
Code:
```{language}
{code}
```

Check whether the logic solves the stated problem, then look for edge cases,
runtime or memory risks, and language-specific issues. Be specific: explain
each likely bug and suggest a practical fix. If you cannot confirm a concern
from the code and statement, label it as a possibility rather than a definite
bug. Do not claim that you compiled or ran the code.

Also evaluate the approach:
- Briefly identify the approach used by the code.
- If a brute-force approach is possible, briefly explain its time complexity.
- State the optimal or expected approach for the given constraints and its
  time complexity.
- Compare the user's approach with the optimal approach and clearly say if
  the current approach is sufficient for the given constraints.
- Do not claim an approach is optimal unless the problem constraints support it.

Keep the review concise."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful competitive-programming code reviewer. "
                        "Base your findings on the supplied problem and code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return response.choices[0].message.content or "No review generated."
    except Exception as exc:
        return f"Groq Error: {exc}"
