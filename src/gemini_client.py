"""Groq API client for hint generation and code review."""

from __future__ import annotations

import os
import streamlit as st
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"


def get_api_key() -> str | None:
    """Retrieve Groq API key from Streamlit secrets or environment variables."""
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY")


def _get_groq_client() -> Groq | None:
    api_key = get_api_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)


def generate_hint(problem: dict, hint_level: int) -> str:
    """Generate Socratic hints based on the selected guidance level."""
    client = _get_groq_client()
    if not client:
        return "Groq API key not configured."

    level_instructions = {
        1: "Give high-level intuition and key observations. Do NOT reveal specific algorithm names or code.",
        2: "Identify the optimal data structures and algorithmic approach (e.g., Dynamic Programming, DSU, Segment Tree). Do NOT provide full implementation.",
        3: "Provide full solution breakdown, key algorithm steps, complete code/pseudocode, and time/space complexity analysis.",
    }

    instruction = level_instructions.get(hint_level, level_instructions[1])

    prompt = f"""
    Problem Title: {problem.get('title', 'Unknown')}
    Problem Statement: {problem.get('statement', problem.get('description', ''))}

    Task:
    {instruction}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a competitive programming coach. Use concise, clear markdown output.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content or "No guidance generated."
    except Exception as exc:
        return f"Groq Error: {exc}"


def review_code(problem: dict, code: str, language: str) -> str:
    """Review submitted code for bugs, edge cases, and complexity bottlenecks."""
    client = _get_groq_client()
    if not client:
        return "Groq API key not configured."

    prompt = f"""
    Problem: {problem.get('title', 'Unknown')}
    Language: {language}

    User Code:
    ```{language}
    {code}
    ```

    Perform a competitive programming code review:
    1. Check for correctness and logical bugs.
    2. Check for time/space complexity bottlenecks and potential TLE (Time Limit Exceeded).
    3. Point out integer overflow risks, corner cases, or array out-of-bounds issues.
    4. Provide actionable suggestions to fix or optimize the code.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert competitive programming code auditor.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return response.choices[0].message.content or "No review generated."
    except Exception as exc:
        return f"Groq Error: {exc}"