"""Codeforces API ingestion and HTML statement scraper."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import argparse
from typing import Any

import cloudscraper
import requests
from bs4 import BeautifulSoup

from config import PROBLEMS_DIR
from src.utils import format_latex

API_URL = "https://codeforces.com/api/problemset.problems"
REQUEST_DELAY = 1.0  # seconds between HTML scrape requests
MAX_SCRAPE_RETRIES = 3
MIN_STATEMENT_LENGTH = 40


def fetch_api_problems() -> list[dict[str, Any]]:
    """Fetch all problems from the official Codeforces API."""
    try:
        resp = requests.get(API_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return []

    if payload.get("status") != "OK":
        print(f"API returned status: {payload.get('comment', 'unknown')}", file=sys.stderr)
        return []

    return payload.get("result", {}).get("problems", [])


def _clean_statement_html(problem_div) -> str:
    """Extract and clean problem statement text with LaTeX preserved."""
    paragraphs = problem_div.find_all("p")
    lines: list[str] = []
    for p in paragraphs:
        text = p.get_text(separator=" ", strip=True)
        text = re.sub(r"\$\$\$(.*?)\$\$\$", r"$$\1$$", text, flags=re.DOTALL)
        text = re.sub(r"\$(.*?)\$", r"$\1$", text)
        lines.append(format_latex(text))
    return "\n\n".join(lines)


def _clean_spec_div(spec_div) -> str:
    """Extract input/output specification text."""
    if not spec_div:
        return ""
    text = spec_div.get_text(separator=" ", strip=True)
    return format_latex(text)


def scrape_statement(
    contest_id: int,
    problem_letter: str,
    scraper=None,
) -> dict[str, str] | None:
    """Scrape problem statement HTML with rate limiting."""
    url = f"https://codeforces.com/contest/{contest_id}/problem/{problem_letter}"
    scraper = scraper or cloudscraper.create_scraper()

    for attempt in range(1, MAX_SCRAPE_RETRIES + 1):
        try:
            resp = scraper.get(url, timeout=30)
            if resp.status_code == 200:
                break
            if resp.status_code not in {429, 500, 502, 503, 504}:
                return None
            if attempt == MAX_SCRAPE_RETRIES:
                return None
        except requests.RequestException as exc:
            if attempt == MAX_SCRAPE_RETRIES:
                print(
                    f"  Scrape failed for {contest_id}{problem_letter}: {exc}",
                    file=sys.stderr,
                )
                return None
        time.sleep(attempt * REQUEST_DELAY)
    else:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    title_div = soup.find("div", class_="title")
    problem_div = soup.find("div", class_="problem-statement")

    if not title_div or not problem_div:
        return None

    time_div = soup.find("div", class_="time-limit")
    memory_div = soup.find("div", class_="memory-limit")
    input_spec = soup.find("div", class_="input-specification")
    output_spec = soup.find("div", class_="output-specification")

    time_limit = ""
    if time_div:
        time_limit = time_div.get_text().replace("time limit per test", "").strip()

    memory_limit = ""
    if memory_div:
        memory_limit = memory_div.get_text().replace("memory limit per test", "").strip()

    statement = _clean_statement_html(problem_div).strip()
    if len(statement) < MIN_STATEMENT_LENGTH:
        return None

    return {
        "title": title_div.get_text(strip=True),
        "statement": statement,
        "input": _clean_spec_div(input_spec),
        "output": _clean_spec_div(output_spec),
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }


def build_tags(api_tags: list[str], rating: int | None) -> list[str]:
    """Combine API tags with difficulty rating tag."""
    tags = [t.strip() for t in api_tags if t.strip()]
    if rating is not None:
        tags.append(f"*{rating}")
    return tags


def problem_exists(problem_id: str) -> bool:
    """Check if a problem JSON file already exists."""
    return os.path.isfile(os.path.join(PROBLEMS_DIR, f"{problem_id}.json"))


def save_problem(data: dict[str, Any]) -> None:
    """Write problem data to JSON."""
    os.makedirs(PROBLEMS_DIR, exist_ok=True)
    path = os.path.join(PROBLEMS_DIR, f"{data['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def ingest_problems(
    contest_min: int = 1000,
    contest_max: int = 2267,
    max_count: int = 1000,
    skip_existing: bool = True,
    scrape_missing: bool = True,
) -> int:
    """
    Ingest problems using Codeforces API metadata and optional HTML scraping.

    Returns the number of newly saved problems.
    """
    api_problems = fetch_api_problems()
    if not api_problems:
        print("No problems fetched from API.", file=sys.stderr)
        return 0

    saved = 0
    seen_ids: set[str] = set()
    scraper = cloudscraper.create_scraper()
    for prob in api_problems:
        if saved >= max_count:
            break

        contest_id = prob.get("contestId")
        index = prob.get("index", "")
        if contest_id is None or not index:
            continue
        if not (contest_min <= contest_id < contest_max):
            continue

        problem_id = f"{contest_id}{index}"
        if problem_id in seen_ids:
            continue
        seen_ids.add(problem_id)
        if skip_existing and problem_exists(problem_id):
            continue

        rating = prob.get("rating")
        tags = build_tags(prob.get("tags", []), rating)
        title = prob.get("name", f"{index}. Problem")

        data: dict[str, Any] = {
            "id": problem_id,
            "contest_id": contest_id,
            "problem_letter": index,
            "title": f"{index}. {title}" if not title.startswith(index) else title,
            "statement": "",
            "input": "",
            "output": "",
            "time_limit": "",
            "memory_limit": "",
            "tags": tags,
            "rating": rating,
        }

        if scrape_missing:
            print(f"Scraping {problem_id} ...")
            scraped = scrape_statement(contest_id, index, scraper=scraper)
            time.sleep(REQUEST_DELAY)

            if not scraped or not scraped.get("statement", "").strip():
                print(f"Skipping {problem_id}: statement unavailable", file=sys.stderr)
                continue
            data.update(scraped)
            if scraped.get("title"):
                data["title"] = scraped["title"]

        save_problem(data)
        saved += 1
        print(f"Saved {problem_id}")

    return saved


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for problem ingestion."""
    parser = argparse.ArgumentParser(description="Fetch Codeforces problems and statements.")
    parser.add_argument("--contest-min", type=int, default=1000)
    parser.add_argument("--contest-max", type=int, default=2267)
    parser.add_argument("--max-count", type=int, default=1000)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Save API metadata without scraping statements.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Rewrite existing problem files instead of skipping them.",
    )
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    count = ingest_problems(
        contest_min=args.contest_min,
        contest_max=args.contest_max,
        max_count=args.max_count,
        skip_existing=not args.include_existing,
        scrape_missing=not args.metadata_only,
    )
    print(f"Ingestion complete. Saved {count} new problems to {PROBLEMS_DIR}")


if __name__ == "__main__":
    main()
