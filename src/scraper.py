"""Codeforces API ingestion and HTML statement scraper."""

from __future__ import annotations

import json
import html
import os
import re
import sys
import time
import argparse
from io import BytesIO
from typing import Any

import cloudscraper
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from config import PROBLEMS_DIR

API_URL = "https://codeforces.com/api/problemset.problems"
REQUEST_DELAY = 1.0  # seconds between HTML scrape requests
MAX_SCRAPE_RETRIES = 3
MIN_STATEMENT_LENGTH = 40
SELENIUM_WAIT_SECONDS = 15


_TEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ϵ", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "ϕ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "lt": "<", "gt": ">", "neq": "≠",
    "ne": "≠", "approx": "≈", "equiv": "≡", "in": "∈", "notin": "∉",
    "ni": "∋", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "supseteq": "⊇", "cup": "∪", "cap": "∩", "cdot": "·",
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "to": "→",
    "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀",
    "exists": "∃", "emptyset": "∅", "angle": "∠", "perp": "⊥",
    "parallel": "∥", "ldots": "…", "dots": "…", "cdots": "⋯",
    "sum": "∑", "prod": "∏", "int": "∫", "lim": "lim",
}
_TEX_STYLE_COMMANDS = {"text", "textrm", "textit", "mathrm", "mathbf", "mathit", "operatorname"}


def _read_tex_group(text: str, start: int) -> tuple[str, int]:
    """Read a brace-delimited TeX group starting at start, including nested groups."""
    if start >= len(text) or text[start] != "{":
        return "", start
    depth = 0
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : position], position + 1
    return text[start + 1 :], len(text)


def _plain_tex(text: str) -> str:
    """Convert Codeforces TeX source to readable Unicode/plain-text notation."""
    text = html.unescape(text)
    for pattern in (
        r"\$\$\$(.*?)\$\$\$",
        r"\$\$(.*?)\$\$",
        r"\\\((.*?)\\\)",
        r"\\\[(.*?)\\\]",
        r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$",
    ):
        text = re.sub(pattern, r"\1", text, flags=re.DOTALL)

    output: list[str] = []
    position = 0
    while position < len(text):
        character = text[position]
        if character == "\\":
            position += 1
            if position >= len(text):
                break
            if text[position].isalpha():
                end = position + 1
                while end < len(text) and text[end].isalpha():
                    end += 1
                command = text[position:end]
                position = end
                if command in {"frac", "dfrac", "tfrac", "binom"}:
                    numerator, next_position = _read_tex_group(text, position)
                    denominator, final_position = _read_tex_group(text, next_position)
                    if final_position > next_position:
                        top = _plain_tex(numerator).strip()
                        bottom = _plain_tex(denominator).strip()
                        if command == "binom":
                            output.append(f"{top} choose {bottom}")
                        else:
                            numerator_text = top if re.fullmatch(r"[\wα-ωΑ-Ω]+", top) else f"({top})"
                            denominator_text = bottom if re.fullmatch(r"[\wα-ωΑ-Ω]+", bottom) else f"({bottom})"
                            output.append(f"{numerator_text}/{denominator_text}")
                        position = final_position
                        continue
                if command == "sqrt":
                    root_index = ""
                    if position < len(text) and text[position] == "[":
                        close = text.find("]", position + 1)
                        if close >= 0:
                            root_index = "[" + _plain_tex(text[position + 1 : close]) + "]"
                            position = close + 1
                    radicand, final_position = _read_tex_group(text, position)
                    if final_position > position:
                        output.append(f"√{root_index}({_plain_tex(radicand)})")
                        position = final_position
                        continue
                    output.append("√")
                    continue
                if command in _TEX_STYLE_COMMANDS:
                    content, final_position = _read_tex_group(text, position)
                    if final_position > position:
                        output.append(_plain_tex(content))
                        position = final_position
                        continue

                output.append(_TEX_SYMBOLS.get(command, command))
                continue

            escaped = text[position]
            position += 1
            if escaped in "{}_%#&$":
                output.append(escaped)
            elif escaped == "\\":
                output.append(" ")
            else:
                output.append(escaped)
            continue

        if character in "_^" and position + 1 < len(text) and text[position + 1] == "{":
            group, final_position = _read_tex_group(text, position + 1)
            if final_position > position + 1:
                value = _plain_tex(group)
                if re.fullmatch(r"[\wα-ωΑ-Ω]+", value):
                    output.extend((character, value))
                else:
                    output.extend((character, "{", value, "}"))
                position = final_position
                continue
        if character in "{}":
            position += 1
            continue
        output.append(character)
        position += 1

    result = "".join(output)
    result = re.sub(r"[ \t\r\f\v]+", " ", result)
    result = re.sub(r"\s*·\s*", "·", result)
    return result.strip()


def _math_attribute(node: Tag) -> str:
    for attribute in ("data-tex", "data-latex", "aria-label", "alt", "title", "text"):
        value = node.get(attribute)
        if value:
            return str(value)
    image = node.find("img")
    if image:
        for attribute in ("alt", "aria-label", "title"):
            value = image.get(attribute)
            if value:
                return str(value)
    return ""


def _html_text(node: PageElement) -> str:
    """Extract visible text, using accessible/TeX alternatives for math markup."""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag) or node.name in {"script", "style"}:
        return ""

    classes = set(node.get_attribute_list("class"))
    if "MathJax" in classes:
        if node.find_previous_sibling(class_="MathJax_Preview"):
            return ""
        return _math_attribute(node)
    if "MathJax_Preview" in classes:
        return node.get_text()

    math_source = node.get("data-tex") or node.get("data-latex")
    if math_source:
        return str(math_source)
    if node.name == "img":
        return _math_attribute(node)

    content = "".join(_html_text(child) for child in node.children)
    if node.name == "sub":
        return "_" + (content if len(content) == 1 else "{" + content + "}")
    if node.name == "sup":
        return "^" + (content if len(content) == 1 else "{" + content + "}")
    if node.name == "br":
        return "\n"
    return content


def _clean_content_blocks(container: Tag) -> str:
    blocks = [
        block
        for block in container.find_all(["p", "pre"])
        if block.name != "pre" or not block.find_parent(class_="sample-tests")
    ]
    if not blocks:
        blocks = [container]

    cleaned: list[str] = []
    for block in blocks:
        if block.name == "pre":
            content = html.unescape(_html_text(block)).strip("\n")
            if content:
                cleaned.append(f"```\n{content}\n```")
        else:
            content = _plain_tex(_html_text(block))
            content = re.sub(r"\s+", " ", content).strip()
            if content:
                cleaned.append(content)

    return "\n\n".join(cleaned)


def _clean_sample_tests(sample_tests: Tag) -> str:
    samples: list[str] = []

    for sample in sample_tests.find_all("div", class_="sample-test"):
        sections: list[str] = []

        for section in sample.find_all("div", recursive=False):
            section_classes = set(section.get_attribute_list("class"))

            if not section_classes.intersection({"input", "output"}):
                continue

            title = section.find(class_="title")
            pre = section.find("pre")

            if pre:
                label = title.get_text(" ", strip=True) if title else "Example"
                code = _html_text(pre).strip("\n")
                sections.append(f"{label}:\n```\n{code}\n```")

        if sections:
            samples.append("\n\n".join(sections))

    return "\n\n".join(samples)


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
    """Extract statement text, math notation, and formatted sample examples."""
    statement = _clean_content_blocks(problem_div)
    sample_tests = problem_div.find(class_="sample-tests")
    samples = _clean_sample_tests(sample_tests) if sample_tests else ""
    return "\n\n".join(part for part in (statement, samples) if part)


def _clean_spec_div(spec_div) -> str:
    """Extract input/output specification text."""
    if not spec_div:
        return ""
    return _clean_content_blocks(spec_div)


def _create_selenium_driver() -> webdriver.Chrome:
    """Create one reusable headless Chrome browser for statement scraping."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )

    return webdriver.Chrome(options=options)


def _transfer_cloudscraper_cookies(
    scraper,
    driver: webdriver.Chrome,
    url: str,
) -> None:
    """Transfer CloudScraper cookies into Selenium's browser session."""
    from urllib.parse import urlparse

    parsed = urlparse(url)

    driver.get(f"{parsed.scheme}://{parsed.netloc}/")

    for cookie in scraper.cookies:
        cookie_data = {
            "name": cookie.name,
            "value": cookie.value,
            "path": cookie.path or "/",
        }

        if cookie.domain:
            domain = cookie.domain.lstrip(".")
            if domain == parsed.netloc:
                cookie_data["domain"] = cookie.domain

        if cookie.expires is not None:
            cookie_data["expiry"] = int(cookie.expires)

        try:
            driver.add_cookie(cookie_data)
        except WebDriverException:
            # Some cookies cannot be transferred to the current browser
            # domain/path. They are safe to ignore.
            continue


def _selenium_page_source(
    url: str,
    scraper,
    driver: webdriver.Chrome,
) -> str | None:
    """Open a problem page in Selenium and return its rendered HTML."""
    try:
        _transfer_cloudscraper_cookies(scraper, driver, url)

        driver.get(url)

        WebDriverWait(driver, SELENIUM_WAIT_SECONDS).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.problem-statement")
            )
        )

        return driver.page_source

    except (TimeoutException, WebDriverException) as exc:
        print(f"  Selenium failed for {url}: {exc}", file=sys.stderr)
        return None



def _fix_pdf_word_spacing(text: str) -> str:
    """Fix common missing/extra spaces caused by PDF text extraction.

    The PDF text layer often loses spaces between normal words and nearby
    variables/numbers, e.g. ``withn``, ``from1`` and ``travelerp``.
    Mathematical tokens such as ``a1`` and ``2≤n≤500000`` are preserved.
    """

    # "Can Y ou Reach" -> "Can You Reach".
    text = re.sub(r"\b([A-Z]) ([a-z])", r"\1\2", text)

    # Longer words first. Singular/plural pairs are handled carefully below
    # so that "travelers" does not become "traveler s".
    prefixes = sorted(
        (
            "from", "with", "given", "contains", "between", "after", "before",
            "each", "any", "exactly", "number", "numbers", "integers",
            "vertices", "vertex", "edges", "edge", "array", "values",
            "maximum", "minimum", "cost", "size", "sum", "modulo",
            "travelers", "points", "cells", "rows", "columns", "steps",
            "tests", "scenarios", "input", "output",
        ),
        key=len,
        reverse=True,
    )

    prefix_pattern = (
        r"\b(" + "|".join(map(re.escape, prefixes)) +
        r")(?=[A-Za-z]|\d)"
    )
    text = re.sub(prefix_pattern, r"\1 ", text, flags=re.IGNORECASE)

    # Singular words: do not match the final "s" of an ordinary plural.
    singular_patterns = {
        r"\btraveler(?!s\b)(?=[A-Za-z]|\d)": "traveler ",
        r"\binteger(?!s\b)(?=[A-Za-z]|\d)": "integer ",
        r"\bscenario(?!s\b)(?=[A-Za-z]|\d)": "scenario ",
        r"\bvalue(?!s\b)(?=[A-Za-z]|\d)": "value ",
        r"\bpoint(?!s\b)(?=[A-Za-z]|\d)": "point ",
        r"\bcell(?!s\b)(?=[A-Za-z]|\d)": "cell ",
        r"\brow(?!s\b)(?=[A-Za-z]|\d)": "row ",
        r"\bcolumn(?!s\b)(?=[A-Za-z]|\d)": "column ",
        r"\bstep(?!s\b)(?=[A-Za-z]|\d)": "step ",
        r"\btest(?!s\b)(?=[A-Za-z]|\d)": "test ",
    }
    for pattern, replacement in singular_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Short words: only repair a missing space before a number.
    short_patterns = {
        r"\bto(?=\d)": "to ",
        r"\bfor(?=\d)": "for ",
        r"\bor(?=\d)": "or ",
        r"\bof(?=\d)": "of ",
        r"\bat(?=\d)": "at ",
        r"\bon(?=\d)": "on ",
        r"\bby(?=\d)": "by ",
        r"\bas(?=\d)": "as ",
    }
    for pattern, replacement in short_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # One PDF font used in the downloaded statements can extract "values"
    # as "value s".
    text = re.sub(r"\bvalue s\b", "values", text, flags=re.IGNORECASE)

    text = re.sub(r" {2,}", " ", text)
    return text


def _clean_pdf_text(text: str) -> str:
    """Normalize common PDF text-extraction artifacts."""
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "–": "-",
        "—": "-",
        "/emdash.cyr": "-",
        "/endash.cyr": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []

    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        line = re.sub(r"\s+", " ", line).strip()
        line = _fix_pdf_word_spacing(line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _extract_pdf_sections(text: str) -> dict[str, str]:
    """Extract title, statement, input, output, and limits from PDF text."""
    text = _clean_pdf_text(text)

    # Remove the repeated page footer added by the PDF extractor.
    text = re.sub(r"\nPage \d+ of \d+\s*$", "", text, flags=re.IGNORECASE)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {}

    title = lines[0]

    time_limit = ""
    memory_limit = ""

    for line in lines[:15]:
        match = re.search(r"^Time limit:\s*(.+)$", line, re.IGNORECASE)
        if match:
            time_limit = match.group(1).strip()

        match = re.search(r"^Memory limit:\s*(.+)$", line, re.IGNORECASE)
        if match:
            memory_limit = match.group(1).strip()

    def find_heading(heading: str) -> int:
        for i, line in enumerate(lines):
            if line.strip().lower() == heading.lower():
                return i
        return -1

    input_index = find_heading("Input")
    output_index = find_heading("Output")
    example_index = find_heading("Example")
    note_index = find_heading("Note")
    scoring_index = find_heading("Scoring")
    additional_index = find_heading("Additional Constraints")

    # Main statement is everything after the limits and before Input.
    first_body = 0
    for i, line in enumerate(lines):
        if line.lower().startswith("memory limit:"):
            first_body = i + 1
            break

    statement_end = input_index if input_index >= 0 else len(lines)
    statement_lines = lines[first_body:statement_end]

    # Remove accidental repeated page markers.
    statement_lines = [
        line for line in statement_lines
        if not re.fullmatch(r"Page \d+ of \d+", line, re.IGNORECASE)
    ]

    input_text = ""
    if input_index >= 0:
        input_end = output_index if output_index > input_index else len(lines)
        input_text = "\n".join(lines[input_index + 1:input_end])

    output_text = ""
    if output_index >= 0:
        output_end_candidates = [
            i for i in (example_index, note_index, scoring_index, additional_index)
            if i > output_index
        ]
        output_end = min(output_end_candidates) if output_end_candidates else len(lines)
        output_text = "\n".join(lines[output_index + 1:output_end])

    # Keep examples/notes/scoring as part of the statement, because the HTML
    # scraper also keeps the useful sample material attached to the statement.
    extras_start = min(
        [i for i in (example_index, note_index, scoring_index) if i >= 0],
        default=-1,
    )
    if extras_start >= 0:
        extras_end = additional_index if additional_index > extras_start else len(lines)
        statement_lines.extend(lines[extras_start:extras_end])

    statement = "\n".join(statement_lines).strip()

    return {
        "title": title,
        "statement": statement,
        "input": input_text.strip(),
        "output": output_text.strip(),
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }


def _scrape_pdf_response(response: requests.Response) -> dict[str, str] | None:
    """Extract a Codeforces problem statement from a PDF HTTP response."""
    try:
        reader = PdfReader(BytesIO(response.content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        data = _extract_pdf_sections(text)
    except Exception as exc:
        print(f"  PDF extraction failed: {exc}", file=sys.stderr)
        return None

    if len(data.get("statement", "").strip()) < MIN_STATEMENT_LENGTH:
        print("  PDF statement is too short", file=sys.stderr)
        return None

    return data

def scrape_statement(
    contest_id: int,
    problem_letter: str,
    scraper=None,
    driver: webdriver.Chrome | None = None,
) -> dict[str, str] | None:
    """Scrape a Codeforces statement using HTTP, PDF extraction, Selenium, and BeautifulSoup."""
    url = (
        f"https://codeforces.com/contest/"
        f"{contest_id}/problem/{problem_letter}"
    )

    scraper = scraper or cloudscraper.create_scraper()

    owns_driver = driver is None
    driver = driver or _create_selenium_driver()

    try:
        cloud_resp = None

        for attempt in range(1, MAX_SCRAPE_RETRIES + 1):
            try:
                cloud_resp = scraper.get(url, timeout=30)

                if cloud_resp.status_code == 200:
                    break

                if cloud_resp.status_code not in {429, 500, 502, 503, 504}:
                    break

            except requests.RequestException as exc:
                if attempt == MAX_SCRAPE_RETRIES:
                    print(
                        f"  CloudScraper failed for {contest_id}{problem_letter}: {exc}",
                        file=sys.stderr,
                    )
                    break

            if attempt < MAX_SCRAPE_RETRIES:
                time.sleep(attempt * REQUEST_DELAY)

        if cloud_resp is not None and cloud_resp.status_code == 200:
            content_type = cloud_resp.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type or cloud_resp.content.startswith(b"%PDF"):
                print(f"  PDF response detected for {contest_id}{problem_letter}")
                return _scrape_pdf_response(cloud_resp)

        rendered_html = _selenium_page_source(url, scraper, driver)
        page_html = rendered_html

        if not page_html and cloud_resp is not None and cloud_resp.status_code == 200:
            content_type = cloud_resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/xhtml" in content_type:
                page_html = cloud_resp.text

        if not page_html:
            return None

        soup = BeautifulSoup(page_html, "lxml")

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
            time_limit = (
                time_div.get_text()
                .replace("time limit per test", "")
                .strip()
            )

        memory_limit = ""
        if memory_div:
            memory_limit = (
                memory_div.get_text()
                .replace("memory limit per test", "")
                .strip()
            )

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

    finally:
        if owns_driver:
            driver.quit()



def build_tags(api_tags: list[str], rating: int | None) -> list[str]:
    """Combine API tags with difficulty rating tag."""
    tags = [t.strip() for t in api_tags if t.strip()]

    if rating is not None:
        tags.append(f"*{rating}")

    return tags


def problem_exists(problem_id: str) -> bool:
    """Check if a problem JSON file already exists."""
    return os.path.isfile(
        os.path.join(PROBLEMS_DIR, f"{problem_id}.json")
    )


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
    driver = _create_selenium_driver()

    try:
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
                "title": (
                    f"{index}. {title}"
                    if not title.startswith(index)
                    else title
                ),
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

                scraped = scrape_statement(
                    contest_id,
                    index,
                    scraper=scraper,
                    driver=driver,
                )

                time.sleep(REQUEST_DELAY)

                if not scraped or not scraped.get("statement", "").strip():
                    print(
                        f"Skipping {problem_id}: statement unavailable",
                        file=sys.stderr,
                    )
                    continue

                data.update(scraped)

                if scraped.get("title"):
                    data["title"] = scraped["title"]

            save_problem(data)
            saved += 1

            print(f"Saved {problem_id}")

    finally:
        driver.quit()

    return saved


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for problem ingestion."""
    parser = argparse.ArgumentParser(
        description="Fetch Codeforces problems and statements."
    )

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

    print(
        f"Ingestion complete. Saved {count} new problems to "
        f"{PROBLEMS_DIR}"
    )


if __name__ == "__main__":
    main()
