"""Fetch Codeforces problems and scrape their statements."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from io import BytesIO
from typing import Any

import cloudscraper
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from config import PROBLEMS_DIR

API_URL = "https://codeforces.com/api/problemset.problems"
REQUEST_DELAY = 1.0
MAX_RETRIES = 3
MIN_STATEMENT_LENGTH = 40
SELENIUM_WAIT = 15

TEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ϵ", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "ϕ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "lt": "<", "gt": ">", "neq": "≠", "ne": "≠", "approx": "≈",
    "equiv": "≡", "in": "∈", "notin": "∉", "ni": "∋",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "cdot": "·", "times": "×", "div": "÷",
    "pm": "±", "mp": "∓", "to": "→", "rightarrow": "→",
    "leftarrow": "←", "leftrightarrow": "↔", "infty": "∞",
    "partial": "∂", "nabla": "∇", "forall": "∀", "exists": "∃",
    "emptyset": "∅", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "ldots": "…", "dots": "…", "cdots": "⋯", "sum": "∑",
    "prod": "∏", "int": "∫", "lim": "lim", "oplus": "⊕",
    "land": "∧", "lor": "∨", "mid": "|", "vert": "|",
}
TEX_STYLE = {"text", "textrm", "textit", "mathrm", "mathbf", "mathit", "operatorname"}


def _group(text: str, start: int) -> tuple[str, int]:
    """Read group inside {}"""
    if start >= len(text) or text[start] != "{":
        return "", start
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)
  

def plain_tex(text: str) -> str:
    """Turn common Codeforces TeX into readable text."""
    text = html.unescape(text)
    text = re.sub(r"\${1,3}(.*?)\${1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.DOTALL)

    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]

        if c == "\\":
            i += 1
            if i >= len(text):
                break
            if text[i].isalpha():
                j = i
                while j < len(text) and text[j].isalpha():
                    j += 1
                cmd = text[i:j]
                i = j

                if cmd in {"frac", "dfrac", "tfrac", "binom"}:
                    a, j = _group(text, i)
                    b, j2 = _group(text, j)
                    if j2 > j:
                        a, b = plain_tex(a).strip(), plain_tex(b).strip()
                        if cmd == "binom":
                            out.append(f"{a} choose {b}")
                        else:
                            aa = a if re.fullmatch(r"[\wα-ωΑ-Ω]+", a) else f"({a})"
                            bb = b if re.fullmatch(r"[\wα-ωΑ-Ω]+", b) else f"({b})"
                            out.append(f"{aa}/{bb}")
                        i = j2
                        continue

                if cmd == "sqrt":
                    index = ""
                    if i < len(text) and text[i] == "[":
                        end = text.find("]", i + 1)
                        if end >= 0:
                            index = "[" + plain_tex(text[i + 1:end]) + "]"
                            i = end + 1
                    value, j = _group(text, i)
                    if j > i:
                        out.append(f"√{index}({plain_tex(value)})")
                        i = j
                        continue
                    out.append("√")
                    continue

                if cmd in TEX_STYLE:
                    value, j = _group(text, i)
                    if j > i:
                        out.append(plain_tex(value))
                        i = j
                        continue

                out.append(TEX_SYMBOLS.get(cmd, cmd))
                continue

            escaped = text[i]
            i += 1
            out.append(" " if escaped == "\\" else escaped)
            continue

        if c in "_^" and i + 1 < len(text) and text[i + 1] == "{":
            value, j = _group(text, i + 1)
            if j > i + 1:
                value = plain_tex(value)
                out.append(c + (value if re.fullmatch(r"[\wα-ωΑ-Ω]+", value) else "{" + value + "}"))
                i = j
                continue

        if c in "{}":
            i += 1
            continue

        out.append(c)
        i += 1
    result = "".join(out)
    result = re.sub(r"\s+", " ", result)
    return result.strip()


def math_text(node: Tag) -> str:
    """Get the original TeX text from a math element."""
    # MathJax often keeps the original formula in an annotation element.
    annotation = node.find("annotation", encoding="application/x-tex")
    if annotation and annotation.get_text(strip=False):
        return annotation.get_text()

    for attr in ("data-tex", "data-latex", "aria-label", "alt", "title", "text"):
        value = node.get(attr)
        if value:
            return str(value)

    image = node.find("img")
    if image:
        for attr in ("alt", "aria-label", "title"):
            value = image.get(attr)
            if value:
                return str(value)
    return ""


def html_text(node: Any) -> str:
    """Extract visible text, lists, equations and line breaks."""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag) or node.name in {"script", "style"}:
        return ""

    classes = set(node.get_attribute_list("class"))

    # Prefer the original TeX instead of MathJax's generated HTML.
    if "MathJax" in classes or "MathJax_Preview" in classes or node.name == "math":
        value = math_text(node)
        if value:
            return value

    value = node.get("data-tex") or node.get("data-latex")
    if value:
        return str(value)
    if node.name == "img":
        return math_text(node)

    content = "".join(html_text(child) for child in node.children)
    if node.name == "sub":
        return "_" + (content if len(content) == 1 else "{" + content + "}")
    if node.name == "sup":
        return "^" + (content if len(content) == 1 else "{" + content + "}")
    if node.name == "br":
        return "\n"
    if node.name in {"li", "p", "div", "tr"}:
        return content + "\n"
    return content


def clean_blocks(container: Tag) -> str:
    """Extract paragraphs, lists and code blocks without losing list items."""
    blocks = []

    for block in container.find_all(["p", "li", "pre", "div"]):
        if block.name == "pre" and block.find_parent(class_="sample-tests"):
            continue
        if block.name == "div" and "tex-div" not in set(block.get_attribute_list("class")):
            continue
        if block.name == "p" and block.find_parent("li"):
            continue
        if block.name == "div" and block.find_parent(["p", "li"]):
            continue
        blocks.append(block)

    if not blocks:
        blocks = [container]

    result = []
    for block in blocks:
        if block.name == "pre":
            text = html.unescape(html_text(block)).strip("\n")
            if text:
                result.append(f"```\n{text}\n```")
            continue

        text = plain_tex(html_text(block))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            result.append(text)

    # Remove consecutive duplicates caused by nested HTML wrappers.
    cleaned = []
    for text in result:
        if not cleaned or text != cleaned[-1]:
            cleaned.append(text)
    return "\n\n".join(cleaned)


def clean_samples(sample_tests: Tag) -> str:
    samples = []
    for sample in sample_tests.find_all("div", class_="sample-test"):
        parts = []
        for section in sample.find_all("div", recursive=False):
            classes = set(section.get_attribute_list("class"))
            if not classes.intersection({"input", "output"}):
                continue
            title = section.find(class_="title")
            pre = section.find("pre")
            if pre:
                label = title.get_text(" ", strip=True) if title else "Example"
                parts.append(f"{label}:\n```\n{html_text(pre).strip(chr(10))}\n```")
        if parts:
            samples.append("\n\n".join(parts))
    return "\n\n".join(samples)


def create_driver() -> webdriver.Chrome:
    options = Options()
    for arg in ("--headless=new", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"):
        options.add_argument(arg)
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def selenium_html(url: str, scraper, driver: webdriver.Chrome) -> str | None:
    try:
        driver.get("https://codeforces.com/")
        for cookie in scraper.cookies:
            try:
                driver.add_cookie({
                    "name": cookie.name,
                    "value": cookie.value,
                    "path": cookie.path or "/",
                })
            except WebDriverException:
                pass

        driver.get(url)
        WebDriverWait(driver, SELENIUM_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.problem-statement"))
        )
        return driver.page_source
    except (TimeoutException, WebDriverException) as exc:
        print(f"Selenium failed for {url}: {exc}", file=sys.stderr)
        return None


def clean_pdf_text(text: str) -> str:
    replacements = {"ﬁ": "fi", "ﬂ": "fl", "–": "-", "—": "-", "\u00a0": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_pdf(response: requests.Response) -> dict[str, str] | None:
    try:
        reader = PdfReader(BytesIO(response.content))
        text = clean_pdf_text("\n".join(page.extract_text() or "" for page in reader.pages))
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if not lines:
            return None

        title = lines[0]
        time_limit = ""
        memory_limit = ""
        for line in lines[:15]:
            m = re.search(r"^Time limit:\s*(.+)$", line, re.I)
            if m:
                time_limit = m.group(1).strip()
            m = re.search(r"^Memory limit:\s*(.+)$", line, re.I)
            if m:
                memory_limit = m.group(1).strip()

        def heading(name: str) -> int:
            for i, line in enumerate(lines):
                if line.lower() == name.lower():
                    return i
            return -1

        inp, out = heading("Input"), heading("Output")
        example = heading("Example")
        note = heading("Note")
        scoring = heading("Scoring")
        additional = heading("Additional Constraints")

        body_start = next((i + 1 for i, x in enumerate(lines) if x.lower().startswith("memory limit:")), 1)
        statement_end = inp if inp >= 0 else len(lines)
        statement = lines[body_start:statement_end]

        input_text = ""
        if inp >= 0:
            end = out if out > inp else len(lines)
            input_text = "\n".join(lines[inp + 1:end])

        output_text = ""
        if out >= 0:
            ends = [i for i in (example, note, scoring, additional) if i > out]
            output_text = "\n".join(lines[out + 1:(min(ends) if ends else len(lines))])

        extras = [i for i in (example, note, scoring) if i >= 0]
        if extras:
            end = additional if additional > min(extras) else len(lines)
            statement.extend(lines[min(extras):end])

        return {
            "title": title,
            "statement": "\n".join(statement).strip(),
            "input": input_text.strip(),
            "output": output_text.strip(),
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }
    except Exception as exc:
        print(f"PDF extraction failed: {exc}", file=sys.stderr)
        return None


def fetch_api_problems() -> list[dict[str, Any]]:
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return []

    if payload.get("status") != "OK":
        print(f"API error: {payload.get('comment', 'unknown')}", file=sys.stderr)
        return []
    return payload.get("result", {}).get("problems", [])


def parse_html_statement(page: str) -> dict[str, str] | None:
    """Parse a Codeforces HTML problem page."""
    soup = BeautifulSoup(page, "lxml")
    problem = soup.find("div", class_="problem-statement")
    title = soup.find("div", class_="title")
    if not problem or not title:
        return None

    statement = clean_blocks(problem)
    samples = problem.find(class_="sample-tests")
    if samples:
        statement = "\n\n".join(x for x in (statement, clean_samples(samples)) if x)
    if len(statement) < MIN_STATEMENT_LENGTH:
        return None

    def limit(cls: str, label: str) -> str:
        node = soup.find("div", class_=cls)
        return node.get_text().replace(label, "").strip() if node else ""

    input_node = soup.find("div", class_="input-specification")
    output_node = soup.find("div", class_="output-specification")

    return {
        "title": title.get_text(strip=True),
        "statement": statement,
        "input": clean_blocks(input_node) if input_node else "",
        "output": clean_blocks(output_node) if output_node else "",
        "time_limit": limit("time-limit", "time limit per test"),
        "memory_limit": limit("memory-limit", "memory limit per test"),
    }


def parse_html_statement(page: str) -> dict[str, str] | None:
    """Parse a Codeforces HTML statement."""
    soup = BeautifulSoup(page, "lxml")
    problem = soup.find("div", class_="problem-statement")
    title = soup.find("div", class_="title")
    if not problem or not title:
        return None

    statement = clean_blocks(problem)
    samples = problem.find(class_="sample-tests")
    if samples:
        statement = "\n\n".join(x for x in (statement, clean_samples(samples)) if x)
    if len(statement) < MIN_STATEMENT_LENGTH:
        return None

    def limit(cls: str, label: str) -> str:
        node = soup.find("div", class_=cls)
        return node.get_text().replace(label, "").strip() if node else ""

    input_div = soup.find("div", class_="input-specification")
    output_div = soup.find("div", class_="output-specification")

    return {
        "title": title.get_text(strip=True),
        "statement": statement,
        "input": clean_blocks(input_div) if input_div else "",
        "output": clean_blocks(output_div) if output_div else "",
        "time_limit": limit("time-limit", "time limit per test"),
        "memory_limit": limit("memory-limit", "memory limit per test"),
    }


def scrape_statement(contest_id: int, problem_letter: str, scraper=None, driver=None) -> dict[str, str] | None:
    """Get a problem statement. Use Selenium only when normal HTTP parsing fails."""
    url = f"https://codeforces.com/contest/{contest_id}/problem/{problem_letter}"
    scraper = scraper or cloudscraper.create_scraper()
    own_driver = driver is None

    try:
        response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = scraper.get(url, timeout=30)
                if response.status_code == 200 or response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    print(f"Request failed for {contest_id}{problem_letter}: {exc}", file=sys.stderr)
                    break
            if attempt < MAX_RETRIES:
                time.sleep(attempt * REQUEST_DELAY)

        if response is not None and response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
                data = extract_pdf(response)
                if data:
                    return data
            else:
                data = parse_html_statement(response.text)
                if data:
                    return data

        # Selenium is a fallback. It is only started if HTTP failed.
        driver = driver or create_driver()
        page = selenium_html(url, scraper, driver)
        if page:
            return parse_html_statement(page)
        return None
    finally:
        if own_driver and driver is not None:
            driver.quit()


def build_tags(api_tags: list[str], rating: int | None) -> list[str]:
    tags = [tag.strip() for tag in api_tags if tag.strip()]
    if rating is not None:
        tags.append(f"*{rating}")
    return tags


def problem_exists(problem_id: str) -> bool:
    return os.path.isfile(os.path.join(PROBLEMS_DIR, f"{problem_id}.json"))


def save_problem(data: dict[str, Any]) -> None:
    os.makedirs(PROBLEMS_DIR, exist_ok=True)
    with open(os.path.join(PROBLEMS_DIR, f"{data['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def ingest_problems(contest_min=1000, contest_max=2267, max_count=1000,
                    skip_existing=True, scrape_missing=True) -> int:
    problems = fetch_api_problems()
    if not problems:
        print("No problems fetched from API.", file=sys.stderr)
        return 0

    saved = 0
    seen = set()
    scraper = cloudscraper.create_scraper()
    driver = None

    try:
        for problem in problems:
            if saved >= max_count:
                break
            contest_id = problem.get("contestId")
            letter = problem.get("index", "")
            if contest_id is None or not letter or not (contest_min <= contest_id < contest_max):
                continue
            problem_id = f"{contest_id}{letter}"
            if problem_id in seen or (skip_existing and problem_exists(problem_id)):
                continue
            seen.add(problem_id)

            rating = problem.get("rating")
            title = problem.get("name", f"{letter}. Problem")
            data = {
                "id": problem_id,
                "contest_id": contest_id,
                "problem_letter": letter,
                "title": title if title.startswith(letter) else f"{letter}. {title}",
                "statement": "",
                "input": "",
                "output": "",
                "time_limit": "",
                "memory_limit": "",
                "tags": build_tags(problem.get("tags", []), rating),
                "rating": rating,
            }

            if scrape_missing:
                print(f"Scraping {problem_id} ...")
                scraped = scrape_statement(contest_id, letter, scraper, driver)
                time.sleep(REQUEST_DELAY)
                if not scraped or not scraped.get("statement", "").strip():
                    print(f"Skipping {problem_id}: statement unavailable", file=sys.stderr)
                    continue
                data.update(scraped)

            save_problem(data)
            saved += 1
            print(f"Saved {problem_id}")
    finally:
        if driver is not None:
            driver.quit()

    return saved


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fetch Codeforces problems and statements.")
    parser.add_argument("--contest-min", type=int, default=1000)
    parser.add_argument("--contest-max", type=int, default=3000)
    parser.add_argument("--max-count", type=int, default=1000)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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
