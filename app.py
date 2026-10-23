"""FairHire - Explainable AI Resume Screening and Fairness Auditing.

Run with:  streamlit run app.py

Design notes
------------
* All logic lives in plain, side-effect-free functions at module level so it can be
  imported and unit-tested without launching the Streamlit interface.
* Streamlit is imported lazily inside ``main()``; the UI only starts when the file is
  executed as a script (which is how Streamlit runs it).
* Scoring is deterministic and transparent: no machine-learning model, no randomness,
  no network calls, and no protected characteristics are ever used to rank candidates.
"""

from __future__ import annotations

import difflib
import hashlib
import html as _html
import io
import json
import math
import random
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

import auth

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB per file (a PDF may hold many resumes)
SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")

# Scoring weights (points out of 100). Components that cannot be evaluated because the
# user did not configure them are dropped and the remaining weights are rescaled.
WEIGHT_SKILLS = 60.0
WEIGHT_EXPERIENCE = 25.0
WEIGHT_KEYWORDS = 15.0
DEFAULT_WEIGHTS = (WEIGHT_SKILLS, WEIGHT_EXPERIENCE, WEIGHT_KEYWORDS)

# canonical skill -> accepted spellings (all matched case-insensitively on word edges)
SKILL_BANK: Dict[str, List[str]] = {
    "python": ["python", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "go": ["golang"],
    "rust": ["rust"],
    "r": ["r programming", "rstudio"],
    "sql": ["sql", "t-sql", "pl/sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular", "angularjs"],
    "node.js": ["node.js", "nodejs", "node"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "streamlit": ["streamlit"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch", "torch"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision"],
    "data analysis": ["data analysis", "data analytics"],
    "data visualization": ["data visualization", "data visualisation", "tableau", "power bi"],
    "statistics": ["statistics", "statistical analysis"],
    "excel": ["excel", "microsoft excel"],
    "git": ["git", "github", "gitlab"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "linux": ["linux", "unix"],
    "rest api": ["rest api", "rest apis", "restful", "rest"],
    "ci/cd": ["ci/cd", "continuous integration", "jenkins", "github actions"],
    "agile": ["agile", "scrum", "kanban"],
    "project management": ["project management", "project manager"],
    "communication": ["communication", "communicating"],
    "leadership": ["leadership", "team lead", "led a team"],
    "problem solving": ["problem solving", "problem-solving"],
}

DEFAULT_REQUIRED_SKILLS = ["python", "sql", "machine learning"]

STOPWORDS = frozenset(
    """
    about above after again also among and any are because been before being below between
    both but can could did does doing down during each etc few for from further had has have
    having here hers him himself his how into its itself just more most must need not now off
    once only other our out over own same should some such than that the their them then there
    these they this those through too under until upon very was were what when where which while
    who whom why will with within without would you your yours able role team work working
    ability candidate candidates company looking required requirements responsibilities strong
    experience years year preferred plus including include includes using use used
    """.split()
)

SAMPLE_RESUMES: Dict[str, str] = {
    "sample_resume_1.txt": (
        "Aarav Mehta\n"
        "Email: aarav.mehta@example.com | Phone: +91 98765 43210\n"
        "Linkedin: https://www.linkedin.com/in/aarav-mehta\n\n"
        "Summary\nData scientist with 5 years of experience building machine learning models.\n"
        "Skills: Python, SQL, pandas, scikit-learn, machine learning, Tableau, Git\n"
        "Experience\n2019 - 2024 Data Scientist, Example Analytics\n"
    ),
    "sample_resume_2.txt": (
        "Name: Priya Sharma\n"
        "Email: priya.s@example.org\nPhone: 9123456780\n\n"
        "Backend developer with 3 years of experience in Java and SQL.\n"
        "Skills: Java, MySQL, REST API, Docker, Agile\n"
    ),
    "sample_resume_3.txt": (
        "Rahul Verma\nrahul.verma@example.net\n\n"
        "Fresher. Completed coursework in Python and statistics. Built a small Streamlit app.\n"
        "Skills: Python, statistics, Excel\n"
    ),
    "sample_resume_4.txt": (
        "Sneha Iyer\nsneha.iyer@example.com | +91 99887 76655\n\n"
        "Senior ML engineer, 8+ years of experience in NLP, deep learning and Python.\n"
        "Skills: Python, PyTorch, NLP, machine learning, SQL, AWS, Docker, leadership\n"
    ),
}

SAMPLE_JOB_DESCRIPTION = (
    "We are hiring a Data Scientist to build and evaluate machine learning models. "
    "You will use Python and SQL to analyse data, communicate findings, and work with "
    "stakeholders. Minimum 2 years of experience. Experience with pandas and "
    "scikit-learn is a plus."
)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp ``value`` into [low, high]; NaN becomes ``low``."""
    if value != value:  # NaN check
        return low
    return max(low, min(high, value))


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def csv_safe(value):
    """Neutralise spreadsheet formula injection in exported text cells."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


# --------------------------------------------------------------------------------------
# File reading
# --------------------------------------------------------------------------------------


SPLIT_MODES = {
    "Detect automatically": "auto",
    "One resume per page": "per_page",
    "Treat each PDF as one resume": "single",
}
MAX_PDF_PAGES = 400
MAX_RESUMES_PER_RUN = 300
MAX_DETAIL_CARDS = 25
MAX_REVIEW_ITEMS = 50
MAX_STRESS_ITEMS = 100

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad"), None)


def clean_extracted_text(text: str) -> str:
    """Tidy text pulled out of PDFs and Word files (ligatures, odd spaces, runs of blank lines)."""
    text = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH).replace("\x00", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _precheck(filename: str, data: bytes) -> Optional[str]:
    name = (filename or "").strip()
    if not data:
        return f"'{name}' is empty (0 bytes)."
    if len(data) > MAX_FILE_BYTES:
        return f"'{name}' is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB and was skipped."
    if not name.lower().endswith(SUPPORTED_EXTENSIONS):
        return f"'{name}' has an unsupported type. Please upload TXT, PDF or DOCX files."
    return None


def read_resume_bytes(filename: str, data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(text, error)`` for one file, treating it as a single resume. Exactly one is ``None``."""
    name = (filename or "").strip()
    lower = name.lower()
    problem = _precheck(name, data)
    if problem:
        return None, problem

    try:
        if lower.endswith(".txt"):
            for encoding in ("utf-8-sig", "utf-16", "latin-1"):
                try:
                    text = data.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:  # pragma: no cover - latin-1 never fails
                return None, f"'{name}' could not be decoded as text."
        elif lower.endswith(".pdf"):
            text = _read_pdf(data)
        else:
            text = _read_docx(data)
    except ImportError as exc:
        return None, f"'{name}' cannot be read because a required library is missing: {exc}."
    except Exception as exc:  # noqa: BLE001 - surface any parser failure to the user
        return None, f"'{name}' could not be read ({type(exc).__name__}). The file may be corrupted or password-protected."

    text = clean_extracted_text(text)
    if not text:
        return None, f"'{name}' contains no extractable text (scanned images are not supported)."
    return text, None


def read_pdf_pages(data: bytes) -> List[str]:
    """Text of every page of a PDF (cleaned). Raises ``ValueError`` with a readable reason."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("the PDF is password-protected") from exc
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"the PDF has more than {MAX_PDF_PAGES} pages")
    return [clean_extracted_text(page.extract_text() or "") for page in reader.pages]


def _read_pdf(data: bytes) -> str:
    return "\n".join(read_pdf_pages(data))


def _read_docx(data: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


# ---- several resumes in one PDF -------------------------------------------------------

_EMAIL_FIND = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_SECTION_LINE_RE = re.compile(
    r"^(?:professional\s+|work\s+)?(?:summary|objective|profile|experience|education|skills|projects|certifications|achievements)\b",
    re.IGNORECASE,
)


def _head_lines(page: str, count: int = 14) -> List[str]:
    return [line.strip() for line in (page or "").splitlines() if line.strip()][:count]


def _page_identities(page: str) -> Set[str]:
    """Emails and phone numbers near the top of the page, used to tell resumes apart."""
    head = "\n".join(_head_lines(page))
    found = {"email:" + m.group(0).lower() for m in _EMAIL_FIND.finditer(head)}
    for phone in _PHONE_RE.finditer(head):
        digits = re.sub(r"\D", "", phone.group(0))
        groups = re.findall(r"\d+", phone.group(0))
        if 10 <= len(digits) <= 15 and not all(len(g) == 4 and 1900 <= int(g) <= 2100 for g in groups):
            found.add("phone:" + digits[-10:])
    return found


def _looks_like_resume_start(page: str) -> bool:
    """A name-like line at the top plus a typical section heading near the top."""
    lines = _head_lines(page, 12)
    return bool(lines) and any(_is_name_line(ln) for ln in lines[:3]) and any(_SECTION_LINE_RE.match(ln) for ln in lines)


def detect_resume_starts(pages: Sequence[str]) -> List[int]:
    """Indexes of the pages that begin a new resume.

    A page starts a new resume when none of its contact details match the current resume's,
    or when it has no contact details but opens with a name and a section heading.
    Continuation pages and running headers that repeat any of the same contact details stay together.
    """
    if not pages:
        return []
    starts = [0]
    current = _page_identities(pages[0])
    for index in range(1, len(pages)):
        identities = _page_identities(pages[index])
        if identities:
            if not current:
                if _looks_like_resume_start(pages[index]):
                    starts.append(index)
                current = set(identities)
            elif identities & current:
                current |= identities
            else:
                starts.append(index)
                current = set(identities)
        elif _looks_like_resume_start(pages[index]):
            starts.append(index)
            current = set()
    return starts


def split_pdf_pages(pages: Sequence[str], mode: str = "auto") -> List[Tuple[int, int]]:
    """Inclusive (first, last) page indexes of each resume in the PDF."""
    count = len(pages)
    if count == 0:
        return []
    if mode == "single":
        return [(0, count - 1)]
    if mode == "per_page":
        return [(i, i) for i in range(count) if pages[i].strip()]
    starts = detect_resume_starts(pages)
    return [(start, (starts[k + 1] - 1 if k + 1 < len(starts) else count - 1)) for k, start in enumerate(starts)]


def _range_text(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def ingest_uploads(
    files: Sequence[Tuple[str, bytes]], split_mode: str = "auto"
) -> Tuple[List[Tuple[str, str]], List[Dict[str, object]], List[str]]:
    """Turn uploaded files into ``(label, text)`` resumes.

    Returns ``(documents, report_rows, problems)``. A PDF holding several resumes becomes
    several documents labelled with their page ranges.
    """
    mode = split_mode if split_mode in ("auto", "per_page", "single") else "auto"
    documents: List[Tuple[str, str]] = []
    rows: List[Dict[str, object]] = []
    problems: List[str] = []

    def skipped(name: str, kind: str, message: str) -> None:
        problems.append(message)
        rows.append({"file": name, "kind": kind, "pages": "-", "resumes": 0, "detail": "", "status": "Skipped"})

    for filename, data in files:
        name = (filename or "").strip() or "unnamed file"
        lower = name.lower()
        kind = lower.rsplit(".", 1)[-1].upper() if "." in lower else "?"

        if not lower.endswith(".pdf"):
            text, error = read_resume_bytes(name, data)
            if error:
                skipped(name, kind, error)
            else:
                documents.append((name, str(text)))
                rows.append({"file": name, "kind": kind, "pages": "-", "resumes": 1, "detail": "", "status": "Read"})
            continue

        precheck = _precheck(name, data)
        if precheck:
            skipped(name, kind, precheck)
            continue
        try:
            pages = read_pdf_pages(data)
        except ImportError as exc:
            skipped(name, kind, f"'{name}' cannot be read because a required library is missing: {exc}.")
            continue
        except ValueError as exc:
            skipped(name, kind, f"'{name}' could not be read: {exc}.")
            continue
        except Exception as exc:  # noqa: BLE001
            skipped(name, kind, f"'{name}' could not be read ({type(exc).__name__}). The file may be corrupted.")
            continue
        if not any(page.strip() for page in pages):
            skipped(name, kind, f"'{name}' contains no extractable text (scanned images are not supported).")
            continue

        kept = []
        for first, last in split_pdf_pages(pages, mode):
            text = "\n\n".join(pages[first : last + 1]).strip()
            if text:
                kept.append((first + 1, last + 1, text))
        blank_pages = sum(1 for page in pages if not page.strip())
        if blank_pages and mode == "per_page":
            problems.append(f"{blank_pages} blank page(s) in '{name}' were skipped.")
        for first, last, text in kept:
            label = f"{name} (page {first})" if len(kept) > 1 and first == last else (
                f"{name} (pages {_range_text(first, last)})" if len(kept) > 1 else name
            )
            documents.append((label, text))
        rows.append(
            {
                "file": name, "kind": "PDF", "pages": len(pages), "resumes": len(kept),
                "detail": ", ".join(_range_text(f, l) for f, l, _ in kept), "status": "Read",
            }
        )

    if len(documents) > MAX_RESUMES_PER_RUN:
        problems.append(f"{len(documents)} resumes were found. Only the first {MAX_RESUMES_PER_RUN} were used.")
        documents = documents[:MAX_RESUMES_PER_RUN]
    return documents, rows, problems


def ingest_report_html(rows: Sequence[Dict[str, object]]) -> str:
    """Collapsible summary of how each uploaded file was read."""
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{_esc(r['file'])}</td><td>{_esc(r['kind'])}</td><td>{_esc(r['pages'])}</td><td>{_esc(r['resumes'])}</td>"
        f"<td>{_esc(r['detail'] or '-')}</td><td>{pill_html(str(r['status']), 'ok' if r['status'] == 'Read' else 'bad')}</td></tr>"
        for r in rows
    )
    total = sum(int(r["resumes"]) for r in rows)
    multi = any(int(r["resumes"]) > 1 for r in rows)
    return (
        f'<details class="fh-cand"{" open" if multi else ""}><summary><span class="grow">How your files were read: '
        f"{total} resume(s) from {len(rows)} file(s)</span></summary><div class=\"fh-cand-body\">"
        '<div class="fh-table-wrap"><table class="fh-table"><thead><tr><th>File</th><th>Type</th><th>Pages</th>'
        "<th>Resumes found</th><th>Pages of each resume</th><th>Status</th></tr></thead><tbody>"
        + body
        + '</tbody></table></div><div class="fh-muted" style="font-size:.85rem">If a PDF was split in the wrong places, change '
        "<b>If a PDF contains several resumes</b> and run the screening again. Scanned (image-only) PDFs cannot be read.</div>"
        "</div></details>"
    )


# --------------------------------------------------------------------------------------
# Skill extraction
# --------------------------------------------------------------------------------------

_PATTERN_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _skill_aliases(skill: str, bank: Optional[Dict[str, List[str]]] = None) -> List[str]:
    key = skill.strip().lower()
    return (bank if bank is not None else SKILL_BANK).get(key, [key])


def _alias_pattern(alias: str) -> "re.Pattern[str]":
    pattern = _PATTERN_CACHE.get(alias)
    if pattern is None:
        pattern = re.compile(
            r"(?<![A-Za-z0-9+#])" + re.escape(alias) + r"(?![A-Za-z0-9+#])", re.IGNORECASE
        )
        _PATTERN_CACHE[alias] = pattern
    return pattern


def has_skill(text: str, skill: str, skill_bank: Optional[Dict[str, List[str]]] = None) -> bool:
    if not text or not skill or not skill.strip():
        return False
    return any(_alias_pattern(alias).search(text) for alias in _skill_aliases(skill, skill_bank))


def canonical_skill(name: str, skill_bank: Optional[Dict[str, List[str]]] = None) -> str:
    """Map a user-typed skill to its canonical name when it is in the skill bank."""
    bank = skill_bank if skill_bank is not None else SKILL_BANK
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in bank:
        return key
    for canonical, aliases in bank.items():
        if key in (a.lower() for a in aliases):
            return canonical
    return key


def extract_skills(text: str, skill_bank: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return the sorted canonical skills found in ``text``."""
    if not text:
        return []
    bank = skill_bank if skill_bank is not None else SKILL_BANK
    found = []
    for canonical, aliases in bank.items():
        if any(_alias_pattern(a).search(text) for a in aliases):
            found.append(canonical)
    return sorted(found)


# --------------------------------------------------------------------------------------
# Experience extraction
# --------------------------------------------------------------------------------------

_STATED_YEARS_PATTERNS = [
    re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?experience", re.IGNORECASE),
    re.compile(r"experience[^.\n]{0,20}?(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE),
]
_DATE_RANGE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:-|\u2013|\u2014|to)\s*((?:19|20)\d{2}|present|current|now)\b",
    re.IGNORECASE,
)
_EDUCATION_WORDS = re.compile(
    r"\b(university|college|school|bachelor|master|b\.?sc|m\.?sc|b\.?tech|m\.?tech|degree|diploma|"
    r"gpa|cgpa|education|graduat\w*)\b",
    re.IGNORECASE,
)


def _date_intervals(text: str, year_now: int) -> List[Tuple[int, int]]:
    """Employment-style date ranges found in ``text`` (lines about education are skipped)."""
    intervals: List[Tuple[int, int]] = []
    for line in (text or "").splitlines():
        if _EDUCATION_WORDS.search(line):
            continue
        for match in _DATE_RANGE.finditer(line):
            start = int(match.group(1))
            end_raw = match.group(2).lower()
            end = year_now if end_raw in ("present", "current", "now") else int(end_raw)
            if start <= end <= year_now + 1 and end - start <= 50:
                intervals.append((start, end))
    return intervals


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def extract_experience_years(text: str, current_year: Optional[int] = None) -> float:
    """Best-effort years of professional experience, between 0 and 40.

    Takes the larger of (a) explicit statements such as "5 years of experience" and
    (b) the union of employment date ranges (lines that mention education are skipped).
    """
    if not text:
        return 0.0
    year_now = current_year if current_year is not None else date.today().year

    stated = 0.0
    for pattern in _STATED_YEARS_PATTERNS:
        for match in pattern.finditer(text):
            try:
                stated = max(stated, float(match.group(1)))
            except ValueError:
                continue

    merged = _merge_intervals(_date_intervals(text, year_now))
    dated = float(sum(end - start for start, end in merged))

    return float(min(40.0, max(stated, dated, 0.0)))


# --------------------------------------------------------------------------------------
# Anonymization (best effort)
# --------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:linkedin\.com|github\.com|twitter\.com|facebook\.com|instagram\.com)/\S+",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\w)\+?\(?\d[\d\s().\-]{8,}\d(?!\w)")
_NAME_LABEL_RE = re.compile(r"(?im)^(\s*(?:candidate\s+name|full\s+name|name)\s*[:\-]\s*).+$")
_PERSONAL_LABEL_RE = re.compile(
    r"(?im)^\s*(?:gender|sex|marital\s+status|nationality|religion|caste|age|date\s+of\s+birth|dob|"
    r"father'?s?\s+name|mother'?s?\s+name|photo|address|location)\s*[:\-].*$"
)
_DOB_RE = re.compile(r"(?i)\b(?:dob|date of birth|born on)\b[^\n]*")
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b\.?"
)
_GENDERED_RE = re.compile(
    r"\b(?:he|she|him|his|hers|her|himself|herself|mr|mrs|ms|miss|male|female)\b", re.IGNORECASE
)
_HEADING_WORDS = frozenset(
    "resume curriculum vitae cv profile summary objective experience education skills projects "
    "contact details personal information work professional".split()
)
_ROLE_WORDS = frozenset(
    "developer engineer engineering analyst analytics scientist science manager management designer design intern fresher "
    "consultant consulting architect administrator specialist student programmer programming lead director officer executive "
    "assistant associate senior junior principal staff head chief technician teacher nurse accountant marketing sales finance "
    "operations development".split()
)
_NAME_PARTICLES = frozenset("de da di du la le van von bin binti al el dos das del".split())
_NAME_LINE_RE = re.compile(
    r"^[A-Z][A-Za-z'\u2019.\-]+(?: (?:[A-Z][A-Za-z'\u2019.\-]*|" + "|".join(sorted(_NAME_PARTICLES)) + r")){1,4}$"
)
_NAME_VALUE_RE = re.compile(r"(?im)^\s*(?:candidate\s+name|full\s+name|name)\s*[:\-]\s*(.+)$")


def _is_name_line(line: str) -> bool:
    """True for a short line that looks like a person's name (and not a title, heading or skill list)."""
    line = (line or "").strip()
    if not line or len(line) > 40 or not _NAME_LINE_RE.match(line):
        return False
    words = {w.lower().strip(".") for w in line.split()}
    return not (words & _HEADING_WORDS or words & _ROLE_WORDS or extract_skills(line))


def _name_tokens(name: str) -> List[str]:
    """The full name and its distinctive parts, for masking later mentions."""
    tokens = [name.strip()] if " " in name.strip() else []
    for part in re.findall(r"[A-Za-z][A-Za-z'\u2019\-]+", name):
        low = part.lower()
        if (
            len(part) >= 3 and low not in STOPWORDS and low not in _HEADING_WORDS and low not in _ROLE_WORDS
            and low not in _NAME_PARTICLES and not extract_skills(part)
        ):
            tokens.append(part)
    return sorted(set(t for t in tokens if t), key=len, reverse=True)


def _mask_phone(match: "re.Match[str]") -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if not 10 <= len(digits) <= 15:
        return raw
    groups = re.findall(r"\d+", raw)
    if all(len(g) == 4 and 1900 <= int(g) <= 2100 for g in groups):
        return raw  # a run of years, not a phone number
    return "[PHONE]"


def anonymize_text(text: str) -> str:
    """Mask names, contact details, personal attributes and gendered words.

    This is best-effort pattern matching. It finds the name from a "Name:" line or the first
    few lines, then masks later mentions of it. It will miss some identifying details (for
    example a different spelling, a nickname, or a name that only appears mid-sentence).
    """
    if not text:
        return ""
    out = text
    out = _EMAIL_RE.sub("[EMAIL]", out)
    out = _URL_RE.sub("[URL]", out)
    out = _PHONE_RE.sub(_mask_phone, out)

    names: List[str] = []
    labelled = _NAME_VALUE_RE.search(out)
    if labelled:
        names.append(labelled.group(1).strip())
    out = _NAME_LABEL_RE.sub(r"\1[NAME]", out)
    out = _DOB_RE.sub("[DOB]", out)
    out = _PERSONAL_LABEL_RE.sub("[REDACTED PERSONAL DETAIL]", out)
    out = _STREET_RE.sub("[ADDRESS]", out)

    lines = out.splitlines()
    seen = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        seen += 1
        if _is_name_line(stripped):
            names.append(stripped)
            lines[index] = "[NAME]"
            break
        if seen >= 3:
            break
    out = "\n".join(lines)

    for name in names:
        for token in _name_tokens(name):
            out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])", "[NAME]", out, flags=re.IGNORECASE)

    return _GENDERED_RE.sub("[G]", out)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _keyword_set(text: str, limit: int = 40) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]{3,}", (text or "").lower())
    counts: Dict[str, int] = {}
    for word in words:
        word = word.strip(".-")
        if len(word) >= 4 and word not in STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


def _normalize_weights(weights: Optional[Sequence[float]]) -> Tuple[float, float, float]:
    """Three non-negative weights, or the defaults when the input is unusable or all zero."""
    try:
        values = [max(0.0, float(w)) for w in weights]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return DEFAULT_WEIGHTS
    if len(values) != 3 or sum(values) <= 0:
        return DEFAULT_WEIGHTS
    return (values[0], values[1], values[2])


def score_candidate(
    resume_text: str,
    required_skills: Sequence[str],
    jd_text: str = "",
    min_years: float = 0.0,
    current_year: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    skill_bank: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    """Deterministic, explainable 0-100 score.

    Components (weights rescaled over the components that can be evaluated):
      * required-skill coverage ........ 60
      * experience vs. minimum years ... 25
      * job-description keyword overlap  15
    """
    text = resume_text or ""
    bank = skill_bank if skill_bank is not None else SKILL_BANK
    reasons: List[str] = []
    required = _dedupe_keep_order(
        s for s in (canonical_skill(x, bank) for x in (required_skills or [])) if s
    )
    if not required and (jd_text or "").strip():
        required = extract_skills(jd_text, bank)
        if required:
            reasons.append("No required skills were configured, so skills found in the job description were used.")

    min_years = max(0.0, float(min_years or 0.0))

    empty = {
        "score": 0.0,
        "skill_points": 0.0,
        "experience_points": 0.0,
        "keyword_points": 0.0,
        "matched_skills": [],
        "missing_skills": list(required),
        "extra_skills": [],
        "years_experience": 0.0,
        "keyword_coverage": 0.0,
        "reasons": reasons,
    }
    if not text.strip():
        empty["reasons"] = reasons + ["Resume is empty, so the score is 0."]
        return empty

    matched = [s for s in required if has_skill(text, s, bank)]
    missing = [s for s in required if s not in matched]
    years = extract_experience_years(text, current_year)
    all_found = extract_skills(text, bank)
    extra = [s for s in all_found if s not in required]

    jd_keywords = _keyword_set(jd_text)
    resume_lower = text.lower()
    kw_hits = [k for k in jd_keywords if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", resume_lower)]
    kw_fraction = (len(kw_hits) / len(jd_keywords)) if jd_keywords else None

    skill_fraction = (len(matched) / len(required)) if required else None
    exp_fraction = min(years / min_years, 1.0) if min_years > 0 else None

    w_skill, w_exp, w_kw = _normalize_weights(weights)
    components = [
        ("skill_points", w_skill, skill_fraction),
        ("experience_points", w_exp, exp_fraction),
        ("keyword_points", w_kw, kw_fraction),
    ]
    active = [(key, w, f) for key, w, f in components if f is not None and w > 0]
    total_weight = sum(w for _, w, _ in active)

    result = dict(empty)
    result.update(
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        years_experience=round(years, 1),
        keyword_coverage=round(kw_fraction, 3) if kw_fraction is not None else 0.0,
    )

    if total_weight <= 0:
        reasons.append("Nothing to score against: add required skills, a job description or minimum years.")
        result["reasons"] = reasons
        return result

    score = 0.0
    for key, weight, fraction in active:
        points = 100.0 * weight * fraction / total_weight
        result[key] = round(points, 1)
        score += points
    result["score"] = round(clamp(score), 1)

    if required:
        reasons.append(f"Matched {len(matched)} of {len(required)} required skills ({result['skill_points']} pts).")
        if missing:
            reasons.append("Missing: " + ", ".join(missing) + ".")
    if min_years > 0:
        reasons.append(
            f"Detected about {result['years_experience']} years of experience against a {min_years:g}-year "
            f"target ({result['experience_points']} pts)."
        )
    if jd_keywords:
        reasons.append(
            f"Covers {len(kw_hits)} of {len(jd_keywords)} job-description keywords ({result['keyword_points']} pts)."
        )
    result["reasons"] = reasons
    return result


def screen_resumes(
    documents: Sequence[Tuple[str, str]],
    required_skills: Sequence[str],
    jd_text: str = "",
    min_years: float = 0.0,
    anonymize: bool = True,
    current_year: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    skill_bank: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Score and rank ``(file_name, text)`` pairs. Returns ``(ranked_results, warnings)``."""
    warnings: List[str] = []
    seen: Dict[str, str] = {}
    results: List[Dict[str, object]] = []

    for file_name, raw_text in documents:
        text = raw_text or ""
        if not text.strip():
            warnings.append(f"'{file_name}' has no text and was skipped.")
            continue
        fingerprint = text_fingerprint(text)
        if fingerprint in seen:
            warnings.append(f"'{file_name}' duplicates '{seen[fingerprint]}' and was skipped.")
            continue
        seen[fingerprint] = file_name

        working_text = anonymize_text(text) if anonymize else text
        detail = score_candidate(working_text, required_skills, jd_text, min_years, current_year, weights, skill_bank)
        detail["file_name"] = file_name
        detail["text_used"] = working_text
        detail["original_text"] = text[:20000]
        detail["fingerprint"] = fingerprint
        detail["redactions"] = count_redactions(working_text) if anonymize else 0
        detail["proxy_flags"] = detect_proxy_risks(working_text, current_year)
        detail["evidence"] = find_skill_evidence(working_text, detail["matched_skills"], skill_bank)
        detail["experience_evidence"] = experience_evidence(working_text, current_year)
        results.append(detail)

    # Deterministic order: score desc, then more matched skills, then file name.
    results.sort(key=lambda r: (-float(r["score"]), -len(r["matched_skills"]), str(r["file_name"])))
    for position, item in enumerate(results, start=1):
        item["rank"] = position
        item["candidate_id"] = f"Candidate {position:03d}"
    return results, warnings


# --------------------------------------------------------------------------------------
# Custom skills
# --------------------------------------------------------------------------------------


def merged_skill_bank(custom: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """Built-in skills plus the user's own (never mutates the global ``SKILL_BANK``)."""
    bank = {name: list(aliases) for name, aliases in SKILL_BANK.items()}
    for name, aliases in (custom or {}).items():
        bank[name] = _dedupe_keep_order(bank.get(name, []) + list(aliases))
    return bank


def parse_custom_skills(text: str, max_skills: int = 100) -> Tuple[Dict[str, List[str]], List[str]]:
    """Parse lines like ``rag = retrieval augmented generation, rag pipeline``.

    Returns ``(skills, warnings)``. Blank lines and lines starting with ``#`` are ignored.
    """
    skills: Dict[str, List[str]] = {}
    warnings: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, alias_part = line.partition("=")
        name = name.strip().lower()
        if not name or len(name) > 40:
            warnings.append(f"Skipped an invalid custom skill line: '{line[:40]}'.")
            continue
        if len(skills) >= max_skills and name not in skills:
            warnings.append(f"Only the first {max_skills} custom skills are used.")
            break
        aliases = [a.strip().lower() for a in alias_part.split(",") if 0 < len(a.strip()) <= 40]
        skills[name] = _dedupe_keep_order([name] + aliases)
    return skills, warnings


def weights_percent(weights: Optional[Sequence[float]]) -> Tuple[float, float, float]:
    """The weights as percentages that add up to 100 (defaults if unusable)."""
    w = _normalize_weights(weights)
    total = sum(w)
    return tuple(round(100.0 * x / total, 1) for x in w)  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# Proxy-risk flags
# --------------------------------------------------------------------------------------

_PROXY_RULES = [
    ("Institution", "medium", re.compile(r"\b(?:university|college|institute)\b[^\n,;|]{0,40}", re.I),
     "Where someone studied can signal region, income or community background."),
    ("Institution", "medium", re.compile(r"\b(?:IIT|IIM|NIT|BITS|MIT)\b[^\n,;|]{0,30}"),
     "Institution prestige can act as a proxy for socioeconomic background."),
    ("Location", "medium", re.compile(r"\b(?:based in|located in|resident of|relocat\w+ to)\b[^\n.;|]{0,40}", re.I),
     "Place of residence can correlate with ethnicity, caste or income."),
    ("Location", "medium", re.compile(r"\b(?:pin|zip|postal)\s*(?:code)?\s*[:\-]?\s*\d{5,6}\b", re.I),
     "Postal codes can correlate with ethnicity, caste or income."),
    ("Age / graduation year", "high", re.compile(r"\b(?:graduated|graduation|class of|batch of|passed out)\b[^\n.;|]{0,25}\b(?:19|20)\d{2}\b", re.I),
     "Graduation year reveals approximate age."),
    ("Age / graduation year", "high", re.compile(r"\b\d{2}\s*years?\s*old\b", re.I), "States age directly."),
    ("Career gap or leave", "high", re.compile(r"\b(?:career break|gap year|sabbatical|maternity|paternity|parental leave|caregiv\w+|time off)\b[^\n.;|]{0,30}", re.I),
     "Gaps and leave correlate with gender, caregiving and health."),
    ("Family / marital status", "high", re.compile(r"\b(?:married|divorced|widowed|spouse|husband|wife|pregnan\w+|my children|kids)\b", re.I),
     "Family status is a protected or sensitive characteristic in many places."),
    ("Religion / community", "high", re.compile(r"\b(?:church|temple|mosque|masjid|gurdwara|synagogue|islamic|hindu|christian|muslim|sikh|jewish|buddhist)\b[^\n,;|]{0,30}", re.I),
     "Can reveal religion or community."),
    ("Nationality / work authorization", "medium", re.compile(r"\b(?:citizen(?:ship)?|visa|work permit|green card|nationality|native speaker|mother tongue)\b[^\n.;|]{0,30}", re.I),
     "Can reveal national origin or immigration status."),
    ("Health / disability", "high", re.compile(r"\b(?:disabilit\w+|wheelchair|chronic|medical leave|health condition)\b", re.I),
     "Health and disability information should not influence screening."),
]


def detect_proxy_risks(text: str, current_year: Optional[int] = None) -> List[Dict[str, str]]:
    """Heuristically list details that can stand in for protected characteristics.

    These are flags for a human to consider hiding or ignoring, not accusations, and the
    list is not exhaustive.
    """
    if not text:
        return []
    flags: List[Dict[str, str]] = []
    seen = set()
    per_category: Dict[str, int] = {}

    def add(category: str, level: str, evidence: str, why: str) -> None:
        key = (category, evidence.lower())
        if key in seen or per_category.get(category, 0) >= 3 or len(flags) >= 15:
            return
        seen.add(key)
        per_category[category] = per_category.get(category, 0) + 1
        flags.append({"category": category, "level": level, "evidence": evidence, "why": why})

    for category, level, pattern, why in _PROXY_RULES:
        for match in pattern.finditer(text):
            evidence = re.sub(r"\s+", " ", match.group(0)).strip()[:70]
            add(category, level, evidence, why)

    year_now = current_year if current_year is not None else date.today().year
    merged = _merge_intervals(_date_intervals(text, year_now))
    for previous, following in zip(merged, merged[1:]):
        gap = following[0] - previous[1]
        if gap >= 2:
            add(
                "Career gap or leave", "high", f"{previous[1]} to {following[0]}: no dated role for {gap} years",
                "Gaps and leave correlate with gender, caregiving and health.",
            )
    return flags


# --------------------------------------------------------------------------------------
# Before / after anonymization view
# --------------------------------------------------------------------------------------

_REDACTION_RE = re.compile(r"\[(?:EMAIL|URL|PHONE|NAME|DOB|ADDRESS|G|REDACTED PERSONAL DETAIL)\]")


def count_redactions(text: str) -> int:
    return len(_REDACTION_RE.findall(text or ""))


def redaction_diff_html(original: str, anonymized: str, limit: int = 3000) -> str:
    """Side-by-side view with removed text (left) and its replacement (right) highlighted."""
    a_tokens = re.findall(r"\s+|\S+", (original or "")[:limit])
    b_tokens = re.findall(r"\s+|\S+", (anonymized or "")[:limit])
    matcher = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    left: List[str] = []
    right: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        a_seg = "".join(a_tokens[i1:i2])
        b_seg = "".join(b_tokens[j1:j2])
        if tag == "equal":
            left.append(_esc(a_seg))
            right.append(_esc(b_seg))
            continue
        if a_seg:
            left.append(f'<mark class="fh-del">{_esc(a_seg)}</mark>')
        if b_seg:
            right.append(f'<mark class="fh-add">{_esc(b_seg)}</mark>')
    return (
        '<div class="fh-diff"><div><h6>Original</h6><pre>' + "".join(left) + "</pre></div>"
        '<div><h6>After anonymization</h6><pre>' + "".join(right) + "</pre></div></div>"
    )


# --------------------------------------------------------------------------------------
# Bias stress test
# --------------------------------------------------------------------------------------

ALT_NAMES = ["Aarav Mehta", "Emily Carter", "Mohammed Al-Farsi", "Wei Zhang", "Chioma Okafor", "Maria Gonzalez"]
_GENDER_SWAPS = {
    "he": "she", "she": "he", "him": "her", "his": "her", "hers": "his", "her": "his",
    "himself": "herself", "herself": "himself", "male": "female", "female": "male",
}
_GENDER_RE = re.compile(r"\b(?:" + "|".join(sorted(_GENDER_SWAPS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_NAME_LABEL_VALUE_RE = re.compile(r"(?i)^(?:candidate\s+name|full\s+name|name)\s*[:\-]\s*(.+)$")


def swap_gender_terms(text: str) -> str:
    """Swap gendered pronouns and words (he/she, his/her, male/female), keeping case."""

    def replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        new = _GENDER_SWAPS[word.lower()]
        if word.isupper() and len(word) > 1:
            return new.upper()
        return new.capitalize() if word[0].isupper() else new

    return _GENDER_RE.sub(replace, text or "")


def detect_name(text: str) -> Optional[str]:
    """Best-effort guess of the candidate's name (a "Name:" line, else a name-like line near the top)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for line in lines[:6]:
        match = _NAME_LABEL_VALUE_RE.match(line)
        if match and match.group(1).strip():
            return match.group(1).strip()[:60]
    for line in lines[:3]:
        if _is_name_line(line):
            return line
    return None


def swap_name(text: str, old: str, new: str) -> str:
    """Replace the full name, then each name part, with the new name (case-insensitive)."""
    out = re.sub(re.escape(old), new, text or "", flags=re.IGNORECASE)
    new_parts = new.split()
    for index, part in enumerate(old.split()):
        if len(part) >= 3 and new_parts:
            out = re.sub(r"\b" + re.escape(part) + r"\b", new_parts[min(index, len(new_parts) - 1)], out, flags=re.IGNORECASE)
    return out


def bias_stress_test(
    resume_text: str,
    required_skills: Sequence[str],
    jd_text: str = "",
    min_years: float = 0.0,
    weights: Optional[Sequence[float]] = None,
    skill_bank: Optional[Dict[str, List[str]]] = None,
    anonymize: bool = False,
    current_year: Optional[int] = None,
    tolerance: float = 0.05,
) -> Dict[str, object]:
    """Re-score the same resume with the name and gendered words changed.

    A deterministic scorer that ignores identity should give the same score every time.
    This checks that property only; it does not prove the whole process is unbiased.
    """
    name = detect_name(resume_text)
    variants: List[Tuple[str, str]] = [("Original", resume_text)]
    alt_names = [n for n in ALT_NAMES if not name or n.lower() != name.lower()]
    if name:
        for alt in alt_names[:4]:
            variants.append((f"Name changed to {alt}", swap_name(resume_text, name, alt)))
    variants.append(("Gender terms swapped", swap_gender_terms(resume_text)))
    if name and alt_names:
        variants.append(("Name and gender terms swapped", swap_gender_terms(swap_name(resume_text, name, alt_names[-1]))))

    rows: List[Dict[str, object]] = []
    base: Optional[float] = None
    for label, variant_text in variants:
        used = anonymize_text(variant_text) if anonymize else variant_text
        score = float(score_candidate(used, required_skills, jd_text, min_years, current_year, weights, skill_bank)["score"])
        if base is None:
            base = score
        rows.append(
            {
                "variant": label,
                "score": score,
                "delta": round(score - base, 1),
                "text_changed": variant_text != resume_text,
            }
        )
    max_abs = max(abs(float(r["delta"])) for r in rows)
    return {"name_detected": name is not None, "rows": rows, "max_abs_delta": max_abs, "passed": max_abs <= tolerance}


# --------------------------------------------------------------------------------------
# Statistics for the fairness audit
# --------------------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson confidence interval for a proportion (unknown when ``n`` is 0)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> Optional[Dict[str, float]]:
    """Two-sided z-test for the difference between two selection rates (group 1 - group 2)."""
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        z, p_value = 0.0, 1.0
    else:
        z = (p1 - p2) / se
        p_value = math.erfc(abs(z) / math.sqrt(2))
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return {
        "z": z,
        "p_value": min(1.0, max(0.0, p_value)),
        "diff": diff,
        "ci_low": diff - 1.96 * se_diff,
        "ci_high": diff + 1.96 * se_diff,
    }


# --------------------------------------------------------------------------------------
# Human review log
# --------------------------------------------------------------------------------------

REVIEW_DECISIONS = ("Not reviewed", "Advance", "Hold", "Reject")


def validate_review_entry(decision: str, reason: str) -> Optional[str]:
    """Return an error message, or ``None`` when the entry is acceptable."""
    if decision not in REVIEW_DECISIONS:
        return "Unknown decision."
    if len(reason or "") > 500:
        return "The reason is longer than 500 characters."
    if decision in ("Hold", "Reject") and not (reason or "").strip():
        return "A job-related reason is required for Hold and Reject."
    return None


def review_log_dataframe(log: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    columns = ["Time (UTC)", "Candidate", "Score", "Decision", "Reason"]
    rows = [
        {
            "Time (UTC)": e.get("time", ""),
            "Candidate": e.get("candidate", ""),
            "Score": e.get("score", ""),
            "Decision": e.get("decision", ""),
            "Reason": e.get("reason", ""),
        }
        for e in log.values()
    ]
    return pd.DataFrame(rows, columns=columns)


def now_utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------------------
# Score distribution
# --------------------------------------------------------------------------------------


def score_histogram_html(scores: Sequence[float], bins: int = 10) -> str:
    bins = max(1, int(bins))
    width = 100.0 / bins
    counts = [0] * bins
    for score in scores:
        counts[min(int(clamp(float(score)) // width), bins - 1)] += 1
    peak = max(counts) or 1
    columns = []
    for index, count in enumerate(counts):
        low, high = index * width, (index + 1) * width
        tone = score_tone((low + high) / 2)
        height = 3 + int(round(100 * count / peak)) if count else 3
        color = _TONE_COLOR[tone] if count else "var(--border)"
        columns.append(
            f'<div class="fh-hist-col"><div class="fh-hist-count">{count if count else ""}</div>'
            f'<div class="fh-hist-bar" style="height:{height}px;background:{color}"></div>'
            f'<div class="fh-hist-label">{low:g}-{high:g}</div></div>'
        )
    return '<div class="fh-hist">' + "".join(columns) + "</div>"


# --------------------------------------------------------------------------------------
# Downloadable audit report (standalone HTML; print to PDF from the browser)
# --------------------------------------------------------------------------------------

REPORT_CSS = (
    "body{font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#0f172a;margin:0;background:#f4f6fb;}"
    ".wrap{max-width:980px;margin:0 auto;padding:32px 24px 64px;}"
    "h1{font-size:28px;margin:0 0 4px;}h2{font-size:19px;margin:32px 0 10px;border-bottom:2px solid #e2e8f0;padding-bottom:6px;}"
    "h3{font-size:15px;margin:18px 0 6px;}.muted{color:#64748b;font-size:13px;}"
    "table{width:100%;border-collapse:collapse;background:#fff;font-size:13px;margin:8px 0;}"
    "th,td{border:1px solid #e2e8f0;padding:7px 9px;text-align:left;vertical-align:top;}th{background:#f1f4fa;}"
    ".card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin:10px 0;page-break-inside:avoid;}"
    ".banner{background:#fef3c7;color:#92400e;padding:10px 14px;border-radius:8px;margin:10px 0;font-size:13px;}"
    ".ok{color:#047857;font-weight:600;}.bad{color:#b91c1c;font-weight:600;}"
    "@media print{body{background:#fff;}.wrap{padding:0;}}"
)

LIMITATIONS_TEXT = (
    "Masking identifying details does not make a hiring process unbiased: skills, schools, employers, gaps and writing style "
    "can act as proxies for protected characteristics. Keyword scoring misses synonyms and context, and experience "
    "extraction is heuristic. Fairness figures from synthetic data are simulations, not evidence about real hiring. The "
    "bias stress test checks only that scores do not change when names and gendered words change. This tool supports "
    "human reviewers and must not be used to reject candidates automatically. Check local employment and data-protection "
    "law, obtain consent for processing resumes, and audit real outcomes with proper statistical testing."
)


def _report_fairness_block(section: Dict[str, object]) -> str:
    report = section["report"]
    simulated = bool(section.get("simulated"))
    rows = []
    for group in sorted(report["rates"]):
        low, high = report["intervals"][group]
        rows.append(
            f"<tr><td>{_esc(group)}</td><td>{report['counts'][group]}</td><td>{report['selected_counts'][group]}</td>"
            f"<td>{report['rates'][group] * 100:.1f}%</td><td>{low * 100:.1f}% to {high * 100:.1f}%</td></tr>"
        )
    ratio = report["disparate_impact_ratio"]
    if ratio is None:
        ratio_text = "undefined"
    else:
        verdict = '<span class="ok">at or above 0.80</span>' if ratio >= 0.8 else '<span class="bad">below 0.80, flag for review</span>'
        ratio_text = f"{ratio:.3f} ({verdict})"
    sig = report.get("significance")
    if sig:
        sig_text = (
            f"{_esc(sig['high_group'])} minus {_esc(sig['low_group'])}: {sig['diff'] * 100:.1f} points "
            f"(95% CI {sig['ci_low'] * 100:.1f} to {sig['ci_high'] * 100:.1f}), p = {sig['p_value']:.3f}. "
            + ("Statistically significant at the 5% level." if sig["significant"] else "Not statistically significant; the gap could be chance.")
        )
    else:
        sig_text = "Not available (needs at least two groups)."
    banner = '<div class="banner">SIMULATED data. This is not evidence about real candidates or a real hiring process.</div>' if simulated else ""
    notes = "".join(f'<div class="muted">{_esc(n)}</div>' for n in report["notes"])
    return (
        f"<h3>{_esc(section['title'])}</h3>{banner}"
        "<table><thead><tr><th>Group</th><th>People</th><th>Selected</th><th>Selection rate</th><th>95% interval</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        f"<div>Demographic parity difference: <b>{report['demographic_parity_difference']:.3f}</b>. "
        f"Disparate impact ratio: <b>{ratio_text}</b>.</div><div>Significance: {sig_text}</div>{notes}"
    )


def build_report_html(
    results: Sequence[Dict[str, object]],
    settings: Dict[str, object],
    fairness_sections: Optional[Sequence[Dict[str, object]]] = None,
    review_log: Optional[Sequence[Dict[str, object]]] = None,
    generated_on: Optional[str] = None,
    show_files: bool = False,
) -> str:
    """Build a self-contained HTML audit report (print it to PDF from the browser)."""
    generated_on = generated_on or now_utc_text()
    scores = [float(r["score"]) for r in results]
    average = sum(scores) / len(scores) if scores else 0.0
    w_skill, w_exp, w_kw = weights_percent(settings.get("weights"))
    required = settings.get("required") or []
    settings_rows = [
        ("Resumes screened", str(len(results))),
        ("Average score", f"{average:.1f}"),
        ("Required skills", ", ".join(required) if required else "None configured (job description and minimum years used)"),
        ("Minimum years of experience", f"{float(settings.get('min_years', 0) or 0):g}"),
        ("Scoring weights", f"skills {w_skill:g}% / experience {w_exp:g}% / keywords {w_kw:g}%"),
        ("Anonymization before scoring", "On" if settings.get("anonymize") else "Off"),
        ("Shortlist size", str(settings.get("shortlist", ""))),
    ] + ([("Job-description wording flags", str(settings["jd_flags"]))] if "jd_flags" in settings else [])
    settings_html = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in settings_rows)

    top_n = int(settings.get("shortlist", 0) or 0)
    rank_rows = []
    cards = []
    for r in results:
        name = _esc(r["candidate_id"]) + (f" ({_esc(r['file_name'])})" if show_files else "")
        flags = r.get("proxy_flags", [])
        rank_rows.append(
            f"<tr><td>{int(r['rank'])}</td><td>{name}</td><td>{float(r['score']):.1f}</td><td>{float(r['years_experience']):g}</td>"
            f"<td>{_esc(', '.join(r['matched_skills']) or '-')}</td><td>{_esc(', '.join(r['missing_skills']) or '-')}</td>"
            f"<td>{len(flags)}</td><td>{'Yes' if int(r['rank']) <= top_n else ''}</td></tr>"
        )
        reasons = "".join(f"<li>{_esc(x)}</li>" for x in r["reasons"]) or "<li>No details.</li>"
        flag_html = "".join(f"<li>{_esc(f['category'])}: {_esc(f['evidence'])}</li>" for f in flags) or "<li>None found by the heuristic checks.</li>"
        cards.append(
            f'<div class="card"><b>#{int(r["rank"])} {name}</b> - {float(r["score"]):.1f} / 100'
            f'<div class="muted">Skills {float(r["skill_points"]):.1f} pts, experience {float(r["experience_points"]):.1f} pts, '
            f'keywords {float(r["keyword_points"]):.1f} pts</div><h3>Why this score</h3><ul>{reasons}</ul>'
            f"<h3>Proxy-risk flags</h3><ul>{flag_html}</ul></div>"
        )

    fairness_html = "".join(_report_fairness_block(s) for s in (fairness_sections or [])) or (
        '<div class="muted">No fairness audit was run in this session.</div>'
    )
    if review_log:
        log_rows = "".join(
            f"<tr><td>{_esc(e.get('time', ''))}</td><td>{_esc(e.get('candidate', ''))}</td><td>{_esc(e.get('score', ''))}</td>"
            f"<td>{_esc(e.get('decision', ''))}</td><td>{_esc(e.get('reason', ''))}</td></tr>"
            for e in review_log
        )
        log_html = (
            "<table><thead><tr><th>Time (UTC)</th><th>Candidate</th><th>Score</th><th>Decision</th><th>Reason</th></tr></thead>"
            f"<tbody>{log_rows}</tbody></table>"
        )
    else:
        log_html = '<div class="muted">No reviewer decisions were recorded.</div>'

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>FairHire audit report</title>'
        f"<style>{REPORT_CSS}</style></head><body><div class=\"wrap\"><h1>FairHire audit report</h1>"
        f'<div class="muted">Generated {_esc(generated_on)}. Decision support only; a human makes every decision.</div>'
        f"<h2>Settings</h2><table>{settings_html}</table>"
        "<h2>Ranking</h2><table><thead><tr><th>Rank</th><th>Candidate</th><th>Score</th><th>Years</th><th>Matched skills</th>"
        "<th>Missing skills</th><th>Proxy flags</th><th>Shortlist</th></tr></thead><tbody>"
        + "".join(rank_rows)
        + "</tbody></table><h2>Score explanations</h2>"
        + "".join(cards)
        + f"<h2>Fairness audit</h2>{fairness_html}<h2>Reviewer decisions</h2>{log_html}"
        f"<h2>Limitations and responsible use</h2><p>{_esc(LIMITATIONS_TEXT)}</p></div></body></html>"
    )


def synthetic_fairness_sections(syn: Optional[pd.DataFrame]) -> List[Dict[str, object]]:
    if syn is None or len(syn) == 0:
        return []
    sections = []
    for title, column in (
        ("Merit-only ranking (simulated)", "selected_merit_only"),
        ("Biased process with simulated reviewer penalty (simulated)", "selected_biased_process"),
    ):
        sections.append({"title": title, "report": fairness_report(syn["group"].tolist(), syn[column].tolist()), "simulated": True})
    return sections



def results_to_dataframe(results: Sequence[Dict[str, object]], include_file_names: bool = False) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "Rank": r["rank"],
            "Candidate": r["candidate_id"],
            "Score": r["score"],
            "Skills pts": r["skill_points"],
            "Experience pts": r["experience_points"],
            "Keyword pts": r["keyword_points"],
            "Years (detected)": r["years_experience"],
            "Matched skills": ", ".join(r["matched_skills"]),
            "Missing skills": ", ".join(r["missing_skills"]),
            "Proxy flags": "; ".join(sorted({f["category"] for f in r.get("proxy_flags", [])})),
            "Why": " ".join(r["reasons"]),
        }
        if include_file_names:
            row["File"] = csv_safe(r["file_name"])
        rows.append(row)
    columns = [
        "Rank", "Candidate", "Score", "Skills pts", "Experience pts", "Keyword pts",
        "Years (detected)", "Matched skills", "Missing skills", "Proxy flags", "Why",
    ] + (["File"] if include_file_names else [])
    return pd.DataFrame(rows, columns=columns)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    safe = df.copy()
    for column in safe.columns:
        series = safe[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            safe[column] = series.map(csv_safe)
    return safe.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------------------
# Fairness metrics
# --------------------------------------------------------------------------------------


def selection_rates(groups: Sequence[str], selected: Sequence[bool]) -> Dict[str, float]:
    """Share of each group that was selected. Groups with no members are absent."""
    if len(groups) != len(selected):
        raise ValueError("groups and selected must have the same length")
    totals: Dict[str, int] = {}
    chosen: Dict[str, int] = {}
    for group, flag in zip(groups, selected):
        totals[group] = totals.get(group, 0) + 1
        chosen[group] = chosen.get(group, 0) + (1 if bool(flag) else 0)
    return {g: chosen[g] / totals[g] for g in totals}


def demographic_parity_difference(rates: Dict[str, float]) -> float:
    """Largest gap in selection rate between any two groups (0 = parity)."""
    if len(rates) < 2:
        return 0.0
    return max(rates.values()) - min(rates.values())


def disparate_impact_ratio(rates: Dict[str, float]) -> Optional[float]:
    """Lowest selection rate divided by the highest. ``None`` when undefined."""
    if len(rates) < 2:
        return None
    highest = max(rates.values())
    if highest <= 0:
        return None
    return min(rates.values()) / highest


def fairness_report(groups: Sequence[str], selected: Sequence[bool]) -> Dict[str, object]:
    rates = selection_rates(groups, selected)
    counts: Dict[str, int] = {}
    chosen: Dict[str, int] = {}
    for group, flag in zip(groups, selected):
        counts[group] = counts.get(group, 0) + 1
        chosen[group] = chosen.get(group, 0) + (1 if bool(flag) else 0)
    dpd = demographic_parity_difference(rates)
    dir_ = disparate_impact_ratio(rates)
    intervals = {g: wilson_interval(chosen[g], counts[g]) for g in counts}
    notes: List[str] = []
    if len(rates) < 2:
        notes.append("At least two groups are needed to compare selection rates.")
    small = [g for g, n in counts.items() if n < 10]
    if small:
        notes.append("Small groups (<10 people): " + ", ".join(sorted(small)) + ". Results are statistically unreliable.")
    if dir_ is None and len(rates) >= 2:
        notes.append("Disparate impact ratio is undefined because nobody was selected.")

    significance = None
    if len(rates) >= 2:
        ordered = sorted(rates, key=lambda g: (rates[g], g))
        low_group, high_group = ordered[0], ordered[-1]
        test = two_proportion_z_test(chosen[high_group], counts[high_group], chosen[low_group], counts[low_group])
        if test is not None:
            significance = {
                "high_group": high_group,
                "low_group": low_group,
                "p_value": test["p_value"],
                "z": test["z"],
                "diff": test["diff"],
                "ci_low": test["ci_low"],
                "ci_high": test["ci_high"],
                "significant": test["p_value"] < 0.05,
            }
        if len(rates) > 2:
            notes.append("The p-value compares only the highest and lowest groups, with no multiple-comparison correction.")
    return {
        "counts": counts,
        "selected_counts": chosen,
        "rates": rates,
        "intervals": intervals,
        "demographic_parity_difference": dpd,
        "disparate_impact_ratio": dir_,
        "four_fifths_pass": (dir_ >= 0.8) if dir_ is not None else None,
        "significance": significance,
        "notes": notes,
    }


def select_top_k(scores: Sequence[float], k: int) -> List[bool]:
    """Mark the ``k`` highest scores as selected (ties broken by position)."""
    k = max(0, min(int(k), len(scores)))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    chosen = set(order[:k])
    return [i in chosen for i in range(len(scores))]


# --------------------------------------------------------------------------------------
# Synthetic data for fairness experiments
# --------------------------------------------------------------------------------------

_SYNTH_SKILLS = [
    "python", "sql", "machine learning", "pandas", "scikit-learn", "excel", "statistics",
    "tableau", "git", "docker", "aws", "communication", "leadership", "java", "nlp",
]


def generate_synthetic_candidates(
    n: int = 200,
    seed: int = 42,
    group_b_share: float = 0.4,
    bias_penalty: float = 0.0,
    shortlist_fraction: float = 0.25,
    required_skills: Optional[Sequence[str]] = None,
    min_years: float = 2.0,
) -> pd.DataFrame:
    """Create simulated candidates whose qualifications are drawn from the SAME
    distribution for both groups. The ``group`` label is used ONLY for auditing; it is
    never an input to ``score_candidate``.

    ``bias_penalty`` (points) simulates a biased human reviewer who marks Group B down,
    so the audit has something to detect. It is a teaching device, not real evidence.
    """
    n = max(0, int(n))
    rng = random.Random(seed)
    required = list(required_skills) if required_skills else list(DEFAULT_REQUIRED_SKILLS)
    rows = []
    for i in range(n):
        group = "Group B" if rng.random() < group_b_share else "Group A"
        skills = rng.sample(_SYNTH_SKILLS, rng.randint(2, 8))
        years = rng.randint(0, 12)
        resume = (
            f"Candidate profile. {years} years of experience.\n"
            f"Skills: {', '.join(skills)}\n"
        )
        detail = score_candidate(resume, required, "", min_years)
        merit = float(detail["score"])
        biased = clamp(merit - (bias_penalty if group == "Group B" else 0.0))
        rows.append(
            {
                "candidate_id": f"Synthetic {i + 1:03d}",
                "group": group,
                "skills": ", ".join(sorted(skills)),
                "years": years,
                "merit_score": merit,
                "biased_score": round(biased, 1),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=["candidate_id", "group", "skills", "years", "merit_score", "biased_score"],
    )
    k = int(round(shortlist_fraction * len(df)))
    df["selected_merit_only"] = select_top_k(df["merit_score"].tolist(), k) if len(df) else []
    df["selected_biased_process"] = select_top_k(df["biased_score"].tolist(), k) if len(df) else []
    return df


def audit_uploaded_outcomes(df: pd.DataFrame, group_col: str, selected_col: str) -> Dict[str, object]:
    """Audit a user-supplied table of real or hypothetical outcomes."""
    if group_col not in df.columns or selected_col not in df.columns:
        raise ValueError("The chosen columns do not exist in the uploaded table.")
    clean = df[[group_col, selected_col]].dropna()
    truthy = {"1", "true", "yes", "y", "selected", "hired", "shortlisted", "t"}
    flags = [str(v).strip().lower() in truthy for v in clean[selected_col]]
    groups = [str(v).strip() for v in clean[group_col]]
    return fairness_report(groups, flags)


# --------------------------------------------------------------------------------------
# Methodology text
# --------------------------------------------------------------------------------------

METHODOLOGY_MD = """
### How scoring works
Each resume receives a score from 0 to 100 made of up to three transparent parts:

| Component | Weight | How it is computed |
|---|---|---|
| Required skills | 60 | Share of the required skills found in the resume (word-boundary text matching with aliases). |
| Experience | 25 | Detected years of experience divided by your minimum, capped at 100%. |
| Job-description keywords | 15 | Share of the job description's most frequent meaningful words found in the resume. |

If a component is not configured (for example no minimum years), its weight is redistributed over the others.
There is no machine-learning model and no randomness: the same input always gives the same score.

### Anonymization
Best-effort regular-expression masking of emails, phones, links, labelled names and personal-detail lines
(gender, age, nationality, religion, address, date of birth), a likely name on the first line, street addresses, and
gendered words. **It will miss some identifying information** (for example a name inside a sentence, a university
that signals background, or a career gap). Treat it as a risk-reducing step, not a guarantee.

### Fairness audit
* **Selection rate** - the share of each group that is shortlisted.
* **Demographic parity difference** - the largest gap between group selection rates (0 means equal rates).
* **Disparate impact ratio** - the lowest selection rate divided by the highest. The "four-fifths rule" of thumb
  flags ratios below 0.80 for further review; it is a screening heuristic, not a legal determination.

The demo experiment uses **synthetic** candidates whose qualifications are drawn from the same distribution for
both groups. The "biased process" is a simulated reviewer penalty added only to show what the audit can detect.
**These results are simulations, not evidence about real hiring.**

### Job description check
A rule-based scan of a job post for wording that can discourage or exclude groups of applicants: gender-coded words, age
references, nationality or language requirements, family status, appearance, health requirements and prestige-school wording.
It also counts masculine- and feminine-coded words from published word lists as a rough indicator. It is not exhaustive and a
flag is a prompt to reconsider the wording, not proof of discrimination. Automatic replacements are limited to safe word swaps;
review the result before using it.

### Skill evidence
For every matched skill the candidate card shows short snippets from the resume (the text the scorer saw, so anonymized when
anonymization is on) with the match highlighted, plus the sentences behind the experience figure. Skills the scorer could not
find are listed as not found. Evidence is shown on screen only and is not copied into saved reports.

### Accounts and storage
You can sign in with a **real account** or a **prototype** (simulated) sign-in. Real accounts store an email and a salted password
hash, plus your saved jobs, reports and audits, in the app's database. With a Postgres `database_url` configured, data survives
restarts. Without one, a local file is used and can be wiped when the app restarts. Administrators can manage accounts but cannot
read anyone's saved data. Resumes are never saved in either mode. The prototype stores nothing outside the browser tab.

### Reading real resumes
Upload TXT, PDF or DOCX files. A single PDF can hold many resumes: FairHire looks at the contact details and layout at the top
of each page to find where each resume starts, keeps continuation pages together, and scores each resume separately. You can
override this with "One resume per page" or "Treat each PDF as one resume". The table "How your files were read" shows what
was found. Limits: scanned (image-only) PDFs cannot be read because there is no text layer; two-column layouts and unusual
templates can scramble the text order; automatic splitting is a heuristic and can be wrong for resumes without contact details.
Always check the file summary before trusting a ranking.

### Adjustable weights
The three weights can be changed with sliders and are rescaled to add up to 100%, so rankings update live. Any component
you have not configured is skipped. Changing weights changes who ranks first, so record the weights you used (the report does).

### Bias stress test
Each resume is re-scored with its name replaced by other names and its gendered words swapped. The score should not move.
This verifies one property of the scoring step only. It cannot detect bias from the criteria you choose, from proxies, or
from the people reviewing the results.

### Proxy-risk flags
Pattern checks list details that can stand in for protected traits (school, location, graduation year, career gaps, family
or religious details, nationality, health). They are heuristics that miss things and can over-flag, and they stay visible
after anonymization. Consider hiding or ignoring them when reviewing.

### Statistical significance
For the highest and lowest selection rates the audit shows a 95% interval for each rate, the difference with its interval,
and a two-sided two-proportion z-test. With small groups the test has little power; a non-significant result does not show
that there is no bias. With three or more groups no multiple-comparison correction is applied.

### Reviewer decisions and report
You record Advance, Hold or Reject for each candidate, with a job-related reason required for Hold and Reject. The
downloadable report bundles the settings, ranking, explanations, flags, fairness results and your decisions.

### Limitations and responsible use
* Masking identity does not make a process unbiased. Skills lists, schools, employers, gaps and writing style can
  act as proxies for protected characteristics.
* Keyword matching misses synonyms, penalises unusual wording, and cannot judge quality, context or potential.
* Experience extraction is heuristic and can mis-read dates; PDFs with complex layouts may extract badly.
* Protected characteristics are never used to score or rank individuals. Group labels are used only to audit outcomes.
* Use the tool to support human reviewers, never to automatically reject people. Check local employment and data-protection
  law, obtain consent for processing resumes, and audit with real outcome data and statistical tests before relying on it.
"""

# --------------------------------------------------------------------------------------
# UI components (pure functions that return HTML strings; easy to test, no Streamlit needed)
# --------------------------------------------------------------------------------------

THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#F4F6FB", "surface": "#FFFFFF", "surface2": "#F1F4FA", "border": "#E2E8F0",
        "text": "#0F172A", "muted": "#64748B", "primary": "#4F46E5", "primary2": "#7C3AED",
        "primary_soft": "#EEF2FF", "ok": "#059669", "ok_soft": "#D1FAE5", "warn": "#B45309",
        "warn_soft": "#FEF3C7", "bad": "#DC2626", "bad_soft": "#FEE2E2",
        "shadow": "0 1px 2px rgba(15,23,42,.06), 0 8px 24px rgba(15,23,42,.06)",
        "hero_a": "#312E81", "hero_b": "#6D28D9", "scheme": "light",
    },
    "dark": {
        "bg": "#0B1020", "surface": "#121A30", "surface2": "#1A2440", "border": "#26335A",
        "text": "#E6E9F5", "muted": "#9AA7C7", "primary": "#818CF8", "primary2": "#A78BFA",
        "primary_soft": "#232C5A", "ok": "#34D399", "ok_soft": "#0F3D33", "warn": "#FBBF24",
        "warn_soft": "#43360F", "bad": "#F87171", "bad_soft": "#4A1D24",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35)",
        "hero_a": "#1E1B4B", "hero_b": "#4C1D95", "scheme": "dark",
    },
}

BASE_CSS = """
html, body, .stApp, button, input, textarea, select {font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: var(--bg) !important; color: var(--text);}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stHeader"] * {color: var(--muted);}
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 1280px; padding: 4.5rem 1.5rem 3rem;}
h1, h2, h3, h4, h5, h6, label, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p {color: var(--text) !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
.stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="textarea"], [data-baseweb="select"] > div {background: var(--surface2) !important; color: var(--text) !important;
  border-color: var(--border) !important; border-radius: 10px !important;}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {color: var(--muted) !important; opacity: .8;}
[data-baseweb="select"] svg, [data-testid="stNumberInput"] button svg {fill: var(--muted); color: var(--muted);}
[data-testid="stNumberInput"] button {background: var(--surface2) !important; border-color: var(--border) !important;}
[data-baseweb="tag"] {background: var(--primary_soft) !important; color: var(--primary) !important; border-radius: 8px !important;}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {color: var(--primary) !important; fill: var(--primary) !important;}
[data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul {background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border); border-radius: 10px;}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {background: var(--surface) !important; color: var(--text) !important;}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover {background: var(--primary_soft) !important;}
[data-testid="stFileUploaderDropzone"] {background: var(--surface2) !important; border: 1.5px dashed var(--border) !important; border-radius: 12px !important;}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] small {color: var(--muted) !important;}
[data-testid="stFileUploaderDropzone"] button {background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important;}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {color: var(--text) !important;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 600; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); transition: transform .08s ease, box-shadow .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color: var(--primary); color: var(--primary); transform: translateY(-1px);}
.stButton > button[kind="primary"] {background: linear-gradient(135deg, var(--primary), var(--primary2)); color: #fff; border: 0;
  box-shadow: 0 6px 18px rgba(79,70,229,.35);}
.stButton > button[kind="primary"]:hover {color: #fff; box-shadow: 0 8px 22px rgba(79,70,229,.5);}
[data-testid="stCheckbox"] label span, [data-testid="stToggle"] label span {color: var(--text) !important;}
.stTabs [data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid var(--border); background: transparent;}
.stTabs button[data-baseweb="tab"] {background: transparent !important; color: var(--muted) !important; font-weight: 600; padding: .6rem 1rem;}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {color: var(--primary) !important;}
.stTabs [data-baseweb="tab-highlight"] {background: var(--primary) !important; height: 3px; border-radius: 3px;}
.stTabs [data-baseweb="tab-border"] {background: transparent !important;}
[data-testid="stSlider"] [role="slider"] {background: var(--primary) !important; box-shadow: none;}
[data-testid="stSlider"] [data-testid="stThumbValue"], [data-testid="stSliderThumbValue"],
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color: var(--muted) !important;}
[data-testid="stColumn"]:has(.fh-card-marker), [data-testid="column"]:has(.fh-card-marker) {background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 1.25rem 1.25rem 1rem; box-shadow: var(--shadow);}
hr {border-color: var(--border) !important;}

.fh-hero {display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: space-between; padding: 1.4rem 1.6rem;
  border-radius: 18px; color: #fff; background: radial-gradient(1200px 300px at 90% -20%, rgba(255,255,255,.18), transparent),
  linear-gradient(120deg, var(--hero_a), var(--hero_b)); box-shadow: var(--shadow);}
.fh-hero-main {display: flex; gap: 1rem; align-items: center;}
.fh-logo {width: 52px; height: 52px; border-radius: 14px; background: rgba(255,255,255,.16); display: flex; align-items: center; justify-content: center; font-size: 1.7rem;}
.fh-title {font-size: 1.8rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; color: #fff;}
.fh-sub {opacity: .88; margin-top: .2rem; font-size: .98rem; color: #fff;}
.fh-chips {display: flex; flex-wrap: wrap; gap: .4rem;}
.fh-chips span {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25); padding: .3rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff;}
.fh-section {display: flex; align-items: center; gap: .6rem; font-weight: 700; font-size: 1.05rem; margin: .2rem 0 .6rem; color: var(--text);}
.fh-step {width: 26px; height: 26px; border-radius: 8px; background: linear-gradient(135deg, var(--primary), var(--primary2)); color: #fff; font-size: .8rem;
  display: inline-flex; align-items: center; justify-content: center;}
.fh-kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin: .4rem 0 1rem;}
.fh-kpi {background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: .9rem 1rem; box-shadow: var(--shadow);}
.fh-kpi-label {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 700;}
.fh-kpi-value {font-size: 1.7rem; font-weight: 800; color: var(--text); margin-top: .15rem; letter-spacing: -.02em;}
.fh-kpi-hint {font-size: .78rem; color: var(--muted); margin-top: .1rem;}
.fh-kpi.tone-ok .fh-kpi-value {color: var(--ok);} .fh-kpi.tone-bad .fh-kpi-value {color: var(--bad);} .fh-kpi.tone-warn .fh-kpi-value {color: var(--warn);}
.fh-table-wrap {overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); margin-bottom: 1rem;}
.fh-table {width: 100%; border-collapse: collapse; font-size: .9rem;}
.fh-table th {text-align: left; padding: .7rem .9rem; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
  background: var(--surface2); border-bottom: 1px solid var(--border); white-space: nowrap;}
.fh-table td {padding: .7rem .9rem; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: middle;}
.fh-table tr:last-child td {border-bottom: 0;}
.fh-table tbody tr:hover td {background: var(--surface2);}
.fh-rank {display: inline-flex; width: 28px; height: 28px; border-radius: 8px; align-items: center; justify-content: center; font-weight: 800; font-size: .85rem;
  background: var(--primary_soft); color: var(--primary);}
.fh-score {display: flex; align-items: center; gap: .6rem; min-width: 150px;}
.fh-score b {min-width: 2.6rem; font-variant-numeric: tabular-nums;}
.fh-bar {flex: 1; height: 8px; border-radius: 999px; background: var(--surface2); overflow: hidden; border: 1px solid var(--border);}
.fh-bar i {display: block; height: 100%; border-radius: 999px;}
.fh-chip {display: inline-block; padding: .15rem .55rem; margin: .12rem .2rem .12rem 0; border-radius: 999px; font-size: .74rem; font-weight: 600; white-space: nowrap;}
.fh-chip.ok {background: var(--ok_soft); color: var(--ok);} .fh-chip.bad {background: var(--bad_soft); color: var(--bad);}
.fh-chip.neutral {background: var(--surface2); color: var(--muted); border: 1px solid var(--border);}
.fh-pill {display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-size: .75rem; font-weight: 700;}
.fh-pill.ok {background: var(--ok_soft); color: var(--ok);} .fh-pill.bad {background: var(--bad_soft); color: var(--bad);}
.fh-pill.warn {background: var(--warn_soft); color: var(--warn);} .fh-pill.neutral {background: var(--surface2); color: var(--muted);}
.fh-muted {color: var(--muted);}
.fh-note {display: flex; gap: .7rem; padding: .8rem 1rem; border-radius: 12px; margin: .4rem 0 .8rem; font-size: .9rem; border: 1px solid var(--border);
  background: var(--surface2); color: var(--text);}
.fh-note.warn {background: var(--warn_soft); border-color: transparent; color: var(--warn);} .fh-note.bad {background: var(--bad_soft); border-color: transparent; color: var(--bad);}
.fh-note.info {background: var(--primary_soft); border-color: transparent; color: var(--primary);}
.fh-empty {text-align: center; padding: 2.5rem 1rem; border: 1.5px dashed var(--border); border-radius: 16px; background: var(--surface); color: var(--muted);}
.fh-empty .big {font-size: 2.2rem;} .fh-empty b {color: var(--text); display: block; margin: .4rem 0 .2rem; font-size: 1.05rem;}
.fh-cand {background: var(--surface); border: 1px solid var(--border); border-radius: 14px; margin-bottom: .6rem; box-shadow: var(--shadow); overflow: hidden;}
.fh-cand summary {cursor: pointer; list-style: none; display: flex; align-items: center; gap: .7rem; padding: .8rem 1rem; font-weight: 600; color: var(--text);}
.fh-cand summary::-webkit-details-marker {display: none;}
.fh-cand summary .grow {flex: 1;}
.fh-cand-body {padding: .2rem 1rem 1rem; border-top: 1px solid var(--border); color: var(--text);}
.fh-cand-body h5 {margin: .9rem 0 .3rem; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted) !important;}
.fh-stack {display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: var(--surface2); border: 1px solid var(--border);}
.fh-stack i {display: block; height: 100%;}
.fh-legend {display: flex; flex-wrap: wrap; gap: 1rem; margin-top: .4rem; font-size: .8rem; color: var(--muted);}
.fh-dot {display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: .35rem;}
.fh-cand ul {margin: .2rem 0 0 1.1rem; padding: 0;} .fh-cand li {margin: .15rem 0; color: var(--text);}
.fh-cand details.fh-raw summary {padding: .5rem 0 0; font-size: .82rem; color: var(--primary); font-weight: 600;}
.fh-cand pre {white-space: pre-wrap; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: .8rem; font-size: .8rem;
  max-height: 260px; overflow: auto; color: var(--text); margin-top: .5rem;}
.fh-panel {background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: var(--shadow); height: 100%;}
.fh-panel-title {font-weight: 700; margin-bottom: .8rem; color: var(--text);}
.fh-rate {margin: .5rem 0;} .fh-rate-head {display: flex; justify-content: space-between; font-size: .85rem; margin-bottom: .25rem; color: var(--text);}
.fh-rate-head span:last-child {color: var(--muted); font-variant-numeric: tabular-nums;}
.fh-rate .fh-bar {height: 12px;}
.fh-theme-marker {display: none;}
.fh-theme-label {font-size: .75rem; color: var(--muted); text-align: right; margin-top: -.4rem;}
.fh-intro {position: fixed; top: 0; right: 0; bottom: 0; left: 0; z-index: 1000001; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; padding: 2rem; pointer-events: none; animation: fh-intro-life 9s ease forwards;
  background: radial-gradient(900px 520px at 15% 10%, rgba(124,58,237,.20), transparent), radial-gradient(900px 520px at 88% 92%, rgba(56,189,248,.18), transparent), var(--bg);}
@keyframes fh-intro-life {0% {opacity: 0; visibility: visible;} 10% {opacity: 1;} 86% {opacity: 1;} 99% {opacity: 0; visibility: visible;} 100% {opacity: 0; visibility: hidden;}}
@keyframes fh-intro-rise {0% {opacity: 0; transform: translateY(14px);} 100% {opacity: 1; transform: translateY(0);}}
@keyframes fh-breathe {0%, 100% {transform: scale(1); box-shadow: 0 0 0 0 rgba(129,140,248,.35);} 50% {transform: scale(1.07); box-shadow: 0 0 0 22px rgba(129,140,248,0);}}
@keyframes fh-dot {0%, 80%, 100% {opacity: .25; transform: scale(.8);} 40% {opacity: 1; transform: scale(1);}}
.fh-intro-logo {width: 92px; height: 92px; border-radius: 28px; display: flex; align-items: center; justify-content: center; font-size: 2.6rem; color: #fff;
  background: linear-gradient(135deg, var(--primary), var(--primary2)); animation: fh-breathe 3s ease-in-out infinite, fh-intro-rise 1.1s ease both;}
.fh-intro-title {margin-top: 1.4rem; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); animation: fh-intro-rise 1.4s ease .5s both;}
.fh-intro-sub {margin-top: .5rem; max-width: 30rem; font-size: 1.02rem; line-height: 1.55; color: var(--muted); animation: fh-intro-rise 1.4s ease 1.3s both;}
.fh-intro-steps {margin-top: .9rem; font-size: .9rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--primary); animation: fh-intro-rise 1.4s ease 2.6s both;}
.fh-intro-dots {margin-top: 1.6rem; display: flex; gap: .45rem; animation: fh-intro-rise 1.4s ease 3.6s both;}
.fh-intro-dots span {width: 9px; height: 9px; border-radius: 50%; background: var(--primary); animation: fh-dot 1.4s ease-in-out infinite;}
.fh-intro-dots span:nth-child(2) {animation-delay: .2s;} .fh-intro-dots span:nth-child(3) {animation-delay: .4s;}
@media (prefers-reduced-motion: reduce) {.fh-intro, .fh-intro * {animation-duration: 5s !important; animation-iteration-count: 1 !important;} .fh-intro-logo, .fh-intro-dots span {animation-name: none !important;}}
.fh-footer {text-align: center; color: var(--muted); font-size: .8rem; margin-top: 2rem;}
.fh-chip.warn {background: var(--warn_soft); color: var(--warn);}
.fh-hist {display: flex; align-items: flex-end; gap: 6px; height: 175px; padding: .4rem .2rem 0; background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; box-shadow: var(--shadow); margin-bottom: 1rem; padding: 1rem .8rem .5rem;}
.fh-hist-col {flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-end; height: 100%;}
.fh-hist-bar {width: 100%; border-radius: 6px 6px 2px 2px; min-height: 3px;}
.fh-hist-count {font-size: .75rem; font-weight: 700; color: var(--text); min-height: 1rem;}
.fh-hist-label {font-size: .66rem; color: var(--muted); margin-top: .25rem; white-space: nowrap;}
.fh-diff {display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin-top: .5rem;}
.fh-diff h6 {margin: 0 0 .25rem; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted) !important;}
.fh-diff pre {margin: 0 !important; max-height: 240px;}
mark.fh-del {background: var(--bad_soft); color: var(--bad); border-radius: 4px; padding: 0 2px;}
mark.fh-add {background: var(--ok_soft); color: var(--ok); border-radius: 4px; padding: 0 2px;}
[data-testid="stExpander"] {background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background: transparent !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color: var(--text) !important;}
[data-testid="stForm"] {border: 1px solid var(--border) !important; border-radius: 14px !important; background: var(--surface); padding: 1rem;}
@media (max-width: 720px) {.fh-diff {grid-template-columns: 1fr;}}
.fh-brand {display: flex; align-items: center; gap: .8rem;}
.fh-brand-logo {width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; color: #fff;
  background: linear-gradient(135deg, var(--primary), var(--primary2));}
.fh-brand-name {font-weight: 800; font-size: 1.25rem; color: var(--text); line-height: 1.1;}
.fh-brand-tag {color: var(--muted); font-size: .85rem;}
.fh-auth-title {font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; margin: 1.1rem 0 .4rem; color: var(--text);}
.fh-auth-sub {color: var(--muted); line-height: 1.55; margin-bottom: 1rem;}
.fh-meter {margin: .2rem 0 .6rem;}
mark.fh-hit {background: var(--ok_soft); color: var(--ok); border-radius: 4px; padding: 0 2px; font-weight: 600;}
.fh-ev {display: flex; gap: .6rem; align-items: baseline; margin: .35rem 0; font-size: .88rem; color: var(--text);}
.fh-ev .fh-chip {flex: none;} .fh-ev-text {line-height: 1.5;}
[data-testid="stRadio"] label, [data-testid="stRadio"] label p {color: var(--text) !important;}
@media (max-width: 720px) {.block-container {padding: 4.2rem .7rem 2rem;} .fh-title {font-size: 1.4rem;} .fh-hero {padding: 1rem;} .fh-kpi-value {font-size: 1.35rem;}}
"""


def _theme_vars(name: str) -> str:
    palette = THEMES[name]
    variables = "".join(f"--{key}: {value};" for key, value in palette.items() if key != "scheme")
    return variables + f"color-scheme: {palette['scheme']};"


def theme_css(dark: bool) -> str:
    """Return one minified <style> block containing BOTH themes.

    The active theme is chosen by a marker element (see ``theme_marker_html``) using the
    ``:has()`` selector, so the CSS text never changes when the user flips the toggle.
    A plain ``:root`` block for the current mode is included as a fallback for browsers
    that do not support ``:has()``.
    """
    mode = "dark" if dark else "light"
    css = (
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
        f":root {{{_theme_vars(mode)}}}"
        f":root:not(:has(.fh-dark-marker)) {{{_theme_vars('light')}}}"
        f":root:has(.fh-dark-marker) {{{_theme_vars('dark')}}}"
        f':root[data-fh-theme="light"] {{{_theme_vars("light")}}}'
        f':root[data-fh-theme="dark"] {{{_theme_vars("dark")}}}'
        + BASE_CSS
    )
    css = re.sub(r"\s+", " ", css).strip()  # one line, no blank lines (Markdown-safe)
    return f"<style>{css}</style>"


def theme_script_html(dark: bool) -> str:
    """A <script> that injects the theme CSS straight into the parent page's <head>.

    Used as a second, independent way to apply the theme in case the <style> block sent
    through Markdown is stripped or ignored by the Streamlit version in use.
    """
    css = theme_css(dark)[len("<style>"):-len("</style>")]
    payload = json.dumps(css).replace("</", "<\\/")
    mode = "dark" if dark else "light"
    return (
        "<script>(function(){try{var d=window.parent.document;var id='fh-theme-style';"
        "var el=d.getElementById(id);if(!el){el=d.createElement('style');el.id=id;d.head.appendChild(el);}"
        f"el.textContent={payload};"
        f"d.documentElement.setAttribute('data-fh-theme','{mode}');"
        "}catch(e){}})();</script>"
    )


def theme_label_html(dark: bool) -> str:
    return f'<div class="fh-theme-label">Theme: {"Dark" if dark else "Light"}</div>'


def theme_marker_html(dark: bool) -> str:
    """Tiny element whose presence switches the page to the dark palette."""
    return '<div class="fh-theme-marker fh-dark-marker"></div>' if dark else '<div class="fh-theme-marker"></div>'


def _esc(value: object) -> str:
    """Escape text for safe use inside HTML that is passed through Markdown."""
    text = "" if value is None else str(value)
    return (
        _html.escape(text, quote=True).replace("$", "&#36;").replace("\r", "").replace("\n", "&#10;")
    )


def score_tone(score: float) -> str:
    if score >= 75:
        return "ok"
    if score >= 50:
        return "warn"
    return "bad"


_TONE_COLOR = {"ok": "var(--ok)", "warn": "var(--warn)", "bad": "var(--bad)", "neutral": "var(--muted)"}
_BAR_GRADIENTS = [
    "linear-gradient(90deg, var(--primary), var(--primary2))",
    "linear-gradient(90deg, #0EA5E9, #22D3EE)",
    "linear-gradient(90deg, #F59E0B, #F97316)",
    "linear-gradient(90deg, #10B981, #34D399)",
]


def hero_html() -> str:
    return (
        '<div class="fh-hero"><div class="fh-hero-main"><div class="fh-logo">⚖️</div><div>'
        '<div class="fh-title">FairHire</div>'
        '<div class="fh-sub">Explainable resume screening with a built-in fairness audit</div></div></div>'
        '<div class="fh-chips"><span>Explainable scores</span><span>Anonymized review</span>'
        "<span>Fairness audit</span></div></div>"
    )


def intro_html() -> str:
    """Calm welcome screen: fades in, holds for a few seconds, then fades away by itself.

    It never intercepts clicks (``pointer-events: none``) and ends up hidden.
    """
    return (
        '<div class="fh-intro"><div class="fh-intro-logo">⚖️</div>'
        '<div class="fh-intro-title">Welcome to FairHire</div>'
        '<div class="fh-intro-sub">Fair, clear and explainable hiring. '
        "Take a deep breath, we&#39;ll handle the heavy lifting together.</div>"
        '<div class="fh-intro-steps">Anonymize · Score · Audit</div>'
        '<div class="fh-intro-dots"><span></span><span></span><span></span></div></div>'
    )


def section_html(title: str, step: Optional[int] = None) -> str:
    badge = f'<span class="fh-step">{int(step)}</span>' if step is not None else ""
    return f'<div class="fh-section">{badge}{_esc(title)}</div>'


def note_html(text: str, tone: str = "neutral", icon: str = "") -> str:
    tone = tone if tone in ("warn", "bad", "info", "neutral") else "neutral"
    icon_html = f"<span>{_esc(icon)}</span>" if icon else ""
    return f'<div class="fh-note {tone}">{icon_html}<div>{_esc(text)}</div></div>'


def empty_state_html(title: str, body: str, icon: str = "📄") -> str:
    return (
        f'<div class="fh-empty"><div class="big">{_esc(icon)}</div><b>{_esc(title)}</b>'
        f"<div>{_esc(body)}</div></div>"
    )


def kpi_card(label: str, value: object, hint: str = "", tone: str = "neutral") -> str:
    tone = tone if tone in ("ok", "warn", "bad") else "neutral"
    hint_html = f'<div class="fh-kpi-hint">{_esc(hint)}</div>' if hint else ""
    return (
        f'<div class="fh-kpi tone-{tone}"><div class="fh-kpi-label">{_esc(label)}</div>'
        f'<div class="fh-kpi-value">{_esc(value)}</div>{hint_html}</div>'
    )


def kpis_html(cards: Sequence[str]) -> str:
    return '<div class="fh-kpis">' + "".join(cards) + "</div>"


def pill_html(text: str, tone: str = "neutral") -> str:
    tone = tone if tone in ("ok", "warn", "bad") else "neutral"
    return f'<span class="fh-pill {tone}">{_esc(text)}</span>'


def chips_html(items: Sequence[str], tone: str = "neutral", limit: Optional[int] = None) -> str:
    items = list(items)
    if not items:
        return '<span class="fh-muted">None</span>'
    shown = items if limit is None else items[:limit]
    out = "".join(f'<span class="fh-chip {tone}">{_esc(i)}</span>' for i in shown)
    if limit is not None and len(items) > limit:
        out += f'<span class="fh-chip neutral">+{len(items) - limit}</span>'
    return out


def _score_cell(score: float) -> str:
    value = clamp(float(score))
    return (
        f'<div class="fh-score"><b>{value:.1f}</b><div class="fh-bar"><i style="width:{value:.1f}%;'
        f'background:{_TONE_COLOR[score_tone(value)]}"></i></div></div>'
    )


def flag_chips_html(flags: Sequence[Dict[str, str]]) -> str:
    if not flags:
        return '<span class="fh-muted">None found by the heuristic checks</span>'
    out = []
    for flag in flags:
        tone = "bad" if flag.get("level") == "high" else "warn"
        out.append(
            f'<span class="fh-chip {tone}" title="{_esc(flag.get("why", ""))}">'
            f'{_esc(flag.get("category", ""))}: {_esc(flag.get("evidence", ""))}</span>'
        )
    return "".join(out)


def results_table_html(results: Sequence[Dict[str, object]], top_n: int, show_files: bool = False) -> str:
    head = ["Rank", "Candidate", "Score", "Years", "Matched skills", "Missing skills", "Proxy flags", "Shortlist"]
    rows = []
    for r in results:
        name = _esc(r["candidate_id"])
        if show_files:
            name += f'<div class="fh-muted" style="font-size:.75rem">{_esc(r["file_name"])}</div>'
        shortlisted = int(r["rank"]) <= top_n
        flag_count = len(r.get("proxy_flags", []))
        rows.append(
            "<tr>"
            f'<td><span class="fh-rank">{int(r["rank"])}</span></td>'
            f"<td><b>{name}</b></td>"
            f"<td>{_score_cell(float(r['score']))}</td>"
            f"<td>{float(r['years_experience']):g}</td>"
            f"<td>{chips_html(r['matched_skills'], 'ok', 6)}</td>"
            f"<td>{chips_html(r['missing_skills'], 'bad', 6)}</td>"
            f"<td>{pill_html(str(flag_count), 'warn') if flag_count else pill_html('0', 'neutral')}</td>"
            f"<td>{pill_html('Shortlisted', 'ok') if shortlisted else pill_html('-', 'neutral')}</td>"
            "</tr>"
        )
    header = "".join(f"<th>{h}</th>" for h in head)
    return (
        '<div class="fh-table-wrap"><table class="fh-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def candidate_card_html(
    item: Dict[str, object],
    show_files: bool = False,
    open_card: bool = False,
    show_original: bool = False,
    skill_bank: Optional[Dict[str, List[str]]] = None,
) -> str:
    score = clamp(float(item["score"]))
    label = _esc(item["candidate_id"])
    if show_files:
        label += f' <span class="fh-muted" style="font-weight:500">({_esc(item["file_name"])})</span>'
    parts = [
        ("Required skills", float(item["skill_points"]), "var(--primary)"),
        ("Experience", float(item["experience_points"]), "#0EA5E9"),
        ("Keywords", float(item["keyword_points"]), "#F59E0B"),
    ]
    stack = "".join(f'<i style="width:{clamp(p):.1f}%;background:{color}"></i>' for _, p, color in parts)
    legend = "".join(
        f'<span><span class="fh-dot" style="background:{color}"></span>{_esc(name)} {p:.1f} pts</span>'
        for name, p, color in parts
    )
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in item["reasons"]) or '<li class="fh-muted">No details.</li>'
    used = str(item.get("text_used", ""))
    original = str(item.get("original_text", used))
    redactions = int(item.get("redactions", 0) or 0)
    if not show_original:
        # blind review: only the text the scorer actually saw (anonymized when anonymization is on)
        diff_summary = f"Text used for scoring ({redactions} items masked)" if redactions else "Text used for scoring"
        diff_body = f"<pre>{highlight_skills_html(used, item.get('matched_skills', []), skill_bank)}</pre>"
    elif redactions or used != original:
        diff_summary = f"Before / after anonymization ({redactions} items masked)"
        diff_body = redaction_diff_html(original, used)
    else:
        diff_summary = "Before / after anonymization"
        diff_body = '<div class="fh-muted" style="margin-top:.5rem">Anonymization is off or nothing needed masking.</div>'
    return (
        f'<details class="fh-cand"{" open" if open_card else ""}><summary>'
        f'<span class="fh-rank">{int(item["rank"])}</span><span class="grow">{label}</span>'
        f'{pill_html(f"{score:.1f} / 100", score_tone(score))}</summary><div class="fh-cand-body">'
        f'<h5>Score breakdown</h5><div class="fh-stack">{stack}</div><div class="fh-legend">{legend}</div>'
        f'<h5>Matched skills</h5>{chips_html(item["matched_skills"], "ok")}'
        f'<h5>Evidence from the resume</h5>{evidence_html(item)}'
        f'<h5>Missing skills</h5>{chips_html(item["missing_skills"], "bad")}'
        f'<h5>Other skills detected</h5>{chips_html(item["extra_skills"], "neutral")}'
        f'<h5>Detected experience</h5><div>{float(item["years_experience"]):g} years</div>'
        f"<h5>Why this score</h5><ul>{reasons}</ul>"
        f'<h5>Proxy-risk flags (still visible after anonymization)</h5>{flag_chips_html(item.get("proxy_flags", []))}'
        f'<details class="fh-raw"><summary>{_esc(diff_summary)}</summary>{diff_body}</details>'
        "</div></details>"
    )


def rates_bars_html(
    rates: Dict[str, float],
    counts: Optional[Dict[str, int]] = None,
    intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    rows = []
    for index, group in enumerate(sorted(rates)):
        rate = clamp(rates[group] * 100.0)
        n = f" (n={counts[group]})" if counts and group in counts else ""
        ci = ""
        if intervals and group in intervals:
            low, high = intervals[group]
            ci = f" | 95% CI {low * 100:.0f}-{high * 100:.0f}%"
        rows.append(
            f'<div class="fh-rate"><div class="fh-rate-head"><span>{_esc(group)}</span>'
            f"<span>{rate:.1f}%{_esc(n)}{_esc(ci)}</span></div>"
            f'<div class="fh-bar"><i style="width:{rate:.1f}%;background:{_BAR_GRADIENTS[index % len(_BAR_GRADIENTS)]}"></i></div></div>'
        )
    return "".join(rows) or '<span class="fh-muted">No data.</span>'


def fairness_panel_html(title: str, report: Dict[str, object]) -> str:
    ratio = report["disparate_impact_ratio"]
    dpd = float(report["demographic_parity_difference"])
    if ratio is None:
        ratio_card = kpi_card("Disparate impact ratio", "n/a", "undefined for this data")
        status = pill_html("Cannot assess", "neutral")
    elif ratio >= 0.8:
        ratio_card = kpi_card("Disparate impact ratio", f"{ratio:.2f}", "at or above the 0.80 guide", "ok")
        status = pill_html("Passes four-fifths guide", "ok")
    else:
        ratio_card = kpi_card("Disparate impact ratio", f"{ratio:.2f}", "below the 0.80 guide", "bad")
        status = pill_html("Flag for review", "bad")

    sig = report.get("significance")
    sig_html = ""
    if sig:
        p_value = float(sig["p_value"])
        sig_pill = (
            pill_html(f"Statistically significant (p = {p_value:.3f})", "warn")
            if sig["significant"]
            else pill_html(f"Could be chance (p = {p_value:.2f})", "neutral")
        )
        sig_html = (
            f'<div style="margin-top:.5rem">{sig_pill}</div>'
            f'<div class="fh-muted" style="font-size:.8rem;margin-top:.3rem">{_esc(sig["high_group"])} minus {_esc(sig["low_group"])}: '
            f'{float(sig["diff"]) * 100:.1f} points (95% CI {float(sig["ci_low"]) * 100:.1f} to {float(sig["ci_high"]) * 100:.1f})</div>'
        )
    notes = "".join(f'<div class="fh-muted" style="font-size:.8rem;margin-top:.3rem">{_esc(n)}</div>' for n in report["notes"])
    return (
        f'<div class="fh-panel"><div class="fh-panel-title">{_esc(title)}</div>'
        f'{rates_bars_html(report["rates"], report["counts"], report.get("intervals"))}'
        + kpis_html([kpi_card("Parity difference", f"{dpd:.3f}", "0 means equal rates"), ratio_card])
        + f"<div>{status}</div>{sig_html}{notes}</div>"
    )


def df_table_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a theme-aware HTML table (first ``max_rows`` rows)."""
    shown = df.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in shown.columns)
    body = []
    for _, row in shown.iterrows():
        cells = []
        for value in row:
            if pd.api.types.is_bool(value):
                text = "✓" if value else "–"
            elif isinstance(value, float):
                text = f"{value:.1f}"
            else:
                text = str(value)
            cells.append(f"<td>{_esc(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="fh-table-wrap" style="max-height:420px;overflow:auto"><table class="fh-table"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------------------
# Streamlit interface
# --------------------------------------------------------------------------------------


def stress_summary_html(results: Sequence[Dict[str, object]], tests: Sequence[Dict[str, object]]) -> str:
    rows = []
    for item, test in zip(results, tests):
        status = (
            pill_html("Unchanged", "ok")
            if test["passed"]
            else pill_html(f"Changed by {float(test['max_abs_delta']):.1f}", "bad")
        )
        rows.append(
            f"<tr><td><b>{_esc(item['candidate_id'])}</b></td>"
            f"<td>{'Yes' if test['name_detected'] else 'No (gender-term test only)'}</td>"
            f"<td>{len(test['rows']) - 1}</td><td>{float(test['max_abs_delta']):.1f}</td><td>{status}</td></tr>"
        )
    return (
        '<div class="fh-table-wrap"><table class="fh-table"><thead><tr><th>Candidate</th><th>Name found</th>'
        "<th>Variants tested</th><th>Largest score change</th><th>Result</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def stress_detail_html(test: Dict[str, object]) -> str:
    rows = []
    for row in test["rows"]:
        delta = float(row["delta"])
        tone = "ok" if abs(delta) <= 0.05 else "bad"
        note = "" if row["text_changed"] or row["variant"] == "Original" else " (text identical: nothing to swap)"
        rows.append(
            f"<tr><td>{_esc(row['variant'])}{_esc(note)}</td><td>{_score_cell(float(row['score']))}</td>"
            f"<td>{pill_html(f'{delta:+.1f}', tone)}</td></tr>"
        )
    return (
        '<div class="fh-table-wrap"><table class="fh-table"><thead><tr><th>Variant</th><th>Score</th><th>Change</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------------------
# Streamlit interface
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Accounts, dashboard and admin (UI)
# --------------------------------------------------------------------------------------


def auth_brand_html() -> str:
    return (
        '<div class="fh-brand"><div class="fh-brand-logo">⚖️</div><div><div class="fh-brand-name">FairHire</div>'
        '<div class="fh-brand-tag">Fair hiring. Explainable AI.</div></div></div>'
        '<div class="fh-auth-title">Smarter hiring starts here.</div>'
        '<div class="fh-auth-sub">Screen candidates transparently, understand every score, and audit your hiring '
        "process for potential bias.</div>"
    )


def job_settings_from_state(state: Dict[str, object]) -> Dict[str, object]:
    """The job set-up worth saving (never any resume content)."""
    return {
        "required": list(state.get("req_skills", [])),
        "extra": str(state.get("extra_skills_text", "")),
        "custom_skills_text": str(state.get("custom_skills_text", "")),
        "min_years": float(state.get("min_years", 0.0) or 0.0),
        "weights": [state.get("w_skill", 60), state.get("w_exp", 25), state.get("w_kw", 15)],
        "shortlist": int(state.get("shortlist", 3) or 3),
    }


def apply_job_to_state(state: Dict[str, object], job: Dict[str, object]) -> None:
    """Load a saved job into the screening form (call before those widgets are created)."""
    settings = job.get("settings") or {}
    weights = list(settings.get("weights") or [60, 25, 15])[:3]
    while len(weights) < 3:
        weights.append(0)
    state["jd_text"] = str(job.get("jd_text", ""))
    state["custom_skills_text"] = str(settings.get("custom_skills_text", ""))
    state["req_skills"] = [str(s) for s in settings.get("required", [])]
    state["extra_skills_text"] = str(settings.get("extra", ""))
    state["min_years"] = float(settings.get("min_years", 0.0) or 0.0)
    state["shortlist"] = max(1, int(settings.get("shortlist", 3) or 3))
    state["w_skill"], state["w_exp"], state["w_kw"] = (max(0, min(100, int(w))) for w in weights)


def audit_summary(sections: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out = []
    for section in sections:
        report = section["report"]
        sig = report.get("significance")
        ratio = report["disparate_impact_ratio"]
        out.append(
            {
                "title": str(section["title"]),
                "simulated": bool(section.get("simulated")),
                "dpd": round(float(report["demographic_parity_difference"]), 3),
                "dir": None if ratio is None else round(float(ratio), 3),
                "p_value": None if not sig else round(float(sig["p_value"]), 4),
                "groups": len(report["rates"]),
            }
        )
    return {"sections": out}


def describe_summary(kind: str, summary: Dict[str, object]) -> str:
    if kind == "audit":
        parts = []
        for section in summary.get("sections", [])[:2]:
            ratio = "n/a" if section.get("dir") is None else f"{section['dir']:.2f}"
            tag = " (simulated)" if section.get("simulated") else ""
            parts.append(f"parity gap {section.get('dpd', 0):.3f}, impact ratio {ratio}{tag}")
        return "; ".join(parts) or "No metrics"
    return f"{summary.get('resumes', '?')} resumes, top score {summary.get('top_score', '?')}"


def build_audit_report_html(sections: Sequence[Dict[str, object]], generated_on: Optional[str] = None) -> str:
    generated_on = generated_on or now_utc_text()
    body = "".join(_report_fairness_block(s) for s in sections) or '<div class="muted">No fairness results.</div>'
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>FairHire fairness audit</title>'
        f'<style>{REPORT_CSS}</style></head><body><div class="wrap"><h1>FairHire fairness audit</h1>'
        f'<div class="muted">Generated {_esc(generated_on)}.</div><h2>Results</h2>{body}'
        f"<h2>Limitations and responsible use</h2><p>{_esc(LIMITATIONS_TEXT)}</p></div></body></html>"
    )


# --------------------------------------------------------------------------------------
# Job description bias checker
# --------------------------------------------------------------------------------------

# (category, severity, regex, why it matters, suggestion, automatic replacement or None)
_BIAS_RULES_RAW = [
    ("Gender-coded", "medium", r"\brock\s?stars?\b", "Macho job titles can put some applicants off.", "Describe the skills you need instead.", "skilled professional"),
    ("Gender-coded", "medium", r"\bninjas?\b", "Macho job titles can put some applicants off.", "Describe the skills you need instead.", "expert"),
    ("Gender-coded", "medium", r"\bgurus?\b", "Macho job titles can put some applicants off.", "Describe the skills you need instead.", "specialist"),
    ("Gender-coded", "medium", r"\bguys\b", "'Guys' reads as male.", "Use 'team' or 'everyone'.", "team"),
    ("Gender-coded", "medium", r"\bmanpower\b", "Gendered noun.", "Use 'workforce' or 'staff'.", "workforce"),
    ("Gender-coded", "medium", r"\bsales(?:man|men)\b", "Gendered job title.", "Use 'salesperson'.", "salesperson"),
    ("Gender-coded", "medium", r"\bchairman\b", "Gendered job title.", "Use 'chair'.", "chair"),
    ("Gender-coded", "medium", r"\bforeman\b", "Gendered job title.", "Use 'supervisor'.", "supervisor"),
    ("Gender-coded", "medium", r"\bbusinessman\b", "Gendered job title.", "Use 'businessperson'.", "businessperson"),
    ("Gender-coded", "low", r"\b(?:he|his|him|himself)\b", "A male pronoun for the candidate.", "Address the reader as 'you' or use 'they'.", None),
    ("Gender-coded", "low", r"\b(?:she|her|hers|herself)\b", "A female pronoun for the candidate.", "Address the reader as 'you' or use 'they'.", None),
    ("Gender-coded", "medium", r"\baggressive\b", "Masculine-coded wording can discourage some applicants.", "Try 'proactive' or 'ambitious'.", "proactive"),
    ("Gender-coded", "medium", r"\bdominant\b", "Masculine-coded wording can discourage some applicants.", "Try 'leading'.", "leading"),
    ("Gender-coded", "medium", r"\bfearless\b", "Masculine-coded wording can discourage some applicants.", "Try 'confident'.", "confident"),
    ("Culture", "medium", r"\bwork[- ]hard,?\s*play[- ]hard\b", "Can signal long hours and a social culture that excludes people with caring duties.", "Describe the working style and support you offer.", "collaborative and supportive team"),
    ("Culture", "medium", r"\bcultur(?:e|al)[- ]fit\b", "'Fit' is subjective and can favour people similar to the existing team.", "Name the values you are looking for.", "values alignment"),
    ("Age", "high", r"\b(?:young(?:er)?|youthful)\b", "Age-related wording can exclude older applicants and is unlawful in many places.", "Remove it and describe the skills needed.", None),
    ("Age", "medium", r"\benergetic\b", "Often read as a code word for youth.", "Try 'motivated'.", "motivated"),
    ("Age", "medium", r"\bdigital natives?\b", "Implies a particular generation.", "Try 'comfortable with digital tools'.", "comfortable with digital tools"),
    ("Age", "medium", r"\b(?:recent|fresh)\s+graduates?\b", "Excludes experienced applicants.", "State the experience level you need (for example 'early-career').", "early-career applicants"),
    ("Age", "high", r"\b(?:no more than|maximum of|max\.?|up to)\s+\d+\s+years?\s+(?:of\s+)?experience\b", "An experience ceiling can exclude older applicants.", "State a minimum, not a maximum.", None),
    ("Age", "high", r"\bage[ds]?\s+(?:between\s+)?\d{2}\b|\bunder\s+\d{2}\s+years?\s+old\b|\bbetween the ages of\b", "States an age requirement.", "Remove it.", None),
    ("Age", "low", r"\boverqualified\b", "Can screen out experienced applicants.", "State what you need rather than screening out experience.", None),
    ("Disability / health", "high", r"\bable[- ]bodied\b", "Excludes people with disabilities.", "State the essential tasks and offer reasonable adjustments.", None),
    ("Disability / health", "medium", r"\b(?:physically fit|perfect health|good health)\b", "Health requirements can exclude people with disabilities.", "State the essential physical tasks and offer reasonable adjustments.", None),
    ("Nationality / language", "medium", r"\bnative english speakers?\b", "Excludes fluent non-native speakers.", "Say 'fluent in English' and the level you need.", "fluent English speaker"),
    ("Nationality / language", "medium", r"\bnative speakers?\b|\bnative[- ]level\b", "Excludes fluent non-native speakers.", "State the language level you need.", None),
    ("Nationality / language", "high", r"\b(?:citizens only|nationals only|local candidates only|must be a (?:citizen|national)|only (?:citizens|nationals))\b", "Can discriminate by national origin.", "Ask about the legal right to work instead.", None),
    ("Family / marital status", "high", r"\b(?:unmarried|married|childless|no children|no (?:family )?commitments|single (?:man|woman|person|male|female))\b", "Family status is a protected characteristic in many places.", "Remove it.", None),
    ("Appearance", "high", r"\b(?:attractive|good[- ]looking|well[- ]groomed|good appearance|pleasant appearance|presentable|send (?:a )?photo)\b", "Appearance is not a job requirement.", "Remove it.", None),
    ("Background", "medium", r"\b(?:top[- ]tier|elite|prestigious)\s+(?:universit(?:y|ies)|schools?|colleges?)\b|\bivy league\b", "Favours people from privileged backgrounds.", "Ask for a relevant degree or equivalent experience.", None),
]
_BIAS_RULES = [(c, s, re.compile(p, re.IGNORECASE), w, g, r) for c, s, p, w, g, r in _BIAS_RULES_RAW]

_MASCULINE_STEMS = [
    "aggress", "ambitio", "assert", "boast", "competitive", "confiden", "courag", "decisive", "determin", "dominan", "driven",
    "fearless", "fight", "headstrong", "hierarch", "hostil", "impulsiv", "independen", "outspoken", "persist", "reckless",
    "self-reliant", "self-sufficien", "stubborn", "superior", "ninja", "rockstar", "rock star", "guru",
]
_FEMININE_STEMS = [
    "affection", "agree", "caring", "cheer", "collaborat", "communal", "compassion", "connect", "considerate", "cooperat",
    "dependab", "emotion", "empath", "gentle", "honest", "interpersonal", "interdependen", "loyal", "modest", "nurtur",
    "pleasant", "polite", "quiet", "sensitiv", "support", "sympath", "tender", "together", "trust", "understand", "warm", "yield",
]
_MASC_RE = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in _MASCULINE_STEMS) + r")\w*", re.IGNORECASE)
_FEM_RE = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in _FEMININE_STEMS) + r")\w*", re.IGNORECASE)


def gender_coding(text: str) -> Dict[str, object]:
    """Count masculine- and feminine-coded words (a rough indicator based on published word lists)."""
    masculine = [m.group(0).lower() for m in _MASC_RE.finditer(text or "")]
    feminine = [m.group(0).lower() for m in _FEM_RE.finditer(text or "")]
    if not masculine and not feminine:
        label = "Neutral"
    elif len(masculine) >= len(feminine) + 3:
        label = "Leans masculine-coded"
    elif len(feminine) >= len(masculine) + 3:
        label = "Leans feminine-coded"
    else:
        label = "Balanced"
    return {
        "masculine": len(masculine), "feminine": len(feminine), "label": label,
        "masculine_words": sorted(set(masculine)), "feminine_words": sorted(set(feminine)),
    }


def check_job_description(text: str) -> Dict[str, object]:
    """Flag wording in a job description that can discourage or exclude groups of applicants.

    Rule-based and not exhaustive. A finding is a prompt to reconsider the wording, not proof of
    discrimination, and an empty result does not mean the posting is free of bias.
    """
    text = text or ""
    findings: List[Dict[str, object]] = []
    seen = set()
    for category, severity, pattern, why, suggestion, replacement in _BIAS_RULES:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen:
                continue
            seen.add(span)
            left = max(0, match.start() - 45)
            right = min(len(text), match.end() + 45)
            findings.append(
                {
                    "term": match.group(0), "category": category, "severity": severity, "why": why,
                    "suggestion": suggestion, "replacement": replacement, "start": match.start(), "end": match.end(),
                    "snippet": re.sub(r"\s+", " ", text[left:right]).strip(),
                }
            )
    findings.sort(key=lambda f: (int(f["start"]), int(f["end"])))
    bullets = len(re.findall(r"(?m)^\s*[-\u2022*\u00b7\u25aa\u25cf]\s+", text))
    if bullets > 12:
        findings.append(
            {
                "term": f"{bullets} bullet points", "category": "Requirements overload", "severity": "low",
                "why": "Very long requirement lists can discourage qualified applicants who do not tick every box.",
                "suggestion": "Mark which requirements are essential and keep the list focused.", "replacement": None,
                "start": len(text), "end": len(text), "snippet": "",
            }
        )
    counts = {level: sum(1 for f in findings if f["severity"] == level) for level in ("high", "medium", "low")}
    score = 3 * counts["high"] + 2 * counts["medium"] + counts["low"]
    if not findings:
        risk = "none"
    elif counts["high"]:
        risk = "high"
    elif score >= 4:
        risk = "medium"
    else:
        risk = "low"
    return {
        "findings": findings, "counts": counts, "risk": risk, "gender": gender_coding(text),
        "word_count": len(re.findall(r"\b\w+\b", text)),
    }


def rewrite_job_description(text: str, findings: Sequence[Dict[str, object]]) -> Tuple[str, int]:
    """Apply the safe automatic replacements. Returns ``(revised_text, number_of_changes)``."""
    out = text or ""
    changes = 0
    for finding in sorted((f for f in findings if f.get("replacement")), key=lambda f: int(f["start"]), reverse=True):
        start, end = int(finding["start"]), int(finding["end"])
        original = out[start:end]
        if original.lower() != str(finding["term"]).lower():
            continue  # the text changed since the check
        replacement = str(finding["replacement"])
        if original.isupper() and len(original) > 1:
            replacement = replacement.upper()
        elif original[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        out = out[:start] + replacement + out[end:]
        changes += 1
    return out, changes


_RISK_TONE = {"none": "ok", "low": "ok", "medium": "warn", "high": "bad"}
_SEVERITY_CHIP = {"high": "bad", "medium": "warn", "low": "neutral"}


def jd_summary_html(result: Dict[str, object]) -> str:
    gender = result["gender"]
    counts = result["counts"]
    risk = str(result["risk"])
    return kpis_html(
        [
            kpi_card("Wording flags", len(result["findings"]), f"{counts['high']} high, {counts['medium']} medium, {counts['low']} low"),
            kpi_card("Overall wording risk", risk.capitalize(), "", _RISK_TONE.get(risk, "neutral")),
            kpi_card("Gender coding", str(gender["label"]), f"{gender['masculine']} masculine-coded, {gender['feminine']} feminine-coded words"),
            kpi_card("Words analysed", result["word_count"]),
        ]
    )


def gender_bar_html(gender: Dict[str, object]) -> str:
    total = max(1, int(gender["masculine"]) + int(gender["feminine"]))
    rows = []
    for label, count, words, color in (
        ("Masculine-coded", int(gender["masculine"]), gender["masculine_words"], _BAR_GRADIENTS[1]),
        ("Feminine-coded", int(gender["feminine"]), gender["feminine_words"], _BAR_GRADIENTS[2]),
    ):
        width = 100.0 * count / total
        rows.append(
            f'<div class="fh-rate"><div class="fh-rate-head"><span>{label}</span><span>{count}</span></div>'
            f'<div class="fh-bar"><i style="width:{width:.1f}%;background:{color}"></i></div>'
            f'<div style="margin-top:.3rem">{chips_html(list(words)[:12], "neutral")}</div></div>'
        )
    return '<div class="fh-panel">' + "".join(rows) + (
        '<div class="fh-muted" style="font-size:.8rem;margin-top:.5rem">A rough indicator based on published word lists. '
        "It is not proof that anyone is excluded, and some of these words are fine in context.</div></div>"
    )


def jd_findings_html(result: Dict[str, object]) -> str:
    findings = result["findings"]
    if not findings:
        return empty_state_html("No flagged wording found", "That does not guarantee the posting is unbiased. Ask a colleague to read it too.", "✅")
    rows = []
    for f in findings:
        snippet = f'<div class="fh-muted" style="font-size:.78rem">{_esc(f["snippet"])}</div>' if f["snippet"] else ""
        rows.append(
            f'<tr><td><span class="fh-chip {_SEVERITY_CHIP[str(f["severity"])]}">{_esc(f["term"])}</span>{snippet}</td>'
            f'<td>{_esc(f["category"])}</td><td>{_esc(f["why"])}</td><td>{_esc(f["suggestion"])}</td></tr>'
        )
    return (
        '<div class="fh-table-wrap"><table class="fh-table"><thead><tr><th>Wording</th><th>Category</th><th>Why it matters</th>'
        "<th>Suggestion</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------------------
# Skill evidence: show where in the resume each matched skill was found
# --------------------------------------------------------------------------------------


def _sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start), text.rfind("! ", 0, start), text.rfind("? ", 0, start))
    left = 0 if left < 0 else left + 1
    stops = [i for i in (text.find("\n", end), text.find(". ", end), text.find("! ", end), text.find("? ", end)) if i != -1]
    right = min(stops) + 1 if stops else len(text)
    return left, right


def _snippet(text: str, start: int, end: int, window: int = 110) -> Dict[str, object]:
    left_bound, right_bound = _sentence_bounds(text, start, end)
    left = max(left_bound, start - window)
    right = min(right_bound, end + window)
    return {
        "before": text[left:start].lstrip(), "hit": text[start:end], "after": text[end:right].rstrip(),
        "left_cut": left > left_bound, "right_cut": right < right_bound,
    }


def find_skill_evidence(
    text: str,
    skills: Sequence[str],
    skill_bank: Optional[Dict[str, List[str]]] = None,
    per_skill: int = 2,
) -> Dict[str, List[Dict[str, object]]]:
    """For each skill, up to ``per_skill`` short snippets from different sentences where it appears."""
    evidence: Dict[str, List[Dict[str, object]]] = {}
    text = text or ""
    for skill in skills:
        matches = []
        for alias in _skill_aliases(skill, skill_bank):
            matches.extend(_alias_pattern(alias).finditer(text))
        matches.sort(key=lambda m: m.start())
        snippets: List[Dict[str, object]] = []
        used_sentences = set()
        for match in matches:
            bounds = _sentence_bounds(text, match.start(), match.end())
            if bounds in used_sentences:
                continue
            used_sentences.add(bounds)
            snippets.append(_snippet(text, match.start(), match.end()))
            if len(snippets) >= per_skill:
                break
        if snippets:
            evidence[skill] = snippets
    return evidence


def experience_evidence(text: str, current_year: Optional[int] = None, limit: int = 3) -> List[Dict[str, object]]:
    """Snippets that explain the detected experience (stated years and employment date ranges)."""
    text = text or ""
    found: List[Tuple[int, Dict[str, object]]] = []
    for pattern in _STATED_YEARS_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), _snippet(text, match.start(), match.end())))
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if not _EDUCATION_WORDS.search(stripped):
            for match in _DATE_RANGE.finditer(stripped):
                found.append((offset + match.start(), _snippet(text, offset + match.start(), offset + match.end())))
        offset += len(line)
    found.sort(key=lambda pair: pair[0])
    out: List[Dict[str, object]] = []
    seen = set()
    for _, snippet in found:
        key = (snippet["before"], snippet["hit"], snippet["after"])
        if key not in seen:
            seen.add(key)
            out.append(snippet)
        if len(out) >= limit:
            break
    return out


def snippet_html(snippet: Dict[str, object]) -> str:
    return (
        ("…" if snippet.get("left_cut") else "") + _esc(snippet.get("before", ""))
        + f'<mark class="fh-hit">{_esc(snippet.get("hit", ""))}</mark>'
        + _esc(snippet.get("after", "")) + ("…" if snippet.get("right_cut") else "")
    )


def evidence_html(item: Dict[str, object]) -> str:
    rows = []
    for skill, snippets in (item.get("evidence") or {}).items():
        for snippet in snippets:
            rows.append(f'<div class="fh-ev"><span class="fh-chip ok">{_esc(skill)}</span><span class="fh-ev-text">{snippet_html(snippet)}</span></div>')
    for snippet in item.get("experience_evidence") or []:
        rows.append(f'<div class="fh-ev"><span class="fh-chip neutral">experience</span><span class="fh-ev-text">{snippet_html(snippet)}</span></div>')
    missing = item.get("missing_skills") or []
    if missing:
        rows.append(f'<div class="fh-muted" style="font-size:.85rem;margin-top:.3rem">Not found anywhere in the resume text: {_esc(", ".join(missing))}</div>')
    return "".join(rows) or '<span class="fh-muted">No evidence to show.</span>'


def highlight_skills_html(text: str, skills: Sequence[str], skill_bank: Optional[Dict[str, List[str]]] = None, limit: int = 3000) -> str:
    """The resume text (escaped) with every mention of the given skills highlighted."""
    text = (text or "")[:limit]
    spans: List[Tuple[int, int]] = []
    for skill in skills:
        for alias in _skill_aliases(skill, skill_bank):
            spans.extend((m.start(), m.end()) for m in _alias_pattern(alias).finditer(text))
    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, cursor = [], 0
    for start, end in merged:
        out.append(_esc(text[cursor:start]))
        out.append(f'<mark class="fh-hit">{_esc(text[start:end])}</mark>')
        cursor = end
    out.append(_esc(text[cursor:]))
    return "".join(out)


# --------------------------------------------------------------------------------------
# Where "saved" items live: this browser tab (prototype) or the database (real accounts)
# --------------------------------------------------------------------------------------


class SessionWorkspace:
    """Saved jobs and reports for the prototype: kept only in the browser session."""

    persistent = False
    label = "this browser tab only"

    def __init__(self, state):
        self.state = state

    def list_jobs(self) -> List[Dict[str, object]]:
        return [dict(j) for j in reversed(self.state.get("saved_jobs", []))]

    def get_job(self, job_id: int) -> Optional[Dict[str, object]]:
        return next((dict(j) for j in self.state.get("saved_jobs", []) if j["id"] == job_id), None)

    def add_job(self, title: str, jd_text: str, settings: Dict[str, object]) -> Optional[str]:
        return add_saved_job(self.state, title, jd_text, settings)

    def delete_job(self, job_id: int) -> bool:
        return delete_saved(self.state, "saved_jobs", job_id)

    def list_reports(self, kind: Optional[str] = None) -> List[Dict[str, object]]:
        items = [r for r in reversed(self.state.get("saved_reports", [])) if kind in (None, r["kind"])]
        return [{k: v for k, v in r.items() if k != "html"} for r in items]

    def get_report_html(self, report_id: int) -> Optional[str]:
        return next((str(r["html"]) for r in self.state.get("saved_reports", []) if r["id"] == report_id), None)

    def add_report(self, kind: str, title: str, html: str, summary: Dict[str, object]) -> Optional[str]:
        return add_saved_report(self.state, kind, title, html, summary)

    def delete_report(self, report_id: int) -> bool:
        return delete_saved(self.state, "saved_reports", report_id)


class DbWorkspace:
    """Saved jobs and reports for a real account, stored in the database."""

    persistent = True

    def __init__(self, db, user_id: int, state):
        self.db = db
        self.user_id = int(user_id)
        self.state = state
        self.label = (
            "your account, in the database" if db.kind == "postgres"
            else "your account (a local file that may be wiped when the app restarts)"
        )

    def _cache(self, key: str, loader):
        cache = self.state.setdefault("_ws_cache", {})
        if key not in cache:
            cache[key] = loader()
        return cache[key]

    def _invalidate(self) -> None:
        self.state["_ws_cache"] = {}

    def list_jobs(self) -> List[Dict[str, object]]:
        return list(self._cache("jobs", lambda: auth.list_jobs(self.db, self.user_id)))

    def get_job(self, job_id: int) -> Optional[Dict[str, object]]:
        return auth.get_job(self.db, self.user_id, job_id)

    def add_job(self, title: str, jd_text: str, settings: Dict[str, object]) -> Optional[str]:
        _id, problem = auth.save_job(self.db, self.user_id, title, jd_text, settings)
        self._invalidate()
        return problem

    def delete_job(self, job_id: int) -> bool:
        ok = auth.delete_job(self.db, self.user_id, job_id)
        self._invalidate()
        return ok

    def list_reports(self, kind: Optional[str] = None) -> List[Dict[str, object]]:
        items = self._cache("reports", lambda: auth.list_reports(self.db, self.user_id, None, 200))
        return [r for r in items if kind in (None, r["kind"])]

    def get_report_html(self, report_id: int) -> Optional[str]:
        return auth.get_report_html(self.db, self.user_id, report_id)

    def add_report(self, kind: str, title: str, html: str, summary: Dict[str, object]) -> Optional[str]:
        _id, problem = auth.save_report(self.db, self.user_id, title, kind, html, summary)
        self._invalidate()
        return problem

    def delete_report(self, report_id: int) -> bool:
        ok = auth.delete_report(self.db, self.user_id, report_id)
        self._invalidate()
        return ok


# --------------------------------------------------------------------------------------
# PROTOTYPE account flow
#
# Everything below is SIMULATED. No account is created, no password is kept, no email is
# sent and nothing is written to disk or a database. "Saved" items live only in this
# browser session (st.session_state) and disappear when the tab closes or you sign out.
# --------------------------------------------------------------------------------------

PROTOTYPE_BANNER = "Sign-in and the dashboard are a simulated prototype. Resume screening is real: files you upload are read in memory and never saved."
PROTOTYPE_FOOTNOTE = "Illustrative prototype. No real accounts are created and nothing you enter is stored or sent."
MAX_SAVED_ITEMS = 25
MAX_SAVED_REPORT_BYTES = 1_000_000

SAMPLE_USERS = [
    {"id": 101, "email": "asha.recruiter@example.com", "role": "recruiter", "verified": True, "jobs": 3, "reports": 5, "created": "2026-09-02"},
    {"id": 102, "email": "daniel.hr@example.com", "role": "recruiter", "verified": True, "jobs": 1, "reports": 2, "created": "2026-09-14"},
    {"id": 103, "email": "mei.talent@example.com", "role": "recruiter", "verified": False, "jobs": 0, "reports": 0, "created": "2026-10-18"},
]

_SIMPLE_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")


def simple_email_ok(email: str) -> bool:
    email = (email or "").strip()
    return bool(email) and len(email) <= 254 and bool(_SIMPLE_EMAIL_RE.match(email))


def password_strength(password: str) -> Dict[str, object]:
    """Feedback for the sign-up meter. Pure function; the password is not kept anywhere."""
    password = password or ""
    checks = [
        ("10 or more characters", len(password) >= 10),
        ("Upper and lower case letters", bool(re.search(r"[a-z]", password)) and bool(re.search(r"[A-Z]", password))),
        ("A number", bool(re.search(r"\d", password))),
        ("A symbol", bool(re.search(r"[^A-Za-z0-9]", password))),
    ]
    score = sum(1 for _, ok in checks if ok)
    if not password:
        label, tone = "Enter a password", "neutral"
    elif len(password) < 10:
        label, tone = "Too short", "bad"
    elif score <= 2:
        label, tone = "Weak", "warn"
    elif score == 3:
        label, tone = "Good", "ok"
    else:
        label, tone = "Strong", "ok"
    return {"score": score, "label": label, "tone": tone, "checks": checks}


def password_meter_html(password: str) -> str:
    info = password_strength(password)
    width = 0 if not password else max(8, 25 * int(info["score"]))
    items = "".join(
        f'<span class="fh-chip {"ok" if ok else "neutral"}">{"✓" if ok else "○"} {_esc(text)}</span>' for text, ok in info["checks"]
    )
    return (
        f'<div class="fh-meter"><div class="fh-bar"><i style="width:{width}%;background:{_TONE_COLOR[str(info["tone"])]}"></i></div>'
        f'<div class="fh-muted" style="font-size:.8rem;margin:.25rem 0">{_esc(info["label"])}</div><div>{items}</div></div>'
    )


def prototype_password_errors(password: str, confirm: str) -> List[str]:
    errors = []
    if len(password or "") < 10:
        errors.append("Use at least 10 characters.")
    if (password or "") != (confirm or ""):
        errors.append("The two passwords do not match.")
    return errors


def prototype_signup_errors(email: str, password: str, confirm: str, agreed: bool) -> List[str]:
    errors = []
    if not simple_email_ok(email):
        errors.append("Enter a valid email address.")
    errors.extend(prototype_password_errors(password, confirm))
    if not agreed:
        errors.append("Tick the box to confirm you understand this is a prototype.")
    return errors


def prototype_signin_errors(email: str, password: str) -> List[str]:
    errors = []
    if not simple_email_ok(email):
        errors.append("Enter a valid email address.")
    if not (password or ""):
        errors.append("Enter your password.")
    return errors


def role_for_email(email: str) -> str:
    """Prototype shortcut: an address starting with 'admin' opens the administrator view."""
    return "admin" if (email or "").strip().lower().startswith("admin") else "recruiter"


def make_demo_user(email: str, role: str) -> Dict[str, object]:
    return {
        "id": 1,
        "email": (email or "").strip().lower(),
        "role": role if role in ("recruiter", "admin") else "recruiter",
        "verified": True,
        "must_change": False,
        "created_at": now_utc_text(),
        "mode": "prototype",
    }


# ---- in-memory "saved items" (session only) ------------------------------------------


def _next_id(state) -> int:
    state["next_id"] = int(state.get("next_id", 1000)) + 1
    return int(state["next_id"])


def _title_error(title: str) -> Optional[str]:
    title = (title or "").strip()
    if not title or len(title) > 120:
        return "Give it a title of up to 120 characters."
    return None


def add_saved_job(state, title: str, jd_text: str, settings: Dict[str, object]) -> Optional[str]:
    problem = _title_error(title)
    if problem:
        return problem
    jobs = state.setdefault("saved_jobs", [])
    if len(jobs) >= MAX_SAVED_ITEMS:
        return f"The prototype keeps up to {MAX_SAVED_ITEMS} saved jobs. Delete one first."
    jobs.append(
        {"id": _next_id(state), "title": title.strip(), "jd_text": (jd_text or "")[:20000], "settings": settings, "created_at": now_utc_text()}
    )
    return None


def add_saved_report(state, kind: str, title: str, html: str, summary: Dict[str, object]) -> Optional[str]:
    problem = _title_error(title)
    if problem:
        return problem
    if kind not in ("screening", "audit"):
        return "Unknown report type."
    if len((html or "").encode("utf-8")) > MAX_SAVED_REPORT_BYTES:
        return "The report is too large to keep in the prototype."
    reports = state.setdefault("saved_reports", [])
    if len(reports) >= MAX_SAVED_ITEMS:
        return f"The prototype keeps up to {MAX_SAVED_ITEMS} saved reports. Delete one first."
    reports.append(
        {"id": _next_id(state), "kind": kind, "title": title.strip(), "html": html or "", "summary": summary, "created_at": now_utc_text()}
    )
    return None


def delete_saved(state, bucket: str, item_id: int) -> bool:
    items = state.get(bucket, [])
    for index, item in enumerate(items):
        if item.get("id") == item_id:
            del items[index]
            return True
    return False


def seed_demo_data(state) -> None:
    """Fill the dashboard with clearly labelled sample items the first time someone 'signs in'."""
    if state.get("demo_seeded"):
        return
    state["demo_seeded"] = True
    state["demo_users"] = [dict(u) for u in SAMPLE_USERS]
    add_saved_job(
        state, "Data Scientist (sample)", SAMPLE_JOB_DESCRIPTION,
        {"required": list(DEFAULT_REQUIRED_SKILLS), "extra": "", "custom_skills_text": "", "min_years": 2.0, "weights": [60, 25, 15], "shortlist": 3},
    )
    add_saved_job(
        state, "Backend Developer (sample)", "We need a backend developer with Java, SQL and REST API experience. Minimum 3 years of experience.",
        {"required": ["java", "sql", "rest api"], "extra": "", "custom_skills_text": "", "min_years": 3.0, "weights": [60, 25, 15], "shortlist": 3},
    )
    results, _ = screen_resumes(list(SAMPLE_RESUMES.items()), DEFAULT_REQUIRED_SKILLS, SAMPLE_JOB_DESCRIPTION, 2.0, True)
    settings = {"required": DEFAULT_REQUIRED_SKILLS, "min_years": 2.0, "weights": DEFAULT_WEIGHTS, "anonymize": True, "shortlist": 3}
    add_saved_report(
        state, "screening", "Data Scientist screening (sample)", build_report_html(results, settings),
        {"resumes": len(results), "top_score": round(float(results[0]["score"]), 1) if results else 0},
    )
    sections = synthetic_fairness_sections(generate_synthetic_candidates(n=200, seed=42, bias_penalty=15))
    add_saved_report(state, "audit", "Simulated audit, 15-point bias (sample)", build_audit_report_html(sections), audit_summary(sections))


def demo_user_action(state, email: str, action: str) -> str:
    """Pretend admin actions on the made-up sample accounts."""
    users = state.setdefault("demo_users", [])
    target = next((u for u in users if u["email"] == email), None)
    if target is None:
        return "Account not found."
    if action == "Make administrator":
        target["role"] = "admin"
        return f"{email} is now an administrator (simulated)."
    if action == "Make recruiter":
        target["role"] = "recruiter"
        return f"{email} is now a recruiter (simulated)."
    if action == "Reset password":
        return "A temporary password would be created and shown once (simulated). Nothing was changed."
    if action == "Delete account":
        users.remove(target)
        return f"{email} was deleted (simulated)."
    return "Unknown action."


def sign_out(state) -> None:
    """Forget the demo account and everything held in memory for this session."""
    for key in list(state.keys()):
        if key not in ("dark_mode", "intro_done", "auth_mode"):
            del state[key]


# ---- screens ------------------------------------------------------------------------


def _set_view(state, view: str) -> None:
    state["auth_view"] = view


def _start_session(state, email: str, role: str) -> None:
    state["user"] = make_demo_user(email, role)
    state["auth_view"] = "login"
    seed_demo_data(state)


def _show(st, text: str, tone: str = "bad", icon: str = "⛔") -> None:
    st.markdown(note_html(text, tone, icon), unsafe_allow_html=True)


def read_app_secrets(st) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str]]:
    """Optional settings from Streamlit secrets: an [smtp] table, ``admin_email`` and ``database_url``."""
    smtp_conf: Optional[Dict[str, str]] = None
    admin_email: Optional[str] = None
    database_url: Optional[str] = None
    try:
        if "smtp" in st.secrets:
            smtp_conf = {str(k): v for k, v in dict(st.secrets["smtp"]).items()}
    except Exception:  # noqa: BLE001 - no secrets file is the normal case
        smtp_conf = None
    try:
        value = st.secrets.get("admin_email")
        admin_email = value.strip() if isinstance(value, str) and value.strip() else None
    except Exception:  # noqa: BLE001
        admin_email = None
    try:
        value = st.secrets.get("database_url")
        database_url = value.strip() if isinstance(value, str) and value.strip() else None
    except Exception:  # noqa: BLE001
        database_url = None
    return smtp_conf, admin_email, database_url


REAL_MODE = "Real account"
PROTOTYPE_MODE = "Prototype (simulated)"


def _apply_revised_jd(state, text: str) -> None:
    state["jd_text"] = text


def _start_real_session(state, user: Dict[str, object]) -> None:
    state["user"] = dict(user, mode="real")
    state["auth_view"] = "login"


def _send_code_email(smtp_conf, user: Dict[str, object], db, purpose: str) -> Optional[str]:
    code = auth.issue_code(db, int(user["id"]), purpose)
    subject = "Your FairHire verification code" if purpose == "verify" else "Your FairHire password reset code"
    body = f"Your FairHire code is {code}. It expires in {auth.CODE_TTL_SECONDS // 60} minutes. If you did not ask for it, ignore this email."
    ok, error = auth.send_email(smtp_conf, str(user["email"]), subject, body)
    return None if ok else error


def render_auth(st, state, db, db_error, smtp_conf, admin_email) -> None:  # pragma: no cover - UI
    """Sign-in screen with a switch between real accounts and the simulated prototype."""
    _, middle, _ = st.columns([1, 1.5, 1])
    with middle:
        st.markdown('<div class="fh-card-marker"></div>' + auth_brand_html(), unsafe_allow_html=True)
        if db is None:
            state["auth_mode"] = PROTOTYPE_MODE
        options = [REAL_MODE, PROTOTYPE_MODE] if db is not None else [PROTOTYPE_MODE]
        if state.get("auth_mode") not in options:
            state["auth_mode"] = options[0]
        mode = st.radio("Sign-in mode", options, key="auth_mode", horizontal=True, on_change=_set_view, args=(state, "login"))
        if db_error:
            _show(st, db_error, "bad")
        if mode == REAL_MODE:
            st.caption("Real account: your email and a salted password hash are stored in this app's database.")
            if db is not None and db.kind == "sqlite":
                _show(
                    st,
                    "Accounts are stored in a local file that may be wiped when the app restarts. Add a database_url in Settings → Secrets "
                    "for lasting storage (see the README).",
                    "warn",
                    "⚠️",
                )
        else:
            st.caption("Prototype: nothing is stored. Any valid-looking email and password works.")
    if mode == REAL_MODE and db is not None:
        _auth_body_real(st, state, middle, db, smtp_conf, admin_email)
    else:
        _auth_body_prototype(st, state, middle)


def _auth_body_real(st, state, middle, db, smtp_conf, admin_email) -> None:  # pragma: no cover - UI
    smtp_ok = auth.smtp_configured(smtp_conf)
    view = state.get("auth_view", "login")
    with middle:
        flash = state.pop("auth_flash", None)
        if flash:
            _show(st, flash, "info", "✅")

        if view == "signup":
            st.markdown(section_html("Create your account"), unsafe_allow_html=True)
            if auth.count_users(db) == 0 and not admin_email:
                st.caption("The first account created becomes the administrator.")
            email = st.text_input("Work email", key="rsu_email", placeholder="Work email", label_visibility="collapsed")
            password = st.text_input("Password", type="password", key="rsu_pw", placeholder="Password (10+ characters)", label_visibility="collapsed")
            st.markdown(password_meter_html(password), unsafe_allow_html=True)
            confirm = st.text_input("Confirm password", type="password", key="rsu_pw2", placeholder="Confirm password", label_visibility="collapsed")
            if not smtp_ok:
                st.caption("Email verification is off because no email service is configured.")
            if st.button("Create account", type="primary", use_container_width=True, key="rsu_submit"):
                if password != confirm:
                    _show(st, "The two passwords do not match.")
                else:
                    user, error = auth.create_user(db, email, password, verified=not smtp_ok, admin_email=admin_email)
                    if error:
                        _show(st, error)
                    elif smtp_ok:
                        problem = _send_code_email(smtp_conf, user, db, "verify")
                        state["pending_email"] = user["email"]
                        state["auth_view"] = "verify"
                        state["auth_flash"] = "We emailed you a 6-digit code." if not problem else f"{problem} Use 'Send a new code'."
                        st.rerun()
                    else:
                        _start_real_session(state, user)
                        st.rerun()
            st.button("Already have an account? Sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "verify":
            st.markdown(section_html("Verify your email"), unsafe_allow_html=True)
            pending = state.get("pending_email", "")
            st.caption(f"Enter the 6-digit code we sent to {pending}.")
            with st.form("rverify_form"):
                code = st.text_input("Code", placeholder="6-digit code", label_visibility="collapsed", max_chars=6)
                submitted = st.form_submit_button("Verify", type="primary", use_container_width=True)
            if submitted:
                user = auth.get_user_by_email(db, pending)
                if user and auth.check_code(db, user["id"], "verify", code):
                    auth.set_verified(db, user["id"])
                    state["auth_flash"] = "Email verified. You can sign in now."
                    state["auth_view"] = "login"
                    st.rerun()
                else:
                    _show(st, "That code is not valid or has expired.")
            if st.button("Send a new code", use_container_width=True):
                user = auth.get_user_by_email(db, pending)
                if user and not user["verified"]:
                    problem = _send_code_email(smtp_conf, user, db, "verify")
                    _show(st, problem or "A new code is on its way.", "bad" if problem else "info", "✉️")
            st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "forgot":
            st.markdown(section_html("Reset your password"), unsafe_allow_html=True)
            if not smtp_ok:
                _show(
                    st,
                    "Password reset by email is not set up for this app. Ask an administrator to reset your password. "
                    "They will give you a temporary one that you change at your next sign-in.",
                    "info",
                    "ℹ️",
                )
            else:
                with st.form("rforgot_form"):
                    email = st.text_input("Email address", placeholder="Email address", label_visibility="collapsed")
                    submitted = st.form_submit_button("Email me a code", type="primary", use_container_width=True)
                if submitted:
                    user = auth.get_user_by_email(db, email)
                    if user:
                        _send_code_email(smtp_conf, user, db, "reset")
                    state["pending_email"] = auth.normalize_email(email)
                    state["auth_flash"] = "If that email has an account, a code has been sent."
                    state["auth_view"] = "reset"
                    st.rerun()
            st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "reset":
            st.markdown(section_html("Choose a new password"), unsafe_allow_html=True)
            with st.form("rreset_form"):
                email = st.text_input("Email address", value=state.get("pending_email", ""), placeholder="Email address", label_visibility="collapsed")
                code = st.text_input("Code", placeholder="6-digit code", label_visibility="collapsed", max_chars=6)
                password = st.text_input("New password", type="password", placeholder="New password (10+ characters)", label_visibility="collapsed")
                confirm = st.text_input("Confirm new password", type="password", placeholder="Confirm new password", label_visibility="collapsed")
                submitted = st.form_submit_button("Update password", type="primary", use_container_width=True)
            if submitted:
                user = auth.get_user_by_email(db, email)
                if password != confirm:
                    _show(st, "The two passwords do not match.")
                elif not (user and auth.check_code(db, user["id"], "reset", code)):
                    _show(st, "That code is not valid or has expired.")
                else:
                    problem = auth.set_password(db, user["id"], password)
                    if problem:
                        _show(st, problem)
                    else:
                        state["auth_flash"] = "Password updated. You can sign in now."
                        state["auth_view"] = "login"
                        st.rerun()
            st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        else:
            st.markdown(section_html("Welcome back"), unsafe_allow_html=True)
            with st.form("rlogin_form"):
                email = st.text_input("Email address", placeholder="Email address", label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                user, error = auth.authenticate(db, email, password)
                if error:
                    _show(st, error)
                else:
                    _start_real_session(state, user)
                    st.rerun()
            st.button("New to FairHire? Create an account", on_click=_set_view, args=(state, "signup"), use_container_width=True)
            st.button("Forgot your password?", on_click=_set_view, args=(state, "forgot"), use_container_width=True)
        st.caption("Resumes you upload are processed in memory and are never saved.")


def render_must_change(st, state, db, user) -> None:  # pragma: no cover - UI
    _, middle, _ = st.columns([1, 1.5, 1])
    with middle:
        st.markdown('<div class="fh-card-marker"></div>' + auth_brand_html(), unsafe_allow_html=True)
        st.markdown(section_html("Choose a new password"), unsafe_allow_html=True)
        st.caption("An administrator gave you a temporary password. Set your own to continue.")
        with st.form("must_change_form"):
            current = st.text_input("Temporary password", type="password", placeholder="Temporary password", label_visibility="collapsed")
            password = st.text_input("New password", type="password", placeholder="New password (10+ characters)", label_visibility="collapsed")
            confirm = st.text_input("Confirm new password", type="password", placeholder="Confirm new password", label_visibility="collapsed")
            submitted = st.form_submit_button("Save password", type="primary", use_container_width=True)
        if submitted:
            if password != confirm:
                _show(st, "The two passwords do not match.")
            else:
                problem = auth.change_password(db, user["id"], current, password)
                if problem:
                    _show(st, problem)
                else:
                    _start_real_session(state, auth.get_user(db, user["id"]))
                    st.rerun()
        st.button("Sign out", on_click=sign_out, args=(state,), use_container_width=True)


def render_dashboard(st, state, user, ws) -> None:  # pragma: no cover - UI
    """One dashboard for both modes; ``ws`` decides whether items live in the session or the database."""
    jobs = ws.list_jobs()
    reports = ws.list_reports("screening")
    audits = ws.list_reports("audit")
    st.markdown(section_html("Your dashboard"), unsafe_allow_html=True)
    st.markdown(
        kpis_html([kpi_card("Saved jobs", len(jobs)), kpi_card("Saved reports", len(reports)), kpi_card("Fairness audits", len(audits))]),
        unsafe_allow_html=True,
    )
    if not ws.persistent:
        _show(
            st,
            "Prototype: items marked (sample) are made up. Anything you save is kept only in this browser tab and disappears when you "
            "close it or sign out. Resumes are never saved.",
            "info",
            "🧪",
        )
    elif ws.db.kind == "postgres":
        _show(st, "Private to your account and kept in the database. Resumes are never saved; reports hold candidate numbers, scores and explanations.", "info", "🔒")
    else:
        _show(
            st,
            "Private to your account, but kept in a local file that may be wiped when the app restarts. Add a database_url in Settings → "
            "Secrets for lasting storage. Resumes are never saved.",
            "warn",
            "⚠️",
        )
    note = state.pop("dash_flash", None)
    if note:
        _show(st, note, "info", "✅")

    st.markdown(section_html("Saved jobs"), unsafe_allow_html=True)
    if not jobs:
        st.markdown(empty_state_html("No saved jobs", "Open the Screening tab and use Save this job.", "💼"), unsafe_allow_html=True)
    for job in jobs:
        c1, c2, c3 = st.columns([5, 1, 1])
        c1.markdown(f'<b>{_esc(job["title"])}</b><div class="fh-muted">{_esc(job["created_at"])}</div>', unsafe_allow_html=True)
        if c2.button("Load", key=f"load_job_{job['id']}"):
            full = ws.get_job(job["id"])
            if full:
                apply_job_to_state(state, full)
                state["dash_flash"] = f"Loaded '{full['title']}'. Open the Screening tab to continue."
                st.rerun()
        if c3.button("Delete", key=f"del_job_{job['id']}"):
            ws.delete_job(job["id"])
            st.rerun()

    for heading, items in (("Saved reports", reports), ("Previous fairness audits", audits)):
        st.markdown(section_html(heading), unsafe_allow_html=True)
        if not items:
            st.markdown(empty_state_html("Nothing saved yet", "Save one from the Review & report or Fairness audit tab.", "📁"), unsafe_allow_html=True)
        for item in items:
            c1, c3 = st.columns([6, 1])
            c1.markdown(
                f'<b>{_esc(item["title"])}</b><div class="fh-muted">{_esc(item["created_at"])} | '
                f'{_esc(describe_summary(item["kind"], item["summary"]))}</div>',
                unsafe_allow_html=True,
            )
            if c3.button("Delete", key=f"del_rep_{item['id']}"):
                ws.delete_report(item["id"])
                st.rerun()

    everything = reports + audits
    if everything:
        st.markdown(section_html("Download a saved report"), unsafe_allow_html=True)
        labels = [f"{r['title']} | {r['created_at']} | #{r['id']}" for r in everything]
        choice = st.selectbox("Saved report", labels, label_visibility="collapsed")
        picked = everything[labels.index(choice)]
        html_text = ws.get_report_html(picked["id"]) or ""
        st.download_button(
            "⬇️ Download", data=html_text.encode("utf-8"), file_name=f"fairhire_{picked['kind']}_{picked['id']}.html", mime="text/html",
        )


def render_admin_real(st, state, db, user) -> None:  # pragma: no cover - UI
    st.markdown(section_html("Administration"), unsafe_allow_html=True)
    try:
        users = auth.list_users(db, int(user["id"]))
    except PermissionError:
        _show(st, "Administrator access is required.")
        return
    _show(st, "You can manage accounts. You cannot see anyone's jobs, reports or resumes, only how many each person has.", "info", "🛡️")
    note = state.pop("admin_flash", None)
    if note:
        _show(st, note, "info", "✅")
    table = pd.DataFrame(
        [
            {"Email": u["email"] + (" (you)" if u["id"] == user["id"] else ""), "Role": u["role"], "Verified": "yes" if u["verified"] else "no",
             "Jobs": u["jobs"], "Reports": u["reports"], "Created": u["created_at"]}
            for u in users
        ]
    )
    st.markdown(df_table_html(table, 200), unsafe_allow_html=True)
    others = [u for u in users if u["id"] != user["id"]]
    if not others:
        st.caption("No other accounts yet.")
        return
    labels = [f"{u['email']} ({u['role']})" for u in others]
    choice = st.selectbox("Account", labels, key="radmin_target")
    action = st.selectbox("Action", ["Make administrator", "Make recruiter", "Reset password", "Delete account"], key="radmin_action")
    confirm = st.checkbox("I understand this cannot be undone", key="radmin_confirm")
    if st.button("Apply", type="primary", key="radmin_apply"):
        target = others[labels.index(choice)]
        if action in ("Reset password", "Delete account") and not confirm:
            _show(st, "Tick the box to confirm.")
        elif action in ("Make administrator", "Make recruiter"):
            problem = auth.set_role(db, user["id"], target["id"], "admin" if action == "Make administrator" else "recruiter")
            if problem:
                _show(st, problem)
            else:
                state["admin_flash"] = f"{target['email']} is now {'an administrator' if action == 'Make administrator' else 'a recruiter'}."
                st.rerun()
        elif action == "Reset password":
            temporary, problem = auth.admin_reset_password(db, user["id"], target["id"])
            if problem:
                _show(st, problem)
            else:
                _show(st, "Temporary password created. Share it with the user privately. They must change it when they sign in. It is shown only now.", "info", "🔑")
                st.code(temporary)
        else:
            problem = auth.delete_user(db, user["id"], target["id"])
            if problem:
                _show(st, problem)
            else:
                state["admin_flash"] = f"{target['email']} was deleted."
                st.rerun()


def _auth_body_prototype(st, state, middle) -> None:  # pragma: no cover - UI
    view = state.get("auth_view", "login")
    with middle:
        flash = state.pop("auth_flash", None)
        if flash:
            _show(st, flash, "info", "✅")

        if view == "signup":
            st.markdown(section_html("Create your account"), unsafe_allow_html=True)
            st.text_input("Full name", key="su_name", placeholder="Full name (optional)", label_visibility="collapsed")
            email = st.text_input("Work email", key="su_email", placeholder="Work email", label_visibility="collapsed")
            st.selectbox("I am a", ["Recruiter", "Administrator (demo)"], key="su_role")
            password = st.text_input("Password", type="password", key="su_pw", placeholder="Password (10+ characters)", label_visibility="collapsed")
            st.markdown(password_meter_html(password), unsafe_allow_html=True)
            confirm = st.text_input("Confirm password", type="password", key="su_pw2", placeholder="Confirm password", label_visibility="collapsed")
            agreed = st.checkbox("I understand this is a prototype and no real account is created.", key="su_terms")
            if st.button("Create account", type="primary", use_container_width=True, key="su_submit"):
                errors = prototype_signup_errors(email, password, confirm, agreed)
                for message in errors:
                    _show(st, message)
                if not errors:
                    role = "admin" if str(state.get("su_role", "")).startswith("Admin") else "recruiter"
                    state["pending_user"] = {"email": email.strip().lower(), "role": role}
                    state["auth_view"] = "verify"
                    st.rerun()
            st.button("Already have an account? Sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "verify":
            st.markdown(section_html("Verify your email"), unsafe_allow_html=True)
            pending = state.get("pending_user")
            if not pending:
                state["auth_view"] = "login"
                st.rerun()
            else:
                st.caption(f"Simulated step: a real app would email a 6-digit code to {pending['email']}. Enter any 6 digits.")
                with st.form("verify_form"):
                    code = st.text_input("Code", placeholder="6-digit code", label_visibility="collapsed", max_chars=6)
                    submitted = st.form_submit_button("Verify and continue", type="primary", use_container_width=True)
                if submitted:
                    if re.fullmatch(r"\d{6}", (code or "").strip()):
                        state.pop("pending_user", None)
                        _start_session(state, pending["email"], pending["role"])
                        st.rerun()
                    else:
                        _show(st, "Enter 6 digits.")
                if st.button("Resend code (simulated)", use_container_width=True):
                    _show(st, "Prototype: no email was sent.", "info", "✉️")
                st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "forgot":
            st.markdown(section_html("Reset your password"), unsafe_allow_html=True)
            with st.form("forgot_form"):
                email = st.text_input("Email address", placeholder="Email address", label_visibility="collapsed")
                submitted = st.form_submit_button("Email me a reset link", type="primary", use_container_width=True)
            if submitted:
                if simple_email_ok(email):
                    state["pending_email"] = email.strip().lower()
                    state["auth_view"] = "forgot_sent"
                    st.rerun()
                else:
                    _show(st, "Enter a valid email address.")
            st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "forgot_sent":
            st.markdown(section_html("Check your inbox"), unsafe_allow_html=True)
            _show(
                st,
                f"Simulated: if this were live, a reset link would be emailed to {state.get('pending_email', 'you')}. Nothing was sent.",
                "info",
                "✉️",
            )
            st.button("Preview the reset screen", on_click=_set_view, args=(state, "reset"), use_container_width=True)
            st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        elif view == "reset":
            st.markdown(section_html("Choose a new password"), unsafe_allow_html=True)
            password = st.text_input("New password", type="password", key="rs_pw", placeholder="New password (10+ characters)", label_visibility="collapsed")
            st.markdown(password_meter_html(password), unsafe_allow_html=True)
            confirm = st.text_input("Confirm new password", type="password", key="rs_pw2", placeholder="Confirm new password", label_visibility="collapsed")
            if st.button("Update password", type="primary", use_container_width=True, key="rs_submit"):
                errors = prototype_password_errors(password, confirm)
                for message in errors:
                    _show(st, message)
                if not errors:
                    state["auth_flash"] = "Password updated (simulated). Sign in to continue."
                    state["auth_view"] = "login"
                    st.rerun()
            st.button("Back to sign in", on_click=_set_view, args=(state, "login"), use_container_width=True)

        else:
            st.markdown(section_html("Welcome back"), unsafe_allow_html=True)
            with st.form("login_form"):
                email = st.text_input("Email address", placeholder="Email address", label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                errors = prototype_signin_errors(email, password)
                for message in errors:
                    _show(st, message)
                if not errors:
                    _start_session(state, email, role_for_email(email))
                    st.rerun()
            st.caption("Prototype tip: any valid-looking email and password works. Start the email with 'admin' to see the administrator view.")
            st.button("New to FairHire? Create an account", on_click=_set_view, args=(state, "signup"), use_container_width=True)
            st.button("Forgot your password?", on_click=_set_view, args=(state, "forgot"), use_container_width=True)
            st.button("Try the demo as a recruiter", on_click=_start_session, args=(state, "demo.recruiter@example.com", "recruiter"), use_container_width=True)
            st.button("Try the demo as an administrator", on_click=_start_session, args=(state, "admin.demo@example.com", "admin"), use_container_width=True)
        st.caption(PROTOTYPE_FOOTNOTE)


def render_admin_prototype(st, state, user) -> None:  # pragma: no cover - UI
    st.markdown(section_html("Administration"), unsafe_allow_html=True)
    _show(
        st,
        "Prototype: these accounts are made up. In the real version an administrator could manage accounts but not read anyone's "
        "saved data, only how many items each person has.",
        "info",
        "🛡️",
    )
    note = state.pop("admin_flash", None)
    if note:
        _show(st, note, "info", "✅")
    yours = {
        "id": 0, "email": user["email"], "role": user["role"], "verified": True,
        "jobs": len(state.get("saved_jobs", [])), "reports": len(state.get("saved_reports", [])), "created": str(user["created_at"])[:10],
    }
    users = [yours] + list(state.get("demo_users", []))
    table = pd.DataFrame(
        [
            {"Email": u["email"] + (" (you)" if u["id"] == 0 else ""), "Role": u["role"], "Verified": "yes" if u["verified"] else "no",
             "Jobs": u["jobs"], "Reports": u["reports"], "Created": u["created"]}
            for u in users
        ]
    )
    st.markdown(df_table_html(table, 200), unsafe_allow_html=True)
    others = list(state.get("demo_users", []))
    if not others:
        st.caption("No other sample accounts left.")
        return
    labels = [f"{u['email']} ({u['role']})" for u in others]
    choice = st.selectbox("Account", labels, key="admin_target")
    action = st.selectbox("Action", ["Make administrator", "Make recruiter", "Reset password", "Delete account"], key="admin_action")
    if st.button("Apply (simulated)", type="primary", key="admin_apply"):
        state["admin_flash"] = demo_user_action(state, others[labels.index(choice)]["email"], action)
        st.rerun()


def main() -> None:  # pragma: no cover - exercised only inside Streamlit
    import streamlit as st

    st.set_page_config(page_title="FairHire", page_icon="⚖️", layout="wide")
    state = st.session_state
    defaults = {
        "docs": [], "read_problems": [], "notice": "", "synthetic": None, "dark_mode": False,
        "review_log": {}, "audit_report": None, "use_sample": False, "jd_text": "", "req_skills": [],
        "min_years": 0.0, "shortlist": 3, "w_skill": 60, "w_exp": 25, "w_kw": 15,
        "custom_skills_text": "", "extra_skills_text": "", "anonymize": True, "reveal_files": False,
        "stress_raw": False, "auth_view": "login", "job_title": "", "report_title": "", "audit_title": "",
        "split_mode": "Detect automatically", "ingest_rows": [], "show_original": False, "auth_mode": REAL_MODE, "jd_check_text": "",
    }
    for key, value in defaults.items():
        state.setdefault(key, value)

    smtp_conf, admin_email, database_url = read_app_secrets(st)
    db, db_error = None, None
    try:
        db = auth.get_database(database_url)
    except Exception as exc:  # noqa: BLE001 - never show the connection string
        db_error = f"Could not open the database ({type(exc).__name__}). Real accounts are unavailable; the prototype still works."
    user = state.get("user")
    if user and user.get("mode") == "real":  # re-read every run so role changes and deleted accounts take effect at once
        fresh = auth.get_user(db, user["id"]) if db is not None else None
        if fresh is None:
            sign_out(state)
            user = None
        else:
            user = dict(fresh, mode="real")
            state["user"] = user

    top_left, top_right = st.columns([6, 1.3])
    with top_right:
        st.toggle("🌙 Dark mode", key="dark_mode")
    dark = bool(state.get("dark_mode"))
    with top_right:
        st.markdown(theme_label_html(dark), unsafe_allow_html=True)
        if user:
            st.markdown(f'<div class="fh-theme-label">{_esc(user["email"])} | {_esc(user["role"])} | {"prototype" if user.get("mode") == "prototype" else "account"}</div>', unsafe_allow_html=True)
            st.button("Sign out", on_click=sign_out, args=(state,), use_container_width=True)
    st.markdown(theme_css(dark) + theme_marker_html(dark), unsafe_allow_html=True)
    try:  # second, independent way to apply the theme (see theme_script_html)
        import streamlit.components.v1 as components

        components.html(theme_script_html(dark), height=0)
    except Exception:  # noqa: BLE001 - purely a progressive enhancement
        pass
    with top_left:
        if user:
            st.markdown(hero_html(), unsafe_allow_html=True)

    if not state.get("intro_done"):  # welcome screen: once per visit
        st.markdown(intro_html(), unsafe_allow_html=True)
        state["intro_done"] = True

    if not user:
        render_auth(st, state, db, db_error, smtp_conf, admin_email)
        return
    if user["must_change"]:
        render_must_change(st, state, db, user)
        return
    ws = SessionWorkspace(state) if user.get("mode") == "prototype" else DbWorkspace(db, user["id"], state)
    flash = state.pop("flash", None)
    if flash:
        st.markdown(note_html(flash[1], flash[0], "✅" if flash[0] == "info" else "⛔"), unsafe_allow_html=True)
    if user.get("mode") == "prototype":
        st.markdown(note_html(PROTOTYPE_BANNER, "info", "🧪"), unsafe_allow_html=True)
    else:
        st.markdown(
            note_html(
                f"Signed in. Saved items are kept in {ws.label}. Resumes are read in memory and never saved.",
                "info" if db.kind == "postgres" else "warn", "🔒" if db.kind == "postgres" else "⚠️",
            ),
            unsafe_allow_html=True,
        )

    is_admin = user["role"] == "admin"
    tab_names = [
        "🏠 Dashboard", "📄 Screening", "🔎 Job description check", "📝 Review & report", "🧪 Bias stress test",
        "📊 Fairness audit", "📘 Methodology",
    ]
    if is_admin:
        tab_names.append("🛡️ Admin")
    tabs = st.tabs(tab_names)
    tab_dash, tab_screen, tab_jd, tab_review, tab_stress, tab_audit, tab_method = tabs[:7]
    tab_admin = tabs[7] if is_admin else None

    with tab_dash:
        render_dashboard(st, state, user, ws)

    results: List[Dict[str, object]] = []
    config: Dict[str, object] = {}
    bank: Dict[str, List[str]] = SKILL_BANK
    required: List[str] = []
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS
    run = False

    # ------------------------------------------------------------------ Screening tab
    with tab_screen:
        left, right = st.columns([1, 2], gap="large")

        def load_demo() -> None:
            state["use_sample"] = True
            state["jd_text"] = SAMPLE_JOB_DESCRIPTION
            state["req_skills"] = list(DEFAULT_REQUIRED_SKILLS)
            state["min_years"] = 2.0
            state["run_demo"] = True

        with left:
            st.markdown('<div class="fh-card-marker"></div>' + section_html("Job setup", 1), unsafe_allow_html=True)
            st.button("✨ One-click demo", on_click=load_demo, use_container_width=True)
            st.checkbox("Include the sample resumes (demo)", key="use_sample")
            st.text_area("Job description", key="jd_text", height=150, placeholder="Paste the job description here...")
            jd_flag_count = len(check_job_description(state["jd_text"])["findings"])
            if jd_flag_count:
                st.caption(f"🔎 {jd_flag_count} wording flag(s) in this job description. See the Job description check tab.")
            with st.expander("➕ Add your own skills"):
                st.caption("One per line: skill = alias 1, alias 2")
                st.text_area(
                    "Custom skills", key="custom_skills_text", height=90, label_visibility="collapsed",
                    placeholder="rag = retrieval augmented generation\nfigma",
                )
            custom_bank, custom_warns = parse_custom_skills(state["custom_skills_text"])
            bank = merged_skill_bank(custom_bank)
            state["req_skills"] = [s for s in state["req_skills"] if s in bank]
            st.multiselect("Required skills", options=sorted(bank), key="req_skills")
            st.text_input("Extra required skills (comma-separated)", key="extra_skills_text")
            n1, n2 = st.columns(2)
            n1.number_input("Min. years", min_value=0.0, max_value=40.0, step=0.5, key="min_years")
            n2.number_input("Shortlist size", min_value=1, max_value=500, step=1, key="shortlist")
            with st.expander("⚖️ Scoring weights (live re-ranking)"):
                st.slider("Required skills", 0, 100, key="w_skill")
                st.slider("Experience", 0, 100, key="w_exp")
                st.slider("Job-description keywords", 0, 100, key="w_kw")
                if state["w_skill"] + state["w_exp"] + state["w_kw"] <= 0:
                    st.caption("All weights are 0, so the defaults are used.")
                else:
                    wp = weights_percent((state["w_skill"], state["w_exp"], state["w_kw"]))
                    st.caption(
                        f"Effective weights: skills {wp[0]:g}% | experience {wp[1]:g}% | keywords {wp[2]:g}%. "
                        "Parts you have not configured are skipped. Rankings update as you move the sliders."
                    )
            with st.expander("💾 Save this job to my dashboard"):
                st.text_input("Job title", key="job_title", placeholder="e.g. Data Scientist, Q4")
                if st.button("Save job", key="save_job_btn"):
                    problem = ws.add_job(state["job_title"], state["jd_text"], job_settings_from_state(state))
                    state["flash"] = ("bad", problem) if problem else ("info", f"Job saved to {ws.label}.")
                    st.rerun()
            st.checkbox("Anonymize resumes before scoring", key="anonymize")
            st.checkbox("Show file names in results", key="reveal_files")
            st.checkbox("Show the original text next to the anonymized text (reveals names)", key="show_original")

            st.markdown(section_html("Upload resumes", 2), unsafe_allow_html=True)
            files = st.file_uploader(
                "TXT, PDF or DOCX (max 25 MB each)", type=["txt", "pdf", "docx"], accept_multiple_files=True
            )
            st.selectbox("If a PDF contains several resumes", list(SPLIT_MODES), key="split_mode")
            st.caption("Resumes are read in memory for this session only and are never saved.")
            run = st.button("Run screening", type="primary", use_container_width=True)

        demo_requested = state.pop("run_demo", False)
        if run or demo_requested:
            uploads = [(u.name, u.getvalue()) for u in files or []]
            documents, ingest_rows, problems = ingest_uploads(uploads, SPLIT_MODES.get(state["split_mode"], "auto"))
            if state["use_sample"]:
                documents.extend(SAMPLE_RESUMES.items())
            state["docs"] = documents
            state["read_problems"] = problems
            state["ingest_rows"] = ingest_rows
            state["notice"] = "" if documents else "Upload at least one readable resume, or tick the sample option."

        extras = [s.strip() for s in state["extra_skills_text"].split(",") if s.strip()]
        required = _dedupe_keep_order(
            s for s in (canonical_skill(x, bank) for x in list(state["req_skills"]) + extras) if s
        )
        weights = (state["w_skill"], state["w_exp"], state["w_kw"])
        min_years = float(state["min_years"])
        anonymize = bool(state["anonymize"])
        reveal_files = bool(state["reveal_files"])
        shortlist = int(state["shortlist"])
        jd_text = state["jd_text"]

        warns: List[str] = []
        if state["docs"]:
            results, warns = screen_resumes(
                state["docs"], required, jd_text, min_years, anonymize, weights=weights, skill_bank=bank
            )
        config = {
            "required": required, "min_years": min_years, "weights": weights, "anonymize": anonymize,
            "shortlist": shortlist, "jd_given": bool(jd_text.strip()), "custom_skills": len(custom_bank),
            "jd_flags": jd_flag_count,
        }

        with right:
            st.markdown(section_html("Results"), unsafe_allow_html=True)
            if state["notice"]:
                st.markdown(note_html(state["notice"], "info", "ℹ️"), unsafe_allow_html=True)
            for message in list(state["read_problems"]) + warns + custom_warns:
                st.markdown(note_html(message, "warn", "⚠️"), unsafe_allow_html=True)
            if state["ingest_rows"]:
                st.markdown(ingest_report_html(state["ingest_rows"]), unsafe_allow_html=True)

            if not results:
                st.markdown(
                    empty_state_html(
                        "No results yet",
                        "Press One-click demo, or set up the job, upload resumes and press Run screening.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                if not required and not config["jd_given"] and min_years <= 0:
                    st.markdown(
                        note_html("Nothing to score against. Add required skills, a job description or minimum years.", "bad", "⛔"),
                        unsafe_allow_html=True,
                    )
                top_n = min(shortlist, len(results))
                average = sum(float(r["score"]) for r in results) / len(results)
                flagged = sum(1 for r in results if r["proxy_flags"])
                st.markdown(
                    kpis_html(
                        [
                            kpi_card("Resumes scored", len(results)),
                            kpi_card("Top score", f"{float(results[0]['score']):.1f}", "out of 100", score_tone(float(results[0]["score"]))),
                            kpi_card("Average score", f"{average:.1f}", "across all resumes"),
                            kpi_card("Shortlisted", top_n, "top-ranked candidates"),
                            kpi_card("Proxy flags", flagged, "resumes with sensitive details", "warn" if flagged else "neutral"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(results_table_html(results, top_n, reveal_files), unsafe_allow_html=True)

                st.markdown(section_html("Score distribution"), unsafe_allow_html=True)
                st.markdown(score_histogram_html([float(r["score"]) for r in results]), unsafe_allow_html=True)

                export_df = results_to_dataframe(results, include_file_names=reveal_files)
                export_df.insert(2, "Shortlist", ["Yes" if r["rank"] <= top_n else "" for r in results])
                st.download_button(
                    "⬇️ Download ranking (CSV)",
                    data=dataframe_to_csv_bytes(export_df),
                    file_name="fairhire_ranking.csv",
                    mime="text/csv",
                )

                st.markdown(section_html("Why each candidate scored as they did"), unsafe_allow_html=True)
                st.markdown(
                    "".join(
                        candidate_card_html(item, reveal_files, open_card=(index == 0), show_original=bool(state["show_original"]), skill_bank=bank)
                        for index, item in enumerate(results[:MAX_DETAIL_CARDS])
                    ),
                    unsafe_allow_html=True,
                )
                if len(results) > MAX_DETAIL_CARDS:
                    rest = results[MAX_DETAIL_CARDS:]
                    _show(st, f"Showing details for the top {MAX_DETAIL_CARDS} of {len(results)} candidates. Pick any other candidate below.", "info", "ℹ️")
                    rest_labels = [f"{r['candidate_id']} ({float(r['score']):.1f})" for r in rest]
                    pick = st.selectbox("Open details for another candidate", rest_labels)
                    st.markdown(candidate_card_html(rest[rest_labels.index(pick)], reveal_files, open_card=True, show_original=bool(state["show_original"]), skill_bank=bank), unsafe_allow_html=True)

    # ------------------------------------------------------------------ Job description check tab
    with tab_jd:
        st.markdown(section_html("Job description check"), unsafe_allow_html=True)
        _show(
            st,
            "Checks the wording of a job post for terms that can discourage or exclude groups of applicants (gender-coded words, age, "
            "nationality, family status, appearance and more). It is rule-based and not exhaustive: a flag is a prompt to reconsider the "
            "wording, not proof of discrimination.",
            "info",
            "🔎",
        )
        st.text_area(
            "Or paste a different job description to check", key="jd_check_text", height=110,
            placeholder="Leave empty to check the job description from the Screening tab",
        )
        checking_own = not state["jd_check_text"].strip()
        jd_source = state["jd_text"] if checking_own else state["jd_check_text"]
        if not jd_source.strip():
            st.markdown(
                empty_state_html("Nothing to check yet", "Paste a job description on the Screening tab or in the box above.", "🔎"),
                unsafe_allow_html=True,
            )
        else:
            jd_result = check_job_description(jd_source)
            st.markdown(jd_summary_html(jd_result), unsafe_allow_html=True)
            left_col, right_col = st.columns([2, 1], gap="large")
            with left_col:
                st.markdown(section_html("Flagged wording"), unsafe_allow_html=True)
                st.markdown(jd_findings_html(jd_result), unsafe_allow_html=True)
            with right_col:
                st.markdown(section_html("Gender-coded wording"), unsafe_allow_html=True)
                st.markdown(gender_bar_html(jd_result["gender"]), unsafe_allow_html=True)
            revised, changes = rewrite_job_description(jd_source, jd_result["findings"])
            if changes:
                st.markdown(section_html("Suggested revision"), unsafe_allow_html=True)
                st.caption(
                    f"{changes} safe replacement(s) applied. Flags without an automatic replacement need your judgement, "
                    "so check the wording reads well."
                )
                st.text_area("Revised job description", value=revised, height=180)
                if checking_own:
                    st.button(
                        "Use this revised text as my job description", on_click=_apply_revised_jd, args=(state, revised),
                        type="primary",
                    )
                else:
                    st.caption("Copy the revised text from the box above.")

    # ------------------------------------------------------------------ Fairness audit tab
    # (built before the review tab so the report can include the latest simulation)
    with tab_audit:
        st.markdown(section_html("Fairness audit"), unsafe_allow_html=True)
        st.markdown(
            note_html(
                "SIMULATION: the data below is synthetic. It shows how the metrics behave; it is not evidence "
                "about real candidates or about your hiring process.",
                "warn",
                "🧪",
            ),
            unsafe_allow_html=True,
        )
        a1, a2, a3, a4 = st.columns(4)
        n_syn = a1.slider("Synthetic candidates", 50, 1000, 300, step=50)
        bias = a2.slider("Simulated bias vs Group B (pts)", 0, 40, 15)
        frac = a3.slider("Shortlist fraction", 0.05, 0.6, 0.25, step=0.05)
        seed = a4.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

        if st.button("Run simulation", type="primary") or state["synthetic"] is None:
            state["synthetic"] = generate_synthetic_candidates(
                n=n_syn, seed=int(seed), bias_penalty=float(bias), shortlist_fraction=float(frac)
            )

        syn = state["synthetic"]
        if syn is not None and len(syn):
            col_m, col_b = st.columns(2, gap="large")
            for column, section in zip((col_m, col_b), synthetic_fairness_sections(syn)):
                with column:
                    st.markdown(fairness_panel_html(section["title"], section["report"]), unsafe_allow_html=True)

            st.markdown(section_html("Synthetic data preview"), unsafe_allow_html=True)
            st.markdown(df_table_html(syn, 50), unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download synthetic data (CSV)",
                data=dataframe_to_csv_bytes(syn),
                file_name="fairhire_synthetic.csv",
                mime="text/csv",
            )

        st.markdown(section_html("Audit your own outcomes table (optional)"), unsafe_allow_html=True)
        st.caption("Upload a CSV with one row per person, a group column and a selected / not-selected column.")
        outcomes = st.file_uploader("Outcomes CSV", type=["csv"], key="outcomes_csv")
        state["audit_report"] = None
        if outcomes is not None:
            table = None
            try:
                table = pd.read_csv(io.BytesIO(outcomes.getvalue()))
            except Exception as exc:  # noqa: BLE001
                st.markdown(note_html(f"Could not read the CSV ({type(exc).__name__}).", "bad", "⛔"), unsafe_allow_html=True)
            if table is not None and len(table.columns) >= 2:
                g_col = st.selectbox("Group column", list(table.columns), index=0)
                s_col = st.selectbox("Selected column", list(table.columns), index=1)
                try:
                    report = audit_uploaded_outcomes(table, g_col, s_col)
                    state["audit_report"] = report
                    st.markdown(fairness_panel_html("Your outcomes", report), unsafe_allow_html=True)
                except ValueError as exc:
                    st.markdown(note_html(str(exc), "bad", "⛔"), unsafe_allow_html=True)
            elif table is not None:
                st.markdown(note_html("The CSV needs at least two columns.", "bad", "⛔"), unsafe_allow_html=True)

        audit_sections = synthetic_fairness_sections(state["synthetic"])
        if state["audit_report"]:
            audit_sections.append({"title": "Uploaded outcomes table", "report": state["audit_report"], "simulated": False})
        if audit_sections:
            st.markdown(section_html("Save this audit"), unsafe_allow_html=True)
            st.text_input("Audit title", key="audit_title", placeholder="e.g. Q4 audit, 15-point simulated bias")
            if st.button("💾 Save audit to my dashboard", key="save_audit_btn"):
                problem = ws.add_report(
                    "audit", state["audit_title"], build_audit_report_html(audit_sections), audit_summary(audit_sections)
                )
                state["flash"] = ("bad", problem) if problem else ("info", f"Audit saved to {ws.label}.")
                st.rerun()

    # ------------------------------------------------------------------ Review & report tab
    with tab_review:
        st.markdown(section_html("Reviewer decisions"), unsafe_allow_html=True)
        if not results:
            st.markdown(
                empty_state_html("Nothing to review yet", "Run a screening first, then record your decisions here.", "📝"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                note_html(
                    "You make the decisions. Record yours for each candidate; Hold and Reject need a reason that is related to the job.",
                    "info",
                    "🧑‍⚖️",
                ),
                unsafe_allow_html=True,
            )
            review_items = results[:MAX_REVIEW_ITEMS]
            if len(results) > MAX_REVIEW_ITEMS:
                st.caption(f"Recording decisions for the top {MAX_REVIEW_ITEMS} of {len(results)} candidates.")
            with st.form("review_form"):
                for item in review_items:
                    fingerprint = str(item["fingerprint"])[:12]
                    saved = state["review_log"].get(fingerprint, {})
                    c1, c2, c3 = st.columns([2, 2, 4])
                    c1.markdown(f"**{item['candidate_id']}** | {float(item['score']):.1f}")
                    c2.selectbox(
                        "Decision", REVIEW_DECISIONS, key=f"dec_{fingerprint}", label_visibility="collapsed",
                        index=REVIEW_DECISIONS.index(saved.get("decision", "Not reviewed")),
                    )
                    c3.text_input(
                        "Reason", value=saved.get("reason", ""), key=f"why_{fingerprint}",
                        label_visibility="collapsed", placeholder="Reason (job-related only)",
                    )
                submitted = st.form_submit_button("Save decisions", type="primary")

            if submitted:
                errors: List[str] = []
                new_log = dict(state["review_log"])
                for item in review_items:
                    fingerprint = str(item["fingerprint"])[:12]
                    decision = state.get(f"dec_{fingerprint}", "Not reviewed")
                    reason = str(state.get(f"why_{fingerprint}", "")).strip()
                    if decision == "Not reviewed":
                        new_log.pop(fingerprint, None)
                        continue
                    problem = validate_review_entry(decision, reason)
                    if problem:
                        errors.append(f"{item['candidate_id']}: {problem}")
                        continue
                    new_log[fingerprint] = {
                        "time": now_utc_text(), "candidate": item["candidate_id"], "score": float(item["score"]),
                        "decision": decision, "reason": reason,
                    }
                state["review_log"] = new_log
                for message in errors:
                    st.markdown(note_html(message, "bad", "⛔"), unsafe_allow_html=True)
                if not errors:
                    st.markdown(note_html("Decisions saved.", "info", "✅"), unsafe_allow_html=True)

            if state["review_log"]:
                log_df = review_log_dataframe(state["review_log"])
                st.markdown(section_html("Decision log"), unsafe_allow_html=True)
                st.markdown(df_table_html(log_df, 100), unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Download decision log (CSV)",
                    data=dataframe_to_csv_bytes(log_df),
                    file_name="fairhire_review_log.csv",
                    mime="text/csv",
                )

            st.markdown(section_html("Audit report"), unsafe_allow_html=True)
            st.caption(
                "One file with the settings, ranking, explanations, proxy flags, fairness results (simulated results are labelled) "
                "and your decisions. Open it in a browser and press Ctrl+P (Cmd+P on a Mac) to save it as a PDF."
            )
            sections = synthetic_fairness_sections(state["synthetic"])
            if state["audit_report"]:
                sections.append({"title": "Uploaded outcomes table", "report": state["audit_report"], "simulated": False})
            report_html = build_report_html(
                results, config, sections, list(state["review_log"].values()), show_files=bool(state["reveal_files"])
            )
            st.download_button(
                "⬇️ Download audit report (HTML)",
                data=report_html.encode("utf-8"),
                file_name="fairhire_audit_report.html",
                mime="text/html",
            )
            st.text_input("Report title", key="report_title", placeholder="e.g. Data Scientist screening, Oct 2026")
            if st.button("💾 Save report to my dashboard", key="save_report_btn"):
                saved_html = build_report_html(
                    results, config, sections, list(state["review_log"].values()), show_files=False
                )
                summary = {
                    "resumes": len(results),
                    "top_score": round(float(results[0]["score"]), 1),
                    "decisions": len(state["review_log"]),
                }
                problem = ws.add_report("screening", state["report_title"], saved_html, summary)
                state["flash"] = ("bad", problem) if problem else ("info", f"Report saved to {ws.label}.")
                st.rerun()

    # ------------------------------------------------------------------ Bias stress test tab
    with tab_stress:
        st.markdown(section_html("Bias stress test"), unsafe_allow_html=True)
        st.markdown(
            note_html(
                "Each resume is re-scored with its name replaced by other names and its gendered words swapped. "
                "A fair, identity-blind scorer should give the same score every time. This checks the scoring step only; "
                "it does not prove the rest of the process is unbiased.",
                "info",
                "🧪",
            ),
            unsafe_allow_html=True,
        )
        if not results:
            st.markdown(
                empty_state_html("No resumes to test", "Run a screening first.", "🧪"), unsafe_allow_html=True
            )
        else:
            st.checkbox("Test the raw text (ignore the anonymization setting)", key="stress_raw")
            use_anonymize = bool(state["anonymize"]) and not state["stress_raw"]
            stress_items = results[:MAX_STRESS_ITEMS]
            if len(results) > MAX_STRESS_ITEMS:
                st.caption(f"Testing the top {MAX_STRESS_ITEMS} of {len(results)} candidates.")
            tests = [
                bias_stress_test(
                    str(item["original_text"]), required, state["jd_text"], float(state["min_years"]),
                    weights, bank, use_anonymize,
                )
                for item in stress_items
            ]
            unchanged = sum(1 for t in tests if t["passed"])
            worst = max(float(t["max_abs_delta"]) for t in tests)
            st.markdown(
                kpis_html(
                    [
                        kpi_card("Resumes tested", len(tests)),
                        kpi_card("Scores unchanged", f"{unchanged} / {len(tests)}", "", "ok" if unchanged == len(tests) else "bad"),
                        kpi_card("Largest change", f"{worst:.1f}", "points", "ok" if worst <= 0.05 else "bad"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            st.markdown(stress_summary_html(stress_items, tests), unsafe_allow_html=True)
            labels = [f"{item['candidate_id']} ({float(item['score']):.1f})" for item in stress_items]
            chosen = st.selectbox("See the variants for one candidate", labels, index=0)
            st.markdown(stress_detail_html(tests[labels.index(chosen)]), unsafe_allow_html=True)

    # ------------------------------------------------------------------ Methodology tab
    with tab_method:
        st.markdown(METHODOLOGY_MD)

    if tab_admin is not None:
        with tab_admin:
            if user.get("mode") == "prototype":
                render_admin_prototype(st, state, user)
            else:
                render_admin_real(st, state, db, user)

    st.markdown(
        '<div class="fh-footer">FairHire is decision support, not a decision maker. A human reviews every outcome.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
