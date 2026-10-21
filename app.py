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
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per resume
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


def read_resume_bytes(filename: str, data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(text, error)``. Exactly one of the two is ``None``."""
    name = (filename or "").strip()
    lower = name.lower()
    if not data:
        return None, f"'{name}' is empty (0 bytes)."
    if len(data) > MAX_FILE_BYTES:
        return None, f"'{name}' is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB and was skipped."
    if not lower.endswith(SUPPORTED_EXTENSIONS):
        return None, f"'{name}' has an unsupported type. Please upload TXT, PDF or DOCX files."

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

    text = text.replace("\x00", " ").strip()
    if not text:
        return None, f"'{name}' contains no extractable text (scanned images are not supported)."
    return text, None


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("encrypted PDF") from exc
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(data: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


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
    "developer engineer analyst scientist manager designer intern fresher consultant architect administrator "
    "specialist student programmer lead director officer".split()
)
_NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z'\u2019.\-]+(?: [A-Z][A-Za-z'\u2019.\-]+){1,3}$")


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

    This is best-effort pattern matching. It will miss some identifying details
    (for example a name in the middle of a sentence), so review before sharing.
    """
    if not text:
        return ""
    out = text
    out = _EMAIL_RE.sub("[EMAIL]", out)
    out = _URL_RE.sub("[URL]", out)
    out = _PHONE_RE.sub(_mask_phone, out)
    out = _NAME_LABEL_RE.sub(r"\1[NAME]", out)
    out = _DOB_RE.sub("[DOB]", out)
    out = _PERSONAL_LABEL_RE.sub("[REDACTED PERSONAL DETAIL]", out)
    out = _STREET_RE.sub("[ADDRESS]", out)

    lines = out.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        words = {w.lower() for w in stripped.split()}
        looks_like_title = bool(words & _HEADING_WORDS) or bool(words & _ROLE_WORDS) or bool(extract_skills(stripped))
        if _NAME_LINE_RE.match(stripped) and not looks_like_title:
            lines[index] = "[NAME]"
        break  # only the first non-empty line is treated as a possible name
    out = "\n".join(lines)

    out = _GENDERED_RE.sub("[G]", out)
    return out


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
    """Best-effort guess of the candidate's name (labelled line, else a name-like first line)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for line in lines[:6]:
        match = _NAME_LABEL_VALUE_RE.match(line)
        if match and match.group(1).strip():
            return match.group(1).strip()[:60]
    if lines:
        first = lines[0]
        words = {w.lower() for w in first.split()}
        if _NAME_LINE_RE.match(first) and not (words & _HEADING_WORDS):
            return first
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
    ]
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

### Accounts (prototype)
The sign-up, sign-in, email-code, password-reset, dashboard and administrator screens are a **prototype**. No account is
created, no password is kept, no email is sent and nothing is written to a database. Saved jobs, reports and audits live only
in this browser tab and disappear when you close it or sign out. Items marked (sample) are made up. A real version would need
password hashing, a database, email delivery and access controls. Resumes are never saved.

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


def candidate_card_html(item: Dict[str, object], show_files: bool = False, open_card: bool = False) -> str:
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
    if redactions or used != original:
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
# PROTOTYPE account flow
#
# Everything below is SIMULATED. No account is created, no password is kept, no email is
# sent and nothing is written to disk or a database. "Saved" items live only in this
# browser session (st.session_state) and disappear when the tab closes or you sign out.
# --------------------------------------------------------------------------------------

PROTOTYPE_BANNER = "Prototype mode: sign-in is simulated and saved items live only in this browser tab."
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
        if key not in ("dark_mode", "intro_done"):
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


def render_auth(st, state) -> None:  # pragma: no cover - UI
    view = state.get("auth_view", "login")
    _, middle, _ = st.columns([1, 1.5, 1])
    with middle:
        st.markdown('<div class="fh-card-marker"></div>' + auth_brand_html(), unsafe_allow_html=True)
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


def render_dashboard(st, state, user) -> None:  # pragma: no cover - UI
    jobs = list(reversed(state.get("saved_jobs", [])))
    reports = [r for r in reversed(state.get("saved_reports", [])) if r["kind"] == "screening"]
    audits = [r for r in reversed(state.get("saved_reports", [])) if r["kind"] == "audit"]
    st.markdown(section_html("Your dashboard"), unsafe_allow_html=True)
    st.markdown(
        kpis_html([kpi_card("Saved jobs", len(jobs)), kpi_card("Saved reports", len(reports)), kpi_card("Fairness audits", len(audits))]),
        unsafe_allow_html=True,
    )
    _show(
        st,
        "Prototype: items marked (sample) are made up. Anything you save is kept only in this browser tab and disappears when you "
        "close it or sign out. Resumes are never saved.",
        "info",
        "🧪",
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
            apply_job_to_state(state, job)
            state["dash_flash"] = f"Loaded '{job['title']}'. Open the Screening tab to continue."
            st.rerun()
        if c3.button("Delete", key=f"del_job_{job['id']}"):
            delete_saved(state, "saved_jobs", job["id"])
            st.rerun()

    for heading, items in (("Saved reports", reports), ("Previous fairness audits", audits)):
        st.markdown(section_html(heading), unsafe_allow_html=True)
        if not items:
            st.markdown(empty_state_html("Nothing saved yet", "Save one from the Review & report or Fairness audit tab.", "📁"), unsafe_allow_html=True)
        for item in items:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            c1.markdown(
                f'<b>{_esc(item["title"])}</b><div class="fh-muted">{_esc(item["created_at"])} | '
                f'{_esc(describe_summary(item["kind"], item["summary"]))}</div>',
                unsafe_allow_html=True,
            )
            c2.download_button(
                "Download", data=str(item["html"]).encode("utf-8"), file_name=f"fairhire_{item['kind']}_{item['id']}.html",
                mime="text/html", key=f"dl_{item['id']}",
            )
            if c3.button("Delete", key=f"del_rep_{item['id']}"):
                delete_saved(state, "saved_reports", item["id"])
                st.rerun()


def render_admin(st, state, user) -> None:  # pragma: no cover - UI
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
    }
    for key, value in defaults.items():
        state.setdefault(key, value)

    user = state.get("user")  # a simulated account that lives only in this browser session

    top_left, top_right = st.columns([6, 1.3])
    with top_right:
        st.toggle("🌙 Dark mode", key="dark_mode")
    dark = bool(state.get("dark_mode"))
    with top_right:
        st.markdown(theme_label_html(dark), unsafe_allow_html=True)
        if user:
            st.markdown(f'<div class="fh-theme-label">{_esc(user["email"])} | {_esc(user["role"])}</div>', unsafe_allow_html=True)
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
        render_auth(st, state)
        return
    flash = state.pop("flash", None)
    if flash:
        st.markdown(note_html(flash[1], flash[0], "✅" if flash[0] == "info" else "⛔"), unsafe_allow_html=True)
    st.markdown(note_html(PROTOTYPE_BANNER, "info", "🧪"), unsafe_allow_html=True)

    is_admin = user["role"] == "admin"
    tab_names = ["🏠 Dashboard", "📄 Screening", "📝 Review & report", "🧪 Bias stress test", "📊 Fairness audit", "📘 Methodology"]
    if is_admin:
        tab_names.append("🛡️ Admin")
    tabs = st.tabs(tab_names)
    tab_dash, tab_screen, tab_review, tab_stress, tab_audit, tab_method = tabs[:6]
    tab_admin = tabs[6] if is_admin else None

    with tab_dash:
        render_dashboard(st, state, user)

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
                    problem = add_saved_job(state, state["job_title"], state["jd_text"], job_settings_from_state(state))
                    state["flash"] = ("bad", problem) if problem else ("info", "Job saved to your dashboard (this browser tab only).")
                    st.rerun()
            st.checkbox("Anonymize resumes before scoring", key="anonymize")
            st.checkbox("Show file names in results", key="reveal_files")

            st.markdown(section_html("Upload resumes", 2), unsafe_allow_html=True)
            files = st.file_uploader(
                "TXT, PDF or DOCX (max 5 MB each)", type=["txt", "pdf", "docx"], accept_multiple_files=True
            )
            run = st.button("Run screening", type="primary", use_container_width=True)

        demo_requested = state.pop("run_demo", False)
        if run or demo_requested:
            documents: List[Tuple[str, str]] = []
            problems: List[str] = []
            for uploaded in files or []:
                text, error = read_resume_bytes(uploaded.name, uploaded.getvalue())
                if error:
                    problems.append(error)
                else:
                    documents.append((uploaded.name, text))
            if state["use_sample"]:
                documents.extend(SAMPLE_RESUMES.items())
            state["docs"] = documents
            state["read_problems"] = problems
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
        }

        with right:
            st.markdown(section_html("Results"), unsafe_allow_html=True)
            if state["notice"]:
                st.markdown(note_html(state["notice"], "info", "ℹ️"), unsafe_allow_html=True)
            for message in list(state["read_problems"]) + warns + custom_warns:
                st.markdown(note_html(message, "warn", "⚠️"), unsafe_allow_html=True)

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
                        candidate_card_html(item, reveal_files, open_card=(index == 0))
                        for index, item in enumerate(results)
                    ),
                    unsafe_allow_html=True,
                )

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
                problem = add_saved_report(
                    state, "audit", state["audit_title"], build_audit_report_html(audit_sections), audit_summary(audit_sections)
                )
                state["flash"] = ("bad", problem) if problem else ("info", "Audit saved to your dashboard (this browser tab only).")
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
            with st.form("review_form"):
                for item in results:
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
                for item in results:
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
                problem = add_saved_report(state, "screening", state["report_title"], saved_html, summary)
                state["flash"] = ("bad", problem) if problem else ("info", "Report saved to your dashboard (this browser tab only).")
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
            tests = [
                bias_stress_test(
                    str(item["original_text"]), required, state["jd_text"], float(state["min_years"]),
                    weights, bank, use_anonymize,
                )
                for item in results
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
            st.markdown(stress_summary_html(results, tests), unsafe_allow_html=True)
            labels = [f"{item['candidate_id']} ({float(item['score']):.1f})" for item in results]
            chosen = st.selectbox("See the variants for one candidate", labels, index=0)
            st.markdown(stress_detail_html(tests[labels.index(chosen)]), unsafe_allow_html=True)

    # ------------------------------------------------------------------ Methodology tab
    with tab_method:
        st.markdown(METHODOLOGY_MD)

    if tab_admin is not None:
        with tab_admin:
            render_admin(st, state, user)

    st.markdown(
        '<div class="fh-footer">FairHire is decision support, not a decision maker. A human reviews every outcome.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
