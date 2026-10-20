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

import hashlib
import html as _html
import io
import json
import random
import re
from datetime import date
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


def _skill_aliases(skill: str) -> List[str]:
    key = skill.strip().lower()
    return SKILL_BANK.get(key, [key])


def _alias_pattern(alias: str) -> "re.Pattern[str]":
    pattern = _PATTERN_CACHE.get(alias)
    if pattern is None:
        pattern = re.compile(
            r"(?<![A-Za-z0-9+#])" + re.escape(alias) + r"(?![A-Za-z0-9+#])", re.IGNORECASE
        )
        _PATTERN_CACHE[alias] = pattern
    return pattern


def has_skill(text: str, skill: str) -> bool:
    if not text or not skill or not skill.strip():
        return False
    return any(_alias_pattern(alias).search(text) for alias in _skill_aliases(skill))


def canonical_skill(name: str) -> str:
    """Map a user-typed skill to its canonical name when it is in the skill bank."""
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in SKILL_BANK:
        return key
    for canonical, aliases in SKILL_BANK.items():
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

    intervals: List[Tuple[int, int]] = []
    for line in text.splitlines():
        if _EDUCATION_WORDS.search(line):
            continue
        for match in _DATE_RANGE.finditer(line):
            start = int(match.group(1))
            end_raw = match.group(2).lower()
            end = year_now if end_raw in ("present", "current", "now") else int(end_raw)
            if start <= end <= year_now + 1 and end - start <= 50:
                intervals.append((start, end))

    dated = 0.0
    if intervals:
        intervals.sort()
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
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
        if _NAME_LINE_RE.match(stripped) and not (words & _HEADING_WORDS):
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


def score_candidate(
    resume_text: str,
    required_skills: Sequence[str],
    jd_text: str = "",
    min_years: float = 0.0,
    current_year: Optional[int] = None,
) -> Dict[str, object]:
    """Deterministic, explainable 0-100 score.

    Components (weights rescaled over the components that can be evaluated):
      * required-skill coverage ........ 60
      * experience vs. minimum years ... 25
      * job-description keyword overlap  15
    """
    text = resume_text or ""
    reasons: List[str] = []
    required = _dedupe_keep_order(
        s for s in (canonical_skill(x) for x in (required_skills or [])) if s
    )
    if not required and (jd_text or "").strip():
        required = extract_skills(jd_text)
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

    matched = [s for s in required if has_skill(text, s)]
    missing = [s for s in required if s not in matched]
    years = extract_experience_years(text, current_year)
    all_found = extract_skills(text)
    extra = [s for s in all_found if s not in required]

    jd_keywords = _keyword_set(jd_text)
    resume_lower = text.lower()
    kw_hits = [k for k in jd_keywords if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", resume_lower)]
    kw_fraction = (len(kw_hits) / len(jd_keywords)) if jd_keywords else None

    skill_fraction = (len(matched) / len(required)) if required else None
    exp_fraction = min(years / min_years, 1.0) if min_years > 0 else None

    components = [
        ("skill_points", WEIGHT_SKILLS, skill_fraction),
        ("experience_points", WEIGHT_EXPERIENCE, exp_fraction),
        ("keyword_points", WEIGHT_KEYWORDS, kw_fraction),
    ]
    active = [(key, w, f) for key, w, f in components if f is not None]
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
        detail = score_candidate(working_text, required_skills, jd_text, min_years, current_year)
        detail["file_name"] = file_name
        detail["text_used"] = working_text
        results.append(detail)

    # Deterministic order: score desc, then more matched skills, then file name.
    results.sort(key=lambda r: (-float(r["score"]), -len(r["matched_skills"]), str(r["file_name"])))
    for position, item in enumerate(results, start=1):
        item["rank"] = position
        item["candidate_id"] = f"Candidate {position:03d}"
    return results, warnings


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
            "Why": " ".join(r["reasons"]),
        }
        if include_file_names:
            row["File"] = csv_safe(r["file_name"])
        rows.append(row)
    columns = [
        "Rank", "Candidate", "Score", "Skills pts", "Experience pts", "Keyword pts",
        "Years (detected)", "Matched skills", "Missing skills", "Why",
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
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    dpd = demographic_parity_difference(rates)
    dir_ = disparate_impact_ratio(rates)
    notes: List[str] = []
    if len(rates) < 2:
        notes.append("At least two groups are needed to compare selection rates.")
    small = [g for g, n in counts.items() if n < 10]
    if small:
        notes.append("Small groups (<10 people): " + ", ".join(sorted(small)) + ". Results are statistically unreliable.")
    if dir_ is None and len(rates) >= 2:
        notes.append("Disparate impact ratio is undefined because nobody was selected.")
    return {
        "counts": counts,
        "rates": rates,
        "demographic_parity_difference": dpd,
        "disparate_impact_ratio": dir_,
        "four_fifths_pass": (dir_ >= 0.8) if dir_ is not None else None,
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
.fh-footer {text-align: center; color: var(--muted); font-size: .8rem; margin-top: 2rem;}
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


def results_table_html(results: Sequence[Dict[str, object]], top_n: int, show_files: bool = False) -> str:
    head = ["Rank", "Candidate", "Score", "Years", "Matched skills", "Missing skills", "Shortlist"]
    rows = []
    for r in results:
        name = _esc(r["candidate_id"])
        if show_files:
            name += f'<div class="fh-muted" style="font-size:.75rem">{_esc(r["file_name"])}</div>'
        shortlisted = int(r["rank"]) <= top_n
        rows.append(
            "<tr>"
            f'<td><span class="fh-rank">{int(r["rank"])}</span></td>'
            f"<td><b>{name}</b></td>"
            f"<td>{_score_cell(float(r['score']))}</td>"
            f"<td>{float(r['years_experience']):g}</td>"
            f"<td>{chips_html(r['matched_skills'], 'ok', 6)}</td>"
            f"<td>{chips_html(r['missing_skills'], 'bad', 6)}</td>"
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
    stack = "".join(
        f'<i style="width:{clamp(p):.1f}%;background:{color}"></i>' for _, p, color in parts
    )
    legend = "".join(
        f'<span><span class="fh-dot" style="background:{color}"></span>{_esc(name)} {p:.1f} pts</span>'
        for name, p, color in parts
    )
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in item["reasons"]) or '<li class="fh-muted">No details.</li>'
    raw = _esc(str(item.get("text_used", ""))[:3000])
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
        f'<details class="fh-raw"><summary>View text used for scoring</summary><pre>{raw}</pre></details>'
        "</div></details>"
    )


def rates_bars_html(rates: Dict[str, float], counts: Optional[Dict[str, int]] = None) -> str:
    rows = []
    for index, group in enumerate(sorted(rates)):
        rate = clamp(rates[group] * 100.0)
        n = f" (n={counts[group]})" if counts and group in counts else ""
        rows.append(
            f'<div class="fh-rate"><div class="fh-rate-head"><span>{_esc(group)}</span>'
            f"<span>{rate:.1f}%{_esc(n)}</span></div>"
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
    notes = "".join(f'<div class="fh-muted" style="font-size:.8rem;margin-top:.3rem">{_esc(n)}</div>' for n in report["notes"])
    return (
        f'<div class="fh-panel"><div class="fh-panel-title">{_esc(title)}</div>'
        f'{rates_bars_html(report["rates"], report["counts"])}'
        + kpis_html([kpi_card("Parity difference", f"{dpd:.3f}", "0 means equal rates"), ratio_card])
        + f"<div>{status}</div>{notes}</div>"
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


def main() -> None:  # pragma: no cover - exercised only inside Streamlit
    import streamlit as st

    st.set_page_config(page_title="FairHire", page_icon="⚖️", layout="wide")
    state = st.session_state
    state.setdefault("results", [])
    state.setdefault("warnings", [])
    state.setdefault("notice", "")
    state.setdefault("synthetic", None)
    state.setdefault("dark_mode", False)

    top_left, top_right = st.columns([6, 1.3])
    with top_right:
        st.toggle("🌙 Dark mode", key="dark_mode")
    dark = bool(state.get("dark_mode"))
    with top_right:
        st.markdown(theme_label_html(dark), unsafe_allow_html=True)
    st.markdown(theme_css(dark) + theme_marker_html(dark), unsafe_allow_html=True)
    try:  # second, independent way to apply the theme (see theme_script_html)
        import streamlit.components.v1 as components

        components.html(theme_script_html(dark), height=0)
    except Exception:  # noqa: BLE001 - purely a progressive enhancement
        pass
    with top_left:
        st.markdown(hero_html(), unsafe_allow_html=True)

    tab_screen, tab_audit, tab_method = st.tabs(["📄 Screening", "📊 Fairness audit", "📘 Methodology"])

    # ------------------------------------------------------------------ Screening tab
    with tab_screen:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown('<div class="fh-card-marker"></div>' + section_html("Job setup", 1), unsafe_allow_html=True)
            use_sample = st.checkbox("Use sample job description and resumes (demo)", value=False)
            jd_text = st.text_area(
                "Job description",
                value=SAMPLE_JOB_DESCRIPTION if use_sample else "",
                height=160,
                placeholder="Paste the job description here...",
            )
            required_selected = st.multiselect(
                "Required skills",
                options=sorted(SKILL_BANK.keys()),
                default=DEFAULT_REQUIRED_SKILLS if use_sample else [],
            )
            custom_skills_text = st.text_input("Extra required skills (comma-separated)", value="")
            n1, n2 = st.columns(2)
            min_years = n1.number_input(
                "Min. years", min_value=0.0, max_value=40.0, value=2.0 if use_sample else 0.0, step=0.5
            )
            shortlist_size = n2.number_input("Shortlist size", min_value=1, max_value=500, value=3, step=1)
            anonymize = st.checkbox("Anonymize resumes before scoring", value=True)
            reveal_files = st.checkbox("Show file names in results", value=False)

            st.markdown(section_html("Upload resumes", 2), unsafe_allow_html=True)
            files = st.file_uploader(
                "TXT, PDF or DOCX (max 5 MB each)",
                type=["txt", "pdf", "docx"],
                accept_multiple_files=True,
            )
            run = st.button("Run screening", type="primary", use_container_width=True)

        if run:
            documents: List[Tuple[str, str]] = []
            problems: List[str] = []
            for uploaded in files or []:
                text, error = read_resume_bytes(uploaded.name, uploaded.getvalue())
                if error:
                    problems.append(error)
                else:
                    documents.append((uploaded.name, text))
            if use_sample:
                documents.extend(SAMPLE_RESUMES.items())

            custom = [s.strip() for s in custom_skills_text.split(",") if s.strip()]
            required = _dedupe_keep_order(
                s for s in (canonical_skill(x) for x in list(required_selected) + custom) if s
            )

            if not documents:
                state["results"] = []
                state["warnings"] = problems
                state["notice"] = "Upload at least one readable resume, or tick the demo option."
            else:
                results, warns = screen_resumes(documents, required, jd_text, float(min_years), anonymize)
                state["results"] = results
                state["warnings"] = problems + warns
                state["notice"] = ""
                state["config"] = {
                    "required": required,
                    "jd_given": bool(jd_text.strip()),
                    "min_years": float(min_years),
                }

        with right:
            st.markdown(section_html("Results"), unsafe_allow_html=True)
            if state["notice"]:
                st.markdown(note_html(state["notice"], "info", "ℹ️"), unsafe_allow_html=True)
            for message in state["warnings"]:
                st.markdown(note_html(message, "warn", "⚠️"), unsafe_allow_html=True)

            results = state["results"]
            if not results:
                st.markdown(
                    empty_state_html(
                        "No results yet",
                        "Set up the job on the left, upload resumes, then press Run screening.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                config = state.get("config", {})
                if not config.get("required") and not config.get("jd_given") and config.get("min_years", 0) <= 0:
                    st.markdown(
                        note_html("Nothing to score against. Add required skills, a job description or minimum years.", "bad", "⛔"),
                        unsafe_allow_html=True,
                    )

                top_n = min(int(shortlist_size), len(results))
                average = sum(float(r["score"]) for r in results) / len(results)
                st.markdown(
                    kpis_html(
                        [
                            kpi_card("Resumes scored", len(results)),
                            kpi_card("Top score", f"{float(results[0]['score']):.1f}", "out of 100", score_tone(float(results[0]["score"]))),
                            kpi_card("Average score", f"{average:.1f}", "across all resumes"),
                            kpi_card("Shortlisted", top_n, "top-ranked candidates"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(results_table_html(results, top_n, bool(reveal_files)), unsafe_allow_html=True)

                export_df = results_to_dataframe(results, include_file_names=bool(reveal_files))
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
                        candidate_card_html(item, bool(reveal_files), open_card=(index == 0))
                        for index, item in enumerate(results)
                    ),
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------------ Audit tab
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
            for column, title, flag in (
                (col_m, "Merit-only ranking (no simulated bias)", "selected_merit_only"),
                (col_b, "Biased process (simulated reviewer penalty)", "selected_biased_process"),
            ):
                report = fairness_report(syn["group"].tolist(), syn[flag].tolist())
                with column:
                    st.markdown(fairness_panel_html(title, report), unsafe_allow_html=True)

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
                    st.markdown(fairness_panel_html("Your outcomes", report), unsafe_allow_html=True)
                except ValueError as exc:
                    st.markdown(note_html(str(exc), "bad", "⛔"), unsafe_allow_html=True)
            elif table is not None:
                st.markdown(note_html("The CSV needs at least two columns.", "bad", "⛔"), unsafe_allow_html=True)

    # ------------------------------------------------------------------ Methodology tab
    with tab_method:
        st.markdown(METHODOLOGY_MD)

    st.markdown(
        '<div class="fh-footer">FairHire is decision support, not a decision maker. A human reviews every outcome.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
