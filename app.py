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
import io
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
# Streamlit interface
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


def main() -> None:  # pragma: no cover - exercised only inside Streamlit
    import streamlit as st

    st.set_page_config(page_title="FairHire", page_icon="⚖️", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; max-width: 1400px;}
        .fh-hero {padding: 1rem 1.25rem; border-radius: 12px;
                  background: linear-gradient(90deg, #1f3a5f, #3b6ea8); color: #fff; margin-bottom: 1rem;}
        .fh-hero h1 {margin: 0; font-size: 1.8rem;}
        .fh-hero p {margin: 0.2rem 0 0 0; opacity: 0.9;}
        </style>
        <div class="fh-hero">
          <h1>⚖️ FairHire</h1>
          <p>Explainable resume screening with a built-in fairness audit. Decision support only - a human decides.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    state = st.session_state
    state.setdefault("results", [])
    state.setdefault("warnings", [])
    state.setdefault("synthetic", None)

    tab_screen, tab_audit, tab_method = st.tabs(["📄 Screening", "📊 Fairness audit", "📘 Methodology"])

    # ------------------------------------------------------------------ Screening tab
    with tab_screen:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("1. Job setup")
            use_sample = st.checkbox("Use sample job description and resumes (demo)", value=False)
            jd_text = st.text_area(
                "Job description",
                value=SAMPLE_JOB_DESCRIPTION if use_sample else "",
                height=170,
                placeholder="Paste the job description here...",
            )
            skill_options = sorted(SKILL_BANK.keys())
            required_selected = st.multiselect(
                "Required skills",
                options=skill_options,
                default=DEFAULT_REQUIRED_SKILLS if use_sample else [],
            )
            custom_skills_text = st.text_input("Extra required skills (comma-separated)", value="")
            min_years = st.number_input("Minimum years of experience", min_value=0.0, max_value=40.0, value=2.0 if use_sample else 0.0, step=0.5)
            shortlist_size = st.number_input("Shortlist size", min_value=1, max_value=500, value=3, step=1)
            anonymize = st.checkbox("Anonymize resumes before scoring", value=True)
            reveal_files = st.checkbox("Show file names in results and export", value=False)

            st.subheader("2. Upload resumes")
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
            required = _dedupe_keep_order(canonical_skill(s) for s in list(required_selected) + custom)

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
                    "anonymize": anonymize,
                    "shortlist": int(shortlist_size),
                    "reveal": reveal_files,
                    "jd_given": bool(jd_text.strip()),
                    "min_years": float(min_years),
                }

        with right:
            st.subheader("Results")
            if state.get("notice"):
                st.info(state["notice"])
            for message in state["warnings"]:
                st.warning(message)

            results = state["results"]
            if not results:
                st.caption("Configure the job on the left, upload resumes, then press **Run screening**.")
            else:
                config = state.get("config", {})
                show_files = bool(reveal_files)
                if not config.get("required") and not config.get("jd_given") and config.get("min_years", 0) <= 0:
                    st.error("Nothing to score against. Add required skills, a job description or minimum years.")

                top_n = min(int(shortlist_size), len(results))
                m1, m2, m3 = st.columns(3)
                m1.metric("Resumes scored", len(results))
                m2.metric("Top score", f"{results[0]['score']:.1f}")
                m3.metric("Average score", f"{sum(r['score'] for r in results) / len(results):.1f}")

                df = results_to_dataframe(results, include_file_names=show_files)
                df.insert(2, "Shortlist", ["Yes" if r["rank"] <= top_n else "" for r in results])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download ranking (CSV)",
                    data=dataframe_to_csv_bytes(df),
                    file_name="fairhire_ranking.csv",
                    mime="text/csv",
                )

                st.markdown("#### Why each candidate scored as they did")
                for item in results:
                    label = f"#{item['rank']} {item['candidate_id']} - {item['score']:.1f}/100"
                    if show_files:
                        label += f" ({item['file_name']})"
                    with st.expander(label):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Matched skills**")
                            st.write(", ".join(item["matched_skills"]) or "None")
                            st.markdown("**Missing skills**")
                            st.write(", ".join(item["missing_skills"]) or "None")
                        with c2:
                            st.markdown("**Other skills detected**")
                            st.write(", ".join(item["extra_skills"]) or "None")
                            st.markdown("**Detected experience**")
                            st.write(f"{item['years_experience']} years")
                        st.progress(int(round(item["score"])))
                        for reason in item["reasons"]:
                            st.write("• " + reason)
                        with st.popover("View text used for scoring"):
                            st.text(item["text_used"][:3000])

    # ------------------------------------------------------------------ Audit tab
    with tab_audit:
        st.subheader("Fairness audit")
        st.warning(
            "SIMULATION: the data below is synthetic. It demonstrates how the metrics behave; "
            "it is **not** evidence about real candidates or about your hiring process."
        )
        a1, a2, a3, a4 = st.columns(4)
        n_syn = a1.slider("Synthetic candidates", 50, 1000, 300, step=50)
        bias = a2.slider("Simulated reviewer bias against Group B (points)", 0, 40, 15)
        frac = a3.slider("Shortlist fraction", 0.05, 0.6, 0.25, step=0.05)
        seed = a4.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

        if st.button("Run simulation", type="primary") or state["synthetic"] is None:
            state["synthetic"] = generate_synthetic_candidates(
                n=n_syn, seed=int(seed), bias_penalty=float(bias), shortlist_fraction=float(frac)
            )

        syn = state["synthetic"]
        if syn is not None and len(syn):
            col_m, col_b = st.columns(2)
            for column, title, flag in (
                (col_m, "Merit-only ranking (no simulated bias)", "selected_merit_only"),
                (col_b, "Biased process (simulated reviewer penalty)", "selected_biased_process"),
            ):
                report = fairness_report(syn["group"].tolist(), syn[flag].tolist())
                with column:
                    st.markdown(f"**{title}**")
                    rates = pd.Series(report["rates"], name="Selection rate").sort_index()
                    st.bar_chart(rates)
                    st.metric("Demographic parity difference", f"{report['demographic_parity_difference']:.3f}")
                    ratio = report["disparate_impact_ratio"]
                    st.metric("Disparate impact ratio", "undefined" if ratio is None else f"{ratio:.3f}")
                    if report["four_fifths_pass"] is True:
                        st.success("Ratio is at or above 0.80 (passes the four-fifths rule of thumb).")
                    elif report["four_fifths_pass"] is False:
                        st.error("Ratio is below 0.80 - flag for review (screening heuristic only).")
                    for note in report["notes"]:
                        st.caption(note)
            st.dataframe(syn, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download synthetic data (CSV)",
                data=dataframe_to_csv_bytes(syn),
                file_name="fairhire_synthetic.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.markdown("#### Audit your own outcomes table (optional)")
        st.caption("Upload a CSV with one row per person, a group column and a selected/not-selected column.")
        outcomes = st.file_uploader("Outcomes CSV", type=["csv"], key="outcomes_csv")
        if outcomes is not None:
            try:
                table = pd.read_csv(io.BytesIO(outcomes.getvalue()))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the CSV ({type(exc).__name__}).")
                table = None
            if table is not None and len(table.columns) >= 2:
                g_col = st.selectbox("Group column", table.columns, index=0)
                s_col = st.selectbox("Selected column", table.columns, index=1)
                try:
                    report = audit_uploaded_outcomes(table, g_col, s_col)
                    st.write("Selection rates", pd.Series(report["rates"], name="rate"))
                    ratio = report["disparate_impact_ratio"]
                    st.metric("Demographic parity difference", f"{report['demographic_parity_difference']:.3f}")
                    st.metric("Disparate impact ratio", "undefined" if ratio is None else f"{ratio:.3f}")
                    for note in report["notes"]:
                        st.caption(note)
                except ValueError as exc:
                    st.error(str(exc))
            elif table is not None:
                st.error("The CSV needs at least two columns.")

    # ------------------------------------------------------------------ Methodology tab
    with tab_method:
        st.markdown(METHODOLOGY_MD)


if __name__ == "__main__":
    main()
