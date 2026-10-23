"""FairHire accounts, roles and per-user storage.

Standard library only, except the optional Postgres driver (psycopg2) which is imported
only when a ``database_url`` is configured. No Streamlit imports, so everything here can
be unit-tested directly.

Storage
-------
* With a Postgres URL (for example a free Supabase or Neon project) data is kept in that
  database and survives restarts and redeploys.
* Without one, a local SQLite file is used. On Streamlit Community Cloud the local disk is
  wiped when the app restarts, so that mode is suitable for demos only.

Security design
---------------
* Passwords are never stored. Only salted PBKDF2-HMAC-SHA256 hashes are kept.
* Sign-in errors are generic, failed attempts lock the account for a while, and password
  checks take similar time whether or not the account exists.
* Every query is parameterised. Every read or write of saved data is scoped to the
  signed-in user's id, which comes from the server-side session, never from the browser.
* Roles are re-read from the database for every admin action. Administrators can manage
  accounts but cannot read recruiters' jobs or reports (only counts are visible).
* Uploaded resumes are never written to the database.
* Email verification and email password reset need an SMTP account. When none is
  configured they are switched off and an administrator resets passwords instead.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import threading
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

DB_PATH_DEFAULT = "fairhire.db"
HASH_ITERATIONS = 310_000
MAX_FAILED_ATTEMPTS = 5
LOCK_SECONDS = 15 * 60
CODE_TTL_SECONDS = 15 * 60
MAX_CODE_ATTEMPTS = 5
MAX_REPORT_BYTES = 1_000_000
MAX_JD_CHARS = 20_000
MAX_SETTINGS_BYTES = 20_000
MAX_REPORTS_PER_USER = 200
MAX_JOBS_PER_USER = 100
ROLES = ("recruiter", "admin")
GENERIC_LOGIN_ERROR = "Incorrect email or password."

_DUMMY_HASH: Optional[str] = None

# {PK} and {REAL} differ between SQLite and Postgres; everything else is shared.
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id {PK},
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'recruiter' CHECK (role IN ('recruiter', 'admin')),
        verified INTEGER NOT NULL DEFAULT 0,
        must_change INTEGER NOT NULL DEFAULT 0,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until {REAL} NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS codes (
        id {PK},
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        purpose TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        expires_at {REAL} NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS jobs (
        id {PK},
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        jd_text TEXT NOT NULL,
        settings TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS reports (
        id {PK},
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('screening', 'audit')),
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        html TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_reports_user ON reports(user_id)",
]


def _now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------------------
# Database layer (SQLite or Postgres behind one small interface)
# --------------------------------------------------------------------------------------


class DbIntegrityError(Exception):
    """A uniqueness or constraint violation (same type for SQLite and Postgres)."""


def _import_psycopg2():
    """Import the Postgres driver on demand (kept separate so tests can substitute a fake)."""
    import psycopg2
    import psycopg2.extras

    return psycopg2, psycopg2.extras


class Database:
    """Minimal wrapper: ``all``, ``one``, ``run`` and ``insert`` with ``?`` placeholders."""

    def __init__(self, kind: str, target: str):
        if kind not in ("sqlite", "postgres"):
            raise ValueError("kind must be 'sqlite' or 'postgres'")
        self.kind = kind
        self.target = target
        self.lock = threading.RLock()
        self._conn: Any = None
        self._driver: Any = None
        self._extras: Any = None
        if kind == "postgres":
            self._driver, self._extras = _import_psycopg2()
        self._connect()
        for statement in _SCHEMA:
            self.run(self._ddl(statement))

    # ---- connection handling ----

    def _ddl(self, statement: str) -> str:
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if self.kind == "sqlite" else "SERIAL PRIMARY KEY"
        real = "REAL" if self.kind == "sqlite" else "DOUBLE PRECISION"
        return statement.replace("{PK}", pk).replace("{REAL}", real)

    def _connect(self) -> None:
        if self.kind == "sqlite":
            conn = sqlite3.connect(self.target, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
        else:
            conn = self._driver.connect(self.target, connect_timeout=10)
            conn.autocommit = True
        self._conn = conn

    def _reconnect(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        self._connect()

    def close(self) -> None:
        with self.lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def _integrity_errors(self) -> tuple:
        if self.kind == "sqlite":
            return (sqlite3.IntegrityError,)
        return (self._driver.IntegrityError,)

    @property
    def _connection_errors(self) -> tuple:
        if self.kind == "sqlite":
            return ()
        return (self._driver.OperationalError, self._driver.InterfaceError)

    def _cursor(self):
        if self.kind == "sqlite":
            return self._conn.cursor()
        return self._conn.cursor(cursor_factory=self._extras.RealDictCursor)

    # ---- statements ----

    def _execute(self, sql: str, params, mode: str):
        text = sql.replace("?", "%s") if self.kind == "postgres" else sql
        if mode == "insert" and self.kind == "postgres":
            text += " RETURNING id"
        with self.lock:
            for attempt in (1, 2):
                try:
                    cursor = self._cursor()
                    try:
                        cursor.execute(text, tuple(params))
                        if mode == "all":
                            result: Any = [dict(row) for row in cursor.fetchall()]
                        elif mode == "one":
                            row = cursor.fetchone()
                            result = dict(row) if row is not None else None
                        elif mode == "insert":
                            result = int(cursor.fetchone()["id"]) if self.kind == "postgres" else int(cursor.lastrowid)
                        else:
                            result = int(cursor.rowcount)
                    finally:
                        cursor.close()
                    if self.kind == "sqlite":
                        self._conn.commit()
                    return result
                except self._integrity_errors as exc:
                    if self.kind == "sqlite":
                        self._conn.rollback()
                    raise DbIntegrityError(str(exc)) from exc
                except self._connection_errors:
                    if attempt == 2:
                        raise
                    self._reconnect()

    def all(self, sql: str, params=()) -> List[Dict[str, Any]]:
        return self._execute(sql, params, "all")

    def one(self, sql: str, params=()) -> Optional[Dict[str, Any]]:
        return self._execute(sql, params, "one")

    def run(self, sql: str, params=()) -> int:
        return self._execute(sql, params, "run")

    def insert(self, sql: str, params=()) -> int:
        return self._execute(sql, params, "insert")


_DATABASES: Dict[Tuple[str, str], Database] = {}
_FACTORY_LOCK = threading.Lock()


def get_database(url: Optional[str] = None, path: Optional[str] = None) -> Database:
    """Return the shared database: Postgres when ``url`` is given, else a SQLite file."""
    if url:
        kind, target = "postgres", url.strip()
    else:
        kind, target = "sqlite", path or os.environ.get("FAIRHIRE_DB") or DB_PATH_DEFAULT
    with _FACTORY_LOCK:
        if target != ":memory:" and (kind, target) in _DATABASES:
            return _DATABASES[(kind, target)]
        database = Database(kind, target)
        if target != ":memory:":
            _DATABASES[(kind, target)] = database
        return database


def open_db(path: Optional[str] = None) -> Database:
    """Backward-compatible helper for a SQLite database."""
    return get_database(None, path)


# --------------------------------------------------------------------------------------
# Validation and password hashing
# --------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
_COMMON_PASSWORDS = frozenset(
    "password password123 passw0rd123 1234567890 12345678910 qwertyuiop letmein123 welcome123 iloveyou123 "
    "admin12345 changeme123 abcdefghij 0123456789 1q2w3e4r5t fairhire123".split()
)


def normalize_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> Optional[str]:
    if not email or len(email) > 254 or not _EMAIL_RE.match(email):
        return "Enter a valid email address."
    return None


def validate_password(password: str, email: str = "") -> Optional[str]:
    """Return a message describing the problem, or ``None`` if the password is acceptable."""
    password = password or ""
    if len(password) < 10:
        return "Use at least 10 characters."
    if len(password) > 128:
        return "Use at most 128 characters."
    if password.isdigit():
        return "A password cannot be only numbers."
    if password.lower() in _COMMON_PASSWORDS:
        return "That password is too common."
    local = normalize_email(email).split("@")[0]
    if len(local) >= 4 and local in password.lower():
        return "The password should not contain your email name."
    kinds = sum(bool(re.search(pattern, password)) for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if kinds < 2:
        return "Mix at least two kinds of characters (lower case, upper case, numbers, symbols)."
    return None


def hash_password(password: str, iterations: Optional[int] = None, salt: Optional[bytes] = None) -> str:
    iterations = iterations or HASH_ITERATIONS
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    return _DUMMY_HASH


# --------------------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------------------


def _user_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "role": row["role"],
        "verified": bool(row["verified"]),
        "must_change": bool(row["must_change"]),
        "created_at": row["created_at"],
    }


def count_users(db: Database) -> int:
    return int(db.one("SELECT COUNT(*) AS n FROM users")["n"])


def get_user(db: Database, user_id: int) -> Optional[Dict[str, Any]]:
    row = db.one("SELECT * FROM users WHERE id = ?", (int(user_id),))
    return _user_dict(row) if row else None


def get_user_by_email(db: Database, email: str) -> Optional[Dict[str, Any]]:
    row = db.one("SELECT * FROM users WHERE email = ?", (normalize_email(email),))
    return _user_dict(row) if row else None


def create_user(
    db: Database,
    email: str,
    password: str,
    role: str = "recruiter",
    verified: bool = True,
    admin_email: Optional[str] = None,
    must_change: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Create an account. Returns ``(user, error)``.

    The first account becomes the administrator, unless ``admin_email`` is given, in
    which case only that address becomes the administrator.
    """
    email = normalize_email(email)
    error = validate_email(email) or validate_password(password, email)
    if error:
        return None, error
    if role not in ROLES:
        return None, "Unknown role."
    with db.lock:
        admin_email = normalize_email(admin_email)
        first_admin = (email == admin_email) if admin_email else (count_users(db) == 0)
        if first_admin:
            role = "admin"
        try:
            user_id = db.insert(
                "INSERT INTO users (email, password_hash, role, verified, must_change, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (email, hash_password(password), role, 1 if verified else 0, 1 if must_change else 0, _now_text()),
            )
        except DbIntegrityError:
            return None, "An account with this email already exists."
    return get_user(db, user_id), None


def authenticate(
    db: Database, email: str, password: str, now: Optional[float] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Check credentials. Returns ``(user, error)`` with deliberately generic errors."""
    now = time.time() if now is None else now
    email = normalize_email(email)
    row = db.one("SELECT * FROM users WHERE email = ?", (email,))
    if row is None:
        verify_password(password or "", _dummy_hash())  # keep timing similar
        return None, GENERIC_LOGIN_ERROR
    if float(row["locked_until"]) > now:
        minutes = max(1, int((float(row["locked_until"]) - now) // 60) + 1)
        return None, f"Too many failed attempts. Try again in about {minutes} minute(s)."
    if not verify_password(password or "", row["password_hash"]):
        attempts = int(row["failed_attempts"]) + 1
        if attempts >= MAX_FAILED_ATTEMPTS:
            db.run("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?", (now + LOCK_SECONDS, row["id"]))
        else:
            db.run("UPDATE users SET failed_attempts = ? WHERE id = ?", (attempts, row["id"]))
        return None, GENERIC_LOGIN_ERROR
    if not row["verified"]:
        return None, "Please verify your email address first."
    db.run("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE id = ?", (row["id"],))
    return _user_dict(row), None


def set_password(db: Database, user_id: int, new_password: str, must_change: bool = False) -> Optional[str]:
    user = get_user(db, user_id)
    if user is None:
        return "Account not found."
    error = validate_password(new_password, user["email"])
    if error:
        return error
    with db.lock:
        db.run(
            "UPDATE users SET password_hash = ?, must_change = ?, failed_attempts = 0, locked_until = 0 WHERE id = ?",
            (hash_password(new_password), 1 if must_change else 0, int(user_id)),
        )
        db.run("DELETE FROM codes WHERE user_id = ?", (int(user_id),))
    return None


def change_password(db: Database, user_id: int, current_password: str, new_password: str) -> Optional[str]:
    row = db.one("SELECT password_hash FROM users WHERE id = ?", (int(user_id),))
    if row is None or not verify_password(current_password or "", row["password_hash"]):
        return "The current password is not correct."
    if hmac.compare_digest((current_password or "").encode(), (new_password or "").encode()):
        return "Choose a new password that is different from the current one."
    return set_password(db, user_id, new_password, must_change=False)


def set_verified(db: Database, user_id: int) -> None:
    db.run("UPDATE users SET verified = 1 WHERE id = ?", (int(user_id),))


# --------------------------------------------------------------------------------------
# One-time codes (email verification and password reset)
# --------------------------------------------------------------------------------------


def _code_hash(user_id: int, purpose: str, code: str) -> str:
    return hashlib.sha256(f"{int(user_id)}:{purpose}:{code}".encode("utf-8")).hexdigest()


def issue_code(db: Database, user_id: int, purpose: str, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    code = f"{secrets.randbelow(10 ** 6):06d}"
    with db.lock:
        db.run("DELETE FROM codes WHERE user_id = ? AND purpose = ?", (int(user_id), purpose))
        db.run(
            "INSERT INTO codes (user_id, purpose, code_hash, expires_at) VALUES (?, ?, ?, ?)",
            (int(user_id), purpose, _code_hash(user_id, purpose, code), now + CODE_TTL_SECONDS),
        )
    return code


def check_code(db: Database, user_id: int, purpose: str, code: str, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    with db.lock:
        row = db.one("SELECT * FROM codes WHERE user_id = ? AND purpose = ?", (int(user_id), purpose))
        if row is None:
            return False
        if float(row["expires_at"]) < now or int(row["attempts"]) >= MAX_CODE_ATTEMPTS:
            db.run("DELETE FROM codes WHERE id = ?", (row["id"],))
            return False
        db.run("UPDATE codes SET attempts = attempts + 1 WHERE id = ?", (row["id"],))
        ok = hmac.compare_digest(_code_hash(user_id, purpose, (code or "").strip()), row["code_hash"])
        if ok:
            db.run("DELETE FROM codes WHERE id = ?", (row["id"],))
        return ok


# --------------------------------------------------------------------------------------
# Optional email (needs an SMTP account in Streamlit secrets)
# --------------------------------------------------------------------------------------

_SMTP_KEYS = ("host", "port", "username", "password", "sender")


def smtp_configured(conf: Optional[Dict[str, Any]]) -> bool:
    return bool(conf) and all(conf.get(key) for key in _SMTP_KEYS)


def send_email(conf: Optional[Dict[str, Any]], to: str, subject: str, body: str) -> Tuple[bool, Optional[str]]:
    if not smtp_configured(conf):
        return False, "Email is not configured."
    message = EmailMessage()
    message["From"] = str(conf["sender"])
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    try:
        port = int(conf["port"])
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(str(conf["host"]), port, timeout=15, context=context) as server:
                server.login(str(conf["username"]), str(conf["password"]))
                server.send_message(message)
        else:
            with smtplib.SMTP(str(conf["host"]), port, timeout=15) as server:
                server.starttls(context=context)
                server.login(str(conf["username"]), str(conf["password"]))
                server.send_message(message)
    except Exception as exc:  # noqa: BLE001 - report any delivery failure without leaking details
        return False, f"The email could not be sent ({type(exc).__name__})."
    return True, None


# --------------------------------------------------------------------------------------
# Administrator actions (the role is re-checked in the database every time)
# --------------------------------------------------------------------------------------


def _require_admin(db: Database, actor_id: int) -> Dict[str, Any]:
    actor = get_user(db, actor_id)
    if actor is None or actor["role"] != "admin":
        raise PermissionError("Administrator access is required.")
    return actor


def list_users(db: Database, actor_id: int) -> List[Dict[str, Any]]:
    """Accounts with counts only. Administrators cannot see anyone's jobs or reports."""
    _require_admin(db, actor_id)
    rows = db.all(
        "SELECT u.*, (SELECT COUNT(*) FROM jobs j WHERE j.user_id = u.id) AS job_count, "
        "(SELECT COUNT(*) FROM reports r WHERE r.user_id = u.id) AS report_count FROM users u ORDER BY u.id"
    )
    out = []
    for row in rows:
        item = _user_dict(row)
        item["jobs"] = int(row["job_count"])
        item["reports"] = int(row["report_count"])
        out.append(item)
    return out


def set_role(db: Database, actor_id: int, target_id: int, role: str) -> Optional[str]:
    _require_admin(db, actor_id)
    if role not in ROLES:
        return "Unknown role."
    if int(actor_id) == int(target_id):
        return "You cannot change your own role."
    if get_user(db, target_id) is None:
        return "Account not found."
    db.run("UPDATE users SET role = ? WHERE id = ?", (role, int(target_id)))
    return None


def delete_user(db: Database, actor_id: int, target_id: int) -> Optional[str]:
    _require_admin(db, actor_id)
    if int(actor_id) == int(target_id):
        return "You cannot delete your own account here."
    if get_user(db, target_id) is None:
        return "Account not found."
    db.run("DELETE FROM users WHERE id = ?", (int(target_id),))
    return None


def admin_reset_password(db: Database, actor_id: int, target_id: int) -> Tuple[Optional[str], Optional[str]]:
    """Set a temporary password the user must change at next sign-in. Returns ``(password, error)``."""
    _require_admin(db, actor_id)
    if get_user(db, target_id) is None:
        return None, "Account not found."
    temporary = secrets.token_urlsafe(12)
    db.run(
        "UPDATE users SET password_hash = ?, must_change = 1, failed_attempts = 0, locked_until = 0 WHERE id = ?",
        (hash_password(temporary), int(target_id)),
    )
    return temporary, None


# --------------------------------------------------------------------------------------
# Per-user saved data (always filtered by user_id)
# --------------------------------------------------------------------------------------


def save_job(db: Database, user_id: int, title: str, jd_text: str, settings: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    title = (title or "").strip()
    if not title or len(title) > 120:
        return None, "Give the job a title of up to 120 characters."
    if len(jd_text or "") > MAX_JD_CHARS:
        return None, "The job description is too long to save."
    settings_text = json.dumps(settings or {})
    if len(settings_text) > MAX_SETTINGS_BYTES:
        return None, "The job settings are too large to save."
    with db.lock:
        count = int(db.one("SELECT COUNT(*) AS n FROM jobs WHERE user_id = ?", (int(user_id),))["n"])
        if count >= MAX_JOBS_PER_USER:
            return None, f"You can save up to {MAX_JOBS_PER_USER} jobs. Delete one first."
        job_id = db.insert(
            "INSERT INTO jobs (user_id, title, jd_text, settings, created_at) VALUES (?, ?, ?, ?, ?)",
            (int(user_id), title, jd_text or "", settings_text, _now_text()),
        )
    return job_id, None


def list_jobs(db: Database, user_id: int) -> List[Dict[str, Any]]:
    rows = db.all("SELECT id, title, created_at FROM jobs WHERE user_id = ? ORDER BY id DESC", (int(user_id),))
    return [{"id": int(r["id"]), "title": r["title"], "created_at": r["created_at"]} for r in rows]


def get_job(db: Database, user_id: int, job_id: int) -> Optional[Dict[str, Any]]:
    row = db.one("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (int(job_id), int(user_id)))
    if row is None:
        return None
    try:
        settings = json.loads(row["settings"])
    except ValueError:
        settings = {}
    return {"id": int(row["id"]), "title": row["title"], "jd_text": row["jd_text"], "settings": settings, "created_at": row["created_at"]}


def delete_job(db: Database, user_id: int, job_id: int) -> bool:
    return db.run("DELETE FROM jobs WHERE id = ? AND user_id = ?", (int(job_id), int(user_id))) > 0


def save_report(
    db: Database, user_id: int, title: str, kind: str, html: str, summary: Dict[str, Any]
) -> Tuple[Optional[int], Optional[str]]:
    title = (title or "").strip()
    if kind not in ("screening", "audit"):
        return None, "Unknown report type."
    if not title or len(title) > 120:
        return None, "Give the report a title of up to 120 characters."
    if len((html or "").encode("utf-8")) > MAX_REPORT_BYTES:
        return None, "The report is too large to save."
    with db.lock:
        count = int(db.one("SELECT COUNT(*) AS n FROM reports WHERE user_id = ?", (int(user_id),))["n"])
        if count >= MAX_REPORTS_PER_USER:
            return None, f"You can save up to {MAX_REPORTS_PER_USER} reports. Delete one first."
        report_id = db.insert(
            "INSERT INTO reports (user_id, kind, title, summary, html, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (int(user_id), kind, title, json.dumps(summary or {}), html or "", _now_text()),
        )
    return report_id, None


def list_reports(db: Database, user_id: int, kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    query = "SELECT id, kind, title, summary, created_at FROM reports WHERE user_id = ?"
    params: List[Any] = [int(user_id)]
    if kind:
        query += " AND kind = ?"
        params.append(kind)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, int(limit)))
    out = []
    for row in db.all(query, params):
        try:
            summary = json.loads(row["summary"])
        except ValueError:
            summary = {}
        out.append({"id": int(row["id"]), "kind": row["kind"], "title": row["title"], "summary": summary, "created_at": row["created_at"]})
    return out


def get_report_html(db: Database, user_id: int, report_id: int) -> Optional[str]:
    row = db.one("SELECT html FROM reports WHERE id = ? AND user_id = ?", (int(report_id), int(user_id)))
    return row["html"] if row else None


def delete_report(db: Database, user_id: int, report_id: int) -> bool:
    return db.run("DELETE FROM reports WHERE id = ? AND user_id = ?", (int(report_id), int(user_id))) > 0
