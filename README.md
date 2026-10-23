# ⚖️ FairHire

Explainable AI resume screening with a built-in fairness audit, built with Streamlit.

> **Decision support only.** FairHire shows *why* a resume scored the way it did so a human reviewer can decide.
> It must not be used to reject candidates automatically.

## Features

**Accounts (your choice of two modes on the sign-in screen)**
- **Real account**: sign up and sign in with email and password (stored only as salted hashes), lockout after repeated failures,
  optional email verification and reset codes, recruiter and administrator roles
- **Prototype (simulated)**: the same screens with nothing stored; any valid-looking email works. Good for demos
- Personal dashboard: saved jobs, saved reports and previous fairness audits
- Administrators manage accounts but **cannot read anyone's saved data**
- **Lasting storage** with a free hosted Postgres database (see below). Resumes are never saved in either mode

**Screening**
- Modern interface with a light / dark toggle and a calm welcome screen
- **Real resumes**: TXT, PDF and DOCX (up to 25 MB each). **One PDF can contain many resumes**; each is found and scored separately
- Job description, required skills, **your own custom skills and aliases**, minimum years
- Deterministic 0-100 score, **adjustable weights with live re-ranking**, matched and missing skills, written reasons
- **Skill evidence**: the exact sentence in the resume behind every matched skill and the experience figure, with matches highlighted
- Best-effort anonymization, **proxy-risk flags**, ranking table, score chart, CSV export, one-click demo

**Fairness and accountability**
- **Job description check**: flags biased wording (gender-coded, age, nationality, family status, appearance, health, prestige
  schools), shows a gender-coding indicator and offers a safe automatic rewrite
- **Bias stress test**: re-scores each resume with names and gendered words changed
- Fairness audit: selection rates with 95% intervals, parity difference, disparate impact ratio, **significance test**
- Synthetic simulation (clearly labelled) and an optional audit of your own outcomes CSV
- **Reviewer decision log** (Hold and Reject need a job-related reason) and a **downloadable audit report**

## Project files

| File | Purpose |
|---|---|
| `app.py` | The application (entry point) |
| `auth.py` | Accounts, roles, password hashing, saved data, database layer (SQLite or Postgres), optional email |
| `requirements.txt` | Dependencies for Streamlit Community Cloud |
| `test_fairhire.py` | Automated tests |
| `README.md` | This file |

## Replace the files on GitHub (no Python install needed)

1. Open your repository on **github.com**.
2. For each file (`app.py`, `requirements.txt`, `README.md`, `test_fairhire.py`): open it, click the **pencil icon**, select all
   (Ctrl+A / Cmd+A), delete, paste the new contents, then **Commit changes**.
   `app.py` must start with `"""FairHire` on line 1 with no spaces before it.
3. Add the file **`auth.py`**: **Add file → Create new file**, name it exactly `auth.py` (replace it if you already have one),
   paste the contents, commit. **`app.py` imports it, so the app crashes without it.**
4. Add a file named `.gitignore` containing these lines so a local database file is never uploaded:
   ```
   fairhire.db
   __pycache__/
   ```
5. Delete any old duplicates or fragments (open the file, trash icon, commit).
6. On **share.streamlit.io**: your app's **⋮ → Reboot app**, then hard-refresh the browser (Ctrl+Shift+R / Cmd+Shift+R).
   `requirements.txt` changed (a database driver was added), so the first start takes a little longer.

## Lasting accounts and saved data (free Postgres database)

Without a database the app keeps accounts in a local file. **Streamlit Community Cloud wipes local files when the app restarts or
redeploys**, so accounts and saved reports would disappear. A hosted Postgres database fixes that.

1. Create a free Postgres project with a provider such as **Supabase** or **Neon**.
2. Copy its **connection string** (a URL starting with `postgresql://`). On Supabase, use the **session pooler** connection string
   rather than the direct one (the direct address is often IPv6-only and may not be reachable from Streamlit Cloud). Replace the
   `[YOUR-PASSWORD]` placeholder with your database password; if the password contains symbols like `@` or `/`, URL-encode them.
   Provider screens change over time, so follow their own "connect" documentation if the labels differ.
3. In Streamlit Community Cloud open **your app → Settings → Secrets** and add (never put this in GitHub):
   ```toml
   database_url = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
   admin_email = "you@example.com"
   ```
4. Reboot the app. The tables are created automatically on first start. The sign-in screen's warning about temporary storage
   disappears, and the dashboard says data is "kept in the database".

Notes: **The first account created becomes the administrator** unless you set `admin_email`, in which case only that address
becomes the administrator (set it before you sign up). Free database projects can be paused by the provider after a period of
inactivity; check your provider's terms. If the database cannot be opened the sign-in screen says so and the prototype still works.

**Honesty note on testing:** the SQLite path is tested directly. The Postgres path is tested against a simulated driver that
checks the SQL and connection logic, but it has **not** been run against a real Postgres server, so please try it once and report
any error message you see (the message never includes your connection string).

### Optional: email codes

Add an `[smtp]` table to the same Secrets box and sign-up will ask for an emailed 6-digit code and "Forgot your password?" will email
a reset code. Without it, new accounts work immediately and an administrator resets forgotten passwords from the **🛡️ Admin** tab.

```toml
[smtp]
host = "smtp.example.com"
port = 465
username = "your-smtp-username"
password = "your-smtp-password-or-app-password"
sender = "FairHire <you@example.com>"
```

## Using real resumes

1. Open **Screening**, enter the job description and required skills (or press **One-click demo** to try it first).
2. Upload your files. **A single PDF can contain many resumes.**
3. Press **Run screening**. Open **How your files were read** to confirm how many resumes were found and which pages belong to each.
4. If a PDF was split in the wrong places, change **If a PDF contains several resumes** to *One resume per page* or
   *Treat each PDF as one resume* and run again.

How splitting works: FairHire looks at the email and phone number near the top of each page. A page with different contact details
starts a new resume; pages with the same details, or no contact details and no name at the top, continue the current one. A page
with no contact details that opens with a name and a section heading (Experience, Skills...) also starts a new resume.

Good to know:
- **Scanned (image-only) PDFs cannot be read** because they have no text layer. Use a text-based PDF or run OCR first.
- Two-column layouts can scramble text order; check the highlighted text in a candidate's details.
- Resumes are read in memory only and **never saved**. Candidates appear as "Candidate 001"; tick **Show file names** to see the source
  and page range. The original text next to the anonymized text stays hidden unless you tick **Show the original text** (it reveals names).
- Up to 300 resumes per run and 400 pages per PDF. Full details show for the top 25 candidates.
- Anonymization is best effort. It can miss nicknames and unusual spellings, and may over-mask a surname that is also a common word.
- Skill evidence is shown on screen only and is not copied into saved reports.

## Run the tests without installing Python (optional)

Create `.github/workflows/tests.yml` in your repo (**Add file → Create new file**) with:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python -m py_compile app.py auth.py test_fairhire.py
      - run: python -m unittest -v test_fairhire
```

Then check the repo's **Actions** tab. With Python locally: `pip install -r requirements.txt`,
`python -m unittest -v test_fairhire`, `streamlit run app.py`.

## How scoring works

| Component | Default weight | Computation |
|---|---|---|
| Required skills | 60 | share of required skills found in the resume |
| Experience | 25 | detected years ÷ minimum years, capped at 100% |
| Job-description keywords | 15 | share of the job description's top meaningful words found |

Weights are adjustable and rescaled to 100%. Components you have not configured are skipped. Scores are clamped to 0-100.
Empty resumes score 0, duplicate uploads are skipped with a warning, and if there is nothing to score against the app says so.

## Responsible use and limitations

- **Masking identity does not make hiring unbiased.** Skills, schools, employers, career gaps and writing style can act as proxies
  for protected characteristics. Anonymization and proxy flags are best-effort heuristics that miss things.
- The **job description check** is rule-based and not exhaustive. A flag is a prompt to reconsider wording, not proof of
  discrimination, and a clean result does not guarantee an unbiased posting. Gender-coding counts are a rough indicator.
- Protected characteristics are **never** used to score or rank individuals. Group labels are used only to audit outcomes.
- The **bias stress test** checks only that the scoring step ignores names and gendered words. It does not prove the process is fair.
- Fairness figures from the synthetic simulation are **not evidence** about real hiring. The four-fifths rule is a screening
  heuristic, not a legal test. With small groups the significance test has little power.
- Keyword matching misses synonyms and context; experience extraction is heuristic and may misread dates or PDF layouts.
- Before real use: get candidate consent, follow local employment and data-protection law (storing account data may carry its own
  obligations), keep a human in the loop, and audit real outcomes with appropriate statistical testing.
