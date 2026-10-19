# ⚖️ FairHire

Explainable AI resume screening with a built-in fairness audit, built with Streamlit.

> **Decision support only.** FairHire helps a human reviewer see *why* a resume scored the way it did.
> It must not be used to automatically reject candidates.

## Features

- Upload resumes as **TXT, PDF or DOCX** (max 5 MB each) with clear error messages for unreadable files
- Job description + required-skill configuration, minimum years of experience, shortlist size
- Deterministic score from **0 to 100** with matched skills, missing skills and written reasons
- Best-effort **anonymization** (names, emails, phones, links, address/DOB/personal-detail lines, gendered words)
- **Fairness audit**: selection rates, demographic parity difference, disparate impact ratio (four-fifths rule of thumb)
- **Synthetic data** simulation, plus optional audit of your own outcomes CSV
- Ranking table and **CSV export**
- Methodology and limitations tab

No API keys, paid services, Docker or environment variables are needed.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Complete application (main entry point) |
| `requirements.txt` | Dependencies for Streamlit Community Cloud |
| `test_fairhire.py` | Automated tests for skills, scoring, anonymization and fairness metrics |
| `README.md` | This file |

## Replace the damaged files using the GitHub website (no Python install needed)

1. Open your repository on **github.com** and sign in.
2. Click `app.py`, then the **pencil icon** (Edit this file).
3. Press **Ctrl+A** (Cmd+A on Mac) to select everything, press **Delete**, then paste the full new `app.py`.
   Make sure the file starts with `"""FairHire - Explainable AI ...` on line 1 with **no spaces before it**.
4. Click **Commit changes…** → **Commit changes**.
5. Repeat for `requirements.txt` and `README.md`. For a file that does not exist yet (for example `test_fairhire.py`),
   use **Add file → Create new file**, type the exact file name, paste the contents and commit.
6. Make sure there is only one app file. If you have old duplicates (for example `app (1).py` or stray fragments),
   open them and use the **trash icon → Commit** to delete them.

## Redeploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Find your app in the list. Click the **⋮** menu → **Reboot app** (this re-reads your code and reinstalls
   `requirements.txt`).
3. If the app does not exist or stays broken, click **Create app** → choose your repo and branch → set
   **Main file path** to `app.py` → **Deploy**.
4. Watch **Manage app** (bottom right) for logs. The first start can take a few minutes while packages install.

## Run the tests without installing Python (optional)

Create the file `.github/workflows/tests.yml` in your repo (Add file → Create new file) with:

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
      - run: python -m py_compile app.py test_fairhire.py
      - run: python -m unittest -v test_fairhire
```

Then open the repo's **Actions** tab to see a green check or the failure details after each commit.

If you do have Python locally: `pip install -r requirements.txt`, `python -m unittest -v test_fairhire`,
then `streamlit run app.py`.

## How scoring works

| Component | Weight | Computation |
|---|---|---|
| Required skills | 60 | share of required skills found in the resume |
| Experience | 25 | detected years ÷ minimum years, capped at 100% |
| Job-description keywords | 15 | share of the job description's top meaningful words found in the resume |

Unconfigured components are dropped and the remaining weights rescaled. Scores are always clamped to 0-100.
Empty resumes score 0, duplicate uploads are skipped with a warning, and if there is nothing to score against
(no skills, job description or minimum years) the app says so instead of inventing a score.

## Responsible use and limitations

- **Masking identity does not make hiring unbiased.** Skills lists, schools, employers, career gaps and writing style
  can act as proxies for protected characteristics. Anonymization here is best effort and will miss some details.
- Protected characteristics are **never** used to score or rank individuals. Group labels are used only to audit outcomes.
- The fairness dashboard uses **simulated** synthetic candidates and a simulated reviewer bias. These results show how
  the metrics behave; they are **not evidence** about real hiring decisions.
- The four-fifths rule is a screening heuristic, not a legal test. Small groups give unreliable rates.
- Keyword matching misses synonyms and context; experience extraction is heuristic and may misread dates or PDF layouts.
- Before any real use: obtain candidate consent, follow local employment and data-protection law, keep a human
  in the loop, and audit real outcomes with proper statistical testing.
