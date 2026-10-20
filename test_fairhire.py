"""Automated tests for FairHire helper logic.

Run with either:
    python -m unittest -v test_fairhire
    pytest -v test_fairhire.py
"""

import io
import sys
import unittest

import app  # must NOT launch the Streamlit interface on import


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_load_streamlit(self):
        self.assertNotIn("streamlit", sys.modules, "importing app must not start/import Streamlit")

    def test_main_is_defined_but_not_called(self):
        self.assertTrue(callable(app.main))


class SkillExtractionTests(unittest.TestCase):
    def test_basic_skills(self):
        found = app.extract_skills("Experienced with Python, SQL and Machine Learning.")
        self.assertEqual(found, ["machine learning", "python", "sql"])

    def test_special_character_skills(self):
        found = app.extract_skills("I write C++ and C# and Node.js.")
        self.assertIn("c++", found)
        self.assertIn("c#", found)
        self.assertIn("node.js", found)

    def test_no_partial_word_matches(self):
        found = app.extract_skills("I know JavaScript and MySQL.")
        self.assertNotIn("java", found)
        self.assertNotIn("sql", found)
        self.assertIn("javascript", found)
        self.assertIn("mysql", found)

    def test_aliases_and_case(self):
        self.assertIn("scikit-learn", app.extract_skills("used SKLEARN daily"))

    def test_empty_inputs(self):
        self.assertEqual(app.extract_skills(""), [])
        self.assertEqual(app.extract_skills(None), [])
        self.assertFalse(app.has_skill("", "python"))
        self.assertFalse(app.has_skill("python", ""))


class ExperienceTests(unittest.TestCase):
    def test_stated_years(self):
        self.assertEqual(app.extract_experience_years("5 years of experience in data"), 5.0)
        self.assertEqual(app.extract_experience_years("8+ years of professional experience"), 8.0)

    def test_date_ranges_merged(self):
        text = "2015 - 2018 Dev at A\n2017 - 2020 Dev at B"
        self.assertEqual(app.extract_experience_years(text, current_year=2024), 5.0)

    def test_present_uses_current_year(self):
        self.assertEqual(app.extract_experience_years("2020 - Present Engineer", current_year=2024), 4.0)

    def test_education_dates_ignored(self):
        text = "2014 - 2018 B.Sc University of X"
        self.assertEqual(app.extract_experience_years(text, current_year=2024), 0.0)

    def test_empty_and_capped(self):
        self.assertEqual(app.extract_experience_years(""), 0.0)
        self.assertLessEqual(app.extract_experience_years("99 years of experience"), 40.0)


class AnonymizationTests(unittest.TestCase):
    SAMPLE = (
        "Jane Doe\n"
        "Email: jane.doe@example.com | Phone: +1 (415) 555-2671\n"
        "https://www.linkedin.com/in/janedoe\n"
        "Gender: Female\n"
        "She worked at Acme from 2018 - 2022.\n"
    )

    def setUp(self):
        self.out = app.anonymize_text(self.SAMPLE)

    def test_email_phone_url_removed(self):
        self.assertNotIn("jane.doe@example.com", self.out)
        self.assertNotIn("555-2671", self.out)
        self.assertNotIn("linkedin.com", self.out)

    def test_name_and_gender_removed(self):
        self.assertNotIn("Jane Doe", self.out)
        self.assertNotIn("Female", self.out)
        self.assertNotIn("She ", self.out)

    def test_years_not_treated_as_phone(self):
        self.assertIn("2018 - 2022", self.out)

    def test_heading_first_line_not_masked(self):
        self.assertTrue(app.anonymize_text("Work Experience\nPython developer").startswith("Work Experience"))

    def test_empty(self):
        self.assertEqual(app.anonymize_text(""), "")
        self.assertEqual(app.anonymize_text(None), "")


class ScoringTests(unittest.TestCase):
    JD = "Data scientist using Python and SQL to build machine learning models."

    def test_bounds(self):
        for text in ["", "python", "python sql machine learning 20 years of experience", "nothing relevant"]:
            result = app.score_candidate(text, ["python", "sql"], self.JD, 3)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 100.0)

    def test_perfect_and_zero(self):
        perfect = app.score_candidate("Python SQL. 10 years of experience.", ["python", "sql"], "", 5)
        self.assertEqual(perfect["score"], 100.0)
        zero = app.score_candidate("Cooking and gardening", ["python", "sql"], "", 5)
        self.assertEqual(zero["score"], 0.0)

    def test_deterministic(self):
        a = app.score_candidate("Python and SQL, 4 years of experience", ["python", "sql", "aws"], self.JD, 3)
        b = app.score_candidate("Python and SQL, 4 years of experience", ["python", "sql", "aws"], self.JD, 3)
        self.assertEqual(a, b)

    def test_explanations(self):
        result = app.score_candidate("Python developer", ["python", "sql"], "", 0)
        self.assertEqual(result["matched_skills"], ["python"])
        self.assertEqual(result["missing_skills"], ["sql"])
        self.assertTrue(any("Missing" in r for r in result["reasons"]))
        self.assertEqual(result["score"], 50.0)

    def test_empty_resume(self):
        result = app.score_candidate("   ", ["python"], self.JD, 2)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["missing_skills"], ["python"])

    def test_empty_job_and_no_skills(self):
        result = app.score_candidate("Python developer", [], "", 0)
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(result["reasons"])

    def test_skills_derived_from_job_description(self):
        result = app.score_candidate("Python and SQL expert", [], self.JD, 0)
        self.assertIn("python", result["matched_skills"])
        self.assertGreater(result["score"], 0.0)

    def test_more_matches_never_lower_score(self):
        low = app.score_candidate("Python", ["python", "sql", "aws"], "", 0)["score"]
        high = app.score_candidate("Python SQL", ["python", "sql", "aws"], "", 0)["score"]
        self.assertGreater(high, low)

    def test_group_labels_have_no_effect(self):
        base = "Python and SQL, 4 years of experience"
        a = app.score_candidate(base + " Group A", ["python", "sql"], "", 3)["score"]
        b = app.score_candidate(base + " Group B", ["python", "sql"], "", 3)["score"]
        self.assertEqual(a, b)


class ScreeningPipelineTests(unittest.TestCase):
    def test_ranking_and_duplicates(self):
        docs = [
            ("a.txt", "Python SQL 5 years of experience"),
            ("b.txt", "Python only"),
            ("dup.txt", "python   only"),  # duplicate of b after normalisation
            ("empty.txt", "   "),
        ]
        results, warnings = app.screen_resumes(docs, ["python", "sql"], "", 2, anonymize=True)
        self.assertEqual([r["file_name"] for r in results], ["a.txt", "b.txt"])
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertEqual(len(warnings), 2)

    def test_no_documents(self):
        results, warnings = app.screen_resumes([], ["python"], "", 0)
        self.assertEqual(results, [])
        self.assertEqual(warnings, [])

    def test_dataframe_and_csv(self):
        results, _ = app.screen_resumes([("x.txt", "Python")], ["python"], "", 0)
        df = app.results_to_dataframe(results)
        self.assertNotIn("File", df.columns)
        self.assertIn("File", app.results_to_dataframe(results, include_file_names=True).columns)
        self.assertIsInstance(app.dataframe_to_csv_bytes(df), bytes)
        self.assertTrue(app.results_to_dataframe([]).empty)

    def test_csv_export_neutralises_formulas(self):
        import pandas as pd

        df = pd.DataFrame({"File": ["=HYPERLINK(1)", "ok.txt"], "Score": [1.0, 2.0]})
        data = app.dataframe_to_csv_bytes(df).decode("utf-8")
        self.assertIn("'=HYPERLINK(1)", data)
        self.assertIn("ok.txt", data)

    def test_csv_injection_guard(self):
        self.assertEqual(app.csv_safe("=SUM(A1)"), "'=SUM(A1)")
        self.assertEqual(app.csv_safe("normal"), "normal")


class FileReadingTests(unittest.TestCase):
    def test_txt(self):
        text, err = app.read_resume_bytes("cv.txt", b"Hello Python")
        self.assertEqual(text, "Hello Python")
        self.assertIsNone(err)

    def test_unsupported_empty_and_corrupt(self):
        self.assertIsNotNone(app.read_resume_bytes("cv.exe", b"abc")[1])
        self.assertIsNotNone(app.read_resume_bytes("cv.txt", b"")[1])
        self.assertIsNotNone(app.read_resume_bytes("cv.pdf", b"not a pdf")[1])
        self.assertIsNotNone(app.read_resume_bytes("cv.docx", b"not a docx")[1])

    def test_too_large(self):
        big = b"a" * (app.MAX_FILE_BYTES + 1)
        self.assertIsNotNone(app.read_resume_bytes("cv.txt", big)[1])

    def test_docx_roundtrip(self):
        try:
            import docx
        except ImportError:
            self.skipTest("python-docx not installed")
        document = docx.Document()
        document.add_paragraph("Python developer with SQL")
        buffer = io.BytesIO()
        document.save(buffer)
        text, err = app.read_resume_bytes("cv.docx", buffer.getvalue())
        self.assertIsNone(err)
        self.assertIn("Python developer", text)

    def test_pdf_roundtrip(self):
        try:
            from pypdf import PdfWriter
        except ImportError:
            self.skipTest("pypdf not installed")
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        text, err = app.read_resume_bytes("blank.pdf", buffer.getvalue())
        self.assertIsNone(text)  # blank page: no extractable text
        self.assertIn("no extractable text", err)


class FairnessMetricTests(unittest.TestCase):
    def test_rates_and_metrics(self):
        groups = ["A"] * 10 + ["B"] * 10
        selected = [True] * 5 + [False] * 5 + [True] * 2 + [False] * 8
        rates = app.selection_rates(groups, selected)
        self.assertAlmostEqual(rates["A"], 0.5)
        self.assertAlmostEqual(rates["B"], 0.2)
        self.assertAlmostEqual(app.demographic_parity_difference(rates), 0.3)
        self.assertAlmostEqual(app.disparate_impact_ratio(rates), 0.4)

    def test_parity(self):
        rates = app.selection_rates(["A", "A", "B", "B"], [True, False, True, False])
        self.assertEqual(app.demographic_parity_difference(rates), 0.0)
        self.assertEqual(app.disparate_impact_ratio(rates), 1.0)

    def test_edge_cases(self):
        self.assertEqual(app.demographic_parity_difference({"A": 0.5}), 0.0)
        self.assertIsNone(app.disparate_impact_ratio({"A": 0.5}))
        self.assertIsNone(app.disparate_impact_ratio({"A": 0.0, "B": 0.0}))
        self.assertEqual(app.selection_rates([], []), {})
        with self.assertRaises(ValueError):
            app.selection_rates(["A"], [])

    def test_report_flags(self):
        groups = ["A"] * 20 + ["B"] * 20
        selected = [True] * 10 + [False] * 10 + [True] * 2 + [False] * 18
        report = app.fairness_report(groups, selected)
        self.assertIs(report["four_fifths_pass"], False)
        small = app.fairness_report(["A", "B"], [True, False])
        self.assertTrue(any("Small groups" in n for n in small["notes"]))

    def test_select_top_k(self):
        self.assertEqual(app.select_top_k([1, 3, 2], 2), [False, True, True])
        self.assertEqual(app.select_top_k([1, 2], 0), [False, False])
        self.assertEqual(app.select_top_k([1, 2], 99), [True, True])

    def test_audit_uploaded_outcomes(self):
        import pandas as pd

        df = pd.DataFrame({"g": ["A", "A", "B", "B"], "s": ["yes", "no", "yes", "yes"]})
        report = app.audit_uploaded_outcomes(df, "g", "s")
        self.assertAlmostEqual(report["rates"]["A"], 0.5)
        self.assertAlmostEqual(report["rates"]["B"], 1.0)
        with self.assertRaises(ValueError):
            app.audit_uploaded_outcomes(df, "g", "missing")


class SyntheticDataTests(unittest.TestCase):
    def test_reproducible(self):
        a = app.generate_synthetic_candidates(n=100, seed=1)
        b = app.generate_synthetic_candidates(n=100, seed=1)
        self.assertTrue(a.equals(b))

    def test_shape_and_bounds(self):
        df = app.generate_synthetic_candidates(n=120, seed=3, shortlist_fraction=0.25)
        self.assertEqual(len(df), 120)
        self.assertTrue(df["merit_score"].between(0, 100).all())
        self.assertTrue(df["biased_score"].between(0, 100).all())
        self.assertEqual(int(df["selected_merit_only"].sum()), 30)

    def test_zero_candidates(self):
        self.assertEqual(len(app.generate_synthetic_candidates(n=0)), 0)

    def test_simulated_bias_is_detected(self):
        df = app.generate_synthetic_candidates(n=600, seed=7, bias_penalty=30, shortlist_fraction=0.25)
        fair = app.fairness_report(df["group"].tolist(), df["selected_merit_only"].tolist())
        biased = app.fairness_report(df["group"].tolist(), df["selected_biased_process"].tolist())
        self.assertGreater(biased["demographic_parity_difference"], fair["demographic_parity_difference"])
        self.assertLess(biased["disparate_impact_ratio"], 0.8)

    def test_group_not_used_in_merit_score(self):
        df = app.generate_synthetic_candidates(n=200, seed=5, bias_penalty=0)
        self.assertTrue((df["merit_score"] == df["biased_score"]).all())


class UiComponentTests(unittest.TestCase):
    def setUp(self):
        self.results, _ = app.screen_resumes(
            [("<script>alert(1)</script>.txt", "Python SQL. 5 years of experience. $100 & <b>bold</b>"), ("b.txt", "Python")],
            ["python", "sql"], "", 2, anonymize=False,
        )

    def test_theme_css_has_both_palettes_and_fallback(self):
        light, dark = app.theme_css(False), app.theme_css(True)
        for css in (light, dark):
            self.assertTrue(css.startswith("<style>") and css.endswith("</style>"))
            self.assertNotIn("\n", css)
            self.assertIn(":root:has(.fh-dark-marker)", css)
            self.assertIn(":root:not(:has(.fh-dark-marker))", css)
            self.assertIn(app.THEMES["light"]["bg"], css)
            self.assertIn(app.THEMES["dark"]["bg"], css)
        # the plain :root fallback block (first --bg occurrence) follows the requested mode
        self.assertTrue(light.split("--bg: ", 1)[1].startswith(app.THEMES["light"]["bg"]))
        self.assertTrue(dark.split("--bg: ", 1)[1].startswith(app.THEMES["dark"]["bg"]))

    def test_theme_marker(self):
        self.assertIn("fh-dark-marker", app.theme_marker_html(True))
        self.assertNotIn("fh-dark-marker", app.theme_marker_html(False))
        self.assertNotIn("\n", app.theme_marker_html(True))

    def test_theme_script_and_label(self):
        for dark in (False, True):
            script = app.theme_script_html(dark)
            self.assertTrue(script.startswith("<script>") and script.endswith("</script>"))
            self.assertEqual(script.count("</script>"), 1)
            self.assertIn("window.parent.document", script)
            self.assertIn(f"'data-fh-theme','{'dark' if dark else 'light'}'", script)
            self.assertIn(app.THEMES["dark"]["bg"], script)
        self.assertIn("Dark", app.theme_label_html(True))
        self.assertIn("Light", app.theme_label_html(False))
        self.assertIn('data-fh-theme="dark"', app.theme_css(False))

    def test_content_clears_streamlit_fixed_header(self):
        # Streamlit's header is fixed at the top; too little top padding puts the
        # dark-mode toggle underneath it, where clicks never reach the toggle.
        import re

        for rule in re.findall(r"\.block-container \{[^}]*padding: ([\d.]+)rem", app.BASE_CSS):
            self.assertGreaterEqual(float(rule), 4.0)

    def test_intro_is_calm_and_never_blocks_clicks(self):
        import re

        html_out = app.intro_html()
        self.assertIn("Welcome to FairHire", html_out)
        self.assertNotIn("\n", html_out)
        self.assertNotIn("<script", html_out)
        rule = re.search(r"\.fh-intro \{([^}]*)\}", app.BASE_CSS).group(1)
        self.assertIn("pointer-events: none", rule)
        self.assertIn("forwards", rule)  # stays hidden after fading out
        self.assertIn("visibility: hidden", app.BASE_CSS)
        self.assertIn("prefers-reduced-motion", app.BASE_CSS)

    def test_html_is_escaped(self):
        table = app.results_table_html(self.results, 1, show_files=True)
        card = app.candidate_card_html(self.results[0], show_files=True)
        for html_out in (table, card):
            self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", table)
        self.assertNotIn("<b>bold</b>", card)
        self.assertNotIn("$", card)

    def test_no_blank_lines_that_break_markdown(self):
        outputs = [
            app.results_table_html(self.results, 1),
            app.candidate_card_html(self.results[0], open_card=True),
            app.hero_html(),
            app.kpis_html([app.kpi_card("A", 1, "hint", "ok")]),
            app.rates_bars_html({"A": 0.5, "B": 0.25}, {"A": 4, "B": 4}),
            app.empty_state_html("t", "b"),
            app.note_html("line1\n\nline2", "warn", "!"),
        ]
        for out in outputs:
            self.assertNotIn("\n", out)

    def test_candidate_card_open_attribute(self):
        self.assertIn("<details class=\"fh-cand\" open>", app.candidate_card_html(self.results[0], open_card=True))
        self.assertNotIn(" open>", app.candidate_card_html(self.results[0], open_card=False))

    def test_fairness_panel_variants(self):
        bad = app.fairness_report(["A"] * 20 + ["B"] * 20, [True] * 10 + [False] * 10 + [True] * 2 + [False] * 18)
        good = app.fairness_report(["A", "A", "B", "B"], [True, False, True, False])
        undefined = app.fairness_report(["A", "B"], [False, False])
        single = app.fairness_report(["A", "A"], [True, False])
        self.assertIn("Flag for review", app.fairness_panel_html("t", bad))
        self.assertIn("Passes four-fifths guide", app.fairness_panel_html("t", good))
        self.assertIn("Cannot assess", app.fairness_panel_html("t", undefined))
        self.assertIn("Cannot assess", app.fairness_panel_html("t", single))

    def test_score_tone_and_bar_bounds(self):
        self.assertEqual([app.score_tone(x) for x in (0, 49.9, 50, 74.9, 75, 100)], ["bad", "bad", "warn", "warn", "ok", "ok"])
        self.assertIn("width:100.0%", app._score_cell(250))
        self.assertIn("width:0.0%", app._score_cell(-5))

    def test_df_table_html(self):
        import pandas as pd

        df = pd.DataFrame({"a": ["<x>", "y"], "ok": [True, False], "v": [1.234, 2.0]})
        out = app.df_table_html(df, max_rows=1)
        self.assertIn("&lt;x&gt;", out)
        self.assertIn("✓", out)
        self.assertNotIn(">y<", out)
        self.assertNotIn("\n", out)


if __name__ == "__main__":
    unittest.main()
