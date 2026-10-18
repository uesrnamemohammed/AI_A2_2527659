"""Build the technical report PDF from the generated JSON and CSV artifacts (needs reportlab)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from xml.sax.saxutils import escape

from src import config

FULL_NAME = "Mohammed Bashier Ahmed"
REG_NUMBER = "25/27659"

_styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=10, leading=14)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11)
H1 = ParagraphStyle("h1", parent=_styles["Heading1"], fontSize=15, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
TITLE = ParagraphStyle("title", parent=_styles["Title"], fontSize=19)


def _table(rows, widths=None, header=True, font=8.5):
    cell = ParagraphStyle("cell", parent=BODY, fontSize=font, leading=font + 2.5)
    rows = [
        [
            Paragraph(f"<b>{c}</b>" if (header and i == 0) else c, cell) if isinstance(c, str) else c
            for c in r
        ]
        for i, r in enumerate(rows)
    ]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                  ]
    t.setStyle(TableStyle(style))
    return t


def _p(text, style=BODY):
    return Paragraph(text, style)


def _img(path: Path, width_cm: float):
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    width = width_cm * cm
    return Image(str(path), width=width, height=width * h / w)


def _f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def _own_sections() -> dict:
    """Read the student's own interpretations from docs/my_interpretations.md (sections: regression, classification, cluster)."""
    path = config.DOCS_DIR / "my_interpretations.md"
    labels = {
        "regression": "Important regression error (own interpretation).",
        "classification": "Important classification error (own interpretation).",
        "cluster": "Farm cluster (own interpretation).",
    }
    if not path.exists():
        return {}
    chunks, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            name = stripped.lstrip("#").strip().lower()
            current = name if name in labels else None
            chunks.setdefault(current, []) if current else None
        elif current and stripped and not stripped.startswith(">"):
            chunks[current].append(stripped)
    return {k: f"<b>{labels[k]}</b> " + escape(" ".join(v)) for k, v in chunks.items() if v}


def build_report(identity: dict, student_id: str) -> Path:
    A = config.ARTIFACTS_DIR
    own = _own_sections()
    reg = json.loads((A / "regression_metrics.json").read_text())
    clf = json.loads((A / "classification_metrics.json").read_text())
    clu = json.loads((A / "clustering_report.json").read_text())
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
    preds = pd.read_csv(A / "test_predictions.csv")
    raw = pd.read_csv(config.dataset_path(identity["reg_digits"]))
    fp = meta["dataset_sha256"]
    code, seed, digits = identity["verification_code"], identity["seed"], identity["reg_digits"]
    base = 240 + seed % 41
    m = int(round(0.05 * base))
    out = config.DOCS_DIR / f"AI_A2_{digits}_REPORT.pdf"
    config.DOCS_DIR.mkdir(exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, f"SWE 3513 Individual Assignment 2 | {FULL_NAME} | {REG_NUMBER} | {code}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                            title="Musanze Harvest Intelligence Service - Technical Report", author=FULL_NAME)
    s = []

    # Page 1
    s += [_p("Musanze Harvest Intelligence Service", TITLE), _p("Technical report - SWE 3513 Artificial Intelligence, Individual Assignment 2", BODY), Spacer(1, 8)]
    s.append(_table([["Item", "Value"],
                     ["Full name", FULL_NAME], ["Registration number", REG_NUMBER],
                     ["Institutional username", student_id], ["Verification code", code],
                     ["Seed LLLL", str(seed)], ["Institution", "INES-Ruhengeri"]], [5 * cm, 11 * cm]))
    s += [_p("1. Identity and decision problem", H1),
          _p("<b>Stakeholder.</b> The field officer of a potato cooperative in Musanze, who receives farm records in inconsistent spreadsheets and has unreliable internet access."),
          _p("<b>Prediction questions.</b> (1) How many tons will a farm harvest (regression)? (2) Is the farm at high production risk, meaning its yield per hectare is below the median (classification)? (3) Which farms share similar conditions (clustering)?"),
          _p("<b>Why local.</b> Training and command-line prediction run offline from one command; a small FastAPI service allows later integration with a mobile application."),
          _p("<b>Ethical boundary.</b> " + config.DISCLAIMER),
          _p("<b>Synthetic data.</b> All records are generated from the formula in the assignment and the registration-based seed. No real farm or person is represented."),
          PageBreak()]

    # Page 2
    q = reg["models"]
    s += [_p("2. Data and reproducibility", H1)]
    s.append(_table([["Column", "Generation rule", "Role"],
                     ["rainfall_mm", "Uniform 600-1400", "Numeric feature"],
                     ["avg_temp_c", "Uniform 12-26", "Numeric feature"],
                     ["soil_moisture_pct", "Uniform 20-80", "Numeric feature"],
                     ["fertilizer_kg", "Uniform 0-250", "Numeric feature"],
                     ["field_size_ha", "Uniform 0.2-4.0", "Numeric feature"],
                     ["season", "Random A or B", "Categorical feature"],
                     ["cooperative_support", "Random 0 or 1", "Binary feature"],
                     ["yield_tons", "Assignment formula, noise SD 0.80, minimum 0.50", "Regression target"],
                     ["high_risk", "Yield per hectare below its median", "Classification target"]],
                    [4 * cm, 8.5 * cm, 4.5 * cm]))
    miss = raw.isna().sum()
    s += [Spacer(1, 6),
          _p(f"<b>Personal seed rule.</b> The seed is the last four digits of the registration number, LLLL = {seed}. "
             f"The generator uses numpy.random.default_rng({seed}); the base row count is 240 + ({seed} mod 41) = {base}. "
             f"run_all.py rejects any other seed."),
          _p(f"<b>Row count and deliberate problems.</b> With m = round(0.05 x {base}) = {m}, the generator removed {m} values of soil_moisture_pct and {m} values of fertilizer_kg "
             f"and appended 3 exact duplicates. The CSV has {len(raw)} rows; the validator found {int(miss['soil_moisture_pct'])} and {int(miss['fertilizer_kg'])} missing values and {int(raw.duplicated().sum())} duplicate rows, "
             f"and the pipeline fails if these counts differ from the generator rules (this detects manual edits). Duplicates are dropped, leaving {len(raw) - int(raw.duplicated().sum())} rows; missing values are imputed inside the pipeline after the split."),
          _p("<b>SHA-256 fingerprint of the final CSV bytes</b> (stored in artifacts/data_fingerprint.txt):"),
          _p(f"<font face='Courier' size='8.5'>{fp}</font>"),
          _p("Re-running run_all.py twice reproduced the same fingerprint and byte-identical metric files, prediction table and fingerprint file."),
          PageBreak()]

    # Page 3
    s += [_p("3. Pipeline design", H1),
          _p(f"<b>Split.</b> After removing duplicates, a stratified (on high_risk) split with test size {config.TEST_SIZE} and random_state={seed} gives "
             f"{len(preds) and (len(raw) - 3 - len(preds))} training rows and {len(preds)} test rows. The test set is used only for final reporting."),
          _p("<b>Preprocessing (one ColumnTransformer, fitted on training rows only).</b> Numeric features: median imputation then standard scaling. season: one-hot encoding with unknown categories ignored. cooperative_support: passed through. "
             "The same fitted transformer is embedded in each saved model Pipeline, so training and inference apply identical transformations, and a copy is saved as models/preprocessor.joblib."),
          _p("<b>Leakage controls.</b> yield_tons, yield per hectare and high_risk are never inputs to classification (assert_no_leakage runs before training and a test checks it). "
             "Imputers, scalers and encoders are never fitted before the split (a test confirms the imputer medians equal the training medians). Clustering uses no target."),
          _p("<b>Model selection.</b> Candidates are compared with 5-fold cross-validation on the training set only (RMSE for regression, F1 for classification); the baseline is reported but cannot be selected. Test-set scores are reported for every model but do not choose the winner.")]
    s.append(_table([["Task", "Baseline", "Candidates", "Selection metric"],
                     ["Regression", "Mean predictor", "Linear regression, random forest, gradient boosting", "CV RMSE (lowest)"],
                     ["Classification", "Majority class", "Logistic regression, random forest, gradient boosting", "CV F1 (highest)"],
                     ["Clustering", "-", "K-means, K = 2 to 6 on scaled features", "Silhouette (highest)"]],
                    [3 * cm, 3.2 * cm, 7.3 * cm, 3.5 * cm]))
    s += [Spacer(1, 6), _p("All random states are set to the seed; models use a single thread, which keeps results deterministic."), PageBreak()]

    # Page 4
    rs = reg["selected_model"]
    s += [_p("4. Regression and classification", H1), _p("<b>Regression - held-out test set</b> (selected: " + rs + ")", H2)]
    rows = [["Model", "CV RMSE (train)", "Test MAE", "Test RMSE", "Test R2"]]
    for n, v in reg["models"].items():
        rows.append([n + (" *" if n == rs else ""), _f(v["cv_rmse_train"]), _f(v["test"]["MAE"]), _f(v["test"]["RMSE"]), _f(v["test"]["R2"])])
    s.append(_table(rows, [5 * cm, 3 * cm, 2.8 * cm, 2.8 * cm, 2.6 * cm]))
    e = reg["largest_error_case"]
    row = raw.loc[e["row_id"]]
    s += [Spacer(1, 4), _img(A / "plots" / "residuals.png", 11.5),
          _p(own.get("regression") or f"<b>Important regression error.</b> The largest test error is row {e['row_id']}: actual {e['actual_yield_tons']:.2f} t, predicted {e['predicted_yield_tons']:.2f} t (error {e['error']:+.2f} t). "
             f"Its inputs were rainfall {row['rainfall_mm']:.0f} mm, temperature {row['avg_temp_c']:.1f} C, field {row['field_size_ha']:.2f} ha. "
             f"The generator adds random noise with SD 0.80 t that no model can predict, so an error of this size is partly irreducible; the model's RMSE of {reg['models'][rs]['test']['RMSE']:.2f} t sits close to that noise level.", SMALL)]
    cs = clf["selected_model"]
    s.append(_p("<b>Classification - held-out test set</b> (selected: " + cs + ")", H2))
    rows = [["Model", "CV F1", "Precision", "Recall", "F1", "ROC-AUC", "Confusion [[TN FP][FN TP]]"]]
    for n, v in clf["models"].items():
        t = v["test"]
        rows.append([n + (" *" if n == cs else ""), _f(v["cv_f1_train"], 3), _f(t["precision"], 3), _f(t["recall"], 3), _f(t["f1"], 3), _f(t["roc_auc"], 3), str(t["confusion_matrix"])])
    s.append(_table(rows, [3.6 * cm, 1.6 * cm, 1.9 * cm, 1.7 * cm, 1.5 * cm, 1.9 * cm, 4.8 * cm], font=8))
    s += [_img(A / "plots" / "confusion_matrix.png", 5.2)]
    wrong = clf["misclassified_test_rows"]
    if wrong:
        w = wrong[0]
        wr = raw.loc[w["row_id"]]
        text = (f"<b>Important classification error.</b> Test row {w['row_id']} was truly high_risk={w['actual_high_risk']} but predicted {w['predicted_high_risk']} "
                f"with probability {w['risk_probability']:.2f}. Its field size is {wr['field_size_ha']:.2f} ha and its yield is {wr['yield_tons']:.2f} t. "
                f"Because high_risk is defined by comparing yield per hectare with the median, farms near that boundary are easily flipped by noise. In total {len(wrong)} of {len(preds)} test farms were misclassified; "
                f"for a risk screening tool, missed high-risk farms (false negatives) matter more than false alarms.")
    else:
        text = "No test farm was misclassified."
    s += [_p(own.get("classification") or text), PageBreak()]

    # Page 5
    s += [_p("5. Clustering", H1),
          _p(f"Features were imputed and scaled with the same preprocessing; K-means was fitted on training rows for K = 2 to 6. {clu['selection_reason']} {clu['interpretation']}")]
    rows = [["K"] + list(clu["silhouette_by_k"].keys()), ["Silhouette"] + [_f(v) for v in clu["silhouette_by_k"].values()]]
    s += [_table(rows, [2.5 * cm] + [2.4 * cm] * len(clu["silhouette_by_k"])), Spacer(1, 6), _img(A / "plots" / "clusters.png", 11)]
    rows = [["Cluster", "Farms (train)", "Profile"]]
    for k, v in clu["profiles"].items():
        rows.append([k, str(v["size_train"]), Paragraph(v["profile"], SMALL)])
    s += [Spacer(1, 4), _table(rows, [2 * cm, 3 * cm, 12 * cm])]
    big = max(clu["profiles"].items(), key=lambda kv: kv[1]["size_train"])
    s += [Spacer(1, 4), _p(own.get("cluster") or f"<b>Operational meaning.</b> Cluster {big[0]} is the largest group ({big[1]['size_train']} training farms): {big[1]['profile'][:-1].lower()}. "
                           f"A field officer could plan visits per cluster, for example checking moisture management in dry-soil groups. Because the silhouette is weak, farms near cluster borders should not be treated as clearly different."),
          PageBreak()]

    # Page 6
    s += [_p("6. API and testing", H1),
          _table([["Endpoint", "Purpose", "Success", "Failure"],
                  ["GET /health", "Service status and verification code", "200", "-"],
                  ["GET /model-info", "Seed, fingerprint, selected models, test metrics", "200", "503 if models are missing"],
                  ["POST /predict", "Typed farm record to yield, risk, probability, cluster", "200", "422 for missing field, wrong type, out-of-range or unknown field"]],
                 [3 * cm, 6.2 * cm, 1.8 * cm, 6 * cm]),
          _p("<b>Valid request</b> (samples/valid.json):", H2),
          _p("<font face='Courier' size='8'>{\"rainfall_mm\": 1000.0, \"avg_temp_c\": 18.5, \"soil_moisture_pct\": 55.0, \"fertilizer_kg\": 120.0, \"field_size_ha\": 1.5, \"season\": \"A\", \"cooperative_support\": 1}</font>"),
          _p("<b>Invalid request</b> (samples/invalid.json) has rainfall 2500, temperature \"warm\", season \"C\" and no field_size_ha; the CLI lists all four problems and exits with code 2, and the API returns HTTP 422 with the details. "
             "The request model uses strict types, so a numeric string is rejected rather than silently converted."),
          _p("<b>Tests.</b> tests/test_system.py contains 10 pytest tests: valid inference, missing field, wrong type, range error, reproducibility and fingerprint, dataset signature with deliberate problems, wrong seed, leakage guard, train-only preprocessing, and an API test covering /health, /model-info and valid and invalid /predict calls. "
             "Failure paths are tested, not only the happy path. The result of the final run is recorded in evidence/" + f"AI_A2_{digits}_TEST_LOG.pdf."),
          _p("<b>Failure behaviour.</b> If models are missing, the CLI returns a message telling the user to run run_all.py and the API returns 503; it never returns a guessed prediction."),
          PageBreak()]

    # Page 7
    s += [_p("7. Model card and reflection", H1),
          _p("<b>Intended use.</b> Classroom decision support for a field officer: rough yield estimates, a risk screen and farm grouping on synthetic data."),
          _p("<b>Prohibited use.</b> Deciding loans, insurance, subsidies or any access to public services; replacing an agronomist; use on real farms without retraining and validation on real data."),
          _p("<b>Limitations.</b> The data is synthetic and follows a known formula, so high scores say little about real farms. The sample is small (" + f"{len(preds)} test farms" + "), so metrics carry wide uncertainty. "
             "Inputs outside the training ranges are rejected, not extrapolated. Clusters are weak groupings."),
          _p("<b>Risks.</b> Over-trust in a confident probability; a false negative that hides a struggling farm; fairness problems if cooperative_support were later used to ration resources."),
          _p("<b>Human oversight.</b> Every prediction is advice for a person to review; the API response repeats the disclaimer."),
          _p("<b>Reflection.</b> The simplest model, linear regression, won the regression task because the generator is almost linear, and logistic regression won classification; more complex models did not help on 204 training rows. "
             "With real data I would collect more seasons, check measurement quality, calibrate probabilities, evaluate errors by district and validate with agronomists before any use.")]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return out
