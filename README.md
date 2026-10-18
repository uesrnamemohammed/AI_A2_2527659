# Musanze Harvest Intelligence Service

| | |
|---|---|
| Full name | Mohammed Bashier Ahmed |
| Registration number | 25/27659 |
| Institutional username | ug2527659@ines.ac.rw |
| Verification code | **AI-A2-7659** |
| Course | SWE 3513 Artificial Intelligence, Individual Assignment 2, INES-Ruhengeri |
| Repository URL | `<add your repository URL>` |
| Final commit hash | `8cc3d13690b09fe1a7b55d5f984c08930fe14732 (code and artifacts as tested)` |
| Python version | 3.14.7 (tested on Windows, 10 passing tests; first developed on Python 3.12, Ubuntu 24) |

## Problem

A potato cooperative in Musanze receives farm records in inconsistent spreadsheets. This prototype estimates harvest quantity (regression), flags farms at high production risk, meaning yield per hectare below the median (classification), and groups similar farms (clustering). Training and command-line prediction work offline; a FastAPI service is provided for a future mobile app.

**The dataset is synthetic.** It is generated from the formula in the assignment and the seed 7659 (last four digits of the registration number). The service is a classroom decision-support prototype: it does not replace an agronomist, must not decide access to finance, insurance or public services, and every output needs human review.

## Setup and run

```
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --student-id ug2527659@ines.ac.rw --seed 7659
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000
```

`run_all.py` rejects any seed other than 7659, recreates the CSV, validates and cleans it, trains and evaluates all three tasks, saves models, plots and metrics, prints the SHA-256 fingerprint and writes the PDF report. Use `--no-report` to skip the PDF. Interactive API docs are at http://127.0.0.1:8000/docs.

Optional evidence helper: `python make_test_log.py --reg-digits 2527659` runs `pytest -q` and saves the test log PDF.

## Artifacts

- `data/AI_A2_2527659.csv` (276 rows = 273 base + 3 duplicates; 14 missing soil_moisture_pct, 14 missing fertilizer_kg)
- `artifacts/data_fingerprint.txt`, `regression_metrics.json`, `classification_metrics.json`, `clustering_report.json`, `test_predictions.csv`
- `artifacts/plots/residuals.png`, `confusion_matrix.png`, `clusters.png`
- `models/` preprocessor, regression, classification and cluster models, `metadata.json`
- `docs/AI_A2_2527659_REPORT.pdf`

## API examples

```
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/valid.json
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/invalid.json
```

Valid input returns yield estimate, risk prediction, probability, cluster, cluster profile and the verification code. Invalid input returns HTTP 422 with details. If models have not been trained, `/predict` and `/model-info` return 503.

## Code layout

`data/generate_data.py` (generator), `src/validation.py`, `src/preprocessing.py`, `src/modeling.py`, `src/evaluation.py`, `src/inference.py`, `src/report.py`, `tests/test_system.py`.

## Extra packages

`reportlab` builds the report and test-log PDFs from the generated artifacts so the report always matches the metrics. `httpx` is required by FastAPI's test client. `uvicorn` runs the API. Everything else is in the list given by the assignment.

## Known limitations

Synthetic data with a known formula, so scores overstate real-world performance; small test set (69 farms); weak cluster structure (silhouette about 0.15); numeric results may differ in the last digits across library versions, so the fingerprint is checked on the data, and metrics should match within floating-point tolerance.
