# Architecture and delivery design (student ID 13130)

## Toolchain decisions (verified 14 Oct 2026)

| Decision | Reason |
|---|---|
| uv-managed CPython 3.12 (`.python-version`, `requires-python >=3.12,<3.15`) | Streamlit Community Cloud documents 3.12 as default and ignores `.python-version`; the user selects 3.12 in the deploy dialog. Wheels for 3.14 exist for everything, so the lock stays universal. |
| `uv.lock` committed, `requirements.txt` exported from it, `Dockerfile` with uv | Satisfies "requirements.txt" (2 marks) and "native locking or full containerization" (3 marks); Docker is built only in CI (no Docker on this machine). |
| `pandas>=2.2,<3` | seaborn 0.13.2 has no released pandas-3 support. |
| Boosting = scikit-learn `HistGradientBoostingClassifier` | No xgboost (57 MB GPU wheel + NCCL on Linux). |
| Charts: Altair in the app (already a Streamlit dependency) + `st.pyplot` for seaborn figures; no plotly | Memory on Cloud (690 MB minimum). |
| Multipage via `st.Page` + `st.navigation` in `app/streamlit_app.py`; Cloud-only `app/requirements.txt` beside the entrypoint | Cloud reads the dependency file in the entrypoint's folder first; `uv.lock` handling on Cloud is undocumented. |
| All heavy work precomputed by `uv run mlda-train` and committed (models, metrics, CV tables, test predictions, figures) | Evaluation page does zero inference; live training page samples <= 20k rows, 3-fold CV, <= 12 grid combinations. |

## Repository layout (`C:\ML\mlda-cw1`)

```
.github/workflows/ci.yml   lint + tests (ubuntu, windows), uv lock --check, export diff, notebook execution, docker build + health
.streamlit/config.toml     headless, no usage stats, theme
.dockerignore .gitattributes (eol=lf) .gitignore .python-version (3.12)
LICENSE (MIT)  README.md  pyproject.toml  uv.lock  requirements.txt  Dockerfile  Makefile
data/README.md             data card: sources, licence, snapshot date, SHA-256, join keys, target definition
data/raw/                  downloaded zips + codebooks (committed if total <= 50 MB; else `uv run mlda-download` with checksum)
data/processed/            clean.parquet (zstd), split_indices.json, feature_schema.json
models/                    <name>.joblib (compress=3, <= 20 MB each) + manifest.json (versions, git SHA, data checksum)
reports/metrics/           <name>.json test metrics + comparison.csv
reports/cv/                <name>_cv_results.csv
reports/predictions/       test_predictions.parquet (y_true + y_proba per model)
reports/figures/           PNGs used by report and app
report/                    MLDA_CW1_13130_report.docx (+ pdf), viva_prep.md, build_report.py
notebooks/01_end_to_end.ipynb   executed notebook: Load -> EDA -> Prepare -> Train -> Evaluate
src/mlda/
  config.py                PROJECT_ROOT, paths, RANDOM_STATE = 42, TARGET, URLS + SHA256, leakage drop-list
  data/download.py         fetch with checksum verification
  data/load.py             typed readers for the pipe-delimited files
  data/join.py             merge_sources(): left joins with validate="many_to_one", indicator, row-count assertions
  data/schema.py           expected columns, dtypes, ranges; validate_schema()
  cleaning.py              pure df -> df: duplicates, type coercion, impossible values, outlier flags (IQR, Z-score)
  features.py              engineer_features() + FeatureEngineer(BaseEstimator, TransformerMixin)
  preprocessing.py         build_preprocessor(num, cat, imputer, scaler, encoder, outlier_rule) -> ColumnTransformer; OPTIONS dicts
  split.py                 make_split(stratify=True) + persisted indices
  models.py                MODEL_ZOO (LogReg, KNN, GaussianNB, SVC, DecisionTree, RandomForest, HistGradientBoosting, MLP) + grids
  tuning.py                GridSearchCV / RandomizedSearchCV with StratifiedKFold(5, shuffle, seed), multi-metric scoring
  evaluation.py            metrics (accuracy, balanced acc, precision, recall/TPR, specificity/TNR, F1, ROC-AUC, PR-AUC, MCC, Brier),
                           confusion, ROC/PR/calibration points, threshold_sweep(), compare_models(), paired tests
  explain.py               permutation importance, coefficients, SHAP (lazy import)
  persistence.py           save/load with manifest
  plots.py                 matplotlib/seaborn figure builders shared by notebook, report and app
  cli.py                   mlda-download, mlda-train, mlda-zip
app/
  streamlit_app.py         entrypoint: sys.path shim, set_page_config, st.navigation, shared sidebar
  requirements.txt         runtime-only export for Cloud
  _common.py               cached loaders (st.cache_data / st.cache_resource), Altair helpers
  pages/1_Data_Explorer.py 2_EDA.py 3_Preprocessing.py 4_Model_Training.py 5_Inference.py 6_Evaluation.py 7_About.py
tests/
  conftest.py + fixtures/sample.csv
  test_load_join.py test_schema.py test_cleaning.py test_features.py test_preprocessing.py test_split.py
  test_pipeline_fit_predict.py test_evaluation.py test_persistence.py test_artifacts.py
  test_app_smoke.py (AppTest over all pages)  test_notebook.py (slow: nbconvert --execute)
docs/                      report_plan.md, architecture.md, dataset_research.md, data_profile.md
```

## Tests mapped to the rubric

| Rubric | Tests |
|---|---|
| A Load + joins | row count preserved on left join; `validate="many_to_one"` raises on duplicated keys; unmatched keys below documented threshold; raw-file checksums match data/README.md |
| B EDA | describe helpers return mean/std/var for all numerics; correlation matrix square, symmetric, unit diagonal; figure builders return Figure objects and PNGs exist |
| C Preparation | duplicates removed; impossible values -> NaN; types coerced; IQR/Z-score flags correct on a hand-built frame; no NaN after transform; scaler output mean ~0 / std ~1 or in [0, 1]; one-hot column count; every OPTIONS combination builds; feature formulas; stratification preserved within 1 pp; train/test indices disjoint; saved indices reproducible |
| D Training | each MODEL_ZOO entry fits on 300 rows, predict_proba in [0, 1] and sums to 1; tune() returns in-grid params; imblearn pipeline leaves test rows untouched |
| E Evaluation | metrics vs hand-computed 2x2 matrix; threshold sweep monotonic recall; comparison columns; committed metrics JSON validates and reloads under the installed sklearn |
| F Deployment | AppTest runs the entrypoint and every page with no exception; inference form renders a probability; CI docker job builds and curls `/_stcore/health` |
| G/H | `uv lock --check`; export diff; fresh venv pip install on Windows and Ubuntu; `jupyter nbconvert --to notebook --execute` under 10 minutes |

## Memory and speed budget for the app

Processed parquet <= 100 MB in memory (float32, category dtypes, <= 300k rows); each joblib <= 20 MB, total models <= 60 MB;
RandomForest capped (n_estimators <= 200, min_samples_leaf >= 5 or max_depth <= 12); SVC on a stratified sample <= 20k rows;
KNN with one-hot `min_frequency` to cap width; `st.dataframe` <= 10k rows; charts on <= 20k-row samples; SHAP imported lazily on
the inference page only; seaborn imported only where `st.pyplot` figures are built.

## Commit plan (real dates, teaching-week labels in messages)

| Week | Commits |
|---|---|
| 12-18 Oct (TW6, built now) | TW1 init (uv, lock, licence, CI); TW1 package skeleton; TW2 download + join + data card; TW2 EDA I; TW3 EDA II + figures; TW3 cleaning + schema tests; TW4 preprocessing, split, LogReg baseline, evaluation; TW5 KNN, NB, SVM + GridSearch |
| 19-25 Oct | TW6 RF + HistGradientBoosting, comparison, persistence; Deploy: Streamlit entrypoint + first pages + AppTest; first Cloud deploy |
| 26 Oct-1 Nov | TW7 Preprocessing page; KMeans profile in EDA |
| 2-8 Nov | TW8 Training + Inference pages |
| 9-15 Nov | TW9 imbalance: class_weight vs SMOTE in imblearn Pipeline (CV only), PR-AUC, threshold analysis; Evaluation dashboard |
| 16-22 Nov | TW10 MLP candidate, calibration; notebook re-executed; figures regenerated |
| 23-29 Nov | TW11 Dockerfile, CI docker build, notebook test, README run instructions; redeploy |
| 30 Nov-4 Dec | TW12 final metric/figure sync, tag v1.0.0, mlda-zip; README polish |

All code is produced in this session; the later weekly commits are real extension and polish commits, so the history is honest.

## Deployment

Primary: Streamlit Community Cloud. Needs a public GitHub repo (ID-based username), `app/streamlit_app.py`, `app/requirements.txt`,
`.streamlit/config.toml` at the root, committed `data/processed`, `models`, `reports`. User steps: sign in at share.streamlit.io with
GitHub, Create app, pick repo/branch/`app/streamlit_app.py`, Advanced settings -> Python 3.12, Deploy, copy URL into README and report.
The app sleeps after 12 h idle; open it before marking and the viva.
Fallback A: Render free web service (reads `uv.lock` and `.python-version`; start command
`streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`).
Fallback B: Hugging Face Spaces Docker SDK (paid plan now required for personal accounts).
Always available: `uv run streamlit run app/streamlit_app.py` or `docker run -p 8501:8501 mlda-cw1`.

## Risk register

| Risk | Mitigation |
|---|---|
| Cloud Python drift / undocumented uv.lock handling | Pin 3.12 in the dialog; `app/requirements.txt` beside the entrypoint; deploy early |
| Cloud out-of-memory | budgets above; no plotly/xgboost; precomputed artefacts |
| Dataset > 50 MB in git | filtered zstd parquet; `mlda-download` with SHA-256 for raw zips; no LFS |
| pandas 3 vs seaborn | `pandas<3` |
| Non-reproducible results | RANDOM_STATE = 42 everywhere; persisted split indices; manifest with versions and git SHA; CI on ubuntu + windows |
| Windows/Linux path and encoding differences | pathlib, `encoding="utf-8"`, forward slashes, `.gitattributes eol=lf`, exact case |
| Notebook > 10 min | small grids, `n_jobs=-1` locally, SVC on a sample, HGB early stopping |
| Leakage | split first; all fitting inside Pipeline; FeatureEngineer in the Pipeline; tests for disjoint indices, duplicate rows across the split, |corr| ~ 1 features; test predictions built only from the held-out set |
| Imbalance handled wrongly | SMOTE only inside imblearn Pipeline; compare with class_weight; report PR-AUC, TPR/TNR, threshold analysis |
| Similarity to online solutions | non-Kaggle source, two-table join, own target and features, cost-based threshold, deployed app; related theses cited and differentiated |
| Name leakage | `mlda-zip` asserts the name is absent from report, notebook metadata, paths; git identity uses the ID only |
