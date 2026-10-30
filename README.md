# Predicting lost-time mine injuries from MSHA open data

Module 6COSC017C Machine Learning and Data Analytics, Coursework 1, student ID 13130.

When a mine operator files an injury on US Form 7000-1, can the case be flagged as a likely lost-time injury on receipt of the form? If it can, the insurer or self-insured operator can open the reserve, the return-to-work plan and medical case management earlier. The data are three open files from the US Mine Safety and Health Administration (MSHA), joined into one table of 101,609 injured persons from 2010 to 2025. Of these, 48.6 % had a lost-time outcome.

The final model is a histogram gradient boosting classifier (`hist_gb`), ranked first by cross-validated PR-AUC. On the 20,320 held-out rows it reaches ROC-AUC 0.843 and PR-AUC 0.842 (`reports/metrics/comparison.csv`). The lookup baseline, which predicts the training-set lost-time rate of each nature of injury x body part x sector cell, reaches ROC-AUC 0.804.

The study reports two operating points separately. The capacity threshold, 0.211, is set on out-of-fold probabilities for specificity 0.40 on all training rows. On all test filings it flags 76.3 % of reports, with recall 0.941, specificity 0.406 and precision 0.600 (`reports/metrics/comparison_operating_point.csv`). The recommended policy is the workload rule: review the three in four reports with the highest scores among those still open at filing (see "Method in brief").

## Deliverables

| Deliverable | Location | How it is produced |
|---|---|---|
| Report | submitted as a Word document through the module intranet; not kept in this repository | |
| Notebook | `notebooks/01_end_to_end.ipynb`, executed, with an HTML export beside it | executed with `jupyter nbconvert --execute`; the committed run used `FULL_RUN=1`, so all eight searches ran inside the notebook |
| Streamlit app | `app/streamlit_app.py`, seven pages: data explorer, exploratory analysis, preprocessing, model training, inference, evaluation, about. The inference page checks the `MINE_ID` before it scores. A malformed id is refused with an error and nothing is scored. A well-formed id that the reference data do not know gives a warning, and scoring continues from the values entered by hand. A known mine without previous-year production gives a notice. The evaluation page has a Capacity policy tab with the capacity-threshold and workload-rule tables | `uv run streamlit run app/streamlit_app.py` |
| Live app | Deployed at: [https://mlda-cw1-8han4medy3o7d8izzcywbm.streamlit.app](https://mlda-cw1-8han4medy3o7d8izzcywbm.streamlit.app) | see "Deployment" |
| Repository | `https://github.com/wiut13130/mlda-cw1` (GitHub) | shared code in `src/mlda`, imported by the notebook and the app |

The repository holds the practical part of the coursework: the package, the notebook, the app, the data, the trained models, their results and the tests. Its commit history shows how each artefact was produced.

## Data

Three files from MSHA Open Government Data (https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp):

| File | Rows x columns | One row is |
|---|---|---|
| `Accidents.zip` | 275,220 x 57 | one injured person on one Form 7000-1 |
| `Mines.zip` | 92,060 x 59 | one mine, with its attributes at the snapshot date |
| `MinesProdYearly.zip` | 658,922 x 11 | employee-hours for one mine, calendar year and subunit |

Each row of the analytic table is one injured person from the accident file. The mine's attributes, its previous-year exposure and its case history are joined on.

MSHA published the three archives on 9 October 2026, and they were downloaded on 14 October 2026 at 12:06 UTC. MSHA refreshes the files every week, so the SHA-256 of every file is pinned in `data/raw/SNAPSHOT.json` and in `mlda.config.SOURCES`, and `mlda-download` refuses a file whose hash differs. The three archives (66 MB) are committed, so every clone analyses the same snapshot without network access. The extracted `.txt` files (about 330 MB) are git-ignored and recreated by `mlda-download`.

Licence: the files are works of the US federal government and are not subject to copyright in the United States (17 U.S.C. 105), so they are in the public domain. Citation: Mine Safety and Health Administration (no date) *MSHA Open Government Data Sets: Accidents, Mines and MinesProdYearly* [data files, version published 9 October 2026]. US Department of Labor. Available at: https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp (Accessed: 14 October 2026).

The data card, [`data/README.md`](data/README.md), gives the sources, keys, joins, target definition and leakage drop-list.

## Reproduce

Run every command from the repository root. Python 3.12 is required (`requires-python = ">=3.12,<3.13"`), and every stochastic call uses `RANDOM_STATE = 42` (`mlda.config`). The repository already holds the data archives, processed tables, tuned models, metrics and figures of the full run, so the notebook, the app and the tests start without retraining. The commands are the same on Windows (PowerShell) and Unix except where marked.

### Route A: uv (locked, preferred)

```
pip install uv
uv python install 3.12
uv sync --locked --all-groups       # builds .venv from uv.lock and fails if the lock is stale
uv run mlda-download                # extracts the committed archives and verifies each SHA-256
uv run mlda-train                   # optional: full study, rebuilds data/processed, models/, reports/
uv run mlda-train --quick           # optional: two-point grids, 10,000 training rows, 3 folds
uv run jupyter lab notebooks/01_end_to_end.ipynb
uv run streamlit run app/streamlit_app.py
uv run pytest -m "not slow"         # fast suite
uv run pytest                       # also executes the notebook and the quick end-to-end runs
```

Both `mlda-train` modes write into `data/processed`, `models` and `reports`, so a quick run overwrites the committed full-run artefacts with smaller-grid results. `git restore data/processed models reports` brings the committed versions back.

### Route B: plain pip

```
python3.12 -m venv .venv            # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt     # pinned export of uv.lock, all dependency groups
pip install -e . --no-deps          # installs the mlda package and its console scripts
mlda-download
mlda-train --quick
jupyter lab notebooks/01_end_to_end.ipynb
streamlit run app/streamlit_app.py
pytest -m "not slow" -k "not app_smoke"
```

`pip install -e . --no-deps` puts the console scripts `mlda-download` and `mlda-train` into the virtual environment. `requirements.txt` (all groups) and `app/requirements.txt` (runtime only, read by Streamlit Community Cloud) are exported from `uv.lock` with these commands:

```
uv export --format requirements-txt --all-groups --no-hashes --no-emit-project -o requirements.txt
uv export --format requirements-txt --no-dev --no-hashes --no-emit-project -o app/requirements.txt
```

The last command of route B selects 382 of the 472 tests (`pytest --collect-only -m "not slow" -k "not app_smoke"`). That count was taken in the uv environment on 20 October 2026; check it on a fresh clone before quoting it. The 73 Streamlit app smoke tests that `-k "not app_smoke"` leaves out run in route A, which has 483 fast tests in total.

### Route C: Docker

```
docker build -t mlda-cw1 .
docker run --rm -p 8501:8501 mlda-cw1     # then open http://localhost:8501
```

The image starts from `python:3.12-slim`, installs the locked runtime dependencies with uv, copies the committed artefacts the app serves, and defines a health check on `/_stcore/health`. The container listens on the port in the `PORT` variable if a host sets one, and on 8501 otherwise; the health check reads the same variable. On a host with 512 MB of RAM, add `-e MLDA_LOW_MEMORY=1` (see "Low-memory mode"). The development machine has no Docker, so the image itself has not been built there. The `CMD` was checked by starting Streamlit through `sh -c` with and without `PORT`.

### Expected run times (8 cores)

| Step | Time |
|---|---|
| `uv run mlda-train --quick` | about 4 minutes |
| `uv run mlda-train` | about 86 minutes (85.9 minutes, 5,154 s, `total_seconds` in `reports/metrics/run_manifest.json`) |
| Notebook, default mode (searches the three cheap models and compares them with the stored tables within 0.001, loads the other five tables, refits every selected pipeline) | under 10 minutes; 7.4 minutes (445 s, section F.6) in a default run on 15 October 2026, before the rerun of 20 October and not repeated since |
| Notebook with `FULL_RUN=1` (repeats all eight hyper-parameter searches; the committed notebook was executed this way) | 41.4 minutes (2,484 s, section F.6) in the committed run; the SVM search alone takes about 18 minutes (1,061 s) |
| `uv run python -m mlda.followup` (follow-up checks, including three wider grids) | about 31 minutes (1,879 s); a few minutes with `--skip-grids` |
| `uv run pytest -m "not slow"` | about two minutes (455 passed, 17 deselected) |

The notebook reads `FULL_RUN` from the environment. On Unix:

```
FULL_RUN=1 uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/01_end_to_end.ipynb
```

In PowerShell, run `$env:FULL_RUN = "1"` first and then the same command without the prefix. The cell timeout has to exceed the time the searches of section D.1 take, which was 39.8 minutes in the committed run. Afterwards, refresh the HTML export with `uv run jupyter nbconvert --to html notebooks/01_end_to_end.ipynb`.

## Repository layout

```
src/mlda/            shared package, imported by the notebook, the app and the tests
  config.py          PROJECT_ROOT, paths, RANDOM_STATE = 42, pinned sources and SHA-256, cost ratios
  data/              download (checksum), load (typed readers), join (merge_sources), schema checks
  cleaning.py        error-correction rules, IQR and Z-score outlier flags
  features.py        FeatureEngineer: the 52 model inputs
  preprocessing.py   ColumnTransformer builder: imputer, scaler, encoder, outlier options
  split.py           grouped stratified train/test split and CV splitter, persisted indices
  models.py          the eight classifiers and their search grids
  tuning.py          GridSearchCV / RandomizedSearchCV, out-of-fold probabilities
  evaluation.py      metrics, confusion matrix, threshold sweep, cost model, McNemar and paired tests
  baselines.py       reference rules (majority, all positive, lookups)
  operating.py       out-of-fold thresholds and the success criteria S1-S6
  sensitivity.py     altered-input, history-audit and mine-grouped refits
  followup.py        checks added after the test scoring (python -m mlda.followup); they change no choice
  explain.py         permutation importance, coefficients, SHAP
  persistence.py     save and load models with a provenance manifest
  plots.py eda.py    figure builders and descriptive statistics
  pipeline.py        train_all(): the whole study
  cli.py             mlda-download, mlda-train
app/                 Streamlit entrypoint, _common.py loaders, pages/1_..7_, requirements.txt for Cloud
notebooks/           01_end_to_end.ipynb and its HTML export
tests/               pytest suite and 300-row fixtures
data/                README.md (data card), raw/ (archives, codebooks, SNAPSHOT.json), processed/ (parquet, split indices)
models/              eight tuned pipelines (.joblib) and manifest.json (versions, git SHA, data checksum)
reports/             metrics/ (JSON, CSV), cv/ (search results), predictions/, figures/ (PNG)
docs/profile/decisions.json   decision log of the data profiling: filters, target, features, error rules, split, success criteria
Dockerfile           container image of the app
```

## Method in brief

### Prediction moment and cohort

The score is computed when Form 7000-1 arrives, at its filing date (a median of 8 days after the accident), from the fields that the form and the mine records hold on that date. At filing, 45.0 % of in-scope reports already carry a return-to-work date and are not scored. The operational cohort is the 55 % still open at filing: 11,290 of the 20,320 test reports, 71.2 % of them lost-time. The extract holds only the final degree code and cannot show what the filer knew at filing. The study is therefore retrospective, and the score needs a prospective trial before anyone relies on it.

### Operating points and the workload rule

The capacity threshold (0.211) is the lowest out-of-fold threshold with specificity 0.40 on all training rows. It flags 76.3 % of all test filings, but it would flag 90.0 % of the open reports, more than a review capacity of three in four. The recommendation is therefore the workload rule: review the top three in four open-at-filing reports by score.

The deployed cut-off, 0.376, is the 25th percentile of the out-of-fold scores of the open training reports, so no test row is used to set it. On the 11,290 open test reports it flags 74.3 % and finds 0.850 of the lost-time cases. At exactly the top 75 % of open reports by test score, recall is 0.855 and precision 0.812. The MLP reaches the same recall (0.855); the lookup baseline reaches 0.828 and random selection 0.75.

The probabilities are calibrated to all filings and run 0.12 below the observed rate on open reports (mean predicted 0.592 against 0.712). The rule uses only the ranking. A paired mine-year bootstrap puts HGB 0.027 ahead of the lookup baseline in recall under the rule (95 % interval 0.021 to 0.033) and level with the MLP (0.000, interval -0.003 to 0.004). An HGB fitted on open-at-filing training rows alone reaches recall 0.877 under the rule, so a prospective trial should fit on open reports.

### Filters and target

Accident years 2010 to 2025 and degree-of-injury codes 01 to 06 are kept. `lost_time` is 1 for fatality, permanent disability, days away, or days away and restricted activity (codes 01 to 04), and 0 for restricted activity only or no days away (05, 06). N = 101,609 and prevalence is 48.6 %.

### Joins

All joins are left joins from the accident rows, each with `validate="many_to_one"` and an asserted row count. Mines joins on `MINE_ID` (100.0 % matched), mine-year production on `MINE_ID` and the previous calendar year (98.8 %), and counts of earlier cases filed before each accident complete the table.

### Cleaning

Thirty error rules (E1 to E30) are documented in `docs/profile/decisions.json`. The 17 implemented in `mlda.cleaning` are logged with row counts in `reports/metrics/error_rules.csv`.

### Features

The models get 52 inputs, which become 546 columns after the default one-hot preprocessor. The three Mines-snapshot flags (safety committee, Part 48 training, portable operation) are not inputs, because a snapshot of the Mines file cannot show their value on the date of each accident. The location and commodity fields stay, and the Section 103(i) status is dated to the accident (`status_103i_at_accident`). Columns that encode the outcome (`DAYS_LOST`, `DAYS_RESTRICT`, `RETURN_TO_WORK_DT`, `CLOSED_DOC_NO` and others) are excluded.

### Split

The split uses `StratifiedGroupKFold` on mine-year groups (32,732 groups), because 82 % of rows share a mine-year with another row. One fold is the 20 % test set (20,320 rows). The 81,289 training rows are cross-validated with five grouped stratified folds. Imputers, scalers, encoders and models are fitted inside scikit-learn Pipelines on training rows only.

### Models

Eight classifiers are compared: logistic regression, k-nearest neighbours, Gaussian naive Bayes, decision tree, random forest, histogram gradient boosting, RBF support vector machine and multi-layer perceptron. Seven are tuned with `GridSearchCV` and the random forest with `RandomizedSearchCV`. Four reference rules serve as baselines.

### Thresholds

Operating thresholds come from out-of-fold training probabilities: specificity 0.40 on all training rows (the review-capacity assumption), a 50 % flag rate, and the cost-optimal point for cost ratios from 2:1 to 30:1. The workload rule's cut-off comes from the out-of-fold scores of the open training reports. The test rows are only scored against these thresholds.

### Checks

A temporal split (tune on 2010 to 2019, score 2020 to 2025) gives `hist_gb` ROC-AUC 0.834. Dropping the one pure level, `NATURE_INJURY_CD=100`, costs 0.003 ROC-AUC, and a split grouped by mine gives 0.838. McNemar's test against the lookup rule gives p < 0.001, and a paired fold test of the top two models gives corrected p = 0.029.

Five altered-input refits of `hist_gb` with the stored hyper-parameters (`without_immed_reportable`, `without_status_103i`, `resolved_lag_counts`, `exposure_blank_if_q4_incomplete` and `history_without_test_outcomes`) move test PR-AUC by between -0.002 and +0.001. The last of these is a history audit. 14.1 % of the training rows (11,426, of which 11,416 have a non-blank history) carry a previous-year case history drawn from a mine-year that is in the test split. Refitting with that history set to missing changes test PR-AUC by less than 0.001 (+0.0006). The temporal split has no such overlap, because every training row precedes every test row.

### Follow-up checks

These checks ran after the test scoring, so they are reported as checks and changed no choice. `uv run python -m mlda.followup` writes `reports/metrics/followup_checks.json`, the three tables `followup_capacity_policy.csv`, `followup_workload_rule.csv` and `followup_open_calibration.csv` in `reports/metrics`, and `reports/cv/followup_*_cv_results.csv`.

They cover:

- wider grids for KNN, the random forest and `hist_gb`, which gain 0.000, 0.001 and 0.002 validation PR-AUC over the stored searches;
- logistic regression given the joint nature x body part x sector cell, with cross-validated PR-AUC 0.829 against 0.820 without it and 0.838 for `hist_gb`;
- S5 among the top 50 % of test scores, and recall at one equal workload;
- the capacity threshold and the exact top 50 % and top 75 %, compared across HGB, the MLP, the lookup baseline and the hand rule, on all test rows and on the 11,290 reports still open at filing (ROC-AUC 0.812);
- the workload rule's cut-off and its test result, and calibration on the open reports;
- the paired mine-year bootstrap of the comparison, and the refit on open-at-filing training rows alone.

### Success criteria

The criteria were fixed before modelling and are evaluated for `hist_gb` on the held-out test rows (`reports/metrics/success_criteria.json`):

| # | Criterion | `hist_gb` | Result |
|---|---|---|---|
| S1 | ROC-AUC >= 0.80 | 0.843 | pass |
| S2 | PR-AUC >= 0.75 | 0.842 | pass |
| S3 | recall >= 0.92 at specificity 0.40 | 0.941 (specificity 0.406) | pass |
| S4 | cost per 1,000 cases below the review-everything floor (102,825) and the lookup rule at 2:1, 3:1, 5:1 | 65,817, 76,781, 87,205 (lookup 73,553, 85,472, 98,061) | pass |
| S5 | recall >= 0.78 at the out-of-fold threshold for a 50 % flag rate (planned as: among the 50 % of cases ranked highest) | 0.777 at a 50.7 % flag rate; 0.770 among the top 50 % of test scores | **fail** |
| S6 | Brier score <= 0.20 | 0.161 | pass |
| S7 | McNemar and paired fold test reported | both reported | pass |

S5 is missed by 0.003. The multi-layer perceptron misses it too, with 0.777 at a 51.1 % flag rate and 0.769 among the top 50 % of test scores (the planned form of S5), against 0.770 for `hist_gb`. No candidate passes all seven criteria. `hist_gb` stays the final model because the selection rule, cross-validated PR-AUC, was fixed before any test score was computed and ranks it first (0.838 against 0.831). The choice was not revised after the test result.

S3 was set for all filings and is met at the capacity threshold. Under the workload rule, recall on open reports is 0.850 (0.855 at exactly the top 75 % by test score), which is below that target. At cost ratios of 10:1 and above the model saves little compared with reviewing every case (4.6 % at 10:1). At 20:1 and 30:1 its cost-optimal threshold costs 0.5 % and 0.3 % more than reviewing every case on the test rows. The model is most useful for ranking cases when review capacity is limited.

## Tests

`pytest` is configured in `pyproject.toml` (`tests/`, 900-second timeout per test, marker `slow`). The 472 tests cover:

- the loaders and joins (300-row fixtures plus counts on the real data), the schema, and the download checksums;
- each error-correction rule on hand-built rows, the IQR and Z-score flags, the feature formulas and the leakage guards;
- every preprocessing option, train-only fitting, and the grouped split (disjoint indices, prevalence within 1 percentage point, reproducibility);
- the model zoo and tuner, metrics against hand-computed confusion matrices, the threshold and cost model, persistence round trips, figure builders and the command-line parsers;
- the follow-up checks (the workload rule's cut-off, the open-cohort calibration bins, the paired bootstrap and the open-only refit), each on hand-built frames;
- a Streamlit `AppTest` run over every page, including the `MINE_ID` validation of the inference page, a live search with the preprocessing chosen on the Preprocessing page, and the caps of low-memory mode.

Tests that need the extracted MSHA files or the committed artefacts skip themselves when those are absent, so run `mlda-download` first.

Seventeen tests carry the `slow` marker. Ten are notebook tests, which check the committed notebook's headings and wording and execute a temporary copy within 15 minutes; the other seven train models through the quick pipeline. `-m "not slow"` leaves them out, and 455 tests remain. Lint with `uv run ruff check src tests app notebooks`.

## Commit history

The commits dated 14 and 15 October 2026 were made in one two-day sprint. Each message names the teaching-week milestone of the brief whose content the commit adds, not the week it was written in. The commits labelled TW7, TW11 and TW12 on those dates (for example `1b32d5c`, `392da03`, `b884487` and `305d012`) therefore delivered their work ahead of the milestone, and the labels are not in date order. No commit date has been changed. From the week of 19 October, each block of work is committed and pushed in the week it is done, with its real date.

On 15 October 2026 the history was rewritten once so that every commit carries the student-ID author identity. The rewrite kept every file, message and date but changed the commit hashes. The manifests of the 20 October 2026 rerun (`models/manifest.json`, `reports/metrics/run_manifest.json`) record the commit that was current when each file was saved: `613d30e` for the first model fitted and `95c56ba` for the rest, both in the present history. The feature change committed in `95c56ba` was already in the working tree when the run started. The run's own log is `reports/logs/train_full.log`.

## Deployment

### Streamlit Community Cloud

The live app runs on Streamlit Community Cloud, deployed from this repository:

1. Sign in at https://share.streamlit.io with the GitHub account and choose "Create app".
2. Pick the repository `wiut13130/mlda-cw1`, the branch `main` and the entrypoint `app/streamlit_app.py`.
3. Under "Advanced settings" select Python 3.12. Cloud ignores `.python-version`.
4. Deploy. Cloud installs `app/requirements.txt`, the runtime-only export beside the entrypoint.
5. Put the URL in the "Deployed at" line above.

The app sleeps after about twelve hours without visitors and wakes on the next visit, so open it shortly before a demonstration or marking. Locally, `uv run streamlit run app/streamlit_app.py` or the Docker route serves the same app at http://localhost:8501.

### Low-memory mode

`app/_common.py` reads `MLDA_LOW_MEMORY` once at import; `1`, `true` or `yes` switches it on. In this mode the Model training page samples at most 5,000 rows (default 3,000; otherwise up to 20,000) and tunes with 2-fold instead of 3-fold cross-validation, using one worker in both modes. The Preprocessing sample is capped at 5,000 rows (otherwise 10,000), the cache keeps one fitted model and one design matrix (otherwise four and two), and the Model training page shows a note. The saved models and the precomputed pages are the same in both modes.

Resident memory of the server process was read with psutil on the development machine (Windows, Streamlit 1.65, `uv run streamlit run app/streamlit_app.py`, all seven pages opened in one session). It was 458 MB without the switch and 462 MB with it. During one live random-forest search the peak was 637 MB at 20,000 rows without the switch and 550 MB at 5,000 rows with it. These figures come from Windows, not from a Linux container. The pages alone use close to 512 MB, and a live search in low-memory mode can exceed that on this machine, so keep the local app ready for a live search during the viva.

## Licence and attribution

The code is released under the MIT licence (`LICENSE`). The data are public-domain works of the US Department of Labor, Mine Safety and Health Administration; please attribute them as in the citation above. The repository, the report and the notebook identify the author by student ID 13130 only.
