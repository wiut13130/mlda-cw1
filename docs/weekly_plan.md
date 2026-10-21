# Weekly plan to the 4 December 2026 deadline (student ID 13130)

Written on Wednesday 14 October 2026, teaching week 6. The commits made so far are dated 14 and 15 October 2026; their messages name the teaching-week milestone each one covers (so TW7, TW11 and TW12 labels appear before some TW3 and TW6 ones), and the dates are not changed. The README section "Commit history" says so. Tag each milestone when its block is committed (for example `git tag tw8-batch-scoring`) and push the tags. Each block below adds one real piece of work with its own tests, so the later commits carry later dates honestly. Message format: `TW<n>: <what changed and why>`. The git identity is the GitHub account `wiut13130 <116626983+wiut13130@users.noreply.github.com>` (set in .git/config); keep it for every commit.

Checks before every push:

```
uv run ruff check src tests app report notebooks
uv run pytest -m "not slow"
uv lock --check
uv run python report/build_report.py --check
```

`uv run mlda-train --quick` and `--models` write into the committed `data/processed`, `models` and `reports`. After a trial run, look at `git status` and `git diff --stat reports/metrics`, and restore with `git restore data/processed models reports` unless the change is intended. Commit regenerated artefacts only from a full `uv run mlda-train` (about 95 minutes).

## Before the first block (by Sunday 18 October)

- [x] Dataset reserved in the module Discussions forum on 10 October 2026 (source URL https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp). Keep the topic title and URL to hand for the viva.
- [x] Public repository: https://github.com/wiut13130/mlda-cw1 (account `wiut13130`; every commit carries that identity and the noreply address). The URL is in README and in `repo_url` of `report/report_meta.json`.
- [x] CI runs on GitHub Actions on every push (seven jobs: lint and tests on Ubuntu and Windows, lock and export check, plain pip installs, notebook execution, Docker build). Run the checks above by hand before each push anyway.

## Week of 19 October (mid-term week): deploy

Commit: the deployment was made on 15 October (TW12 commits); this week's commit records the live-app measurements.

- [x] `app/`, `notebooks/` and `tests/` committed with their tests (14 October).
- [x] Deployed on Streamlit Community Cloud on 15 October: https://mlda-cw1-8han4medy3o7d8izzcywbm.streamlit.app (entrypoint `app/streamlit_app.py`, Python 3.12). Render (Docker, free plan, `MLDA_LOW_MEMORY=1`, health check `/_stcore/health`) stays documented in README "Deployment" as the fallback.
- [ ] Visit all seven pages on the live URL. Read the resident memory on the About page (it uses psutil) and time a cold start after the app has slept. Add both numbers to "Memory and speed budget" in `docs/architecture.md` and beside the local figures in README "Deployment" (458 MB without and 462 MB with `MLDA_LOW_MEMORY` after seven pages, measured on Windows). If a live search restarts the app, lower the caps in `app/_common.py` and keep the live search for the local app.
- [x] README "Deployed at" line, `repo_url` and `app_url` in `report/report_meta.json` set (15 October).
- [x] Forum reservation date (10 October 2026) recorded in `report/sections/01_introduction.md`; report rebuilt.
- [x] `report/sections/90_references.md` exists and renders as the reference list.

## Week of 19 October: response to an external review (started 20 October)

A strict external marking of the 15 October submission (estimate 83/100) was checked item by item; every point below was verified against the artefacts before it was changed.

- [x] Prediction moment defined once (receipt of Form 7000-1, its filing date); reports that already carry a return-to-work date are not scored; the open-at-filing cohort is the operational population; the study is stated as retrospective with prospective validation as the condition for use (report Sections 1.1, 4.4, 5, model card).
- [x] The capacity threshold (0.210) flags 90.0% of open-at-filing test reports, so the recommendation became a workload rule (top three in four open reports by score) evaluated at equal workloads for HGB, the MLP, the lookup baseline and the hand rule (`reports/metrics/followup_capacity_policy.csv`, report Table "Equal-workload capacity policy").
- [x] Three Mines-snapshot flags (SAFETY_COMMITTEE_IND, PART48_TRAINING, PORTABLE_OPERATION) removed from the model inputs (52 remain); full study rerun with `uv run mlda-train`, `python -m mlda.followup` and the notebook with `FULL_RUN=1`.
- [x] History audit: 14.1% of training rows carry a previous-year history drawn from a test mine-year; sensitivity variant `history_without_test_outcomes` refits HGB without them; the temporal split is the leakage-free check.
- [x] Statistical wording corrected (mean and standard deviation under skew, raw values and trees, association as a leakage screen, specificity and workload, Brier score and calibration); S5 random reference 0.50.
- [x] Report build: static contents list cached in the TOC field with fields updated on opening, repeated table headers, rows kept together, landscape pages for wide tables, short caption titles in the lists of figures and tables.
- [x] App: MINE_ID validated (malformed input refused, unknown mine warned, batch rows checked).
- [ ] Follow-up checks extended: out-of-fold cut-off for the workload rule, calibration on the open cohort, open-only refit, paired mine-year bootstrap of the equal-workload comparison; numbers written into the report, notebook re-executed, README and viva pack refreshed, submission zip rebuilt.
- [ ] You: export the report PDF from Word after the rebuild and keep making one dated commit per week with real changes until 4 December (the review awards the weekly-commit marks only for dated progress).

## Week of 26 October (TW7): mine-profile view on the EDA page

Commit: `TW7: k-means mine profiles on the exploratory analysis page`

- [ ] Add a function to `src/mlda/eda.py` that clusters mine-year rows of the training split on standardised exposure and history features already in `features.parquet` (`prev_hours`, `prev_employees`, `prev_incidents`, `prev_lost_time_share`). Use a Pipeline of `StandardScaler` and `KMeans(n_init=10, random_state=42)`, fitted on training rows only.
- [ ] Choose k from 2 to 8 by silhouette score on a 10,000-row sample (silhouette is quadratic in rows) and show the inertia curve beside it.
- [ ] Show, on `app/pages/2_EDA.py`, the cluster sizes, the lost-time rate per cluster and a PCA scatter coloured by cluster, reusing the pattern in `plots.pca_projection`. The target describes the clusters and never builds them.
- [ ] Tests in `tests/test_eda.py`: same labels for the same seed, the cluster count asked for, no test-split index used. `tests/test_app_smoke.py` must still visit the page without an exception.
- [ ] Say on the page that the clusters are descriptive and not a model input. If the lost-time rates differ little between clusters, write that.

## Week of 2 November (TW8): batch scoring on the inference page

Commit: `TW8: batch scoring report with row validation and a review list by capacity`

- [ ] Today the page offers a template CSV, an upload limited by `MAX_BATCH_ROWS`, three summary metrics and a download with `probability`, `flag_operating_point` and `flag_at_0_5`. `prepare_batch` (`app/_common.py`) refuses the whole file when a required column is missing, the file is too long or any `ACCIDENT_DT` is invalid. Change it to score the valid rows and show a table of the rejected rows with the reason (unparsable date, `MINE_ID` absent from the Mines table).
- [ ] Add a capacity control: review the top k % of cases. Show the probability cut-off, the number of rows flagged and the recall at that flag rate from the stored held-out predictions, so the S5 trade-off is visible to the user.
- [ ] Add `rank` and `flag_at_capacity` to the downloadable CSV.
- [ ] Tests: `prepare_batch` on a frame with one bad date and one unknown mine; an `AppTest` run that uploads a three-row CSV. The page loads the saved pipeline and fits nothing.

## Week of 9 November (TW9): class weights and SMOTE

Commit: `TW9: class_weight and SMOTE comparison for the selected model, recorded in sensitivity.json`

- [ ] Prevalence is 48.6 %, so little gain is expected. The commit records the measurement and states the result whatever it is.
- [ ] Add an `imbalance` block to `mlda.sensitivity` for `hist_gb` (`class_weight="balanced"`) and for `logreg`, compared with the unweighted fit. SMOTE sits inside an `imblearn` Pipeline after the preprocessor, so it resamples training folds only; note that SMOTE on one-hot columns produces fractional dummy values.
- [ ] Same protocol as the other variants: grouped five-fold cross-validation on the training rows, out-of-fold threshold for specificity 0.40, one scoring of the test rows. Report ROC-AUC, PR-AUC, recall at that threshold and Brier score; Brier shows the loss of calibration that resampling causes.
- [ ] Test on the synthetic base in `tests/test_sensitivity.py`: SMOTE changes the training rows inside the pipeline and leaves the rows passed to `predict_proba` untouched.
- [ ] Regenerate artefacts from one full `uv run mlda-train`, then add one sentence to README "Method in brief" and to the report if its numbers change.

## Week of 16 November (TW10): MLP calibration study

Commit: `TW10: calibration study for the multi-layer perceptron and its effect on the cost thresholds`

- [ ] Evidence: the MLP is second on cross-validated PR-AUC, passes all seven criteria, and has a Brier score of 0.163 against 0.161 for `hist_gb`. Its out-of-fold cost-optimal threshold at 2:1 is 0.299 against 0.335 for `hist_gb`, so the two models place probabilities on different scales; the study tests whether calibration closes that gap.
- [ ] Fit `CalibratedClassifierCV` with `method="sigmoid"` and `method="isotonic"` around the tuned MLP pipeline on training rows, with explicit mine-year-grouped splits (`CalibratedClassifierCV` is already used for the SVM in `models.py`).
- [ ] Compare reliability tables (`evaluation.calibration_table`), Brier score, log loss and the out-of-fold cost-optimal thresholds before and after. A monotone sigmoid leaves ROC-AUC unchanged; isotonic can introduce ties, so report both.
- [ ] Selection stays as it was: `hist_gb` is the final model. The calibrated MLP is reported as an extra, post hoc candidate. Do not change the final model after seeing test scores.
- [ ] Regenerate `F19_calibration.png`, rebuild and execute the notebook in default mode (`uv run python notebooks/build_notebook.py`, then `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end.ipynb`), and refresh the HTML export with `uv run jupyter nbconvert --to html notebooks/01_end_to_end.ipynb`.

## Week of 23 November (TW11): Docker verified in CI, README

Commit: `TW11: Docker image verified in CI, README run instructions checked on clean clones`

- [ ] The image has never been built (no Docker on the development machine). With the repository on GitLab the `docker` job does not run, so the first build is the Render deploy of the week of 19 October: read its build log and fix failures. One thing to check: `.dockerignore` excludes `data/raw/*.txt`, while the Dockerfile copies `data/raw/*_Definition_File.txt`; add `!data/raw/*_Definition_File.txt` to `.dockerignore` or drop the pattern from the `COPY` line.
- [ ] Extend the job: after the health check run `docker run --rm mlda-cw1 python -c "import mlda; print(mlda.__version__)"` and print `docker image ls mlda-cw1`. Add the CI badge to the README title block once the repository URL is known.
- [ ] Clone the repository into a new folder on Windows and follow README Route A and Route B line by line, then Route C where Docker is available. Fix any line that fails and update the test counts and run times if they changed.
- [ ] Write `report/viva_prep.md` from the 25 questions in `docs/report_plan.md` section 7, one number and one table or figure per answer.
- [ ] You: build the docx (`uv run python report/build_report.py`), open it in Word and read it once for layout, page count and figure placement.

## Week of 30 November (TW12): final sync, v1.0.0, zip. Deadline Friday 4 December

Commit: `TW12: final metric and figure sync; README and report numbers checked against reports/metrics`

- [ ] If code or data changed since the last full run, run `uv run mlda-train` once more so that `models/manifest.json` and every metric come from one run.
- [ ] Rebuild and execute the notebook, refresh its HTML export, then check each number in `README.md` (introduction, success-criteria table, test counts, run times) against `reports/metrics` and `uv run pytest --collect-only -q`.
- [ ] `uv lock --check`; `make export` leaves `requirements.txt` and `app/requirements.txt` unchanged; the last `ci` run on `main` is green.
- [ ] Tag: `git tag -a v1.0.0 -m "CW1 submission, student ID 13130"` and `git push origin main v1.0.0`.
- [ ] You: open `report/MLDA_CW1_13130_report.docx` in Word, update the table of contents and the lists of figures and tables (right-click, Update Field, Update entire table), and check File, Info for an author name in the document properties. Export the PDF to `report/MLDA_CW1_13130_report.pdf`.
- [ ] You: run the anti-plagiarism check on the report and keep the similarity report.
- [ ] Commit the docx and the PDF, then build the archive. Set `MLDA_FORBIDDEN` to the strings that must not appear (comma-separated, supplied at run time and never written to a file), run `uv run mlda-zip` after everything is committed (the bundle holds committed history only, and the packer prints a warning for uncommitted changes), and list the archive to confirm it holds the README, notebook, HTML export, docx, PDF, data archives, `mlda-cw1/repository.bundle` and `mlda-cw1/reports/commit_history.txt`. `.github/` is excluded from it by design. Unzip it in a scratch folder and run `git clone repository.bundle mlda-cw1` beside the bundle to confirm that the history restores. The packer scans the commit history text for the forbidden strings but cannot read file contents inside older commits, so run `git log --all -i -G"<string>" --oneline` for each forbidden string as well.
- [ ] You: upload `MLDA.CW1.13130.zip` to the intranet by Wednesday 2 December, two days before the deadline.
- [ ] You: open the live app the day before marking and again a few minutes before the viva, because a Render free instance spins down after 15 minutes without a request and needs about one minute to start (a Cloud app sleeps after about 12 hours idle). Fallback if it will not wake, and for the live search in the viva: `uv run streamlit run app/streamlit_app.py` or the Docker route on the presenting machine.
