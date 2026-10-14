# Module contracts for src/mlda (binding for all builders)

All constants come from `mlda.config` (target codes, year window, drop lists, cost settings). Every
function below is pure where possible, typed, documented, seeded with `RANDOM_STATE`, and never fits
anything on test rows. Decisions and the numbers behind them: `docs/data_profile.md`,
`docs/profile/decisions.json` (error rules E1-E30, feature formulas, split design, success criteria).

## 1. `mlda.data.join` : analytic base table

`build_base_table(accidents, mines, production) -> pd.DataFrame` (also `load_base_table()` which loads
the three tables with `mlda.data.load`, builds, and caches to `data/processed/base.parquet`).

Steps, in order, each logged with row counts:
1. Filter `CAL_YR` between `YEAR_MIN` and `YEAR_MAX`; keep `DEGREE_INJURY_CD` in POSITIVE + NEGATIVE codes
   (drop EXCLUDED_CODES and NaN). Expected N = 101,609, positives 49,367.
2. `lost_time` = 1 if code in POSITIVE_CODES else 0 (int8).
3. `filing_date` = DOCUMENT_NO[1:8] parsed as Julian YYYYDDD (NaT when unparsable, 18 rows in the full file).
4. `group_id` = MINE_ID + "_" + CAL_YR; `accident_group` = MINE_ID + "_" + ACCIDENT_DT.strftime("%Y-%m-%d").
5. Left-join Mines on MINE_ID (`validate="many_to_one"`, `indicator="_m_mines"`), keeping only
   `MINES_KEEP_COLUMNS` plus `COAL_METAL_IND` renamed `mine_coal_metal_ind`; assert row count unchanged and
   match rate 100 %; record agreement between the two COAL_METAL_IND columns.
6. Production aggregated to mine-year: `prod_year = production.groupby(["MINE_ID","CALENDAR_YR"]).agg(
   hours=("ANNUAL_HRS","sum"), employees=("AVG_ANNUAL_EMPL","sum"), subunits=("SUBUNIT_CD", frozenset))`.
   Join on (MINE_ID, CAL_YR - 1) -> `prev_hours`, `prev_employees`, `prev_subunits`, `prev_year_matched`
   (bool from indicator). Expected match 98.81 % (1,206 unmatched rows kept with NaN).
   Join on (MINE_ID, CAL_YR) -> `same_year_hours`, `same_year_employees` (EDA only, never features).
7. `type_at_time`: from `prev_subunits`: if {01,02} intersects -> "Underground"; elif subset of {30,99,17}
   -> "Facility"; else "Surface"; if no t-1 production: fallback from the row's SUBUNIT_CD
   (01/02 -> Underground; 30/99/17 -> Facility; else Surface).
8. Lagged case history from the FULL accidents file (all degree codes, all years >= 2009): for each base row,
   cases at the same MINE_ID with CAL_YR == row CAL_YR - 1 and `filing_date < ACCIDENT_DT`:
   `prev_incidents` (any code), `prev_valid_cases` (codes 01-06), `prev_lost_time_cases` (codes 01-04).
   Value 0 when t-1 production exists but no such case; NaN when `prev_year_matched` is False.
   Implement with a merge + vectorised comparison (no Python row loops over 100k rows x cases).
9. Persist helper tables for the app: `data/processed/mine_year_features.parquet`
   (MINE_ID, CALENDAR_YR, hours, employees, subunits as "|"-joined string, type) and
   `data/processed/mine_case_history.parquet` (MINE_ID, CAL_YR, filing_date, DEGREE_INJURY_CD) for 2009-2025.

Output keeps every raw accident column (leakage columns included, because EDA and label-validity checks
need them) plus the added columns above. `mlda.data.schema.validate_base(df)` asserts: expected columns
present, N and prevalence within tolerance, MINE_ID 7 chars, no NaN in lost_time/group_id, CAL_YR in window.

## 2. `mlda.cleaning` : error-correction rules (E1-E12, E21-E25 of decisions.json)

`apply_error_rules(base) -> tuple[pd.DataFrame, pd.DataFrame]`: returns the corrected frame and a log
frame with columns `rule, condition, action, rows_affected` (this becomes Table 4 in the report).
Rules: JOB_EXPER/MINE_EXPER > TOT_EXPER -> NaN + `exper_inconsistent`=1; any experience > 60 -> NaN;
ACCIDENT_TIME/SHIFT_BEGIN_TIME codes 9990-9999 -> NaN, 2400 -> 0, three-character values parsed as integers;
NO_INJURIES capped at 10 (`no_injuries_capped`); coordinates outside (17.5<=lat<=72, -180<=lon<=-64) and
STATE != "AS" -> NaN; `status_103i_at_accident` derived from CURRENT_103I / CURRENT_103I_DT vs ACCIDENT_DT
("Never had 103I", "In 103I at accident", "Formerly 103I", "Unknown"); lagged rates winsorised at RATE_CAP.
Also `flag_outliers(series, method="iqr"|"zscore", k=1.5|3.0) -> pd.Series[bool]` used by EDA and the app.

## 3. `mlda.features` : row-wise feature engineering

`engineer_features(base_clean) -> pd.DataFrame` returns ONLY the model inputs listed below, in this order,
with these exact names (the ColumnTransformer and the app rely on them):

CATEGORICAL (object/str, NaN -> "none" for PLACEHOLDER_AS_LEVEL, otherwise "missing"):
SUBUNIT_CD, FIPS_STATE_CD, UG_LOCATION_CD, UG_MINING_METHOD_CD, MINING_EQUIP_CD, EQUIP_MFR_NAME,
CLASSIFICATION_CD, ACCIDENT_TYPE_CD, OCCUPATION, ACTIVITY_CD, INJURY_SOURCE_CD, NATURE_INJURY_CD,
INJ_BODY_PART_CD, COAL_METAL_IND, PRIMARY_CANVASS, type_at_time, status_103i_at_accident, mine_size_band,
SAFETY_COMMITTEE_IND, PART48_TRAINING, PORTABLE_OPERATION, accident_weekday, accident_month
(weekday 0-6 and month 1-12 as strings).

NUMERIC (float, NaN allowed, imputed inside the pipeline):
TOT_EXPER, MINE_EXPER, JOB_EXPER, log1p_tot_exper, exper_ratio_mine, exper_ratio_job, no_injuries_capped,
accident_hour, shift_begin_hour, hours_into_shift, latitude, longitude, prev_hours, log1p_prev_hours,
prev_employees, prev_hours_per_employee, prev_incidents, prev_incident_rate_200k, prev_lost_time_rate_200k,
prev_lost_time_share.

BINARY (int8 0/1, never NaN):
is_night, is_weekend, new_to_mine, exper_inconsistent, is_contractor, immed_reportable, prev_year_matched,
tot_exper_missing, accident_hour_missing, hours_into_shift_missing, prev_share_missing, coords_missing.

Formulas are in decisions.json `features.engineered`. Key details: hours_into_shift =
((acc_min - shift_min) mod 1440)/60, NaN if either time invalid or result > 16; is_night = hour in
{0-5, 18-23} (0 when hour missing); immed_reportable = IMMED_NOTIFY_CD.notna() & (code != "13");
mine_size_band = pd.cut(prev_employees, [0,5,20,50,200,inf], right=False, labels
["1-4","5-19","20-49","50-199","200+"]) with "missing" when NaN; rates = log1p(min(count/prev_hours*200000, 50));
prev_lost_time_share = prev_lost_time_cases/prev_valid_cases (NaN when 0 valid cases; prev_share_missing=1).
Expose `FEATURE_SPEC = {"categorical": [...], "numeric": [...], "binary": [...]}` and
`feature_frame(base_clean) -> (X, y, groups, meta)` where meta holds CAL_YR, COAL_METAL_IND, MINE_ID,
ACCIDENT_DT, accident_group for stratified reporting and the temporal split.
`FeatureEngineer(BaseEstimator, TransformerMixin)` wraps engineer_features for use on a single
inference row (no fitting; `fit` returns self).

## 4. `mlda.preprocessing`

`build_preprocessor(numeric_imputer="median"|"mean"|"knn", scaler="standard"|"minmax"|"robust"|"none",
encoder="onehot"|"ordinal", outlier_rule=None|"iqr_clip"|"zscore_clip", min_frequency=ONE_HOT_MIN_FREQUENCY)
-> ColumnTransformer` over FEATURE_SPEC: numeric -> optional clipper (fitted on train quantiles / mean+-3sd)
-> SimpleImputer/KNNImputer -> scaler; categorical -> SimpleImputer(constant "missing") ->
OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=..., sparse_output=False) or
OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1); binary -> passthrough.
`OPTIONS` dict lists the allowed values and a one-line rationale for each (shown in the app).
`get_feature_names(preprocessor) -> list[str]` after fitting.

## 5. `mlda.split`

`make_split(X, y, groups, test_fraction=0.2, random_state=42) -> SplitIndices(train_idx, test_idx)` using
StratifiedGroupKFold(n_splits=5, shuffle=True) on `group_id`, taking fold 0 as test. Asserts: disjoint
groups, no accident_group straddles, prevalence within 1 pp of the full set. `save/load_split_indices()`
to `data/processed/split_indices.json`. `cv_splitter(n_splits=5) -> StratifiedGroupKFold` for tuning on
the training rows. `temporal_split(meta) -> (train_idx, test_idx)` on CAL_YR < TEMPORAL_SPLIT_YEAR.

## 6. `mlda.models`

`MODEL_ZOO: dict[str, ModelSpec]` with ModelSpec(name, estimator factory, param_grid, search="grid"|"random",
n_iter, preprocess_overrides (e.g. scaler="none" for trees), sample_cap (20_000 for SVC), description,
taught_in_week). Models: logreg (LogisticRegression, C, penalty l2, class_weight None/balanced, max_iter 2000),
knn (n_neighbors 5-51 odd, weights, p), gaussian_nb (var_smoothing), decision_tree (max_depth,
min_samples_leaf, criterion), random_forest (n_estimators <= 300, max_depth, min_samples_leaf, max_features),
hist_gb (learning_rate, max_iter, max_leaf_nodes, l2_regularization, early_stopping), svc_rbf (C, gamma,
probability=True, on <= 20,000 stratified training rows), mlp (hidden_layer_sizes, alpha, early_stopping).
Grids small enough that every model tunes in < 10 minutes with n_jobs=-1 on 20 cores.
`build_pipeline(spec, preprocessor) -> Pipeline`.

## 7. `mlda.tuning`

`tune(spec, X_train, y_train, groups_train, cv, quick=False) -> TuneResult(best_pipeline, best_params,
cv_results_df, oof_proba)`: GridSearchCV / RandomizedSearchCV with scoring = {"roc_auc", "average_precision",
"f1", "recall", "precision", "accuracy", "neg_brier_score"}, refit="average_precision", return_train_score=True;
then out-of-fold probabilities via cross_val_predict with the best pipeline on the same StratifiedGroupKFold.
`quick=True` shrinks grids to 2 points and subsamples 10,000 rows (used by tests and the app).

## 8. `mlda.evaluation`

`classification_metrics(y_true, proba, threshold) -> dict` with accuracy, balanced_accuracy, precision,
recall (tpr), specificity (tnr), f1, mcc, roc_auc, pr_auc (average_precision), brier, log_loss, tp/fp/tn/fn,
flag_rate. `threshold_for_specificity(y, proba, 0.40)`, `threshold_for_flag_rate(y, proba, 0.5)`,
`cost_optimal_threshold(y, proba, ratio)`, `expected_cost_per_1000(y, pred, ratio)`,
`threshold_sweep(y, proba) -> DataFrame` (threshold, recall, specificity, precision, f1, flag_rate, cost per ratio),
`curves(y, proba)` (roc, pr, calibration bins), `mcnemar_test(y, pred_a, pred_b)`,
`paired_fold_test(scores_a, scores_b)` (paired t and Wilcoxon), `compare_models(metrics_by_model) -> DataFrame`
in the brief's column order (TPR, TNR, AUC, Recall, Precision, F1, PR-AUC, Accuracy), `by_sector(...)`.
Baselines in `mlda.baselines`: majority, all_positive, rule_not_cut_or_dust (NATURE_INJURY_CD not in
{180,320,310}), lookup_nature_bodypart_coal (fitted on train with fall-backs) each exposing predict_proba-like
scores.

## 9. `mlda.eda`

`describe_numeric(df, cols) -> DataFrame` (n, missing %, mean, median, trimmed mean 10 %, sample variance,
sample std, min, Q1, Q3, max, IQR, skewness, kurtosis), `describe_categorical`, `correlations(df, cols)`
(Pearson, Spearman), `point_biserial(df, cols, target)`, `cramers_v_table(df, cols, target)` (bias-corrected),
`vif_table(X_numeric)`, `missingness_table`.

## 10. `mlda.plots`

matplotlib/seaborn builders that return a Figure and optionally save to `FIGURES_DIR` at 200 dpi:
target_distribution, histograms_grid, class_boxplots, pairplot_top, correlation_heatmaps (Pearson + Spearman),
categorical_vs_target, missingness, qq_plots, outlier_fences, pca_projection, roc_curves, pr_curves,
validation_curve, learning_curve, confusion_matrices, feature_importance, threshold_cost_curve, calibration.
Consistent palette, British-English labels, no titles that contain the student's name.

## 11. `mlda.explain`, `mlda.persistence`, `mlda.pipeline`

`explain.permutation_importance_table`, `coefficient_table`, `shap_summary(model, X_sample)` (lazy import,
returns None when SHAP cannot handle the estimator). `persistence.save_model(pipeline, name)` ->
models/<name>.joblib (compress=3) + entry in models/manifest.json (sklearn/python versions, git SHA,
data checksum, timestamp, best params); `load_model(name)`. `pipeline.train_all(quick=False)` runs the
whole study and writes every artefact listed in docs/architecture.md (reports/metrics/*.json,
comparison.csv, baselines.json, thresholds.json, reports/cv/*.csv, reports/predictions/test_predictions.parquet,
reports/predictions/oof_predictions.parquet, reports/figures/*.png, data/processed/*.parquet, models/*).
