# Viva voce preparation (student ID 13130)

WIUT 6COSC017C Machine Learning and Data Analytics, Coursework 1: predicting lost-time mine injuries from MSHA open data.

**How to use this file.** Each of the 32 questions has a spoken answer of about 30 seconds (90 to 120 words at a calm pace), a pointer to the place where you can prove it, and one likely follow-up with a two-sentence answer. Table and Figure numbers are those of the rendered report; `report/build_report.py` assigns them in document order, so check them again after any rebuild that adds a table or figure. "Notebook" means `notebooks/01_end_to_end.ipynb` (sections A to F). "App" means the Streamlit app, whose pages are Data explorer, Exploratory analysis, Preprocessing, Model training, Inference, Evaluation and About. Numbers come from `report/sections/*.md` and `reports/metrics`; the few that come from elsewhere are listed at the end under "Where the other numbers come from". Say "HGB" for histogram-based gradient boosting and "the lookup baseline" for the nature x body part x sector table of lost-time rates.

---

## Numbers to memorise

**Data and split**

| Item | Value |
|---|---|
| In-scope reports N | 101,609 (2010-2025, degree codes 01 to 06), from 8,303 mines |
| Lost-time (positive) / other (negative) | 49,367 / 52,242; prevalence 48.6%; majority-class accuracy 0.514 |
| By sector | coal 41,461 reports, 63.0% lost-time; metal and non-metal 60,148 reports, 38.6% |
| Prediction moment | receipt of Form 7000-1, at its filing date; the median form is filed 8 days after the accident (quartiles 5 and 13), 80.9% within 14 days |
| Cohort at that moment | 45.0% of reports already carry a return-to-work date and are not scored; 55% are open at filing (11,290 of the 20,320 test reports, 71.2% lost-time, against 20.3% for the 9,030 filed after the worker was back). Lost-time share of open reports by filing lag: 0.62 within a day, 0.84 after 30 days |
| Features | 52 (20 categorical, 20 numeric, 12 binary); 546 columns after encoding; the three Mines-snapshot flags (safety committee, Part 48 training, portable operation) are not inputs |
| Split | StratifiedGroupKFold on 32,732 mine-years, seed 42, fold 0 as test: train 81,289, test 20,320 (6,724 mine-years) |
| Validation | five StratifiedGroupKFold folds inside the 81,289 training rows, about 16,000 rows each |
| Criteria fixed before modelling | S1 ROC-AUC 0.80 or more; S2 PR-AUC 0.75 or more; S3 recall 0.92 or more at specificity 0.40; S4 cost below the review-everything floor and the lookup at 2:1, 3:1, 5:1; S5 recall 0.78 or more at a 50% flag rate; S6 Brier 0.20 or less; S7 tests reported |

**The eight models** (ordered by cross-validated PR-AUC, which was the selection rank; the test AUC and PR-AUC are on the 20,320 held-out reports)

| Model | CV PR-AUC (mean ± SD) | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| HGB (selected) | 0.838 ± 0.003 | 0.843 | 0.842 |
| MLP | 0.831 ± 0.003 | 0.839 | 0.836 |
| Random forest | 0.821 ± 0.004 | 0.833 | 0.827 |
| Logistic regression | 0.820 ± 0.003 | 0.829 | 0.825 |
| Decision tree | 0.803 ± 0.004 | 0.817 | 0.805 |
| SVM, RBF kernel | 0.795 ± 0.014 | 0.820 | 0.811 |
| KNN | 0.764 ± 0.009 | 0.773 | 0.769 |
| Gaussian naive Bayes | 0.735 ± 0.007 | 0.745 | 0.742 |
| Lookup baseline (nature x body part x sector) | not cross-validated | **0.804** | 0.790 |

The decision tree and the SVM swap places between cross-validation and the test set (Tables 12 and 13). The lookup baseline's ROC-AUC has three values: 0.804 on the test set (use this one), 0.794 on the temporal split and 0.810 in-sample on all reports. Name the split when you quote it.

**The final model (HGB) at the capacity threshold**

| Item | Value |
|---|---|
| Settings | learning rate 0.05, at most 400 rounds with early stopping, 31 leaves, no L2 penalty |
| Capacity threshold | **0.211**, the lowest out-of-fold threshold with specificity 0.40 on all training rows; flag rate 0.763 on all test filings (about three in four), but 0.900 on the reports open at filing |
| Test recall / specificity / precision | **0.941 / 0.406 / 0.600**; F1 0.733; Brier 0.161; recall interval by mine-year bootstrap 0.936 to 0.947 |
| Test confusion counts | TP 9,294, FN 579, FP 6,204, TN 4,243 (Table 17) |
| At the default threshold 0.5 | recall 0.740, precision 0.766, accuracy 0.764 |
| S5 (recall at a 50% flag rate) | **0.777 against the 0.78 target, missed**; flag rate 0.507; 0.747 on the temporal split, 0.762 on the mine-grouped split; the MLP also misses it (0.777 at a 51.1% flag rate) |
| Sector, same threshold | coal recall 0.970, specificity 0.395, precision 0.739, PR-AUC 0.913; metal and non-metal recall 0.909, specificity 0.411, precision 0.487, PR-AUC 0.708 |
| Temporal split (train 2010-2019, test 2020-2025) | PR-AUC 0.843, recall 0.945, specificity 0.373 |

**The recommended policy: the workload rule** (review the top three in four of the reports open at filing, ordered by score)

| Item | Value |
|---|---|
| Deployed cut-off | 0.376, the 25th percentile of the out-of-fold scores of the 44,640 open training reports, so no test row enters it |
| On the 11,290 open test reports | flag rate 0.743, recall 0.850, specificity 0.522, precision 0.815 |
| At exactly the top 75% by test score | recall 0.855 and precision 0.812 for HGB; MLP 0.854 (precision 0.811); lookup baseline 0.828; random selection 0.75 (expected recall equals the share reviewed) |
| Capacity threshold on the same reports | flag rate 0.900, recall 0.949, specificity 0.219: more than the three in four that capacity allows |
| Paired mine-year bootstrap of recall under the rule | HGB minus lookup 0.027 (0.021 to 0.033); HGB minus MLP 0.000 (-0.003 to 0.004) |
| Calibration on open reports | mean predicted 0.592 against observed 0.712, so 0.12 low; the rule uses the ranking only |
| HGB fitted on open-at-filing training rows alone | recall 0.877 under the rule, against 0.855 for the model fitted on all filings; a prospective trial should fit on open reports |
| History audit | 14.1% of training rows (11,426) have a previous mine-year in the test split; refitting without that history changes test PR-AUC by +0.0006, under 0.001 |

**Costs**

| Item | Value |
|---|---|
| False positive C_FP | USD 200 (range 105 to 260) |
| False negative C_FN | USD 4,000 (range 3,100 to 6,200), an assumption |
| Cost ratio | 20:1 base; sensitivity set 2:1, 3:1, 5:1, 10:1, 20:1, 30:1; implied threshold on a calibrated probability 1 / (1 + ratio) |
| Review-everything floor | USD 102,829 per 1,000 cases (102,825 on the test set); cost-optimal from about 10:1 upwards |
| HGB cost per 1,000 at 2:1 / 3:1 / 5:1 | USD 65,817 / 76,781 / 87,205 |
| HGB cost per 1,000 at 10:1 / 20:1 / 30:1 | USD 98,130 / 103,346 / 103,110; the last two are above the floor, because the cost-optimal threshold comes from out-of-fold probabilities |
| Lookup baseline at 2:1 / 3:1 / 5:1 | USD 73,553 / 85,472 / 98,061 |
| Lost-time cases missed per 1,000 filings | HGB 28.5 at the capacity threshold; lookup 40.8; one-field hand rule built for this study 49.5 (Table 21) |

**Tests and run.** HGB against the MLP on the five folds: paired t-test p = 0.007, corrected resampled t-test p = 0.029, Wilcoxon p = 0.0625 (the smallest value possible with five folds). HGB against the lookup baseline: clustered McNemar p below 0.001. `mlda-train` took 5,154 s (85.9 minutes, about 86); `python -m mlda.followup` took about 31 minutes (1,879 s) and the notebook with `FULL_RUN=1` 41.4 minutes (2,484 s). The suite has 494 tests, 477 of them fast (17 are marked slow). The history has 40 commits: 17 on 14 October, 21 on 15 October and 2 on 20 October (read `git log` again before the viva).

---

## Part 1. Data and framing (questions 1 to 4)

### 1. Why this dataset and question, who acts on it, why not Kaggle, when was it reserved?

**Answer (30 seconds).** The question is whether a US mine injury report can be classified, on receipt of Form 7000-1, as a lost-time case, which is when the insurer or self-insured operator decides on an indemnity reserve and a return-to-work plan. Table 1 prices a missed case at about USD 4,000 and an unnecessary review at USD 200. The data are three open MSHA files joined into 101,609 reports from 2010 to 2025 (Table 4). The brief bans Kaggle and I found no Kaggle copy of the accident file, so I pinned the version of 9 October 2026 from the government site by SHA-256 digest. I reserved the topic in the module Discussions forum on [DATE AND TOPIC TITLE: fill in once the post exists]. The claims adjuster acts on the score, which only orders the review queue.

**Where to show it.** Report: Section 1.1, Table 1 (costs), Table 4 (sources). Notebook: "A. Load dataset" and "A.1 Raw files and checksums". App: About > Purpose; Data explorer > Sources.

**Follow-up.** *Why not use MSHA's published injury rates instead of a model?* Those rates are aggregates per 200,000 employee-hours, which suit an inspector choosing mines but cannot say which of today's filed claims needs a reserve. The insurer decides claim by claim when the form arrives, so I scored single reports.

### 2. How was the licence verified and attributed?

**Answer (30 seconds).** All three files are works of the US federal government, so they are public domain under 17 U.S.C. 105 and may be copied, modified and redistributed. I checked this on the MSHA open-data page and the data.gov entry, and the licence column of Table 4 records it for each file. MSHA asks for credit to the US Department of Labor, and the report and the README give it. The files are refreshed weekly, so I pinned the 9 October 2026 version and stored the SHA-256 digests in `data/raw/SNAPSHOT.json`; anyone can check they hold the same bytes. My own code carries the MIT licence.

**Where to show it.** Report: Table 4, Section 1.3, Section 5.5. Notebook: "A.1 Raw files and checksums" and "F.5 Licence, data citation and deployed address". App: About > Data and licence (file sizes and checksums); Data explorer > Sources.

**Follow-up.** *If the data are public domain, why does attribution matter?* Copyright does not require it, but MSHA requests it and a reader needs the source and the snapshot date to reproduce the numbers. The model also uses no names or narrative text, so public-domain status does not stand in for privacy care.

### 3. How were the sources joined and how do you know the join is correct?

**Answer (30 seconds).** I used left joins from the accident rows and asserted the row count after each one. Accidents to Mines is many-to-one on MINE_ID: 101,609 rows in, 101,609 out, and all 101,609 matched. Production was summed to 370,794 mine-years and joined on MINE_ID and the previous year; 100,403 rows, 98.8%, matched and the other 1,206 were kept with a flag (Table 4). The argument validate="many_to_one" makes pandas raise an error if a right-hand key repeats, which is how a join silently multiplies rows. The coal or metal flag also agrees between the tables on 100.0% of rows (Table 8, step 3).

**Where to show it.** Report: Table 4, Table 8 step 3. Notebook: "A.3 Keys and the many-to-one rule" (shows what the validation catches), "A.5 Build the analytic base table", "A.6 Validate the base table and compare it with the stored copy". App: Data explorer > Joins.

**Follow-up.** *Why join production on the previous year and not the same year?* Hours are reported quarterly after each quarter ends, so same-year hours are not known when a first-quarter accident is filed, and using them would leak information the deployed model cannot have. The previous year's hours are always available, at the cost of 1,206 unmatched reports.

### 4. What is one row and what exactly does the label mean?

**Answer (30 seconds).** One row is one injured person on one MSHA Form 7000-1, so a two-victim accident gives two rows. The label lost_time comes from DEGREE_INJURY_CD: codes 01 to 04 (fatality, permanent disability, days away) are positive and codes 05 and 06 (restricted duty only, no lost time) are negative, which gives 49,367 positives among 101,609 reports, a prevalence of 48.6% (Table 2). It matches the point at which wage replacement starts, so it fits the reserve decision. It also records whether the operator offered restricted duty, because a back sprain becomes days away or restricted duty depending on that choice. That is why coal is 63.0% lost-time against 38.6% for metal and non-metal, and why I report results by sector.

**Where to show it.** Report: Section 1.2, Table 2 (label mapping). Notebook: "A.4 Row filters and the target", "B.4 Target distribution", "B.12 Evidence for three claims that later sections use" (restricted-duty coding by sector). App: Data explorer > Target; About > Limits and ethics.

**Follow-up.** *Why not the DART label, which does not depend on restricted-duty practice?* The DART label (days away, restricted or transferred) is 69.0% positive and hardly varies by sector (Cramer's V 0.015 against 0.240 for lost_time), so it suits inspection targeting. Restricted duty triggers no indemnity payment, so it does not match the insurer's decision, and I kept it as a sensitivity label.

---

## Part 2. Exploratory analysis (questions 5 to 10)

### 5. Why mean and median both, and when the trimmed mean?

**Answer (30 seconds).** The mean assumes symmetry and is pulled up by a long tail, the median ignores how far the tail reaches, and the trimmed mean sits between them. For total experience in Table 6 the mean is 9.8 years, the median 6.0 and the 10% trimmed mean 8.0, with a skewness of 1.37, so the three disagree because the data are right-skewed. I impute skewed features with the median, because a mean fill would give unrecorded workers a tenure of 9.8 years when the typical worker has 6. I read the trimmed mean as a check: it discards a fifth of the rows, 10% from each tail, so it resists outliers but still uses most of the data.

**Where to show it.** Report: Table 6 (descriptives), Table 23 in Appendix A. Notebook: "B.2 Descriptive statistics of the numeric inputs" (mean, median and 10% trimmed mean in one table). App: Exploratory analysis > Summary statistics.

**Follow-up.** *Why not impute with the trimmed mean?* It still sits above the median (8.0 against 6.0 years), and scikit-learn's SimpleImputer offers mean, median, most frequent and constant, so the median is the nearest standard choice for skewed data. The bias would also fall on the costly cases, since reports with blank experience are lost-time more often (59.3% against 48.5%).

### 6. Why sample variance uses n-1?

**Answer (30 seconds).** Sample variance divides the sum of squared deviations by n minus 1, because the deviations are measured from the sample mean, which is fitted to the same data and sits closer to the points than the true mean does. Dividing by n would underestimate the population variance, and n minus 1 removes that bias. My tables use it throughout: total experience in Table 6 has a variance of 108.6 and a standard deviation of 10.4 years. At 81,289 training rows the choice changes the variance by about one part in 81,289, so it makes no practical difference there. It matters for small groups, for example a cell of 19 cases, where the factor is 19 over 18.

**Where to show it.** Report: caption of Table 6 ("variance and SD use the n - 1 denominator"). Notebook: "B.2 Descriptive statistics of the numeric inputs" (the text states `ddof = 1`). App: Exploratory analysis > Summary statistics.

**Follow-up.** *What are the defaults in pandas and numpy?* pandas `var` and `std` use n minus 1 (ddof = 1), whereas numpy uses n (ddof = 0) unless you pass ddof. I set ddof = 1 explicitly in the notebook so both agree.

### 7. When would Pearson mislead; what did Spearman show?

**Answer (30 seconds).** Pearson's r measures linear association and is pulled by extreme points, so it misleads when a relation is monotone but curved, or when a few very large mines dominate. In my data log1p of previous-year hours against previous-year employees has a Pearson r of 0.645 but a Spearman rank correlation of 0.994, so Pearson understates the redundancy (Figure 3 shows both matrices). Raw hours and employees reach r = 0.996 with a variance inflation factor above 160 (Table 7). I therefore read redundancy from Spearman and kept Pearson beside it. Because redundant size measures let logistic-regression coefficients trade weight, I kept both columns under the L2 penalty and did not interpret that pair.

**Where to show it.** Report: Figure 3 (corr), Table 7 (VIF columns). Notebook: "B.6 Association with the target and between inputs", "B.8 Collinearity: variance inflation factors". App: Exploratory analysis > Correlations (switch between Pearson and Spearman; the table "Largest gaps between Pearson and Spearman"; the VIF expander).

**Follow-up.** *Why not drop one of the two size columns?* Dropping one assumes it adds nothing, and principal components would give up readable features. Trees are unaffected and logistic regression is regularised, so keeping both costs little.

### 8. Which correlation for numeric vs binary target and why?

**Answer (30 seconds).** For a numeric feature against the binary target I used point-biserial correlation, which is Pearson's r with one dichotomous variable, and for categorical features I used bias-corrected Cramer's V. Table 7 ranks them: previous-year lost-time share leads the numeric features at r = 0.292, and nature of injury leads the categorical ones at V = 0.391. Point-biserial r carries a sign that gives the direction, whereas V runs from 0 to 1. I did not rank by chi-square p-values, because at 81,289 rows trivial effects are significant. Both leaders are far below 0.95, the level at which I would suspect leakage, and I checked field by field that both are available when the form is filed.

**Where to show it.** Report: Table 7 (associations), Figure 7 (lost-time rate per category level). Notebook: "B.6 Association with the target and between inputs", "B.7 Categorical inputs against the target". App: Exploratory analysis > Associations with the target.

**Follow-up.** *Is r = 0.292 a strong feature?* It is the strongest numeric one, but its square is 0.085, so it explains under 9% of the variance alone. That is why the classes overlap in the pair plot and the PCA plot (Figures 6 and 10) and a model has to combine many fields.

### 9. How does bin count change a histogram's story?

**Answer (30 seconds).** The bin count is a smoothing choice: too few bins hide structure and too many draw noise. I used 30 bins in every panel of Figure 4 so that 20 features and two classes can be compared on the same footing. The Freedman-Diaconis rule would have set anything from one bin for the capped victim count to 274 bins for hours per employee, so one rule for all panels does not work. The 60-bin log-experience panel in Figure 9 shows a comb pattern consistent with experience recorded in whole years. Each caption states its bin count.

**Where to show it.** Report: Figure 4 (hist, 30 bins), Figure 9 (outliers, 60 bins). Notebook: "B.5 Distributions and class differences"; to show the effect live, call `plots.histograms_grid` in that cell with `bins=10` and then `bins=100` (an optional demonstration that I have not run). App: Exploratory analysis > Distributions (one feature at a time; the page has no bin control).

**Follow-up.** *How would you choose the bin count for a single variable?* I would use the Freedman-Diaconis rule, which sets the bin width from the IQR and the cube root of n, and compare it with a coarser and a finer histogram. If the story changes between them, I would report the choice and show both.

### 10. What did box plots show that the table did not?

**Answer (30 seconds).** Table 6 gives pooled summary numbers, whereas the box plots in Figure 5 split each feature by outcome, so I could see which features differ between the classes. Lost-time cases have the higher median previous-year lost-time share, and they lie further east in longitude, near the Appalachian coal fields, which fits the sector gap of 63.0% against 38.6%. For most other features the boxes overlap, so no single feature separates the classes, and the PCA plot in Figure 10 agrees. Box plots hide the shape inside the box, so I read them with the histograms of Figure 4: the previous-year lost-time share piles up at zero among non-lost-time cases, which a box cannot show.

**Where to show it.** Report: Figure 5 (box), Figure 4 (hist), Figure 10 (pca). Notebook: "B.5 Distributions and class differences". App: Exploratory analysis > Distributions (histogram and box plot side by side).

**Follow-up.** *Why did you hide the outlier points?* With whiskers at 1.5 times the IQR, the outlier points of skewed features would swamp the boxes. Outliers are treated separately in Figure 9, where the IQR rule flags 4,020 raw experience values.

---

## Part 3. Preparation (questions 11 to 16)

### 11. Drop vs impute per column, why median?

**Answer (30 seconds).** I decided column by column. Columns more than 60% blank were dropped unless the gap was structural, so the equipment model number, 76.7% blank, went. Structural gaps, such as underground fields on a surface accident, became their own level, "none". For numeric gaps I used SimpleImputer with the median plus explicit missing-value indicators, because blank total experience is informative: those reports are lost-time in 59.3% of training cases against 48.5% of recorded ones (Table 8, step 11). The median suits the skew, since total experience has a mean of 9.8 years and a median of 6.0 (Table 6), and a mean fill would overstate the tenure of unrecorded workers.

**Where to show it.** Report: Table 8 steps 7 and 11, Figure 8 (missing), Table 6. Notebook: "B.9 Missing values", "B.12 Evidence for three claims that later sections use" (lost-time rate where experience is blank), "C.3 Missing-value plan". App: Exploratory analysis > Missingness; Preprocessing (the "Numeric imputer" control).

**Follow-up.** *Why not KNNImputer or IterativeImputer?* They predict gaps from other features, which would invent histories for quiet mines: previous-year lost-time share is missing on 19.8% of training rows because it is undefined (zero cases over zero) at mines with no case in the previous year. They are also much slower on 81,289 rows inside every cross-validation fold.

### 12. IQR or Z-score, why?

**Answer (30 seconds).** I compared both rules on total experience (Figure 9). IQR fences flag 4,020 raw values, 5.0% of training rows, and the Z-score rule flags 714, 0.9%, because the long tail inflates the standard deviation behind the Z-score. The Z-score rule assumes a roughly normal variable, and at a skewness of 1.37 that assumption fails and its lower fence falls below zero. After log1p neither rule flags any value. I kept all values because they are real veterans and large mines, added log1p versions, and left clipping as an app option.

**Where to show it.** Report: Figure 9 (outliers), Table 8 step 12, Section 3.5. Notebook: "B.10 Normality and outliers". App: Preprocessing (the "Outlier rule" control, and "One numeric column before and after").

**Follow-up.** *Why keep the outliers at all?* Mines with 200 or more employees supply 36.7% of training rows, so treating their exposure as error would discard the insurer's largest group of cases. Impossible values, such as job experience above total experience, went through separate error rules and were set to missing.

### 13. Why StandardScaler for LR/KNN/SVM and nothing for trees?

**Answer (30 seconds).** Logistic regression, KNN, the SVM, the MLP and naive Bayes depend on distances, gradients or variances, so a feature in large units dominates. Previous-year hours run from 1 to 7,735,308 while experience runs from 0.02 to 60 years (Table 8, step 14), and unscaled hours would make KNN flags follow mine size. Decision trees, the forest and HGB split on rank order, so a monotone rescaling changes nothing and they get no scaler. I chose StandardScaler over MinMaxScaler because a few very large mines would squeeze most previous-year hours into the bottom tenth of the range. The scaler sits inside the Pipeline, so its mean and standard deviation come from training rows only.

**Where to show it.** Report: Table 8 step 14, Section 3.8, Table 9 (scaling column). Notebook: "C.6 The preprocessing pipeline" (fitted statistics come from training rows), "C.7 Do the preprocessing choices matter?" (seven variants with logistic regression). App: Preprocessing (the "Scaler" control with the before-and-after histogram).

**Follow-up.** *Does the scaler choice change the result?* In the notebook's C.7 table the standard, robust and min-max scalers give logistic-regression average precision within 0.001 of each other (0.808 to 0.809), less than one fold standard deviation. I chose StandardScaler for the argument above, not because the others were worse here.

### 14. One-hot vs ordinal vs label with examples?

**Answer (30 seconds).** One-hot creates a 0/1 column per level, ordinal encoding maps levels to integers and so implies an order, and label encoding is the same mapping meant for the target. Nearly all my 20 categorical features are nominal codes, such as nature of injury with 27 levels and occupation with 193, so one-hot is right: it creates no false order, and the matrix grows to 546 columns (Table 8, step 13). I set handle_unknown to infrequent_if_exist and min_frequency to 0.1%, so rare and unseen levels pool into one column instead of failing at inference. The only ordered feature, the five employee bands of mine_size_band, stayed one-hot because its lost-time rate is not monotonic. Label encoding is used for the target alone.

**Where to show it.** Report: Table 5 (dictionary), Table 8 step 13, Section 3.8. Notebook: "C.6 The preprocessing pipeline", "C.7 Do the preprocessing choices matter?" (ordinal against one-hot). App: Preprocessing (the "Categorical encoder" control and the min_frequency slider).

**Follow-up.** *What goes wrong with ordinal codes for nature of injury?* A linear model reads code 180 (cut) as larger than code 100 (amputation), which has no meaning; in C.7 swapping one-hot for ordinal codes lowers logistic-regression average precision from 0.808 to 0.756. Trees can split a code repeatedly, but I did not test that and used one-hot for all eight models.

### 15. Where could leakage enter and how was it prevented?

**Answer (30 seconds).** Leakage could enter in four places. Outcome fields: DAYS_LOST has a Cramer's V of 0.971 with the target and the closing-document number is present for 73.5% of positives against 33.6% of negatives, so both were dropped (Table 8, step 7). Future information: for 40.4% of rows the Mines status date post-dates the accident, so I re-dated the 103(i) status, took the three undated flags out of the inputs, and used previous-year hours and earlier cases filed before the accident. Related rows: I split by mine-year groups. Fitting: every imputer, scaler, encoder and threshold is fitted on training folds inside a Pipeline, and the test set was scored once.

**Where to show it.** Report: Table 8 steps 7 to 9, Figure 11 (pipeline), Sections 3.7 and 3.9. Notebook: "C.5 Leakage check" (asserts the dropped columns are absent), "C.6 The preprocessing pipeline" (medians and scaler statistics from training rows), "C.8 The grouped, stratified split". App: Preprocessing and Model training (both state that fitting uses training rows only).

**Follow-up.** *How do you know the grouped split mattered?* The mean label of a row's later-dated siblings in the same mine-year ranks it with a ROC-AUC of 0.667, close to the best planned feature, and the numeric features identify the mine-year for 99.9% of rows, so KNN and the tree ensembles could memorise outcomes (Section 3.9). The split asserts that no mine-year or same-day group crosses between train and test. The one path it leaves open, a previous-year history drawn from a test mine-year, is audited in question 29.

### 16. Why stratify; what is the validation set?

**Answer (30 seconds).** I stratified so the lost-time share stays at 48.6% in both the training and test sets; unstratified group folds ranged from 46.5% to 50.9% (Table 8, step 9). Because reports from one mine-year resemble each other, I used StratifiedGroupKFold on 32,732 mine-years and held out one fold as the test set: 81,289 training rows and 20,320 test rows from 6,724 mine-years. The validation set is the five StratifiedGroupKFold folds inside the training rows, about 16,000 rows each, used for the hyperparameter search and for the out-of-fold thresholds. The test set stayed untouched until the final scoring. That gives the three roles the brief asks for: training, validation and test.

**Where to show it.** Report: Table 8 step 9, Section 3.9, Section 4.3. Notebook: "C.8 The grouped, stratified split" (invariants asserted), "D.1 Hyper-parameter search". App: Model training (states five-fold StratifiedGroupKFold on training rows); About > Reproduce (split indices stored in `data/processed/split_indices.json`).

**Follow-up.** *Why not a plain stratified train_test_split as in the seminars?* 82.3% of rows share a mine-year with another report, so a random split puts siblings on both sides and lets KNN and the trees memorise outcomes, which flatters the scores (Table 8, step 9 and Section 3.9). The grouped version keeps the stratification and adds the guarantee that no mine-year crosses the split.

---

## Part 4. Algorithms and tuning (questions 17 to 21)

### 17. Logistic regression cost function and why not MSE?

**Answer (30 seconds).** Logistic regression passes a weighted sum of the inputs through the sigmoid, and I fit the weights by minimising binary cross-entropy (log loss) plus an L2 penalty with lbfgs. Cross-entropy is the negative log-likelihood of the labels under a Bernoulli model and is convex in the weights, so there is one optimum. Squared error on a sigmoid output is not convex, and its gradient vanishes when the model is confidently wrong, so learning stalls on the cases that matter. C is the inverse of the penalty strength: I searched 0.01 to 10 and chose C = 1, with a validation PR-AUC of 0.820 ± 0.003 (Table 10). The coefficients convert to odds ratios (Figure 19).

**Where to show it.** Report: Section 4.2, Table 10 (first block), Figure 19 (coefficients). Notebook: "D.1 Hyper-parameter search", "E.12 What drives the predictions". App: Model training (select Logistic regression); Evaluation > Importance > "Logistic regression coefficients".

**Follow-up.** *What does the C curve look like?* Logistic regression is flat from C = 0.1 upwards, with training and validation scores close together (Figure 13). That pattern marks a high-bias model, so a weaker penalty would not help.

### 18. Why K odd; what happens at K = 1 and K = N?

**Answer (30 seconds).** K is odd so that a vote between two classes cannot tie. At K = 1 a report takes the label of its single nearest neighbour, so the boundary follows noise: my KNN scores 1.000 on its own training folds under distance weighting, because every training row is its own nearest neighbour (Table 10). At K = N with uniform weights every report receives the same vote, the majority class, so it collapses to the majority baseline with accuracy 0.514 and recall 0 (Table 13). K is the bias-variance dial, small K meaning low bias and high variance. I searched 11, 25 and 51 and chose 51, with a validation PR-AUC of 0.764 ± 0.009, well below HGB's 0.838.

**Where to show it.** Report: Table 10 (KNN block), Figure 13 (validation curves), Table 13. Notebook: "D.1 Hyper-parameter search", "D.2 Validation curves and the learning curve". App: Model training (select KNN; the page shows the reduced grid and the training score against the validation score of the live run).

**Follow-up.** *Why is your best K the largest value on the grid?* The validation score was still rising at K = 51, so after the test scoring I searched K = 101 and 151 as a follow-up check: they scored 0.763 and 0.761 against 0.764 at K = 51, so 51 was the peak (Table 10, follow-up row; reports/metrics/followup_checks.json). The check changed no choice.

### 19. Naive Bayes assumes independence; your heatmap shows correlation; why keep it?

**Answer (30 seconds).** Gaussian naive Bayes assumes the features are independent given the class, and my heatmaps and Table 7 show they are not: previous-year hours and employees correlate at r = 0.996 with variance inflation factors above 160. The effect is that size evidence is counted twice and the probabilities become overconfident, which shows in a Brier score of 0.291 against 0.161 for HGB (Table 14) and in the calibration curves (Figure 15). It ranks poorly as well: its test ROC-AUC of 0.745 is the lowest of the eight (Table 13). I kept it deliberately as a fast generative baseline and as a test of calibration, not as a candidate for the final model.

**Where to show it.** Report: Figure 3 (corr), Table 7, Table 14, Figure 15 (calibration). Notebook: "B.8 Collinearity: variance inflation factors", "E.7 Calibration". App: Exploratory analysis > Correlations; Evaluation > Curves (calibration panel, with naive Bayes selected).

**Follow-up.** *Would you ever use naive Bayes in production here?* No, because the app shows probabilities and the cost rule takes them at face value, and a Brier score of 0.291 fails the S6 limit of 0.20. Its capacity threshold of 0.001 (Table 14) is another sign that its scores sit on an unusable scale.

### 20. How does a random forest reduce variance vs one tree?

**Answer (30 seconds).** A single deep tree has low bias and high variance: a small change in the rows can change every split below the root. A random forest grows many deep trees on bootstrap samples and lets each split see only a random subset of features, which decorrelates the trees, and averaging decorrelated estimates cuts the variance without raising the bias much. The numbers show it: the single tree has a validation PR-AUC of 0.803 and a recall spread of ± 0.047 across folds, whereas the forest has 0.821 and ± 0.008 (Tables 10 and 12). The forest's training PR-AUC of 0.946 against 0.821 on validation shows a gap remains, so averaging reduces the variance but does not remove it.

**Where to show it.** Report: Table 10 (decision tree and random forest blocks), Table 12, Figure 13. Notebook: "D.1 Hyper-parameter search", "D.2 Validation curves and the learning curve". App: Model training (run the decision tree and the random forest at the same sample size and compare the training and validation columns).

**Follow-up.** *Why did boosting beat bagging here?* HGB adds shallow trees in stages, each fitted to the gradient of the log loss, so it also reduces bias, and it scored 0.838 against the forest's 0.821 in cross-validation. The forest's search covered only 4 of 24 grid points; the full grid, searched afterwards as a check, reached 0.822 with 200 trees, so the gap to HGB stands (Table 10).

### 21. Walk through one hyperparameter: meaning, range, validation curve.

**Answer (30 seconds).** Take the decision tree's max_depth, the longest root-to-leaf path. I searched 4, 8, 12 and unlimited, from a readable tree to unlimited growth. In Figure 13 the tree underfits at depth 4, where training and validation scores are equal, overfits at unlimited depth, where the gap widens, and peaks at depth 12. The selected tree has a validation PR-AUC of 0.803 ± 0.004, against 0.793 for unlimited depth (Table 10). I paired it with min_samples_leaf of 50, which makes each leaf's lost-time rate steadier. Depth buys flexibility and costs stability.

**Where to show it.** Report: Figure 13 (validation), Table 10 (decision tree block, columns "Values searched", "Why these bounds", "Selected", "Best other value"). Notebook: "D.2 Validation curves and the learning curve". App: Model training (select Decision tree and press Run tuning).

**Follow-up.** *Why does the random forest's validation score keep rising with depth when a single tree overfits?* Each deep tree overfits, but averaging many decorrelated trees cancels much of that variance, so the forest's validation score rises although its training score is far higher (0.946 against 0.821, Table 10). That is why I let the forest grow to unlimited depth and capped the single tree at 12.

---

## Part 5. Evaluation and decision (questions 22 to 25)

### 22. Why is accuracy inadequate; what is the primary metric?

**Answer (30 seconds).** Accuracy prices both errors alike, and here they are not alike: a missed lost-time case costs about USD 4,000 and an unnecessary review about USD 200 (Table 1). The majority-class rule is 51.4% accurate while finding no lost-time case (Table 13). Accuracy would also choose the wrong threshold: HGB at 0.5 has accuracy 0.764 and recall 0.740, whereas at the capacity threshold accuracy falls to 0.666 (from the counts in Table 17) while recall rises to 0.941. The primary metric is PR-AUC (average precision), 0.842 for HGB, because the adjuster works down a ranked queue with limited capacity. I report accuracy but do not select on it.

**Where to show it.** Report: Table 13 (default threshold), Table 14 and Table 17 (capacity threshold). Notebook: "E.2 Comparison at threshold 0.5", "E.3 Operating point at specificity 0.40", "D.1 Hyper-parameter search" (the search refits on average precision). App: Evaluation > Comparison (both tables); Evaluation > Threshold explorer (move the threshold and watch recall and cost).

**Follow-up.** *Why not select on F1?* F1 weights precision and recall equally, whereas a miss costs twenty times a false alarm in my base case, and expected cost states that asymmetry directly. I report F1 (0.733 at the capacity threshold) but do not optimise it.

### 23. ROC-AUC vs PR-AUC at your prevalence?

**Answer (30 seconds).** ROC-AUC is the probability that a random lost-time case scores above a random other case, and it does not depend on prevalence. PR-AUC averages precision over recall levels, and its no-skill level equals the prevalence, 0.486. At 48.6% prevalence the classes are nearly balanced, so the two metrics tell the same story: HGB scores 0.843 on ROC-AUC and 0.842 on PR-AUC, and all nine ranked models, the eight plus the lookup baseline, come out in the same order on both (Table 13; Figures 16 and 17). I made PR-AUC primary because precision is the review cost per lost-time case. With rare positives the two would diverge.

**Where to show it.** Report: Table 13 (AUC and PR-AUC columns), Figure 16 (roc), Figure 17 (pr), Section 4.4. Notebook: "E.4 ROC and precision-recall curves". App: Evaluation > Curves.

**Follow-up.** *At 5% prevalence which metric would you trust?* PR-AUC, because ROC-AUC could stay high while precision collapsed under the large negative class, and the no-skill PR-AUC would be 0.05. At my prevalence either would do, which is why Section 4.4 says the two rank models alike.

### 24. Are the top two models statistically different; why choose one anyway?

**Answer (30 seconds).** They are close, and the report says so. HGB leads the MLP by 0.006 PR-AUC over the five folds and wins every one (paired t-test p = 0.007), but the corrected resampled t-test gives p = 0.029 and the Wilcoxon test cannot go below 0.0625 with five folds (Table 15). The MLP's test PR-AUC of 0.836 lies inside HGB's mine-year bootstrap interval. I chose HGB by the rule fixed before testing, the highest cross-validated PR-AUC; Table 16 only checks that choice, with lower expected cost at 2:1, 3:1 and 5:1 and a higher PR-AUC on the temporal split (0.843 against 0.838). Against the lookup baseline the difference is clear, with a clustered McNemar p below 0.001.

**Where to show it.** Report: Table 15 (stats), Table 16 (decision), Table 18 (criteria results). Notebook: "E.9 Is the best model really better? Statistical tests", "E.11 The final-model decision". App: Evaluation > Criteria and tests (McNemar table and bootstrap intervals).

**Follow-up.** *Neither model passes S5, and the MLP scores faster; why not pick the MLP?* The MLP reaches 0.777 on S5 at a 51.1% flag rate and HGB 0.777 at 50.7%; among the top half of test scores, the planned form of S5, they find 0.769 and 0.770, so both miss (Table 18). Under the workload rule they are level (question 31). HGB was selected by the rule fixed before testing, the highest cross-validated PR-AUC (0.838 against 0.831); Table 16 checks that choice and did not make it. HGB also needs no scaler and has SHAP values for the app's explanations.

### 25. What harm could the model cause, who bears it, what mitigates it, how would you detect drift?

**Answer (30 seconds).** The main harm is a missed lost-time case: at the capacity threshold HGB misses 579 of 9,873 (Table 19). Metal and non-metal workers bear most of it, 9.1% of their lost-time cases against 3.0% in coal. A second harm is misuse: the label records restricted-duty practice, so a score could be used to contest a claim or to rank mines, which the model card rules out (Table 20). Mitigation is an adjuster confirming every flag, no use for reducing benefits, and a quarterly recall comparison by sector. For drift I track the flag rate against the plan (0.763 of all filings, 0.743 of open reports under the workload rule), and PR-AUC and Brier score by sector on closed cases.

**Where to show it.** Report: Table 19 (errors), Table 20 (model card), Table 22 (uncertainty and generalisability checks), Section 5.5. Notebook: "E.8 Results by commodity", "E.13 Temporal split: train on 2010-2019, test on 2020-2025", "F.4 Limitations". App: Evaluation > Threshold explorer ("By sector at this threshold"); About > Limits and ethics (the warning "Decision support only"); the sidebar notice on every page.

**Follow-up.** *Labels arrive only when cases close, so how do you detect drift in time?* The flag rate and the score distribution need no labels, so I watch them on each daily batch against the plan (0.763 at the capacity threshold, 0.743 under the workload rule), and PR-AUC and Brier score are checked on closed cases by sector, with retraining below a PR-AUC of 0.75 or above a Brier score of 0.20 (Table 20). The temporal split shows why this matters: on 2020 to 2025 the realised specificity was 0.373 against the planned 0.40, and about one in twenty late-2025 reports may still be re-coded (Section 5.3).

---

## Part 6. Questions from the external review (questions 26 to 32)

### 26. At exactly what moment is the model used, and what evidence shows that the lost-time outcome is still unknown then?

**Answer (30 seconds).** The model scores a report on receipt of Form 7000-1, at its filing date, a median of 8 days after the accident. The extract holds only the final degree and days values, with nothing as they stood at filing, so the data cannot show that the outcome was still unknown. The study is therefore retrospective, and a prospective trial is needed. Two facts bear on the question. At filing 45.0% of reports already carry a return-to-work date and are not scored, which leaves the 55% still open. Among those, the lost-time share rises with filing lag, from 0.62 within a day to 0.84 after 30 days, so the outcome partly shapes when the form is filed.

**Where to show it.** Report: Section 1.1 (the paragraph that fixes the moment of scoring), Section 5.3 (Limitations), Table 22 (row for the reports still open at filing). Artefact: the `return_to_work` block of `reports/metrics/followup_checks.json`. App: About > Limits and ethics.

**Follow-up.** *What would a prospective trial need?* Every report scored on the day it is received, the score stored, and the outcome compared once the claim closes. The model should be fitted on open-at-filing reports, which gives a recall of 0.877 under the rule on the retrospective data.

### 27. Using Table 22, calculate the flag rate for open-at-filing cases: how does it fit a 75% review capacity?

**Answer (30 seconds).** The flag rate is the lost-time share times recall, plus the other share times the false-positive rate. In Table 22 the 11,290 open test reports have a lost-time share of 0.712, and at the capacity threshold of 0.211 recall is 0.949 and specificity 0.219. The flag rate is therefore 0.712 x 0.9485 + 0.288 x 0.7813 = 0.900: nine reports in ten are flagged, fifteen points above a capacity of three in four. The threshold was set for specificity 0.40 on all filings, where prevalence is 0.486 and the flag rate 0.763. Open reports are more often lost-time, so the same specificity flags more of them. That is why I recommend the workload rule, which flags three in four.

**Where to show it.** Report: Table 22 (open-at-filing row), the capacity and equal-workload policies table in Section 4.3, Section 4.4 (the flag-rate identity). Artefact: `reports/metrics/followup_capacity_policy.csv` (rows `open_at_filing`, `hist_gb`, `capacity_threshold`). App: Evaluation > Capacity policy.

**Follow-up.** *Could a higher threshold fit the capacity?* Yes, and the workload rule does that: its cut-off of 0.376 flags 0.743 of the open test reports with recall 0.850. The cost is that this recall is below the 0.92 level of S3, which I set for all filings.

### 28. How do you establish the historical values of the retained Mines snapshot flags?

**Answer (30 seconds).** I cannot, and that is why I removed them. Mines.zip is a single snapshot taken at the pull date. The safety committee, Part 48 training and portable operation flags carry no effective date, so for an accident in 2012 the file does not say what they were then. I took the three out of the model inputs, 52 remain, and kept them in the base table for description only. The location and commodity fields stay, because a mine rarely moves, and the coal or metal flag agrees with the production table on 100.0% of rows. The 103(i) status has a status date, so I re-dated it to the accident. Test PR-AUC is 0.842 after the rerun, against 0.841 with the flags.

**Where to show it.** Report: Section 3.7, Table 8 step 7, Table 5 (the data dictionary of the 52 model features). Notebook: "C.4 Feature engineering" and "C.5 Leakage check". App: Data explorer (the three flags remain in the base table).

**Follow-up.** *Why keep the 103(i) status when it also comes from the snapshot?* The Mines file gives a status and the date it began, so I can state its value on the accident date. Setting the status to one constant level moves test PR-AUC by -0.0002 (sensitivity variant `without_status_103i`), so the result does not depend on it.

### 29. Can a test-year outcome contribute to a previous-year-history feature in a later training row? Show the data flow.

**Answer (30 seconds).** Yes. The history features of a row in mine-year Y count the mine's cases in year Y minus 1 from the full accident file, together with their lost-time outcomes. The split keeps each mine-year on one side, but not the year before it. The path therefore runs from a test row's label, through the lagged counts of the same mine's next-year row, into the training matrix. It affects 11,426 of the 81,289 training rows, 14.1%, and 11,416 of them carry a history computed from test outcomes. I refitted HGB with that history set to missing: test PR-AUC moved from 0.8419 to 0.8424, a change of +0.0006, under 0.001. The temporal split has no such path and gives PR-AUC 0.843.

**Where to show it.** Report: Section 3.9 (the paragraph that begins "Mine-year grouping leaves one path between the splits"), Section 4.5, Table 22 (history row). Artefact: `reports/metrics/sensitivity.json`, variant `history_without_test_outcomes`; code: `rows_with_test_history` in `src/mlda/sensitivity.py`.

**Follow-up.** *Why is the primary split left as it is?* The history definition is the operational one: a score for a 2026 filing also uses the outcomes of 2025, which are known by then. The overlap only affects how clean the test estimate is, the audit puts the effect under 0.001, and the temporal and mine-grouped splits (PR-AUC 0.843 and 0.837) are reported beside it.

### 30. Why is expected recall under random selection of half the cases 0.50 rather than the class prevalence?

**Answer (30 seconds).** Recall is the share of lost-time cases that are flagged. If I choose half of all reports at random, each lost-time case has a one-in-two chance of being chosen, so on average half of them are flagged and expected recall is 0.50 whatever the prevalence. The prevalence, 0.486, is the expected precision, because a random half has the same mix as all reports. The same holds for open reports: reviewing three in four at random gives recall 0.75 and precision 0.712. Those are the lines that my 0.855 recall and 0.812 precision at three in four have to beat. Prevalence would be the answer only for precision, or for the no-skill PR-AUC.

**Where to show it.** Report: Section 4.3 (the paragraph under the policies table: a random ranking gives recall equal to the share reviewed and precision equal to prevalence), Section 4.4 (the no-skill level of PR-AUC). Notebook: "E.4 ROC and precision-recall curves".

**Follow-up.** *What is the lift of HGB over random at half of all filings?* At the top 50% of test scores HGB finds 0.770 of the lost-time cases against 0.50 for a random half, a lift of 1.54, and its precision of 0.749 against the prevalence of 0.486 gives the same ratio.

### 31. What changes when HGB and MLP are compared at exactly the same workload?

**Answer (30 seconds).** At exactly the same workload nothing separates them. At their own capacity thresholds they flag different shares, 0.763 for HGB and 0.768 for the MLP, so recall is not strictly comparable. Reviewing exactly three in four of the open test reports, HGB finds 0.855 of the lost-time cases and the MLP 0.854, with precision 0.812 and 0.811. The paired mine-year bootstrap puts the recall difference at 0.000, interval -0.003 to 0.004; on PR-AUC for open reports HGB is ahead by 0.003, interval 0.001 to 0.005. HGB stays the choice because the rule fixed before testing ranks it first, and its lead over the lookup baseline holds: 0.027 in recall, interval 0.021 to 0.033.

**Where to show it.** Report: the capacity and equal-workload policies table in Section 4.3, Table 22 (bootstrap row). Artefacts: `reports/metrics/followup_capacity_policy.csv` and the `paired_bootstrap_open_at_filing` block of `followup_checks.json`. App: Evaluation > Capacity policy.

**Follow-up.** *Does the equal-workload comparison use the test labels to set the cut-off?* The cut-off is read from the test scores alone, and the labels only count the hits. The deployed cut-off, 0.376, comes from the out-of-fold scores of the open training reports, so no test row enters it.

### 32. Which notebook steps recompute results, which reuse saved results, and what command reproduces the full search?

**Answer (30 seconds).** In default mode the notebook recomputes the exploratory tables and the three cheap searches, for Gaussian naive Bayes, logistic regression and the decision tree, and checks that each reproduces the stored table within 0.001 of the mean score. It also refits every selected pipeline on the training split. For the other five models it loads the stored cross-validation tables and chosen settings. That takes under ten minutes. With `FULL_RUN=1` it reruns all eight searches inside the notebook, which took 41.4 minutes in the committed run. The command `uv run mlda-train` reproduces the whole study, with the temporal and sensitivity studies, in 86 minutes.

**Where to show it.** Notebook: "D.1 Hyper-parameter search" (the agreement check), "D.3 Refit the selected pipelines on the training split", "F.3 Reproducibility checklist", "F.6 Run time of this notebook". README: "Reproduce" and "Expected run times". App: About > Reproduce.

**Follow-up.** *Why compare within 0.001 and not exactly?* Floating-point sums differ in the last digits between the BLAS builds on Windows and Linux. The tolerance is a third of the smallest fold standard deviation, and the stored choice must lie within it of the live best.

---

## Weak points and how to answer honestly

State the fact first, give the evidence second, then say what you did and what you would do next. Do not defend a weak point by changing the subject.

**1. S5 is missed (0.777 against 0.78).**
*Answer.* "HGB finds 0.777 of lost-time cases at the out-of-fold threshold for a 50% flag rate, against my pre-set target of 0.78, a shortfall of about one case per 1,000 reports; among the top half of test scores, the planned wording, it finds 0.770. The miss repeats on the temporal split (0.747) and the mine-grouped split (0.762), so it is not a quirk of one test fold. I left the target where it was. S3 passes at the capacity threshold with recall 0.941, but that target was set for all filings; under the recommended workload rule recall on open reports is 0.850. The MLP misses S5 as well, with 0.777 at a 51.1% flag rate and 0.769 at exactly half, so no model passes all seven criteria."
*Show.* Table 18 (S5 rows), Table 22 (S5 row); notebook "E.10 Success criteria"; app Evaluation > Criteria and tests.

**2. The label depends on restricted-duty practice.**
*Answer.* "Yes. Coal is 63.0% lost-time and metal and non-metal 38.6%, while the DART label hardly differs by sector (Cramer's V 0.015 against 0.240), so part of what the model learns is how employers code restricted duty and not only severity. I disclosed this in Sections 1.2 and 5.3, report all results by sector, and rule out using the score to rank mines. The limit I cannot remove is that this data does not separate severity from practice."
*Show.* Table 2; Table 20 (limitations, out-of-scope uses); notebook "B.12 Evidence for three claims that later sections use" and "E.8 Results by commodity"; app About > Limits and ethics.

**3. The random forest search covered 4 of 24 grid points.**
*Answer.* "That is right. RandomizedSearchCV sampled four candidates because each forest is costly, and it never tried 200 trees or the log2 feature subset, so the forest's 0.821 was a lower bound. After the test scoring I ran the full 24-point grid as a check (585 s): its best, 200 trees with the other settings unchanged, reached 0.822, still 0.016 behind HGB, so the choice stands (Table 10, follow-up row)."
*Show.* Table 10 (random forest block); `src/mlda/models.py` (`n_iter=4`); app Model training with the random forest selected ("Full search: 4 candidates (random)").

**4. KNN's best K sits at the grid edge.**
*Answer.* "K = 51 was the largest value in the planned grid (11, 25, 51), and the validation score was still rising, so I searched K = 101 and 151 afterwards as a check: 0.763 and 0.761 against 0.764, so 51 was the peak (Table 10, follow-up row). I fitted KNN on a 39,999-row sample of whole mine-years because scoring cost grows with the rows stored. Section 4.3 reports the check."
*Show.* Table 10 (KNN block); Figure 13.

**5. The SVM was fitted on a 10,000-row subsample.**
*Answer.* "Training time grows at least quadratically with rows and each fit already took about two and a half minutes, so I used 9,913 rows of whole mine-years (Table 10). That makes its 0.795 validation PR-AUC a statement about a small-sample SVM, and it has the largest fold-to-fold spread, ± 0.014. It also takes 31.0 s to score the 20,320 test reports against 0.28 s for HGB (Table 16). It would need to gain 0.043 PR-AUC from more data to match HGB, which I did not test."
*Show.* Table 10 (SVM block); Table 16 (prediction time); Table 12.

**6. Most commits are dated 14 and 15 October, with weekly follow-ups planned.**
*Answer.* "The history has 40 commits: 17 on 14 October, 21 on 15 October and 2 on 20 October. The first 38 come from a two-day sprint. Their messages carry the teaching-week milestone each covers, but the dates show the work was assembled in two days, and I did not edit them. The two commits of 20 October are dated revision commits after an external review: the feature-timing audit, and the rerun of the study with its artefacts. `docs/weekly_plan.md` sets out follow-up commits for each week from 19 October to 30 November, each with its own tests, and I will show those dates."
*Before the viva.* Run `git log --date=short --format="%ad %s"` and quote only what the log shows. If the planned commits are missing, say so and say what was done instead. Do not describe a plan as completed work.
*Show.* `git log`; `docs/weekly_plan.md`.

**7. Precision is below 0.5 in metal and non-metal mines.**
*Answer.* "At the capacity threshold precision is 0.487 for metal and non-metal mines against 0.739 for coal, and their PR-AUC is 0.708 against 0.913. Their lost-time prevalence is 0.381, so a threshold that flags about three reports in four sends more non-lost-time cases to review. I used one threshold for both sectors and did not test sector-specific thresholds; I report the gap in Table 20 and propose quarterly monitoring by sector. Their recall is also lower (0.909 against 0.970), so metal and non-metal workers bear more of both errors."
*Show.* Table 20 (subgroup performance), Table 22 (by sector); notebook "E.8 Results by commodity"; app Evaluation > Threshold explorer, "By sector at this threshold".

**8. The probabilities run low on open reports (0.592 against 0.712).**
*Answer.* "The model is fitted on all filings, where 48.6% are lost-time, but the cohort it serves is the 55% still open at filing, where 71.2% are. On the 11,290 open test reports the mean predicted probability is 0.592 against an observed rate of 0.712, so the probabilities for open reports are about 0.12 too low, and for reports filed after the return to work they are too high (0.354 predicted, 0.203 observed). The workload rule uses only the ranking, so its recall does not depend on the level, but a cost threshold applied to open-report probabilities would. A prospective deployment should recalibrate on open reports or fit on them (point 9)."
*Show.* `reports/metrics/followup_open_calibration.csv` and the `workload_rule` block of `followup_checks.json`; app Evaluation > Capacity policy (reliability of the reports still open at filing).

**9. A model fitted on open reports alone does better.**
*Answer.* "I fitted the final model on all filings, because the 45% filed after the return to work carry the same filing-time fields and nearly double the labelled rows. A check run after the test scoring fitted HGB on the 44,640 open-at-filing training reports only. Under the workload rule its recall on the open test reports is 0.877 against 0.855 for the model fitted on all filings, and its PR-AUC on those reports is 0.932 against 0.920. I did not change the final model, because the check came after the test rows were scored and the study stays as it was. A prospective trial should fit on open reports."
*Show.* The `open_only_refit` block of `reports/metrics/followup_checks.json`; Table 22 (the row for the refit on open-at-filing training rows).

**Further points a marker may raise**

- *The C_FN of USD 4,000 is an assumption.* True, and Table 1 says so (range 3,100 to 6,200). From about 10:1 upwards reviewing every case is the cheapest policy: HGB's cost-optimal threshold costs USD 103,346 per 1,000 at 20:1, USD 521 above the floor of 102,825 (the threshold comes from out-of-fold probabilities and does slightly worse than reviewing everyone on the test rows), so review capacity sets the threshold and cost separates models only at the low ratios that S4 tests.
- *The test set is one held-out fold and no external data were scored.* True (Section 5.3). The mine-year bootstrap interval of PR-AUC (0.826 to 0.856), the temporal split (0.843) and the mine-grouped split (0.837) are the checks I ran. A prospective trial on reports scored at receipt, fitted on open-at-filing reports, is the first item of future work (question 26).
- *The top two models are close.* See question 24: the corrected paired test gives p = 0.029 and the Wilcoxon test cannot show less than 0.0625 with five folds, so the lead is thin, and the choice rests on the selection rule fixed before testing, with lower cost at 2:1, 3:1 and 5:1 and the higher temporal PR-AUC as checks.
- *The app's live training differs from the full run.* It searches reduced grids on smaller samples so the hosted app stays within memory, and the page says so (Section 5.3). The committed models and the report come from the full run.
- *The cross-validation order of the SVM and the decision tree differs from the test order.* In cross-validation the tree is fifth and the SVM sixth, on the test set the reverse (Tables 12 and 13), and the differences are a few hundredths. Say that the gap between them is small and that neither is near the top.

---

## Open items to fill in before the viva

- The Discussions forum post: the report states 10 October 2026 (Section 1.3). Check that the topic title and the forum timestamp agree with that date and keep the URL to hand (question 1).
- Repository https://github.com/wiut13130/mlda-cw1 and live app https://mlda-cw1-8han4medy3o7d8izzcywbm.streamlit.app are on the report cover, in the README and in Appendix C.
- Open the live app the day before and again 30 minutes before the viva, because a Community Cloud app sleeps when idle; the fallback is `uv run streamlit run app/streamlit_app.py`.
- Re-check Table and Figure numbers against the final docx if any section changes.

*Note (20 October 2026).* The table and figure numbers in this file follow the key below, which has not been renumbered since the rebuild after the external review. The rebuild adds one table (the capacity threshold and equal-workload policies, in Section 4.3), so tables from Table 12 onwards may be one higher in the rebuilt report; check each number against the docx.

## Table and figure key (numbers used above)

Tables: 1 costs; 2 label mapping; 3 success criteria S1 to S7 and baselines; 4 sources and joins; 5 data dictionary; 6 descriptive statistics; 7 associations and VIF; 8 preprocessing decision log; 9 algorithms; 10 hyperparameter search; 11 capacity threshold and equal-workload policies (new on 20 October); 12 cross-validated scores; 13 test scores at threshold 0.5; 14 test scores at the capacity threshold; 15 statistical comparisons; 16 multi-criteria decision; 17 confusion counts; 18 success criteria results; 19 errors; 20 model card; 21 outcomes per 1,000 cases; 22 uncertainty and generalisability checks; 23 remaining descriptive statistics (Appendix A); 24 SHAP values (Appendix B); 25 commit history (Appendix D).

Figures: 1 target distribution; 2 Q-Q plots; 3 correlation heatmaps; 4 histograms; 5 box plots; 6 pair plot; 7 categorical against target; 8 missingness; 9 outlier fences; 10 PCA; 11 pipeline; 12 learning curve; 13 validation curves; 14 threshold and cost curves; 15 calibration; 16 ROC curves; 17 precision-recall curves; 18 permutation importance; 19 logistic coefficients; 20 confusion matrices.

## Where the other numbers come from

Everything above is in `report/sections/*.md` or `reports/metrics` except the following.

- Git history (40 commits: 17 on 14 October, 21 on 15 October and 2 on 20 October): `git log`, read on 20 October 2026. The count rises as you commit.
- No Kaggle copy of the accident file, and the data.gov access level "public": `docs/dataset_research.md` (research log of 14 October 2026).
- HGB cost at 10:1, 20:1 and 30:1 (USD 98,130, 103,346 and 103,110) and the floor of 102,825 on the test set: `reports/metrics/success_criteria.json`.
- The workload rule, the open-report calibration, the paired bootstrap, the refit on open reports and the capacity-threshold figures for open reports (questions 26 to 32 and weak points 8 and 9): `reports/metrics/followup_checks.json` and `followup_capacity_policy.csv`, `followup_workload_rule.csv` and `followup_open_calibration.csv` in the same folder. The history audit: `reports/metrics/sensitivity.json`, variant `history_without_test_outcomes`.
- The logistic-regression preprocessing comparison in questions 13 and 14 (average precision 0.808 to 0.809 across scalers, 0.756 with ordinal codes): output of the notebook cell "C.7 Do the preprocessing choices matter?" on a 12,000-row sample.
- Derived by arithmetic from report numbers: 0.292 squared = 0.085 (question 8); one part in 81,289 and the factor 19 over 18 (question 6); accuracy 0.666 at the capacity threshold = (9,294 + 4,243) / 20,320 (question 22); 0.016 = 0.838 - 0.822 and 0.043 = 0.838 - 0.795 (weak points 3 and 5); 0.900 = 0.712 x 0.9485 + 0.288 x 0.7813 (question 27); 1.54 = 0.770 / 0.50 (question 30).
- Not available: the date of the Discussions forum post (question 1).
