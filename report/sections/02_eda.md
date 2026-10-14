# 2 Exploratory Data Analysis

## 2.1 Summary

Medians, quartiles and box plots were read before any model was fitted (Tukey, 1977), on training rows only except for the target count. The joined table holds 101,609 rows, of which 48.6% are lost-time cases ({fig:target}), and {tab:dictionary} gives the categorical, numeric and binary feature counts. Raw experience and exposure measures are right-skewed ({tab:descriptives}, {fig:hist}, {fig:qq}), which led to median imputation, log1p features and retained outliers in Section 3 ({fig:outliers}). The mine's previous-year lost-time share and the nature of the injury are most associated with the target ({tab:associations}, {fig:box}, {fig:cat}). Previous-year hours and employees measure mine size twice ({fig:corr}), a problem for logistic regression and naive Bayes. Missing values sit mainly in structural placeholders ({fig:missing}), and the classes overlap in every two-dimensional view ({fig:pair}, {fig:pca}).

## 2.2 Structure, variable types and target

Each row is one injured worker on one Form 7000-1; the statistics describe the 81,289 training rows. I classified every feature by measurement scale, because scale decides the summary statistic, the plot and the encoder (Han et al., 2022); {tab:dictionary} gives each type. Categorical features are nominal codes, summarised by mode and level count, except mine_size_band, whose five employee bands are ordinal. The modal nature of injury, sprain or strain, covers 32.4% of rows, and OCCUPATION has 193 levels, which motivated rare-level pooling. Integer-coded weekday and month were also treated as categorical, since their code values carry no magnitude. Type corrections converted numeric text, kept zero-padded codes such as MINE_ID as strings and parsed HHMM clock fields as integers (Section 3.2).

The target lost_time (Section 1.2) is close to balanced, with negatives outnumbering positives by 1.06 to one ({fig:target}). The majority-class baseline, predicting no lost time for every case, is correct for 51.4% of cases yet finds none, so accuracy is reported but not used for selection. Resampling methods such as SMOTE target minority classes too rare to learn (He and Garcia, 2009), and this target is not one. The cost gap between a missed lost-time case and an unnecessary review is handled by the decision threshold instead (Section 3.6).

![Counts and shares of non-lost-time and lost-time cases for all 101,609 in-scope rows before the split, drawn as a bar chart because two class sizes are compared. Source: author's analysis of MSHA open data.](reports/figures/F02_target_distribution.png){#fig:target width=3.8}

## 2.3 Descriptive statistics and distributions

{tab:descriptives} covers the numeric features with the strongest target association or most extreme shape, and {tab:descriptives_rest} in Appendix A covers the rest. Measures of centrality diverge under right skew, and total experience has a mean of 9.8 years, a median of 6.0 and a trimmed mean between them. The mean assumes symmetry and is pulled up by long careers, the median ignores how far the tail reaches, and the trimmed mean discards a fifth of the rows to resist both (Bruce et al., 2020). I chose the median as the imputation centre for skewed features. A mean fill would inflate the tenure of unrecorded rows, which are lost-time more often (Section 3.4), so the bias would fall on the costliest misses.

Table: Sample statistics of 12 of the 20 numeric features on the 81,289 training rows; the trimmed mean cuts 10% from each tail, variance and SD (sample standard deviation) use the n - 1 denominator, prev_hours is in thousands of employee-hours, and skewness is read against the bands below 0.5 symmetric, 0.5 to 1 moderate and above 1 high. Source: author's analysis of MSHA open data. {#tab:descriptives}

| Feature (unit) | Missing % | Mean | Median | Trimmed mean | Variance | SD | Min | Q1 | Q3 | Max | IQR | Skewness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TOT_EXPER (years) | 1.2 | 9.8 | 6.0 | 8.0 | 108.6 | 10.4 | 0.02 | 1.9 | 14.4 | 60.0 | 12.5 | 1.37 |
| log1p_tot_exper | 1.2 | 1.89 | 1.95 | 1.90 | 1.09 | 1.04 | 0.02 | 1.07 | 2.74 | 4.11 | 1.66 | -0.07 |
| MINE_EXPER (years) | 1.4 | 5.8 | 2.6 | 4.0 | 64.6 | 8.0 | 0.02 | 0.7 | 7.5 | 58.0 | 6.8 | 2.25 |
| JOB_EXPER (years) | 4.5 | 6.2 | 3.0 | 4.5 | 65.0 | 8.1 | 0.02 | 0.9 | 8.1 | 60.0 | 7.1 | 2.09 |
| exper_ratio_mine (ratio) | 1.7 | 0.72 | 1.00 | 0.77 | 0.15 | 0.38 | 0.00 | 0.33 | 1.00 | 1.00 | 0.67 | -0.83 |
| hours_into_shift (hours) | 3.8 | 4.8 | 4.5 | 4.7 | 8.4 | 2.9 | 0.0 | 2.5 | 7.0 | 16.0 | 4.5 | 0.42 |
| longitude (degrees) | 0.2 | -92.6 | -87.5 | -90.9 | 217.9 | 14.8 | -166.6 | -104.0 | -81.5 | -64.9 | 22.5 | -1.20 |
| prev_hours (thousand hours) | 1.2 | 612.7 | 247.0 | 413.6 | 1,044,971 | 1,022.2 | 0.001 | 69.6 | 829.6 | 7,735.3 | 760.0 | 4.27 |
| prev_employees (persons) | 1.2 | 281.6 | 113.0 | 186.4 | 239,644 | 489.5 | 1 | 33 | 367 | 3,837 | 334 | 4.59 |
| prev_incidents (count) | 1.2 | 15.8 | 6.0 | 9.9 | 667.0 | 25.8 | 0 | 1 | 19 | 190 | 18 | 3.11 |
| prev_lost_time_rate_200k (log1p) | 1.2 | 0.86 | 0.79 | 0.78 | 0.65 | 0.81 | 0.00 | 0.00 | 1.46 | 3.93 | 1.46 | 0.60 |
| prev_lost_time_share (ratio) | 19.8 | 0.48 | 0.50 | 0.48 | 0.11 | 0.33 | 0.00 | 0.22 | 0.75 | 1.00 | 0.53 | 0.01 |

Variance, a measure of variability, is in squared units, so the table adds the sample standard deviation in original units. For prev_hours it exceeds the mean, so a few very large mines dominate the spread. The standard deviation assumes symmetric spread, whereas the interquartile range (IQR) covers the middle half of the rows and assumes no shape (Han et al., 2022). I used the IQR for skewed features and outlier fences. Mines with 200 or more employees supply 36.7% of training rows, and treating their exposure as error would discard the insurer's largest group of cases.

The skewness columns of {tab:descriptives} and {tab:descriptives_rest} put nine of the 20 numeric features in the high band above 1 in absolute value, with prev_employees the most skewed continuous measure at 4.59. A log1p transform assumes positive values with multiplicative spread. A Yeo-Johnson transform fits its exponent but is harder to interpret, and raw values suit only trees, which split on rank order (Kuhn and Johnson, 2019). I added log1p versions of total experience and previous-year hours beside the raw columns. Compressing the tail keeps small and medium mines distinguishable in the insurer's score.

Normality was checked with Q-Q plots of six features ({fig:qq}). Total experience bends away from the reference line at both ends, its log1p version follows the line except at its bounds, and prev_lost_time_share has point masses at both extremes. A Shapiro-Wilk test would reject normality for trivial departures at about 80,000 rows, whereas a Q-Q plot shows where a distribution departs (Bruce et al., 2020). Pearson's r is pulled by tail points, so Spearman's rank correlation is reported beside it. Gaussian naive Bayes probabilities are likely to be miscalibrated, which would move a cost-based threshold and the review workload.

![Normal Q-Q plots of six numeric features on the training split, with 5,000 seeded points per panel and skewness computed on all training rows; Q-Q plots are used because departures from the dashed reference line show where each distribution differs from a normal one. Source: author's analysis of MSHA open data.](reports/figures/F09_qq_plots.png){#fig:qq width=6.0}

## 2.4 Correlation and association

The correlation matrix of the 20 numeric features was computed with Pearson's r for linear and Spearman's rank correlation for monotone association ({fig:corr}). The two disagree most for log1p_prev_hours against prev_employees, 0.645 against 0.994, because the relation is monotone but curved. Pearson assumes linearity and reacts to extreme points, whereas Spearman assumes only monotonicity (Han et al., 2022). I therefore read redundancy from Spearman, since a Pearson-only reading would understate it. Redundant size measures let logistic-regression coefficients trade weight from one refit to the next, which would make the reasons the application shows beside each flagged claim unstable.

![Pearson (left) and Spearman (right) correlation matrices of the 20 numeric features on the training split, drawn as heatmaps on one diverging scale centred at zero so that the two coefficients can be compared cell by cell. Source: author's analysis of MSHA open data.](reports/figures/F06_correlation_heatmaps.png){#fig:corr width=6.3}

Point-biserial r, Pearson's r with one dichotomous variable, ranked the numeric features against the target, and bias-corrected Cramer's V, which corresponds to the Phi coefficient for two levels, ranked the categorical features ({tab:associations}). The leaders are prev_lost_time_share at r = 0.292 and NATURE_INJURY_CD at V = 0.391, far below the 0.95 level at which I would suspect leakage. Mutual information would capture non-monotone effects but carries no sign, and chi-square p-values cannot rank features when trivial effects are significant at this sample size (Tan et al., 2019). Total experience and shift start hour also enter the top ten, consistent with Yedla et al. (2020). Every leading field is known at filing, so the score is available when the reserve is set.

Table: Association of the numeric features (point-biserial r, top 10 by absolute value) and the categorical features (bias-corrected Cramer's V, top 10, number of levels in brackets) with the lost-time target, and the numeric features whose variance inflation factor exceeds 5 with their most correlated partner, all on the 81,289 training rows; every listed r and V has p < 0.001. Source: author's analysis of MSHA open data. {#tab:associations}

| Rank | Numeric feature | r | Categorical feature (levels) | V | Feature with VIF > 5 | VIF | Partner (Pearson r) |
|---|---|---|---|---|---|---|---|
| 1 | prev_lost_time_share | 0.292 | NATURE_INJURY_CD (27) | 0.391 | prev_hours | 176.4 | prev_employees (0.996) |
| 2 | prev_lost_time_rate_200k | 0.205 | INJ_BODY_PART_CD (46) | 0.364 | prev_employees | 164.8 | prev_hours (0.996) |
| 3 | longitude | 0.135 | INJURY_SOURCE_CD (124) | 0.311 | prev_lost_time_rate_200k | 8.3 | prev_lost_time_share (0.772) |
| 4 | prev_incidents | 0.099 | ACCIDENT_TYPE_CD (41) | 0.282 | TOT_EXPER | 7.5 | log1p_tot_exper (0.895) |
| 5 | exper_ratio_mine | -0.096 | FIPS_STATE_CD (54) | 0.243 | log1p_tot_exper | 6.0 | TOT_EXPER (0.895) |
| 6 | exper_ratio_job | -0.083 | PRIMARY_CANVASS (5) | 0.239 | | | |
| 7 | prev_incident_rate_200k | 0.073 | COAL_METAL_IND (2) | 0.237 | | | |
| 8 | log1p_tot_exper | 0.068 | ACTIVITY_CD (96) | 0.225 | | | |
| 9 | TOT_EXPER | 0.060 | CLASSIFICATION_CD (21) | 0.217 | | | |
| 10 | shift_begin_hour | 0.058 | OCCUPATION (193) | 0.215 | | | |

Multicollinearity is worst for mine size. Previous-year hours and employees correlate at r = 0.996, both with a variance inflation factor (VIF) above 160, and {tab:associations} lists three more features with a high VIF. In logistic regression this inflates coefficient variance and destabilises signs (Hosmer et al., 2013). In naive Bayes it breaks conditional independence, so size evidence is counted twice and posteriors become overconfident, although the ranking can survive (Domingos and Pazzani, 1997). Dropping one column assumes it adds nothing, and principal components give up interpretable features. I kept both, under the L2 penalty in logistic regression, and did not interpret the pair's coefficients. Overconfident probabilities would misplace the threshold, so calibration is checked in Section 4.

## 2.5 Visual analysis and plot justification

Figures were drawn with Matplotlib (Hunter, 2007) and seaborn (Waskom, 2021). Bars in {fig:target} compare two counts by length, which reads more accurately than pie angles. The heatmaps in {fig:corr} share one diverging scale but miss interactions and categorical features, which {fig:cat} covers. I did not plot a full scatter matrix of the 20 numeric features, because 190 panels would be unreadable.

{fig:hist} overlays class-wise density histograms and kernel density curves, because density puts classes of unequal size on one scale. All panels share 30 bins, since the Freedman-Diaconis rule would set between one bin for the capped victim count and 274 for hours per employee. The 60-bin log-experience panel in {fig:outliers} repeats the comb seen here, consistent with whole-year recording. The classes overlap except on the previous-year lost-time share, where non-lost-time cases pile up at zero. Box plots ({fig:box}) give the medians and quartiles that skewed features need; lost-time cases have the higher median lost-time share and lie further east, near the Appalachian coal fields.

![Density histograms (30 bins) with kernel density estimates of the 20 numeric features by outcome class on all 81,289 training rows, drawn as densities so that classes of unequal size share one scale. Source: author's analysis of MSHA open data.](reports/figures/F03_histograms.png){#fig:hist width=6.3}

![Box plots of the 20 numeric features by outcome class on all 81,289 training rows, with whiskers at 1.5 times the IQR and outlier points omitted, chosen because medians and quartiles are the comparison that skewed features need. Source: author's analysis of MSHA open data.](reports/figures/F04_class_boxplots.png){#fig:box width=6.3}

The corner pair plot ({fig:pair}) draws the five features most correlated with the target on 5,000 seeded rows to avoid overplotting; the lost-time history measures fan out together and the classes mix in every panel. {fig:cat} plots lost-time rates per level for the eight categorical features with the highest V, since a rate shows how far a level moves the probability. Cuts and lacerations sit near 18% against 48.6% overall, coal mines above it, and printed sizes flag thin data.

![Corner pair plot of the five numeric features with the largest absolute point-biserial correlation, for 5,000 seeded training rows without missing values, coloured by outcome to show pairwise structure and class overlap. Source: author's analysis of MSHA open data.](reports/figures/F05_pairplot_top.png){#fig:pair width=5.8}

![Lost-time rate per level for the eight categorical features with the highest Cramer's V on the training split, showing the twelve largest levels and a pooled bar for levels under 100 rows, with the overall rate dashed; proportion bars are used because the question is the rate per level. Source: author's analysis of MSHA open data.](reports/figures/F07_categorical_vs_target.png){#fig:cat width=5.1}

Because imputation is decided per column, {fig:missing} ranks the most incomplete columns as bars; underground, equipment and contractor gaps are structural, and one indicator covers the 985 rows missing all exposure columns (Section 3.4). In {fig:outliers}, IQR fences flag 4,020 raw experience values and Z-score fences only 714, because the tail inflates the standard deviation. After log1p neither rule flags any experience value, and Section 3.5 kept these real long careers.

![Percentage of missing values in the 30 most incomplete columns of the joined base table on the training rows, ranked as bars so that structural and sporadic gaps can be told apart. Source: author's analysis of MSHA open data.](reports/figures/F08_missingness.png){#fig:missing width=5.6}

![Histograms (60 bins) of total mining experience and previous-year employee-hours on the training split before (left) and after (right) the log1p transform, with the right-hand axes in log1p units, IQR fences (k = 1.5) and Z-score fences (|z| > 3) and the number of rows each flags, used to compare the two outlier rules on skewed data. Source: author's analysis of MSHA open data.](reports/figures/F10_outlier_fences.png){#fig:outliers width=6.3}

Two principal components of the standardised numeric features, explaining 18.5% and 16.1% of the variance, show whether the classes separate linearly ({fig:pca}). The classes overlap throughout, although a plane holding a third of the variance cannot rule out separation in more dimensions.

![First two principal components of the 20 median-imputed and standardised numeric features, fitted on the training split with 5,000 seeded points shown and coloured by outcome, used to check for linear class separation in two dimensions. Source: author's analysis of MSHA open data.](reports/figures/F11_pca_projection.png){#fig:pca width=4.6}
