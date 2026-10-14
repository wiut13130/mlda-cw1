# Report plan and marking rubric: 6COSC017C CW1 (student ID 13130)

Target: Report >= 32/35, Practical >= 59/65, total >= 91.

## 0. Facts from the brief that shape the plan

- Four deliverables: report, reproducible notebook, deployed multi-page Streamlit app, Git repository (MIT/Apache licence,
  README, weekly commits matching milestones, requirements.txt or pyproject.toml, and "native locking or full containerization").
- Practical part (65) is itemised: Data load 5 (joins must be correct when two sources are used); EDA 10 (statistical summary 4,
  correlation matrix 3, other graphs 3); Preparation 10 (missing values 2, scaling 2, correcting error data 2, feature engineering 2,
  train/test split 2); Training 10 (3+ models 6, hyperparameter tuning 4); Evaluation 10; Deployment 10 (working link 2,
  clean solution 3, dashboard 5); Version control 5 (weekly commits 3, README 2); Reproducibility 5 (requirements 2, lock or container 3).
- Report (35): A Introduction 5; B EDA summary 1 + statistics/correlations 2 + visualisation 2; C Preparation summary 1 + per-step
  justification 2; D Modelling summary 1 + algorithm selection 4 + model selection per algorithm (hyperparameters) 4 + evaluation
  metrics 4 + comparison and final model 4; E Conclusion findings 2 + interpretation 3.
- Dataset must be reserved in the module Discussions forum (topic + URL); Kaggle banned; a solution similar to one on the internet scores 0.
- The brief says the data "should be split into training, test and validation sets": name all three (StratifiedKFold on the
  training portion is the validation set).
- Example comparison table columns are TPR, TNR, AUC, Recall, Precision. Keep those exact columns (footnote TPR = Recall) and add
  F1, PR-AUC, Accuracy, CV mean +/- std.
- Each rubric block has a 1-mark "Summary of ..." line (EDA, Preparation, Modelling): each of those sections opens with a
  paragraph headed "Summary".
- Module-recommended texts (Week 1 slides): Tan, Steinbach and Kumar; Han, Pei and Tong; Bishop; Provost and Fawcett; pandas and
  scikit-learn documentation.
- Format: Times New Roman or Arial 12, single-spaced, A4; cover sheet with student ID, module title and marker's name (Ruslan Isaev);
  the student's name must not appear anywhere; ID on every page; table of contents with page numbers; all pages numbered; Harvard
  referencing; submitted inside MLDA.CW1.13130.zip with the notebook and the dataset.

## 1. Justification Unit (JU) template

Every "justified" step in sections 2 to 4 is one paragraph containing, in order:
(a) the evidence in this dataset that triggered the decision (a number plus a Table/Figure reference);
(b) at least two alternatives and the assumption each makes;
(c) the trade-off and why the chosen option wins here;
(d) one citation;
(e) one sentence linking to the business decision or to the false-negative/false-positive costs;
(f) optionally, what would reverse the decision.
A paragraph missing (a), (b) or (e) is descriptive, not critical, and sits in the 50 to 69 band.

## 2. Report skeleton with word budgets

Total 6,800 to 7,400 words excluding front matter, references and appendices (30 to 45 pages with figures).

### Front matter (not counted)
Cover sheet (Student ID 13130, "6COSC017C Machine Learning and Data Analytics", "Coursework 1 (50%)", "Marker: Ruslan Isaev",
submission date, word count, repository and app URLs; no name). Table of contents. List of figures. List of tables.
Abbreviations (TPR, TNR, AUC, PR-AUC, CV, IQR, SMOTE, VIF, MSHA).

### Executive summary (120 to 150 words)
Business question in one sentence; dataset and size; final model and headline test metrics (recall, precision, PR-AUC at the chosen
threshold); recommendation to the stakeholder; one limitation.

### 1 Introduction (850 to 950 words; rubric A, 5 marks; LO1)
1.1 Business case (250): who decides what today (mine operator, workers'-compensation insurer, MSHA inspector), how often, with what
information, and what each wrong decision costs (unreserved lost-time claim vs unnecessary case review). Why prediction rather than
rules (Mitchell's P, T, E framing stated for this task). One or two domain sources (MSHA statistics, NIOSH). Position the problem as
supervised binary classification in batch and interactive scoring modes; cite Provost and Fawcett (2013) on framing business problems
as data-mining tasks.
1.2 Problem framing (200 to 250): unit of analysis (one row = one injured worker in one reported accident), exact target definition and
how the label is produced from DEGREE_INJURY, positive class, prevalence (Table 1). Success criteria set before modelling (for example
recall >= 0.80 with precision >= 0.60, or an expected-cost target) and why they follow from the error costs. Scope exclusions.
Baselines the model must beat: majority class and a simple rule the stakeholder uses today.
1.3 Dataset origin, provenance, licensing and integration (300): publisher, collection method, period, geography, URLs, access date,
snapshot date and SHA-256, licence (US federal public domain) and what it permits. Original purpose vs this reuse; collection biases
(reporting rules under 30 CFR Part 50). Join keys, standardisation, join types, row counts before and after, orphan handling. Table 1
data dictionary reference. Cite Gebru et al. (2021) datasheets as the frame; note the Discussions reservation post.
1.4 Aims, objectives and report structure (120 to 150): three or four measurable objectives mapping to sections 2 to 5, in prose.

### 2 Exploratory Data Analysis (1,450 to 1,550 words; rubric B, 5 marks; LO1, LO4)
2.1 Summary of EDA (130 to 150) [1 mark, headed "Summary"]: shape after join; variable-type counts; target balance; four or five
headline findings with figure/table numbers; which findings feed section 3. Cite Tukey (1977).
2.2 Structure, variable types and target (220 to 250): per-variable type classification and why type drives statistic, plot and
encoder; target counts and proportions, imbalance ratio, majority-class accuracy baseline; dtype corrections observed.
2.3 Descriptive statistics and distributions (380 to 420): Table 2 with mean, median, 10 % trimmed mean, sample variance, sample standard
deviation, range, Q1, Q3, IQR, skewness, missing % for numeric variables; mode and cardinality for categoricals. JU for each statistic
family (mean vs median under skew; trimmed mean robustness; std over variance for units; IQR as shape-free). Skewness thresholds
(< 0.5 symmetric, 0.5 to 1 moderate, > 1 high) and consequences (log transform, median imputation, IQR not Z-score). Normality check with
histogram and Q-Q plot for two or three variables and what it implies for Pearson and Gaussian Naive Bayes. Group summaries by class.
2.4 Correlation and association (330 to 370): Pearson and Spearman where they disagree and why; point-biserial for numeric vs binary
target; Phi/Cramer's V for categorical vs target; multicollinearity (|r| > 0.8, VIF) and its consequence for logistic regression and
Naive Bayes; what the heatmap cannot show; link each notable correlation to a modelling implication (redundancy, interaction, leakage
suspicion at |r| > 0.95).
2.5 Visual analysis and plot justification (330 to 370) [2 marks]: one paragraph per figure family with what it shows, why this plot
type, what insight it gave: histograms (frequency vs density, bin sensitivity, KDE), class box plots, scatter/pair plot on a chosen
subset, categorical-vs-target proportion bars, correlation heatmap, missingness plot, optional PCA projection. Include one "did not
plot because" sentence. Cite Hunter (2007) and Waskom (2021) once.

### 3 Dataset preparation (1,050 to 1,150 words; rubric C, 3 marks; LO3)
3.1 Summary (130 to 150) [1 mark, headed "Summary"]: ordered list of steps, row/column counts after each, and the sentence "all learned
transformations are fitted on the training split only inside a scikit-learn Pipeline with ColumnTransformer". Reference Table 4
(decision log) and Figure 1 (pipeline diagram).
3.2 Integration and error correction (120 to 150): key standardisation, join validation, unit/format consistency, type coercion,
impossible values and the rule applied (correct, set to NaN, drop) with counts.
3.3 Duplicates and impossible combinations (90 to 120): exact vs near duplicates, detection, counts, why removal or retention.
3.4 Missing values (170 to 200): % per column, pattern, mechanism argued as MCAR/MAR/MNAR (Rubin, 1976) with evidence; per-column decision
(drop if > 60 % missing, median for skewed numerics, mode or "Unknown" for categoricals, missing-indicator where absence is informative);
KNNImputer/IterativeImputer considered; fitted on train only. Cite van Buuren (2018).
3.5 Outliers (120 to 150): IQR vs Z-score tied to skewness; counts flagged; error vs rare real event; action; which models are sensitive.
3.6 Instance selection and class balance (90 to 120): rows excluded by scope; options none / class_weight / under- or over-sampling /
SMOTE inside CV folds only / threshold tuning; test set never resampled. Cite Chawla et al. (2002), Lemaitre et al. (2017).
3.7 Feature engineering and transformation (130 to 160): date decomposition, ratios (experience ratios, hours-based rates), counts,
discretisation where interpretability matters, log1p for skew > 1, dropping constant/duplicate columns, leakage audit of every
engineered feature (known at decision time?). Cite Kuhn and Johnson (2019); Kaufman et al. (2012).
3.8 Encoding and scaling (130 to 160): one-hot (nominal, handle_unknown="ignore", rare-level grouping), ordinal (ordered categories),
label encoding only for the target; StandardScaler vs MinMaxScaler vs Normalizer and the choice per model family (none for trees).
3.9 Train/validation/test split and leakage control (100 to 130): stratified 80/20 with fixed random_state; 5-fold StratifiedKFold on
the 80 % as the validation set; group-aware split if rows share an accident; Table 4 closes the section.

### 4 Modelling (2,500 to 2,700 words; rubric D, 17 marks; LO2, LO5)
4.1 Summary (180 to 220) [1 mark, headed "Summary"]: task type, candidate algorithms, validation protocol, selection metric, final
model and headline numbers, pointer to Tables 5 to 8.
4.2 Algorithm selection (600 to 700) [4 marks]: dataset characteristics that constrain the choice; per algorithm (Logistic Regression,
KNN, Naive Bayes, Decision Tree, Random Forest, HistGradientBoosting, SVM and MLP where used) one JU paragraph: paradigm (parametric or
not, generative or discriminative, lazy or eager), mechanism in the lecturer's terms (sigmoid, binary cross-entropy, gradient descent,
learning rate, L2 and C; K, distance, scale sensitivity, curse of dimensionality; Bayes theorem, conditional independence, Gaussian
likelihood, Laplace smoothing; impurity, greedy splits, variance; bagging and feature subsampling (Breiman, 2001); additive boosting;
margin, C, kernel, gamma (Cortes and Vapnik, 1995)), assumptions and where this dataset violates them, the bias introduced, the reason
it still earns a place. Why not others (deep MLP on tabular data, Grinsztajn et al., 2022; unsupervised methods; time-series models).
Wolpert (1996) no-free-lunch as the reason for empirical comparison; bias-variance (Hastie et al., 2009) with the learning curve as evidence.
4.3 Model selection per algorithm, hyperparameters (600 to 700) [4 marks]: protocol (GridSearchCV or RandomizedSearchCV, StratifiedKFold
k = 5, Kohavi 1995, refit on full training split, scoring = primary metric, fixed seed, pipeline inside the search). Table 5 lists every
searched hyperparameter: name, meaning, values, why those bounds, best value, CV mean +/- std. Defaults left untouched are listed with
a reason. Per algorithm: validation-curve reading, over/underfitting diagnosis, train vs CV gap, sensitivity. Threshold as a tuned
parameter on out-of-fold probabilities. Why the grid was not wider; the one-standard-error rule.
4.4 Evaluation metrics (450 to 550) [4 marks]: confusion matrix in domain words; accuracy reported but not used (majority baseline);
precision, recall/TPR, specificity/TNR, F1 (F-beta if costs asymmetric), balanced accuracy, MCC; ROC-AUC (Fawcett, 2006) vs PR-AUC
(Saito and Rehmsmeier, 2015; Davis and Goadrich, 2006) and which is primary given prevalence; calibration (Brier) if probabilities are
shown; expected cost per 1,000 cases and how it moves with threshold; CV mean +/- std vs held-out test, never conflated; what the
metrics cannot capture (subgroup fairness, drift).
4.5 Comparison and final model (550 to 650) [4 marks]: Table 6 (CV mean +/- std, all models, all metrics), Table 7 (test set, brief's
columns plus F1, PR-AUC, accuracy); paired test across folds or McNemar on test predictions (Dietterich, 1998; Demsar, 2006); multi-criteria
decision (performance, cost, interpretability, training and inference time, robustness, deployment simplicity); final model with chosen
threshold, confusion matrix (Table 8), performance against the success criteria and both baselines; error analysis; model card
(Mitchell et al., 2019); monitoring and technical debt (Sculley et al., 2015).

### 5 Conclusion (850 to 950 words; rubric E, 5 marks; LO4)
5.1 Key findings (220 to 260) [2 marks]: five or six numbered findings, each one sentence with a number and a figure/table reference,
each tied to the business question.
5.2 Interpretation (280 to 320) [3 marks]: operational meaning per 1,000 cases; why the winning algorithm won in terms of assumptions
matching the data; confidence and generalisability (CV std, bootstrap CI on the main metric, test size); the answer to 1.1 and the
recommendation.
5.3 Limitations (140 to 170): data, method, evaluation, tooling.
5.4 Future work (90 to 120): prioritised and tied to limitations.
5.5 Ethical considerations (140 to 170): licence and attribution; personal data; representational and measurement bias in reporting
(Mehrabi et al., 2021); subgroup performance; misuse risk; transparency via model card and app disclaimer; human in the loop.
End on the last concrete recommendation, not a recap.

### References (Harvard, alphabetical) and Appendices
A additional EDA figures; B full hyperparameter search results; C environment, seed, run instructions, repo and app URLs, commit
timeline screenshot; D full data dictionary; E dataset licence text; F declaration of tools used if WIUT requires one.

### Lecturer vocabulary to echo at least once
Section 2: measures of centrality; measures of variability (dispersion); trimmed mean; sample standard deviation; interquartile range;
skewness; Q-Q plot; Pearson; Spearman's rank; point-biserial; Phi coefficient; Cramer's V; frequency vs density histogram; number of
bins; pair plot; box plot quartiles; correlation matrix.
Section 3: exact and near-duplicates; consistency; impossible data combinations; percentage missing per column; median resists
outliers; most_frequent; missing indicator; IQR method; Z-score; min-max normalisation; z-score normalisation (StandardScaler);
Normalizer; one-hot; ordinal; label encoding for the target; discretisation; feature construction; stratified train-test split;
data leakage; Pipeline; ColumnTransformer; fit on train only; SMOTE; class_weight.
Section 4: supervised; sigmoid; binary cross-entropy; gradient descent; learning rate; regularisation; decision boundary; K odd to
avoid ties; Euclidean/Manhattan/Minkowski; instance-based (lazy) learning; conditional independence; prior, likelihood, posterior;
Gaussian Naive Bayes; underfitting/overfitting; K-fold cross-validation; GridSearchCV; confusion matrix; TPR/TNR; F-measure; ROC/AUC;
PR-AUC; bias-variance.

## 3. Figures and tables

Caption rule: a full sentence stating content, data split and the reason for the plot type; every figure and table is referenced in
the text before it appears; source line "Source: author's analysis of MSHA open data".

Figures: F1 pipeline diagram; F2 target distribution; F3 density histograms with KDE; F4 class box plots; F5 pair plot of top
variables by point-biserial correlation; F6 Pearson and Spearman heatmaps side by side; F7 categorical variables vs target; F8
missingness; F9 Q-Q plots; F10 outlier fences before/after; F11 PCA projection (optional); F12 ROC curves; F13 PR curves with
prevalence baseline; F14 validation curves per algorithm; F15 learning curves; F16 confusion matrices at 0.50 and tuned threshold;
F17 feature importance / SHAP; F18 threshold vs precision, recall, F1 and expected cost; F19 calibration curve; F20 app screenshots
and commit timeline (appendix).

Tables: T1 data dictionary; T2 descriptive statistics; T3 association with target and multicollinearity; T4 preprocessing decision log
(step, problem found with count, options, chosen, why, rows x cols after); T5 hyperparameter grid per algorithm; T6 CV results mean +/-
std; T7 held-out test results in the brief's layout (TPR, TNR, AUC, Recall, Precision, F1, PR-AUC, Accuracy); T8 confusion matrix of the
final model; T9 statistical comparison; T10 model card; T11 baselines vs final model vs success criteria.

## 4. Harvard reference list (verified entries)

In-text: (Breiman, 2001); (Saito and Rehmsmeier, 2015); three or more authors (Pedregosa et al., 2011); several sources chronological.

1. Bishop, C.M. (2006) Pattern Recognition and Machine Learning. New York: Springer.
2. Breiman, L. (2001) 'Random forests', Machine Learning, 45(1), pp. 5-32. doi:10.1023/A:1010933404324.
3. Bruce, P., Bruce, A. and Gedeck, P. (2020) Practical Statistics for Data Scientists. 2nd edn. Sebastopol, CA: O'Reilly Media.
4. Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic minority over-sampling technique', Journal of Artificial Intelligence Research, 16, pp. 321-357. doi:10.1613/jair.953.
5. Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system', Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. San Francisco, CA, 13-17 August. New York: ACM, pp. 785-794. doi:10.1145/2939672.2939785.
6. Cortes, C. and Vapnik, V. (1995) 'Support-vector networks', Machine Learning, 20(3), pp. 273-297. doi:10.1007/BF00994018.
7. Cover, T. and Hart, P. (1967) 'Nearest neighbor pattern classification', IEEE Transactions on Information Theory, 13(1), pp. 21-27. doi:10.1109/TIT.1967.1053964.
8. Davis, J. and Goadrich, M. (2006) 'The relationship between Precision-Recall and ROC curves', Proceedings of the 23rd International Conference on Machine Learning. Pittsburgh, PA, 25-29 June. New York: ACM, pp. 233-240. doi:10.1145/1143844.1143874.
9. Demsar, J. (2006) 'Statistical comparisons of classifiers over multiple data sets', Journal of Machine Learning Research, 7, pp. 1-30.
10. Dietterich, T.G. (1998) 'Approximate statistical tests for comparing supervised classification learning algorithms', Neural Computation, 10(7), pp. 1895-1923. doi:10.1162/089976698300017197.
11. Domingos, P. and Pazzani, M. (1997) 'On the optimality of the simple Bayesian classifier under zero-one loss', Machine Learning, 29(2-3), pp. 103-130. doi:10.1023/A:1007413511361.
12. Fawcett, T. (2006) 'An introduction to ROC analysis', Pattern Recognition Letters, 27(8), pp. 861-874. doi:10.1016/j.patrec.2005.10.010.
13. Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J.W., Wallach, H., Daume III, H. and Crawford, K. (2021) 'Datasheets for datasets', Communications of the ACM, 64(12), pp. 86-92. doi:10.1145/3458723.
14. Geron, A. (2022) Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. 3rd edn. Sebastopol, CA: O'Reilly Media.
15. Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022) 'Why do tree-based models still outperform deep learning on typical tabular data?', Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track.
16. Han, J., Pei, J. and Tong, H. (2022) Data Mining: Concepts and Techniques. 4th edn. Cambridge, MA: Morgan Kaufmann.
17. Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020) 'Array programming with NumPy', Nature, 585(7825), pp. 357-362. doi:10.1038/s41586-020-2649-2.
18. Hastie, T., Tibshirani, R. and Friedman, J. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction. 2nd edn. New York: Springer.
19. He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', IEEE Transactions on Knowledge and Data Engineering, 21(9), pp. 1263-1284. doi:10.1109/TKDE.2008.239.
20. Hosmer, D.W., Lemeshow, S. and Sturdivant, R.X. (2013) Applied Logistic Regression. 3rd edn. Hoboken, NJ: Wiley.
21. Hunter, J.D. (2007) 'Matplotlib: A 2D graphics environment', Computing in Science and Engineering, 9(3), pp. 90-95. doi:10.1109/MCSE.2007.55.
22. James, G., Witten, D., Hastie, T., Tibshirani, R. and Taylor, J. (2023) An Introduction to Statistical Learning: with Applications in Python. Cham: Springer. doi:10.1007/978-3-031-38747-0.
23. Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012) 'Leakage in data mining: Formulation, detection, and avoidance', ACM Transactions on Knowledge Discovery from Data, 6(4), Article 15. doi:10.1145/2382577.2382579.
24. Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', Proceedings of the 14th International Joint Conference on Artificial Intelligence. Montreal, 20-25 August. San Francisco, CA: Morgan Kaufmann, vol. 2, pp. 1137-1145.
25. Kuhn, M. and Johnson, K. (2019) Feature Engineering and Selection: A Practical Approach for Predictive Models. Boca Raton, FL: CRC Press.
26. Lemaitre, G., Nogueira, F. and Aridas, C.K. (2017) 'Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning', Journal of Machine Learning Research, 18(17), pp. 1-5.
27. Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', Advances in Neural Information Processing Systems 30, pp. 4765-4774.
28. McKinney, W. (2010) 'Data structures for statistical computing in Python', Proceedings of the 9th Python in Science Conference. Austin, TX, 28 June-3 July, pp. 56-61. doi:10.25080/Majora-92bf1922-00a.
29. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K. and Galstyan, A. (2021) 'A survey on bias and fairness in machine learning', ACM Computing Surveys, 54(6), Article 115. doi:10.1145/3457607.
30. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D. and Gebru, T. (2019) 'Model cards for model reporting', Proceedings of the Conference on Fairness, Accountability, and Transparency. Atlanta, GA, 29-31 January. New York: ACM, pp. 220-229. doi:10.1145/3287560.3287596.
31. Mitchell, T.M. (1997) Machine Learning. New York: McGraw-Hill.
32. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011) 'Scikit-learn: Machine learning in Python', Journal of Machine Learning Research, 12, pp. 2825-2830.
33. Provost, F. and Fawcett, T. (2013) Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking. Sebastopol, CA: O'Reilly Media.
34. Rubin, D.B. (1976) 'Inference and missing data', Biometrika, 63(3), pp. 581-592. doi:10.1093/biomet/63.3.581.
35. Saito, T. and Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets', PLoS ONE, 10(3), e0118432. doi:10.1371/journal.pone.0118432.
36. Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F. and Dennison, D. (2015) 'Hidden technical debt in machine learning systems', Advances in Neural Information Processing Systems 28, pp. 2503-2511.
37. Streamlit (2026) Streamlit documentation. Available at: https://docs.streamlit.io/ (Accessed: 14 October 2026).
38. Tan, P.-N., Steinbach, M., Karpatne, A. and Kumar, V. (2019) Introduction to Data Mining. 2nd edn. New York: Pearson.
39. Tukey, J.W. (1977) Exploratory Data Analysis. Reading, MA: Addison-Wesley.
40. US Department of Labor, Mine Safety and Health Administration (2026) Accident Injuries Data Set [Dataset]. Available at: https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp (Accessed: 14 October 2026). Public domain (US Government work).
41. US Department of Labor, Mine Safety and Health Administration (2026) Mines Data Set and Mines Production Yearly Data Set [Datasets]. Available at: https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp (Accessed: 14 October 2026).
42. van Buuren, S. (2018) Flexible Imputation of Missing Data. 2nd edn. Boca Raton, FL: CRC Press.
43. Waskom, M.L. (2021) 'seaborn: statistical data visualization', Journal of Open Source Software, 6(60), 3021. doi:10.21105/joss.03021.
44. Wolpert, D.H. (1996) 'The lack of a priori distinctions between learning algorithms', Neural Computation, 8(7), pp. 1341-1390. doi:10.1162/neco.1996.8.7.1341.

Add during writing if used: MSHA/NIOSH domain sources; Brown (2023) and the NCI (2023) thesis as related work.

## 5. Marker-simulation audit rubric (pass criteria)

Scoring rule: full marks only if every pass criterion is met; half if one is missing; zero if two or more are missing. Cite evidence.

Report (35)
- A Introduction (5): stakeholder and decision named with error costs; target and positive class defined with prevalence; both sources
  with publisher, period, URL, licence; join described with before/after counts; numeric success criteria; dataset cited in Harvard;
  original purpose vs reuse discussed.
- B1 EDA summary (1): paragraph headed "Summary" at the start of section 2 listing shape, types, target balance, headline findings with refs.
- B2 Statistics and correlations (2): Table 2 includes mean, median, trimmed mean, variance, std, IQR, skewness; each statistic family
  has a JU; Pearson and Spearman both reported and compared; point-biserial and Cramer's V (or Phi) for the target; multicollinearity
  discussed; every finding links to a downstream decision.
- B3 Visualisation (2): each figure family has what / why this plot / insight; bins and density choice justified; box plot and pair
  plot present and justified; heatmap present; at least one "did not plot because" sentence.
- C1 Preparation summary (1): "Summary" paragraph with ordered steps, counts, fit-on-train-only pipeline statement.
- C2 Per-step justification (2): Table 4 decision log complete; missing-value mechanism argued; IQR vs Z-score tied to skewness; scaler
  choice tied to model family; encoder choice tied to variable type; stratified split with seed; leakage controls explicit; resampling on
  train only.
- D1 Modelling summary (1): "Summary" paragraph naming task type, algorithms, validation protocol, primary metric, final model.
- D2 Algorithm selection (4): >= 3 algorithms (recommend six) each with paradigm, mechanism, assumptions, dataset-specific violations,
  bias, reason included; >= 2 rejected alternatives with reasons; no-free-lunch or bias-variance framing with citation.
- D3 Hyperparameters (4): Table 5 lists every searched hyperparameter with meaning, range, rationale, best value, CV mean +/- std;
  untouched defaults listed and justified; validation curve read per algorithm with over/underfitting diagnosis; threshold tuned on
  OOF predictions; CV scheme named and justified.
- D4 Metrics (4): confusion-matrix definitions in domain words; accuracy shown inadequate with the majority baseline number;
  precision, recall/TPR, specificity/TNR, F1, ROC-AUC, PR-AUC defined and justified for this prevalence; primary metric tied to error
  costs; CV vs test roles explained; expected-cost metric present.
- D5 Comparison and final model (4): Tables 6 and 7 complete with the brief's columns; statistical test of differences; multi-criteria
  ranking including interpretability and cost; final model with tuned threshold meets or honestly misses the criteria; error analysis;
  model card.
- E1 Findings (2): five or six numbered findings, each with a number and a reference, each linked to the business question.
- E2 Interpretation (3): operational translation per 1,000 cases; why the winner won; uncertainty quantified; limitations, future work
  and ethics specific; ends on a concrete recommendation.

Practical (65)
- A Data load (5): both sources loaded by code from files in the zip; join key standardised; row counts asserted before/after; orphan
  check; result equals Table 1 shape.
- B EDA (10): describe() plus a custom table with trimmed mean, variance, std, IQR, skewness (4); Pearson and Spearman heatmaps plus
  target-association table (3); histograms, box plots, pair plot, categorical bars, missingness plot, Q-Q (3).
- C Preparation (10): missing values handled per column with stated % (2); scaler inside ColumnTransformer fit on train (2); impossible
  values detected and corrected with counts (2); >= 3 engineered features with leakage check (2); stratified split with seed plus a
  named validation strategy (2).
- D Training (10): >= 3 (recommend six) models on identical splits via Pipeline (6); GridSearchCV/RandomizedSearchCV with StratifiedKFold
  for every model, results saved (4).
- E Evaluation (10): test-set confusion matrices, TPR, TNR, precision, recall, F1, ROC-AUC, PR-AUC, accuracy for every model; ROC and PR
  curves; CV mean +/- std table; threshold analysis; statistical comparison; final model saved with joblib and version manifest.
- F Deployment (10): live URL opens and runs inference within seconds (2); code modular (src package reused by notebook and app, no
  duplicated preprocessing, no hard-coded paths, cached loads) (3); multi-page app with data exploration, preprocessing choices that
  re-run the pipeline, model training/inference (form and batch CSV), evaluation (metrics, curves, confusion matrix) (5).
- G Version control (5): commits across the teaching weeks with messages matching milestones (3); README with purpose, setup, run
  commands for notebook and app, data provenance and licence (2).
- H Reproducibility (5): requirements.txt pinned (2); uv.lock or Dockerfile that builds and runs the app (3); fixed seeds, environment
  printed in the notebook.

LO band checks (yes/no with evidence)
- LO1 80+: each technique tied to the business case and placed in a wider data-science frame with citations.
- LO2 80+: assumptions and biases of every algorithm stated and checked against this dataset's evidence.
- LO3 80+: pipeline single-sourced, tested, reproducible from a clean environment in one command.
- LO4 80+: every conclusion sentence points to a table or figure; uncertainty quantified; recommendation concrete.
- LO5 80+: metric choice derives from prevalence and error costs; PR-AUC vs ROC-AUC explained; threshold choice shown; interpretation
  domain-specific.

## 6. Writing checklist (no-ai-slop adapted for an academic report)

A. Banned outright (zero occurrences): delve, foster, leverage, utilise/utilize, facilitate, empower, streamline, cutting-edge,
paradigm shift, game changer, tapestry, realm, beacon, multifaceted, meticulous, intricate, paramount, transformative, elevate, embark,
supercharge, harness, ever-evolving; "robust" only as a technical term (RobustScaler, robust statistics, robust to outliers with
evidence). Phrases: it's worth noting, it is important to note, at the end of the day, when it comes to, at its core, in today's world,
in the age of, in the world of, the reality is, the truth is, in terms of, with regard to, in order to, going forward, let's dive in,
plays a crucial/vital/pivotal role, stands as a testament, marks a pivotal moment, underscores the importance, in the realm of, a wide
array of, a variety of, a range of (when the items could be named), key insights, deep dive, seamless, holistic, nuanced (as filler),
navigate (metaphorical), landscape (metaphorical). Often-empty adverbs to cut unless carrying contrast or uncertainty: just, literally,
honestly, simply, actually, truly, fundamentally, importantly, crucially, inherently, inevitably, notably, significantly (unless a test
is reported).

B. Patterns to cut: binary contrasts ("It is not X but Y"); negative listing; throat-clearing openers ("It is essential to understand
that"); faux-insight setups ("What is often overlooked is"); colon reveals; superficial "-ing" analysis ("highlighting the importance
of"); importance puffery; interpretive metadiscourse ("As can be seen", "It is clear that", "This distinction matters", redundant "In
other words"); weasel attribution ("Studies show"); fake-strong verbs ("serves as", "acts as"); synonym cycling (one term per concept:
"row", "feature", "test set"); dramatic fragments; robotic rhythm (<= 15 % of paragraphs open with a connective; sentence-length std
>= 6 words within a section); rhetorical setups; rule-of-three padding; fake-profound kickers; summary-recap endings inside sections;
formatting slop (no emoji, no decorative bold, bullets only in tables/decision log/findings list, no heading over < 3 sentences,
sentence case after colons); em dashes <= 2 in the whole report; portability test (a sentence that could sit in any other report is
filler); protect the specific fact (never "performance improved significantly" for "recall rose from 0.62 to 0.81"); active voice with
a human or named-system subject for decisions.

C. Academic voice: (1) first person sparingly and only for decisions ("I chose median imputation because ..."), never "we", never "the
author"; (2) hedge with evidence, not adverbs (banned: clearly, obviously, it is evident; allowed: suggests, is consistent with, within
one standard deviation); (3) define each term once with its abbreviation, then use the abbreviation; (4) past tense for what was done,
present for what data and tables show, future only in 5.4; (5) numbers in tables, interpretations in prose: <= 3 numbers per prose
paragraph, three decimals for metrics, one for percentages, units always, thousands separators, "n = 1,234"; (6) one idea per
paragraph, topic sentence first, four to eight sentences; (7) refer to figures and tables by number only; citations at the end of the
clause they support.

D. Quantitative audit script: banned-word count = 0; em-dash count <= 2; "not only ... but also" <= 1; connective openers <= 15 %;
vague "This" sentence openers <= 5 per section; mean sentence length 15 to 24 words with std >= 6; every Figure N / Table N in text
exists and vice versa; every (Author, Year) in the reference list and vice versa; no occurrence of the student's name anywhere
(text, headers, metadata, URLs, alt text, paths); British spelling consistent; section word counts within budget.

## 7. Viva voce preparation (25 questions; each answer names one number and one table/figure)

1 Why this dataset and question, who acts on it, why not Kaggle, when was it reserved? 2 How was the licence verified and attributed?
3 How were the sources joined and how do you know the join is correct? 4 What is one row and what exactly does the label mean?
5 Why mean and median both, and when the trimmed mean? 6 Why sample variance uses n-1? 7 When would Pearson mislead; what did Spearman
show? 8 Which correlation for numeric vs binary target and why? 9 How does bin count change a histogram's story? 10 What did box plots
show that the table did not? 11 Drop vs impute per column, why median? 12 IQR or Z-score, why? 13 Why StandardScaler for LR/KNN/SVM and
nothing for trees? 14 One-hot vs ordinal vs label with examples? 15 Where could leakage enter and how was it prevented? 16 Why stratify;
what is the validation set? 17 Logistic regression cost function and why not MSE? 18 Why K odd; what happens at K = 1 and K = N? 19 Naive
Bayes assumes independence; your heatmap shows correlation; why keep it? 20 How does a random forest reduce variance vs one tree? 21 Walk
through one hyperparameter: meaning, range, validation curve. 22 Why is accuracy inadequate; what is the primary metric? 23 ROC-AUC vs
PR-AUC at your prevalence? 24 Are the top two models statistically different; why choose one anyway? 25 What harm could the model cause,
who bears it, what mitigates it, how would you detect drift?
