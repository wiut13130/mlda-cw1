# 4 Modelling

## 4.1 Summary

The task is supervised binary classification of each Form 7000-1 report as lost-time or not, with a probability that orders the adjuster's review queue. Eight algorithms ({tab:algorithms}) were tuned by GridSearchCV, or RandomizedSearchCV for the forest, over five StratifiedGroupKFold folds of mine-year groups inside the training split, with PR-AUC (average precision) as the refit metric ({tab:grids}). Histogram-based gradient boosting (HGB) ranked first in cross-validation ({tab:cv}) and on the test set ({tab:test}), with a test PR-AUC of 0.842. At its capacity threshold of 0.211 it flags 76.3% of all filings in the test set with recall 0.941 and precision 0.600 ({tab:operating}; {tab:confusion}). In use it would score only open reports (Section 1.1). On them the workload rule, review of the top three in four by score, gives recall 0.850 at a cut-off of 0.376 fixed from out-of-fold scores; reviewing exactly 75%, a comparison whose cut-offs come from the test scores, gives 0.855 against 0.828 for the lookup baseline ({tab:policy}). HGB passes five of the six performance criteria and misses recall at a 50% flag rate by 0.003 ({tab:criteria_results}).

## 4.2 Algorithm selection

Encoding yields 546 columns, mostly zero-valued one-hot indicators beside skewed numeric and binary features ({tab:decisionlog}). Size relations are monotone but curved and collinear ({tab:associations}). Nominal codes lead the associations, and the classes are near balance. An adjuster needs a reason for every flag and the application scores forms on demand, which made interpretability and speed criteria too ({tab:algorithms}).

Table: Candidate algorithms, assumptions and violations || The eight candidate algorithms with their paradigm, key assumption, the violation found in the training split and the bias it introduces, scaling need, interpretability and the module week in which each was taught; violations refer to the exploratory analysis of the training split in Section 2. Source: author's analysis of MSHA open data. {#tab:algorithms}

| Algorithm | Paradigm | Key assumption | Violation in this data and resulting bias | Scaling | Interpretability | Taught week |
|---|---|---|---|---|---|---|
| Logistic regression | parametric; discriminative; eager | log-odds linear and additive in the encoded columns | curved size effects (Pearson r 0.645 against Spearman 0.994) and nature x body part interactions; underfits (high bias) | StandardScaler | high: coefficients convert to odds ratios | 4 |
| KNN | non-parametric; discriminative; lazy (instance-based) | rows close in scaled Euclidean space share outcomes | 546 mostly binary dimensions flatten distances, so votes drift towards the local base rate | StandardScaler | low: a flag is explained only by similar past reports | 5 |
| Gaussian naive Bayes | parametric; generative; eager | features independent given the class, each Gaussian within a class | collinear size measures (VIF above 160) and nine highly skewed numeric features; evidence counted twice, overconfident posteriors | StandardScaler (var_smoothing is relative to the largest variance) | medium: class-conditional means | 5 |
| Decision tree (CART) | non-parametric; discriminative; eager | classes separate in axis-aligned boxes | curved effects need many steps; greedy splits give high variance | none | high when shallow: if-then rules | 6 |
| Random forest | non-parametric bagging ensemble; discriminative; eager | averaging decorrelated deep trees cancels their variance | few violations; averaged leaf frequencies pull probabilities towards the middle | none | medium: permutation importance | 6 |
| Histogram-based gradient boosting (HGB) | non-parametric boosting ensemble; discriminative; eager | log-odds is a sum of small trees on binned features | rare levels pooled into one column; greedy binned splits; tied to the training-period mix of codes | none | medium: permutation importance and SHAP values | 6 |
| SVM, RBF kernel | non-parametric (kernel); discriminative; eager | a smooth wide-margin boundary in kernel space | training cost at least quadratic in rows, so fitted on 10,000 rows; scores need Platt scaling | StandardScaler | low: margin scores only | 5 |
| MLP | parametric; discriminative; eager | a smooth mapping learnable by gradient descent | tens of thousands of weights on sparse one-hot inputs; overfits without alpha and early stopping | StandardScaler | low: needs post-hoc explanation | 10 |

Logistic regression passes a weighted sum of the inputs through the sigmoid and minimises binary cross-entropy by lbfgs, a quasi-Newton relative of gradient descent, under an L2 penalty that weakens as C rises (Hosmer et al., 2013); its odds ratios made it the transparency benchmark despite its bias. Gaussian naive Bayes combines the class prior with per-feature Gaussian likelihoods by Bayes' theorem. Independence fails for the duplicated size measures and normality for the skewed and binary features ({fig:qq}), which makes posteriors overconfident; ranking can survive this (Domingos and Pazzani, 1997), and I kept it as a fast generative baseline and calibration test.

KNN, a lazy instance-based learner, labels a report by the vote of its K nearest training rows under Euclidean distance (Cover and Hart, 1967), and distances lose contrast across 546 mostly binary dimensions, the curse of dimensionality. It shows whether similarity to past claims adds signal; because scoring compares each report with every stored row, I fitted it on whole mine-years capped at 40,000 rows. An SVM maximises the margin between the classes, with C penalising rows inside the margin and the width gamma of its radial basis function (RBF) kernel setting how local the boundary is (Cortes and Vapnik, 1995); that smooth boundary is the alternative to axis-aligned tree splits. Training time grows at least quadratically with rows, so I capped it at 10,000 rows and obtained probabilities by Platt scaling.

A decision tree splits greedily on the axis-aligned threshold that most reduces Gini impurity or entropy. It needs no scaling and nests interactions, yet a small change in the rows can alter every split below the root. A random forest averages deep trees grown on bootstrap samples with a random feature subset per split, which decorrelates them and cuts that variance (Breiman, 2001).

HGB adds shallow trees in stages, fitting each to the gradient of the log loss and shrinking it by the learning rate, with capped leaf nodes and early stopping. It follows the second-order boosting of XGBoost (Chen and Guestrin, 2016), and its sum of small trees can follow curved effects and code interactions. Because tree ensembles lead on medium-sized tabular data (Grinsztajn et al., 2022), HGB was expected to rank the review queue best.

The MLP trains one or two hidden layers of rectified linear units by Adam, a gradient-descent variant, with alpha as an L2 penalty and early stopping (Bishop, 2006). It is the one neural candidate taught in the module, and it stayed small because tens of thousands of weights on mostly zero inputs invite overfitting.

Three families were rejected. Time-series models such as ARIMA assume one quantity observed at regular intervals, but each row is a separate claim and time enters through lagged mine history. Clustering would discard the label, so unsupervised methods stayed in the exploratory analysis ({fig:pca}). Deep tabular networks trail tree ensembles at this scale and need more tuning than the MLP (Grinsztajn et al., 2022). Because no algorithm wins on every problem (Wolpert, 1996), the eight were compared under one protocol.

The learning curve places that comparison in bias-variance terms (Hastie et al., 2009). On a 10,000-row subsample HGB's validation PR-AUC rises from 0.773 to 0.818 as rows are added and its training PR-AUC falls from 0.949 to 0.912, with one small rise at the last step. The gap narrows at every step from 0.175 to 0.094 ({fig:learning}) and is variance, which more data reduces. Naive Bayes shows almost no gap at a low score, the mark of an error due to bias ({tab:grids}).

![Learning curve of HGB || Mean training-fold and validation-fold PR-AUC (average precision) of HGB with its selected settings, with one standard deviation bands over five grouped folds of a 10,000-row subsample of the training split, drawn against training size because the gap between the two curves separates variance from bias. Source: author's analysis of MSHA open data.](reports/figures/F15_learning_curve_hist_gb.png){#fig:learning width=5.5}

## 4.3 Model selection per algorithm

Every search tuned the whole pipeline, refitting imputers, encoder, scaler and classifier on the training rows of each fold ({fig:pipeline}). The grouped folds keep each mine-year on one side and the lost-time share near that of the whole; grouping mirrors deployment, where the insurer scores mine-years the model has not seen. Kohavi (1995) favoured ten stratified folds; five halved the fits and left about 16,000 rows per validation fold. Seed 42 fixed the folds, subsamples and estimators.

Most grids contain the scikit-learn default and step towards stronger and weaker regularisation; other settings kept defaults for the reasons in {tab:grids}. Both HGB caps on boosting rounds scored identically at the faster learning rate, as early stopping ended training first.

Table: Hyperparameter search per model || Grids searched on the 81,289 training rows with five grouped folds; the first row of each model gives the search, the number of candidates, the selected candidate's validation PR-AUC (mean ± SD) and its training-fold PR-AUC, the last column the best validation PR-AUC with another value (in brackets), and the rows marked as follow-up give wider grids searched after the test scoring, which changed no choice. Source: author's analysis of MSHA open data. {#tab:grids}

| Model (search; validation PR-AUC; training PR-AUC) | Hyperparameter: meaning | Values searched | Why these bounds | Selected | Best other value |
|---|---|---|---|---|---|
| Logistic regression (grid, 8 candidates; 0.820 ± 0.003; 0.828) | C: inverse strength of the L2 penalty | 0.01, 0.1, 1, 10 | four decades around the default of 1 | 1 | 0.820 (10); 0.814 (0.01) |
| | class_weight: reweights the classes in the loss | None, balanced | tests reweighting as the alternative to resampling at near balance | None | 0.820 (balanced) |
| | Fixed settings | L2 penalty, lbfgs solver, max_iter 2,000 | L2 suits collinear columns; no convergence warning was logged | | |
| KNN (grid, 6; 0.764 ± 0.009; 1.000) | n_neighbors (K): number of voters | 11, 25, 51 | odd values from a local to a smooth vote | 51 | 0.759 (25) |
| | weights: equal or inverse-distance votes | uniform, distance | lets close neighbours count more | distance | 0.757 (uniform) |
| | p: Minkowski power | 2 | Euclidean distance only, to keep six candidates | 2 | none |
| | Fixed settings | brute-force search; 39,999-row subsample of whole mine-years | tree indexes do not help in 546 dimensions; scoring cost grows with stored rows | | |
| | Follow-up grid (6 candidates, 286 s) | n_neighbors 51, 101, 151 with both weightings | the original optimum sat at the grid edge | 51, distance (0.764) | 0.763 (101); 0.761 (151) |
| Gaussian naive Bayes (grid, 5; 0.735 ± 0.007; 0.736) | var_smoothing: share of the largest variance added to every variance | 1e-9, 1e-6, 1e-3, 0.1, 1 | log-spaced from the default upwards, since near-constant one-hot columns have tiny variances | 0.1 | 0.703 (1) |
| | Fixed settings | class priors from training frequencies | the priors then equal the 48.6% prevalence | | |
| Decision tree (grid, 24; 0.803 ± 0.004; 0.826) | max_depth: longest root-to-leaf path | 4, 8, 12, None | from a readable tree to unlimited growth | 12 | 0.793 (None) |
| | min_samples_leaf: smallest leaf | 5, 20, 50 | larger leaves give steadier lost-time rates | 50 | 0.794 (20) |
| | criterion: impurity measure | gini, entropy | the two impurity measures taught | gini | 0.802 (entropy) |
| | Fixed settings | best split over all features; no cost-complexity pruning | depth and leaf size already prune the tree | | |
| Random forest (random search, 4 of 24 grid points; 0.821 ± 0.004; 0.946) | n_estimators: number of trees | 100, 200 | the default and double | 100 | not sampled |
| | max_depth: longest path in each tree | 8, 12, None | as for the single tree | None | 0.808 (12) |
| | min_samples_leaf: smallest leaf | 5, 20 | small leaves suit averaged deep trees | 5 | 0.815 (20) |
| | max_features: features tried per split | sqrt, log2 | the two usual subset sizes for decorrelation | sqrt | not sampled |
| | Fixed settings | bootstrap samples, Gini impurity | the design of Breiman (2001) | | |
| | Follow-up grid (all 24 points, 585 s) | every combination of the values above | the random search sampled four points | 200 trees, otherwise as selected (0.822 ± 0.004) | 0.821 (100 trees); 0.811 (log2) |
| HGB (grid, 16; 0.838 ± 0.003; 0.867) | learning_rate: shrinkage applied to each tree | 0.05, 0.1 | the default and half of it | 0.05 | 0.837 (0.1) |
| | max_iter: cap on boosting rounds | 200, 400 | caps only, since early stopping ends training | 400 | 0.837 (200) |
| | max_leaf_nodes: leaves per tree | 15, 31 | the default and about half | 31 | 0.836 (15) |
| | l2_regularization: penalty on leaf values | 0, 1 | no penalty or a unit penalty | 0 | 0.838 (1) |
| | early_stopping | True | stops after 10 rounds without loss improvement on a 10% validation split | True | none |
| | Fixed settings | min_samples_leaf 20, 255 bins, no depth limit | the leaf count already limits tree size | | |
| | Follow-up grid (24 candidates, 951 s) | max_leaf_nodes 15, 31, 63; min_samples_leaf 20, 100 in place of the inert max_iter (fixed at 400); learning_rate and l2_regularization as above | 31 leaves sat at the grid edge, and leaf size was not searched | 63 leaves, min_samples_leaf 100, l2 1 (0.840 ± 0.003; training 0.887) | 0.838 (selected settings) |
| SVM, RBF kernel (grid, 4; 0.795 ± 0.014; 0.856) | C: penalty on margin violations | 1, 10 | the default and a weaker penalty; each fit took about two and a half minutes | 1 | 0.788 (10) |
| | gamma: RBF kernel width | scale, 0.01 | the default heuristic and one fixed value | 0.01 | 0.792 (scale) |
| | Fixed settings | Platt scaling over five inner folds; 9,913-row subsample of whole mine-years | probabilities are needed for thresholds; training time is quadratic in rows | | |
| MLP (grid, 9; 0.831 ± 0.003; 0.871) | hidden_layer_sizes: units per hidden layer | (64), (128), (128, 64) | one or two small layers for about 65,000 rows per fold | (128) | 0.831 (64) |
| | alpha: L2 weight penalty | 1e-4, 1e-2, 1e-1 | log-spaced from the default upwards | 1e-4 | 0.831 (0.01); 0.831 (0.1) |
| | Fixed settings | ReLU units; Adam with initial learning rate 0.001; batches of 200; early stopping on a 10% split; at most 200 epochs | standard settings, with early stopping against overfitting | | |

In {fig:validation} the decision tree underfits at depth 4 with equal training and validation scores, overfits at unlimited depth with a wide gap and peaks at depth 12 with large leaves. The random forest's deep trees open a wider gap still, yet its validation score rises with depth, because averaging cancels much of each tree's variance. The SVM overfits as C grows and has the largest fold-to-fold spread, a cost of its small subsample.

![Validation curves of the eight tuned models || Mean training-fold and validation-fold PR-AUC (average precision) with one standard deviation bands over the five grouped folds of the training split against the hyperparameter with most searched values, the other hyperparameters held at their selected values (for the random forest, the best sampled candidate per value), with the selected value dashed; curves are drawn because the train-validation gap and its shape show over- and underfitting. Source: author's analysis of MSHA open data.](reports/figures/F14_validation_curves.png){#fig:validation width=5.5}

Logistic regression is flat from C = 0.1 upwards, with training and validation scores close together, the signature of high bias. KNN scores 1.000 on its training folds because under distance weighting each training row is its own nearest neighbour; K = 101 and 151, tried after testing, scored below the grid-edge value of 51. Gaussian naive Bayes peaks sharply in var_smoothing without a gap; one smoothing constant decides its score. HGB and the MLP have flat curves and moderate gaps; for them the model class matters more than the selected values.

The threshold was tuned as a further hyperparameter on out-of-fold probabilities, never on test rows, whose labels would then make the reported recall optimistic. The capacity threshold is the lowest threshold whose out-of-fold specificity on all training rows reaches 0.40, a second threshold flags half the cases, and the cost rule minimises out-of-fold expected cost at each cost ratio ({fig:cost}). For HGB the test specificity (0.406) and flag rate (0.763) stayed close to the out-of-fold plan because test prevalence matched training.

The flag rate rises with prevalence (Section 4.4), and the case mix at receipt differs: by then 45.0% of all filings already record a return to work. Of the 11,290 open test reports, 71.2% are lost-time, and the capacity threshold flags 90.0% of them (recall 0.949, specificity 0.219), close to reviewing them all; the workload rule exists for that reason. {tab:policy} compares models on open reports in two ways: at exactly the top 50% or 75% by score, with the cut-off read from the test scores, and under the workload rule at a cut-off fixed before any test row was scored, the 25th percentile of each model's out-of-fold scores on open training reports.

Table: Capacity threshold and equal-workload policies on the test set || Recall, specificity and precision of HGB, the MLP, the lookup baseline and the one-field hand rule on all filings (the 20,320 test reports) and on the 11,290 open test reports: at the 0.211 capacity threshold, with exactly the top 50% or 75% by score reviewed (cut-off from the test scores), and under the workload rule at cut-offs fixed from out-of-fold scores; at exact workloads, tied scores of the lookup baseline and hand rule are broken by row order. Source: reports/metrics/followup_capacity_policy.csv and reports/metrics/followup_workload_rule.csv. {#tab:policy}

| Reports and workload | Model | Flag rate | Recall | Specificity | Precision | Score cut-off | Lost-time cases missed per 1,000 |
|---|---|---|---|---|---|---|---|
| All filings (20,320; prevalence 0.486), capacity threshold 0.211 | HGB | 0.763 | 0.941 | 0.406 | 0.600 | 0.211 | 28.5 |
| All filings, top 75% reviewed | HGB | 0.750 | 0.936 | 0.425 | 0.606 | 0.222 | 31.3 |
| | MLP | 0.750 | 0.934 | 0.424 | 0.605 | 0.200 | 32.0 |
| | Lookup baseline (ties broken by row order) | 0.750 | 0.915 | 0.406 | 0.593 | 0.228 | 41.3 |
| | Hand rule (ties broken by row order) | 0.750 | 0.906 | 0.397 | 0.587 | 0.000 | 45.9 |
| All filings, top 50% reviewed | HGB | 0.500 | 0.770 | 0.755 | 0.749 | 0.464 | 111.6 |
| | MLP | 0.500 | 0.769 | 0.754 | 0.747 | 0.458 | 112.3 |
| | Lookup baseline (ties broken by row order) | 0.500 | 0.734 | 0.721 | 0.713 | 0.447 | 129.4 |
| | Hand rule (ties broken by row order) | 0.500 | 0.611 | 0.605 | 0.594 | 1.000 | 189.0 |
| Open reports (11,290; prevalence 0.712), capacity threshold 0.211 | HGB | 0.900 | 0.949 | 0.219 | 0.750 | 0.211 | 36.7 |
| Open reports, top 75% reviewed (cut-off from the test scores) | HGB | 0.750 | 0.855 | 0.509 | 0.812 | 0.369 | 103.3 |
| | MLP | 0.750 | 0.854 | 0.508 | 0.811 | 0.358 | 103.6 |
| | Lookup baseline (ties broken by row order) | 0.750 | 0.828 | 0.442 | 0.786 | 0.435 | 122.8 |
| | Hand rule (ties broken by row order) | 0.750 | 0.766 | 0.290 | 0.728 | 1.000 | 166.3 |
| Open reports, workload rule at the cut-off fixed from out-of-fold scores | HGB | 0.743 | 0.850 | 0.522 | 0.815 | 0.376 | 106.7 |
| | MLP | 0.756 | 0.858 | 0.496 | 0.808 | 0.350 | 101.3 |
| | Lookup baseline | 0.752 | 0.829 | 0.438 | 0.785 | 0.433 | 121.8 |
| Open reports, top 50% reviewed | HGB | 0.500 | 0.640 | 0.846 | 0.911 | 0.610 | 256.3 |
| | MLP | 0.500 | 0.637 | 0.840 | 0.908 | 0.626 | 258.2 |
| | Lookup baseline (ties broken by row order) | 0.500 | 0.618 | 0.792 | 0.880 | 0.548 | 271.8 |
| | Hand rule (ties broken by row order) | 0.500 | 0.506 | 0.515 | 0.721 | 1.000 | 351.6 |

For open reports the operating policy is the workload rule at HGB's cut-off of 0.376, which flags 74.3% of open test reports with recall 0.850. The exact workloads take their cut-offs from the test scores and serve only to compare models: with three in four reviewed, recall is 0.855 for HGB, 0.854 for the MLP and 0.828 for the lookup baseline, against 0.75 for a random ranking, and with half reviewed 0.640, 0.637 and 0.618. At its own pre-fixed cut-off the MLP flags 75.6% with recall 0.858, its gain bought with 1.3 points more workload. Reviewing three in four open reports misses 103 lost-time cases per 1,000 against 36.7 at the capacity threshold, but when only open reports are scored it takes about 42% of all filings (0.75 x 11,290 / 20,320) against 50% (0.900 x 11,290 / 20,320).

The models are fitted on all filings, because reports filed after the return to work carry the same filing-time fields and nearly double the labelled rows. A refit on the 44,640 open training reports alone nevertheless ranks open test reports better (recall 0.877 against 0.855 with exactly 75% reviewed; {tab:robustness}). It has no pre-fixed cut-off yet, which makes it the candidate for the prospective trial of Section 5.4 and leaves the all-filings model, with selection and thresholds fixed before testing, as the reported model and the trial's comparator at 0.376.

The grids stayed narrow for compute reasons ({tab:decision}): searches took up to 15 minutes per model, and the temporal split repeated them all. Wider grids for KNN, the forest and HGB, searched after testing, raised validation PR-AUC by at most 0.0025, within one fold standard deviation ({tab:grids}). The one-standard-error rule, which picks the simplest candidate within one standard error of the best mean (Hastie et al., 2009), would give logistic regression a tenfold stronger penalty, the MLP its smaller layer and HGB the faster learning rate with fewer rounds; I kept the highest mean, since each alternative lies within fold noise and none changes the ranking.

## 4.4 Evaluation metrics

A true positive (TP) is a lost-time case flagged at filing, whose reserve and return-to-work plan then start early. A false negative (FN) is a lost-time case left in routine handling, the miss priced at USD 4,000 in {tab:costs}. A false positive (FP) sends a restricted-duty or no-lost-time case to an unnecessary review costing about USD 200, and a true negative (TN) is such a case left alone. Accuracy, the share of correct cells, was reported but not used, because it prices both errors alike and the majority class reaches 0.514 while flagging nothing ({tab:test}).

Recall, the true positive rate (TPR), is TP / (TP + FN), the share of lost-time cases flagged, which the insurer most needs. Precision, TP / (TP + FP), is the share of flags that are lost-time and sets the review effort per catch. Specificity, the true negative rate (TNR), is TN / (TN + FP). The flag rate, the review workload, equals prevalence x recall + (1 - prevalence) x (1 - specificity), so a specificity target leaves it free to move with prevalence. F1, the harmonic mean of precision and recall, weights both errors equally and was reported but not optimised; the expected cost states the asymmetry that an F-beta score would only approximate. Balanced accuracy averages TPR and TNR, and the Matthews correlation coefficient (MCC) is zero for any constant prediction; both expose a model that flags everything.

ROC-AUC is the probability that a random lost-time case scores above a random other case and does not depend on prevalence (Fawcett, 2006). PR-AUC averages precision over all recall levels, and its no-skill level equals the prevalence of 0.486. A model dominates in precision-recall space exactly when it dominates in ROC space (Davis and Goadrich, 2006), and at this prevalence, where the imbalance argument of Saito and Rehmsmeier (2015) does not apply, the two rank the models alike. PR-AUC is primary because precision is the review yield along the adjuster's queue, while S3 measures recall at the capacity threshold.

Because the application shows each probability and the cost rule takes it at face value, calibration was checked. The Brier score, the mean squared gap between probability and outcome, is 0.250 when every report receives the prevalence as its probability. A score under the S6 limit of 0.20 also reflects discrimination, which leaves the reliability diagram ({fig:calibration}) as the calibration evidence. Gaussian naive Bayes places many lost-time cases in its lowest bin, the overconfidence expected in Section 2.4, while HGB follows the diagonal on all filings. That fit averages two opposite biases: on open test reports HGB's mean prediction of 0.592 runs 0.12 below the observed rate of 0.712 (Brier 0.172), and on reports filed after the return to work 0.354 runs 0.15 above 0.203 (Brier 0.147). For open reports the probability is therefore a ranking score, not a calibrated chance; the workload rule uses only the ranking, and cost thresholds must not be applied to open reports without recalibration.

Expected cost per 1,000 cases is 1,000 x (FN x C_FN + FP x C_FP) / N, with C_FP from {tab:costs} and C_FN set by the cost ratio ({fig:cost}). At high ratios the cheapest threshold flags nearly every case at about the cost of reviewing everything, and cost then separates models only at the low ratios that S4 tests.

![Rates and expected cost against the HGB threshold || Recall, specificity, precision, F1 and flag rate (top) and expected cost per 1,000 cases at six cost ratios on a log scale (bottom) against the decision threshold of HGB on the 20,320 test reports, with the three out-of-fold thresholds dashed and the cost of reviewing everything dotted; one threshold axis lets each rate and cost be read at the same cut-off. Source: author's analysis of MSHA open data.](reports/figures/F18_threshold_cost_curve_hist_gb.png){#fig:cost width=6.3}

![Reliability diagram of the tuned models || Calibration curves of the eight tuned models and the lookup baseline on the 20,320 test reports, plotting the observed lost-time rate against the mean predicted probability in ten equal-width bins with marker size proportional to the cases in each bin and Brier scores in the legend; a reliability diagram is used because the application displays these probabilities. Source: author's analysis of MSHA open data.](reports/figures/F19_calibration.png){#fig:calibration width=5.0}

No metric here captures fairness between sectors, whose label partly records employer practice, or drift after the training years; results are therefore also reported by sector and on a temporal split ({tab:robustness}).

## 4.5 Comparison and final model

HGB leads both the cross-validated scores ({tab:cv}) and the test scores at the default threshold of 0.5 ({tab:test}), with the MLP second and logistic regression nearly level with the random forest. The curves of the top six models lie close together, while KNN and naive Bayes fall below the lookup baseline ({fig:roc}; {fig:pr}). At the capacity thresholds recall is close among the top six ({tab:operating}).

Table: Cross-validated scores of the selected candidates || Scores as mean ± SD over five grouped folds of the 81,289-row training split for each model's selected candidate at the default threshold of 0.5, ordered by PR-AUC, the refit metric; KNN and the SVM were cross-validated on their 39,999-row and 9,913-row subsamples, and TNR is pooled over the out-of-fold predictions of all training rows, so it has no SD. Source: author's analysis of MSHA open data. {#tab:cv}

| Model | PR-AUC | ROC-AUC | F1 | Recall (TPR) | TNR (pooled) | Precision | Accuracy | Brier |
|---|---|---|---|---|---|---|---|---|
| HGB | 0.838 ± 0.003 | 0.838 ± 0.002 | 0.743 ± 0.002 | 0.723 ± 0.007 | 0.788 | 0.764 ± 0.005 | 0.757 ± 0.002 | 0.164 ± 0.001 |
| MLP | 0.831 ± 0.003 | 0.834 ± 0.003 | 0.738 ± 0.005 | 0.714 ± 0.011 | 0.791 | 0.763 ± 0.004 | 0.753 ± 0.002 | 0.166 ± 0.002 |
| Random forest | 0.821 ± 0.004 | 0.828 ± 0.002 | 0.732 ± 0.003 | 0.704 ± 0.008 | 0.793 | 0.763 ± 0.004 | 0.750 ± 0.002 | 0.173 ± 0.001 |
| Logistic regression | 0.820 ± 0.003 | 0.825 ± 0.003 | 0.738 ± 0.002 | 0.739 ± 0.007 | 0.751 | 0.737 ± 0.005 | 0.745 ± 0.002 | 0.171 ± 0.001 |
| Decision tree | 0.803 ± 0.004 | 0.812 ± 0.004 | 0.724 ± 0.015 | 0.716 ± 0.047 | 0.752 | 0.735 ± 0.021 | 0.735 ± 0.004 | 0.176 ± 0.002 |
| SVM, RBF kernel | 0.795 ± 0.014 | 0.807 ± 0.011 | 0.723 ± 0.016 | 0.722 ± 0.019 | 0.753 | 0.724 ± 0.016 | 0.732 ± 0.015 | 0.179 ± 0.005 |
| KNN | 0.764 ± 0.009 | 0.766 ± 0.008 | 0.648 ± 0.011 | 0.584 ± 0.014 | 0.795 | 0.727 ± 0.012 | 0.692 ± 0.008 | 0.197 ± 0.003 |
| Gaussian naive Bayes | 0.735 ± 0.007 | 0.739 ± 0.003 | 0.602 ± 0.004 | 0.527 ± 0.007 | 0.787 | 0.701 ± 0.006 | 0.661 ± 0.003 | 0.298 ± 0.003 |

Table: Test-set scores at the default threshold || Scores of the eight tuned models and four baselines on the 20,320 test reports at the default threshold of 0.5, ordered by PR-AUC; TPR and recall are both TP / (TP + FN), and TNR is specificity, TN / (TN + FP). Source: author's analysis of MSHA open data. {#tab:test}

| Model | TPR | TNR | AUC | Recall | Precision | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|---|---|---|
| HGB | 0.740 | 0.786 | 0.843 | 0.740 | 0.766 | 0.753 | 0.842 | 0.764 |
| MLP | 0.737 | 0.789 | 0.839 | 0.737 | 0.768 | 0.752 | 0.836 | 0.764 |
| Random forest | 0.710 | 0.798 | 0.833 | 0.710 | 0.769 | 0.738 | 0.827 | 0.756 |
| Logistic regression | 0.744 | 0.759 | 0.829 | 0.744 | 0.745 | 0.744 | 0.825 | 0.752 |
| SVM, RBF kernel | 0.731 | 0.762 | 0.820 | 0.731 | 0.744 | 0.737 | 0.811 | 0.747 |
| Decision tree | 0.676 | 0.799 | 0.817 | 0.676 | 0.761 | 0.716 | 0.805 | 0.739 |
| Lookup baseline (nature x body part x sector) | 0.655 | 0.801 | 0.804 | 0.655 | 0.756 | 0.702 | 0.790 | 0.730 |
| KNN | 0.584 | 0.805 | 0.773 | 0.584 | 0.739 | 0.652 | 0.769 | 0.698 |
| Gaussian naive Bayes | 0.524 | 0.804 | 0.745 | 0.524 | 0.717 | 0.605 | 0.742 | 0.668 |
| Baseline: hand rule (not a cut, dust in the eye or a scratch) | 0.898 | 0.423 | 0.660 | 0.898 | 0.595 | 0.716 | 0.584 | 0.654 |
| Baseline: majority class | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.486 | 0.514 |
| Baseline: review everything | 1.000 | 0.000 | 0.500 | 1.000 | 0.486 | 0.654 | 0.486 | 0.486 |

![ROC curves on the test reports || ROC curves of the eight tuned models and the lookup baseline on the 20,320 test reports, with test ROC-AUC in the legend and the chance diagonal dotted; ROC space is shown because it does not depend on prevalence. Source: author's analysis of MSHA open data.](reports/figures/F12_roc_curves.png){#fig:roc width=5.0}

![Precision-recall curves on the test reports || Precision-recall curves of the eight tuned models and the lookup baseline on the 20,320 test reports, with average precision (PR-AUC) in the legend and the prevalence of 0.486 dotted as the no-skill level; precision-recall space is shown because precision along the curve is the review cost of each lost-time case found. Source: author's analysis of MSHA open data.](reports/figures/F13_pr_curves.png){#fig:pr width=5.0}

Table: Test-set scores at the capacity thresholds || Scores on the 20,320 test reports at each model's capacity threshold, the lowest threshold whose out-of-fold specificity reached 0.40 on the training split, with the resulting flag rate, balanced accuracy, MCC and Brier score; the threshold-free AUC and PR-AUC are in {tab:test}, and the two constant baselines are omitted because no threshold changes a constant score. Source: author's analysis of MSHA open data. {#tab:operating}

| Model | Threshold | Flag rate | TPR (recall) | TNR | Precision | F1 | Balanced accuracy | MCC | Brier |
|---|---|---|---|---|---|---|---|---|---|
| HGB | 0.211 | 0.763 | 0.941 | 0.406 | 0.600 | 0.733 | 0.674 | 0.408 | 0.161 |
| MLP | 0.184 | 0.768 | 0.942 | 0.396 | 0.596 | 0.730 | 0.669 | 0.400 | 0.163 |
| Random forest | 0.290 | 0.757 | 0.936 | 0.412 | 0.601 | 0.732 | 0.674 | 0.405 | 0.170 |
| Logistic regression | 0.223 | 0.760 | 0.934 | 0.404 | 0.597 | 0.728 | 0.669 | 0.395 | 0.169 |
| SVM, RBF kernel | 0.235 | 0.754 | 0.927 | 0.409 | 0.597 | 0.726 | 0.668 | 0.389 | 0.173 |
| Decision tree | 0.239 | 0.773 | 0.932 | 0.377 | 0.586 | 0.719 | 0.654 | 0.369 | 0.174 |
| Lookup baseline (nature x body part x sector) | 0.225 | 0.753 | 0.916 | 0.402 | 0.591 | 0.719 | 0.659 | 0.368 | 0.179 |
| KNN | 0.300 | 0.732 | 0.887 | 0.414 | 0.589 | 0.707 | 0.650 | 0.340 | 0.195 |
| Gaussian naive Bayes | 0.001 | 0.730 | 0.875 | 0.407 | 0.582 | 0.699 | 0.641 | 0.318 | 0.291 |
| Baseline: hand rule (not a cut, dust in the eye or a scratch) | not applicable (fixed rule) | 0.733 | 0.898 | 0.423 | 0.595 | 0.716 | 0.660 | 0.363 | 0.346 |

{tab:stats} tests whether these differences exceed noise (Dietterich, 1998; Demšar, 2006). McNemar's test rejects equal error rates between HGB and the lookup baseline at both thresholds, also when clustered by mine-year (Durkalski et al., 2003). Between the top two, the paired fold test gives p = 0.029 after the Nadeau and Bengio (2003) correction for overlapping training folds. With HGB higher in all five folds the Wilcoxon statistic is W = 0, the strongest result possible, yet its p-value cannot fall below 0.0625 with five folds. On open reports with exactly 75% reviewed, a paired mine-year bootstrap puts HGB's recall above the lookup baseline's by 0.027 (95% interval from 0.021 to 0.033) and level with the MLP's (from -0.003 to 0.004).

Table: Statistical comparisons of HGB || McNemar tests against the lookup baseline on the 20,320 test reports at the default and capacity thresholds, row-wise and clustered by the 6,724 test mine-years; paired tests against the MLP on the five cross-validation folds; 95% bootstrap intervals (1,000 resamples) by rows and by mine-years; and a paired bootstrap over the 4,537 mine-years of the 11,290 open test reports with exactly 75% reviewed. Source: author's analysis of MSHA open data. {#tab:stats}

| Comparison | Test | Result | p-value |
|---|---|---|---|
| HGB against lookup baseline, threshold 0.5 | McNemar, rows treated as independent | chi-square 148.7; HGB alone correct on 1,946 reports, lookup baseline alone on 1,255 | below 0.001 |
| HGB against lookup baseline, threshold 0.5 | McNemar clustered by mine-year (Durkalski et al., 2003) | chi-square 87.5 | below 0.001 |
| HGB against lookup baseline, capacity thresholds (0.211 and 0.225) | McNemar, rows treated as independent | chi-square 36.9; 1,318 against 1,023 reports | below 0.001 |
| HGB against lookup baseline, capacity thresholds | McNemar clustered by mine-year (Durkalski et al., 2003) | chi-square 24.5 | below 0.001 |
| HGB against MLP, cross-validated PR-AUC | paired t-test | mean difference 0.0062 (SD 0.0027); HGB higher in all five folds; t = 5.02 | 0.007 |
| HGB against MLP, cross-validated PR-AUC | corrected resampled t-test (Nadeau and Bengio, 2003) | t = 3.34 | 0.029 |
| HGB against MLP, cross-validated PR-AUC | Wilcoxon signed-rank | W = 0 | 0.0625, the smallest value possible with five folds |
| HGB, test PR-AUC 0.842 | bootstrap of rows; bootstrap of mine-years | 0.835 to 0.849; 0.826 to 0.856 | |
| HGB, test ROC-AUC 0.843 | bootstrap of rows; bootstrap of mine-years | 0.838 to 0.849; 0.836 to 0.852 | |
| HGB, recall 0.941 at the capacity threshold | bootstrap of rows; bootstrap of mine-years | 0.937 to 0.946; 0.936 to 0.947 | |
| HGB against lookup baseline and MLP, open reports, top 75% reviewed | paired bootstrap of 4,537 mine-years (1,000 resamples) | recall difference 0.027 (0.021 to 0.033) against the lookup baseline and 0.000 (-0.003 to 0.004) against the MLP; PR-AUC difference 0.038 (0.032 to 0.044) and 0.003 (0.001 to 0.005) | |

I selected HGB by the rule fixed before any test score was computed, the highest cross-validated PR-AUC, where its lead over the MLP exceeds one fold standard deviation ({tab:cv}). The test-set evidence in {tab:decision} checks this choice but did not make it: HGB has the lowest expected cost at the three low cost ratios, and on the temporal split it kept the higher PR-AUC while neither model's specificity moved by more than 0.008 without the 103(i) status or the immediate-reporting flag. Under the mine-year split a training row's mine history can draw on the same mine's previous-year test rows (Section 3.9), a path the temporal split, trained on earlier years, does not have. Blanking those histories and refitting HGB gives test PR-AUC 0.842 in both fits (the refit is higher by less than 0.001) and recall 0.941 at the capacity threshold in both ({tab:robustness}).

Its permutation importance puts nature of injury and body part first ({fig:importance}), as do most of the largest logistic coefficients, and two of its four largest mean SHAP values ({tab:shap}; Lundberg and Lee, 2017), for cuts and finger injuries, match negative coefficients ({fig:coefficients}). The final pipeline uses the settings of {tab:grids}, refitted on all training rows, at its out-of-fold capacity threshold ({tab:confusion}; {fig:confusion}), where it misses 579 of 9,873 lost-time cases against 830 for the lookup baseline at about the same specificity.

Table: Multi-criteria comparison of the strongest models || Comparison of the four strongest tuned models and the SVM, compiled after HGB was selected by cross-validated PR-AUC: test-set cost at the out-of-fold cost-optimal thresholds, transfer of out-of-fold threshold targets to the test set, recall on open reports and the temporal split (top two only), explanations, run times on eight workers and the size of the saved pipeline. Source: author's analysis of MSHA open data. {#tab:decision}

| Criterion | HGB | MLP | Random forest | Logistic regression | SVM, RBF kernel |
|---|---|---|---|---|---|
| Cross-validated PR-AUC (rank of eight) | 0.838 (1) | 0.831 (2) | 0.821 (3) | 0.820 (4) | 0.795 (6) |
| Test PR-AUC | 0.842 | 0.836 | 0.827 | 0.825 | 0.811 |
| Test cost per 1,000 cases at 2:1 / 3:1 / 5:1 (USD) | 65,817 / 76,781 / 87,205 | 67,047 / 77,864 / 90,000 | 67,736 / 78,947 / 90,059 | 68,622 / 80,138 / 91,299 | 70,659 / 82,195 / 92,697 |
| Test specificity at the out-of-fold threshold for specificity 0.40 | 0.406 | 0.396 | 0.412 | 0.404 | 0.409 |
| Test flag rate at the out-of-fold threshold for a 50% flag rate | 0.507 | 0.511 | 0.497 | 0.497 | 0.500 |
| Test recall on the 11,290 open reports under the workload rule at the out-of-fold cut-off (exactly 75% reviewed) | 0.850 (0.855) | 0.858 (0.854) | not run | not run | not run |
| Temporal split: PR-AUC on 2020-2025 | 0.843 | 0.838 | not run | not run | not run |
| Temporal split: specificity at the capacity threshold across three feature variants | 0.373 to 0.381 | 0.388 to 0.394 | not run | not run | not run |
| Explanation available | permutation importance; SHAP values | permutation importance | permutation importance | coefficients as odds ratios | permutation importance |
| Scaler needed | no | yes | no | yes | yes |
| Tuning time (s) | 505 | 161 | 139 | 66 | 871 |
| Prediction time for 20,320 reports (s) | 0.28 | 0.09 | 0.57 | 0.09 | 30.96 |
| Saved pipeline (MB) | 0.48 | 0.80 | 20.32 | 0.01 | 1.22 |

![Permutation importance of HGB || Permutation importance of the 20 most important input features of HGB, measured as the mean drop in test PR-AUC (average precision) over five shuffles on 5,000 test reports with one standard deviation whiskers; importance is computed on input features rather than one-hot columns so that each nominal code counts once. Source: author's analysis of MSHA open data.](reports/figures/F17_feature_importance_hist_gb.png){#fig:importance width=5.5}

![Largest logistic regression coefficients || The 20 largest coefficients of the tuned logistic regression fitted on the training split, with numeric inputs standardised and one-hot columns coded 0 or 1 and labelled by their codebook meaning, where positive values (blue) raise and negative values (orange) lower the odds of a lost-time case; signed bars are used because each coefficient converts to an odds ratio. Source: author's analysis of MSHA open data.](reports/figures/F17b_coefficients_logreg.png){#fig:coefficients width=5.5}

Table: Confusion counts on the test reports || Counts on the 20,320 test reports for HGB at its capacity threshold and at 0.5, and for the lookup baseline at its capacity threshold, where a true positive is a lost-time case flagged for early review and a false negative a lost-time case left in routine handling. Source: author's analysis of MSHA open data. {#tab:confusion}

| Model and threshold | TP: lost-time, flagged | FN: lost-time, not flagged | FP: other, flagged | TN: other, not flagged | Recall | Precision | Flag rate |
|---|---|---|---|---|---|---|---|
| HGB, capacity threshold 0.211 | 9,294 | 579 | 6,204 | 4,243 | 0.941 | 0.600 | 0.763 |
| HGB, threshold 0.5 | 7,306 | 2,567 | 2,231 | 8,216 | 0.740 | 0.766 | 0.469 |
| Lookup baseline, capacity threshold 0.225 | 9,043 | 830 | 6,248 | 4,199 | 0.916 | 0.591 | 0.753 |

![Confusion matrices at two thresholds || Confusion matrices on the 20,320 test reports for HGB (top) and the lookup baseline (bottom) at the default threshold of 0.5 (left) and at each one's out-of-fold capacity threshold (right), with counts and row percentages; matrices are shown because each cell is a distinct outcome for the adjuster. Source: author's analysis of MSHA open data.](reports/figures/F16_confusion_matrices.png){#fig:confusion width=6.0}

HGB passes criteria S1 to S4, S6 and S7 ({tab:criteria_results}; {tab:stats}). It misses S5 at the out-of-fold threshold for a 50% flag rate (0.777 against 0.78), as does the MLP, and the miss recurs on the temporal and mine-grouped splits and among the top half of test scores, the planned form of S5. The lookup baseline passes S1, S2 and S6 and fails S5 like the learned models; only S3 and S4 separate them from it. S3 is defined at the capacity threshold on all filings and is met there; the same threshold also reaches it on open test reports (recall 0.949), but only by flagging 90.0% of them. The workload rule lowers recall by design, to 0.850 at a flag rate of 74.3% ({tab:policy}), and S3 was not set for it.

Table: Results against the success criteria || Success criteria S1 to S7, fixed before modelling, with S7 a reporting criterion, evaluated on the 20,320 test reports for HGB, the MLP and the lookup baseline with thresholds fixed on out-of-fold probabilities, with S5 repeated on the temporal and mine-grouped splits; the lookup baseline's tied scores stop the planned top-half cut-off at a flag rate of 0.490. Source: author's analysis of MSHA open data. {#tab:criteria_results}

| Criterion | Target | HGB | Result | MLP | Lookup baseline |
|---|---|---|---|---|---|
| S1 ROC-AUC | 0.80 or more | 0.843 | pass | 0.839 | 0.804 |
| S2 PR-AUC | 0.75 or more | 0.842 | pass | 0.836 | 0.790 |
| S3 recall on all filings at the capacity threshold (out-of-fold specificity 0.40) | 0.92 or more | 0.941 (specificity 0.406) | pass | 0.942 (0.396) | 0.916 (0.402) |
| S4 cost per 1,000 cases at the out-of-fold cost-optimal threshold, 2:1 / 3:1 / 5:1 (USD) | below the review-everything floor (102,825 on the test set) and below the lookup baseline | 65,817 / 76,781 / 87,205 | pass | 67,047 / 77,864 / 90,000 | 73,553 / 85,472 / 98,061 |
| S5 recall at the out-of-fold threshold for a 50% flag rate | 0.78 or more | 0.777 (flag rate 0.507) | fail | 0.777 (0.511) | 0.753 (0.520) |
| S5 as planned: recall among the 50% of test reports with the highest scores (cut-off from the test scores, labels only count hits) | 0.78 or more | 0.770 (flag rate 0.500) | fail | 0.769 (0.500) | 0.725 (0.490) |
| S5 on the temporal split (2020-2025) and the mine-grouped split | 0.78 or more | 0.747 and 0.762 | fail | 0.730 (temporal) | 0.710 (mine-grouped) |
| S6 Brier score on the test set, and by sector on the temporal split | 0.20 or less | test 0.161 (coal 0.135, metal and non-metal 0.179); temporal split: coal 0.125, metal and non-metal 0.186 | pass | test 0.163 (0.137 and 0.181); temporal split: 0.126 and 0.189 | test 0.179 (0.150 and 0.199) |
| S7 statistical comparison | McNemar test against the lookup baseline and a paired fold test between the top two | clustered McNemar p below 0.001; corrected paired t p = 0.029 ({tab:stats}) | reported | | |

No fatality or permanent disability was missed at the capacity threshold on all filings ({tab:errors}). Most misses are cuts and lacerations, above all to fingers, thumbs and hands, where more than half of the lost-time cases score below the threshold; such injuries are usually minor, and the few that keep a worker away resemble the many that do not. Metal and non-metal mines hold 420 of the 579 misses and have the lower precision ({tab:modelcard}), consistent with their lower lost-time rate. Most false positives are restricted-duty cases, which differ from lost-time ones mainly in whether restricted work was offered.

Table: Errors of HGB at the capacity threshold || Missed lost-time cases at the threshold of 0.211 on the 20,320 test reports by sector, by degree-of-injury code and for the six nature x body part cells with most misses, and flagged non-lost-time cases by degree code. Source: author's analysis of MSHA open data. {#tab:errors}

| Group | Cases | Misclassified | Share |
|---|---|---|---|
| All lost-time cases | 9,873 | 579 missed | 0.059 |
| Lost-time cases, coal | 5,280 | 159 missed | 0.030 |
| Lost-time cases, metal and non-metal | 4,593 | 420 missed | 0.091 |
| Fatality (code 01) | 142 | 0 missed | 0.000 |
| Permanent disability (02) | 224 | 0 missed | 0.000 |
| Days away only (03) | 7,380 | 388 missed | 0.053 |
| Days away and restricted activity (04) | 2,127 | 191 missed | 0.090 |
| Cut, laceration or puncture x finger or thumb | 324 | 175 missed | 0.540 |
| Cut, laceration or puncture x hand | 104 | 63 missed | 0.606 |
| Fracture x finger or thumb | 392 | 52 missed | 0.133 |
| Fracture x mouth, lips, teeth or throat | 19 | 17 missed | 0.895 |
| Burn or chemical injury x eye | 46 | 17 missed | 0.370 |
| Cut, laceration or puncture x forearm | 27 | 16 missed | 0.593 |
| Non-lost-time: restricted activity only (05) | 4,212 | 3,273 flagged | 0.777 |
| Non-lost-time: no days away and no restriction (06) | 6,235 | 2,931 flagged | 0.470 |

{tab:modelcard} is the model card of HGB (Mitchell et al., 2019). Its monitoring entries follow Sculley et al. (2015), who show that deployed models accrue hidden debt through shifting inputs and feedback loops. A feedback loop is plausible here, since return-to-work plans prompted by flags could turn lost-time cases into restricted-duty ones and change the label the model learns from.

Table: Model card of the HGB pipeline || Fields follow Mitchell et al. (2019); intended-use results, and the first half of the subgroup results, are on the 11,290 open test reports under the workload rule at its cut-off of 0.376 fixed from out-of-fold scores, and the other results are on the 20,320 test reports at the capacity threshold, the protocol fixed before testing. Source: author's analysis of MSHA open data. {#tab:modelcard}

| Field | Content |
|---|---|
| Model | HistGradientBoostingClassifier from scikit-learn 1.9.1 inside the preprocessing pipeline of {fig:pipeline}; learning rate 0.05, at most 400 rounds with early stopping, 31 leaf nodes, no L2 penalty; refitted on the 81,289 training rows on 20 October 2026, 52 inputs, 546 encoded columns; saved pipeline 0.48 MB |
| Intended use | scoring each Form 7000-1 report at receipt when it carries no recorded return-to-work date (an open report), and ranking open reports for early review by workers'-compensation adjusters at insurers and self-insured operators under the workload rule, the top three in four by score; an adjuster confirms every flag; operational use waits for the prospective trial of Section 5.4 |
| Out-of-scope uses | ranking mines for inspection, which the DART label suits better; setting benefit levels or denying claims; judging individual workers; reports outside 2010-2025 or outside degree codes 01 to 06 |
| Data | 101,609 MSHA reports of 2010-2025 (Section 1.3), split by 32,732 mine-years with 6,724 of them in the test set; secondary temporal split (2010-2019 against 2020-2025) and mine-grouped split |
| Metrics | PR-AUC (primary), ROC-AUC, recall, specificity, precision, Brier score and expected cost per 1,000 cases, with thresholds from out-of-fold probabilities |
| Overall performance | PR-AUC 0.842 (mine-year bootstrap 95% interval 0.826 to 0.856), ROC-AUC 0.843 and Brier score 0.161; recall 0.941, specificity 0.406 and precision 0.600 at the capacity threshold of 0.211, which flags 76.3% of all filings in the test set |
| Intended-use performance | 11,290 open test reports (prevalence 0.712): flag rate 0.743, recall 0.850, precision 0.815 and 107 lost-time cases missed per 1,000; with exactly 75% reviewed, recall 0.855; probabilities rank but are not calibrated (mean 0.592 against an observed rate of 0.712, Brier 0.172); a refit on open training reports reaches recall 0.877 with exactly 75% reviewed |
| Subgroup performance | open test reports under the workload rule: coal (4,773 reports, prevalence 0.912) flag rate 0.915, recall 0.930, precision 0.928, PR-AUC 0.974, Brier 0.111; metal and non-metal (6,517, prevalence 0.565) flag rate 0.617, recall 0.755, precision 0.692, PR-AUC 0.797, Brier 0.216; all filings at the capacity threshold: coal (8,270, prevalence 0.638) recall 0.970, precision 0.739, PR-AUC 0.913, Brier 0.135; metal and non-metal (12,050, prevalence 0.381) recall 0.909, precision 0.487, PR-AUC 0.708, Brier 0.179 |
| Stability checks | temporal split: PR-AUC 0.843 and recall 0.945 at a realised specificity of 0.373; mine-grouped split: PR-AUC 0.837 and recall 0.934; removing immed_reportable or the 103(i) status, counting only earlier cases closed before the accident or blanking early-January exposure lowered test PR-AUC by at most 0.0025; without the 175 amputation reports, all lost-time, test PR-AUC is 0.835; history audit, with the 11,416 test-derived mine histories blanked on training rows: PR-AUC 0.842 |
| Limitations | misses S5 (recall 0.777 at a 50% flag rate); probabilities run 0.12 low on open reports and 0.15 high on reports filed after the return to work; on open metal and non-metal reports workload-rule recall is 0.755 against 0.930 in coal, and the Brier score 0.216; the label records whether restricted duty was offered; self-reported data omit unreported injuries; scores describe association, not cause |
| Caveats | the evaluation is retrospective, classifying reports whose outcome is now known from fields the form carries at filing; the extract holds the final degree code and no filing-time value of the days-away, days-restricted or degree fields, so whether the filer already knew the outcome cannot be reconstructed; prospective validation on reports scored at receipt is required before operational use |
| Ethical considerations | no names or narratives are used, but occupation, experience and state can stand in for groups of workers; a flag must never reduce benefits; the sector gap is disclosed; the application states the intended use and leaves the decision to a person |
| Monitoring | recalibrate on open reports before use (metal and non-metal Brier score 0.216 there); then track the flag rate among open reports against the three in four planned, and recall, PR-AUC and Brier score by sector once those reports close; retrain or recalibrate when a sector's PR-AUC falls more than 0.03 below its test value (twice the 0.015 half-width of the mine-year bootstrap interval of PR-AUC), its Brier score exceeds 0.20 or the share of pooled infrequent levels grows (Sculley et al., 2015) |
