# 4 Modelling

## 4.1 Summary

The task is supervised binary classification of each Form 7000-1 report as lost-time or not, with a probability that ranks the adjuster's review queue. Eight algorithms were compared on the same split and fold scheme ({tab:algorithms}): logistic regression, KNN, Gaussian naive Bayes, a decision tree, a random forest, histogram-based gradient boosting (HGB), an SVM with a radial basis function (RBF) kernel and an MLP. Each was tuned by GridSearchCV, or RandomizedSearchCV for the forest, over five StratifiedGroupKFold folds of mine-year groups inside the training split. PR-AUC, computed as average precision, was the primary metric and the refit criterion ({tab:grids}). Decision thresholds were then fixed on out-of-fold probabilities, so the test set was scored once. HGB ranked first in cross-validation ({tab:cv}) and on the test set ({tab:test}), with a test PR-AUC of 0.841. At its capacity threshold it flags about three reports in four and finds 0.943 of lost-time cases with a precision of 0.599 ({tab:operating}; {tab:confusion}). It passes five of the six performance criteria and misses recall at half capacity by 0.005 ({tab:criteria_results}).

## 4.2 Algorithm selection

Encoding yields 552 columns, mostly zero-valued one-hot indicators beside skewed numeric and binary features ({tab:decisionlog}). Size relations are monotone but curved and collinear ({tab:associations}). Nominal codes lead the associations, and the classes are near balance. An adjuster needs a reason for every flag and the application scores forms on demand, so interpretability and speed also counted ({tab:algorithms}).

Table: The eight candidate algorithms with their paradigm, key assumption, the violation found in the training split and the bias it introduces, scaling need, interpretability and the module week in which each was taught; violations refer to the exploratory analysis of the training split in Section 2. Source: author's analysis of MSHA open data. {#tab:algorithms}

| Algorithm | Paradigm | Key assumption | Violation in this data and resulting bias | Scaling | Interpretability | Taught week |
|---|---|---|---|---|---|---|
| Logistic regression | parametric; discriminative; eager | log-odds linear and additive in the encoded columns | curved size effects (Pearson r 0.645 against Spearman 0.994) and nature x body part interactions; underfits (high bias) | StandardScaler | high: coefficients convert to odds ratios | 4 |
| KNN | non-parametric; discriminative; lazy (instance-based) | rows close in scaled Euclidean space share outcomes | 552 mostly binary dimensions flatten distances, so votes drift towards the local base rate | StandardScaler | low: a flag is explained only by similar past reports | 5 |
| Gaussian naive Bayes | parametric; generative; eager | features independent given the class, each Gaussian within a class | collinear size measures (VIF above 160) and nine highly skewed numeric features; evidence counted twice, overconfident posteriors | StandardScaler (var_smoothing is relative to the largest variance) | medium: class-conditional means | 5 |
| Decision tree (CART) | non-parametric; discriminative; eager | classes separate in axis-aligned boxes | curved effects need many steps; greedy splits give high variance | none | high when shallow: if-then rules | 6 |
| Random forest | non-parametric bagging ensemble; discriminative; eager | averaging decorrelated deep trees cancels their variance | few violations; averaged leaf frequencies pull probabilities towards the middle | none | medium: permutation importance | 6 |
| Histogram-based gradient boosting (HGB) | non-parametric boosting ensemble; discriminative; eager | log-odds is a sum of small trees on binned features | rare levels pooled into one column; greedy binned splits; tied to the training-period mix of codes | none | medium: permutation importance and SHAP values | 6 |
| SVM, RBF kernel | non-parametric (kernel); discriminative; eager | a smooth wide-margin boundary in kernel space | training cost at least quadratic in rows, so fitted on 10,000 rows; scores need Platt scaling | StandardScaler | low: margin scores only | 5 |
| MLP | parametric; discriminative; eager | a smooth mapping learnable by gradient descent | tens of thousands of weights on sparse one-hot inputs; overfits without alpha and early stopping | StandardScaler | low: needs post-hoc explanation | 10 |

Logistic regression passes a weighted sum of the inputs through the sigmoid and minimises binary cross-entropy by lbfgs, a quasi-Newton relative of gradient descent, under an L2 penalty that weakens as C rises (Hosmer et al., 2013). Linear log-odds miss curved size effects and code interactions, so it underfits. Its readable odds ratios made it the transparency benchmark. Gaussian naive Bayes applies Bayes' theorem, combining the class prior with per-feature Gaussian likelihoods into a posterior. Conditional independence fails for the duplicated size measures and normality for the skewed and binary features ({fig:qq}), so evidence is counted twice and posteriors become overconfident. Ranking can survive this (Domingos and Pazzani, 1997), and I kept it as a fast generative baseline and calibration test.

KNN, a lazy instance-based learner, labels a report by the vote of its K nearest training rows under Euclidean distance, with K odd to avoid ties (Cover and Hart, 1967). Distances need standardised inputs and lose contrast across 552 mostly binary dimensions, the curse of dimensionality. It tests whether similarity to past claims adds signal; since scoring compares each report with every stored row, I fitted it on whole mine-years capped at 40,000 rows. An SVM maximises the margin between the classes, with C penalising rows inside the margin and the RBF width gamma setting how local the boundary is (Cortes and Vapnik, 1995). Training time grows at least quadratically with rows, so I fitted it on whole mine-years capped at 10,000 rows and obtained probabilities by Platt scaling. It was included to check whether a smooth boundary ranks lost-time cases better than tree splits.

A decision tree splits greedily on the axis-aligned threshold that most reduces Gini impurity or entropy. It needs no scaling and nests interactions, yet a small change in the rows can alter every split below the root. A random forest averages deep trees grown on bootstrap samples with a random feature subset per split, which decorrelates them and cuts that variance (Breiman, 2001). The tree is the single-tree reference for the ensembles, and the forest the bagging rival to HGB.

HGB adds shallow trees in stages, fitting each to the gradient of the log loss and shrinking it by the learning rate, with capped leaf nodes and early stopping. It follows the second-order boosting of XGBoost (Chen and Guestrin, 2016). As a sum of small trees, its log-odds can follow curved effects and code interactions. Tree ensembles lead on medium-sized tabular data (Grinsztajn et al., 2022), so it was expected to rank the review queue best.

The MLP trains one or two hidden layers of rectified linear units by Adam, a gradient-descent variant, with alpha as an L2 penalty and early stopping (Bishop, 2006). Tens of thousands of weights on mostly zero inputs invite overfitting, so the network stayed small. It tests whether learned code combinations add recall at the capacity threshold.

Three families were rejected. Time-series models such as ARIMA assume one quantity observed at regular intervals, but each row is a separate claim and time enters through lagged mine history. Clustering would discard the label, so unsupervised methods stayed in the exploratory analysis ({fig:pca}). Deep tabular networks trail tree ensembles at this scale and need more tuning than the MLP (Grinsztajn et al., 2022). Because no algorithm wins on every problem (Wolpert, 1996), the eight were compared under one protocol.

The learning curve places that comparison in bias-variance terms (Hastie et al., 2009). On a 10,000-row subsample HGB's training score falls and validation score rises as rows are added ({fig:learning}), so its gap is variance that more data removes. Naive Bayes shows almost no gap at a low score, so its error is bias ({tab:grids}). A flexible, regularised learner therefore suits this volume of data.

![Learning curve of HGB with its selected settings, giving mean training-fold and validation-fold PR-AUC (average precision) with one standard deviation bands over five grouped folds of a 10,000-row subsample of the training split, drawn against training size because the gap between the two curves separates variance from bias. Source: author's analysis of MSHA open data.](reports/figures/F15_learning_curve_hist_gb.png){#fig:learning width=5.5}

## 4.3 Model selection per algorithm

Every search tuned the whole pipeline, which refitted imputers, encoder, scaler and classifier on the training rows of each fold ({fig:pipeline}). I used five StratifiedGroupKFold folds over mine-year groups inside the training split, which keep each mine-year on one side and the lost-time share near that of the whole. Kohavi (1995) favoured ten stratified folds; five halved the fits and left about 16,000 rows per validation fold. Grouping mirrors deployment, where the insurer scores mine-years the model has not seen, so no setting is rewarded for memorising a mine's other claims. PR-AUC was the refit metric, with the other metrics recorded alongside. Seed 42 fixed the folds, subsamples and estimators.

{tab:grids} lists every searched hyperparameter with its meaning, bounds, selected value and the best score reached with any other value. Most grids contain the scikit-learn default and step towards stronger and weaker regularisation, and settings outside them kept defaults for the reasons listed. The random forest is the exception, since RandomizedSearchCV sampled only part of its grid; the full grid, searched after testing, added 0.001 with 200 trees. For HGB, both caps on boosting rounds scored identically at the faster learning rate, because early stopping ended training before either cap.

Table: Hyperparameter search per model on the 81,289 training rows with five grouped folds; the first row of each model gives the search, the number of candidates, the selected candidate's validation PR-AUC (mean ± SD) and its training-fold PR-AUC, the last column the best validation PR-AUC with another value (in brackets), and the rows marked as follow-up give wider grids searched after the test scoring, which changed no choice. Source: author's analysis of MSHA open data. {#tab:grids}

| Model (search; validation PR-AUC; training PR-AUC) | Hyperparameter: meaning | Values searched | Why these bounds | Selected | Best other value |
|---|---|---|---|---|---|
| Logistic regression (grid, 8 candidates; 0.820 ± 0.003; 0.828) | C: inverse strength of the L2 penalty | 0.01, 0.1, 1, 10 | four decades around the default of 1 | 1 | 0.820 (10); 0.815 (0.01) |
| | class_weight: reweights the classes in the loss | None, balanced | tests reweighting as the alternative to resampling at near balance | None | 0.820 (balanced) |
| | Fixed settings | L2 penalty, lbfgs solver, max_iter 2,000 | L2 suits collinear columns; no convergence warning was logged | | |
| KNN (grid, 6; 0.764 ± 0.009; 1.000) | n_neighbors (K): number of voters | 11, 25, 51 | odd values from a local to a smooth vote | 51 | 0.758 (25) |
| | weights: equal or inverse-distance votes | uniform, distance | lets close neighbours count more | distance | 0.757 (uniform) |
| | p: Minkowski power | 2 | Euclidean distance only, to keep six candidates | 2 | none |
| | Fixed settings | brute-force search; 39,999-row subsample of whole mine-years | tree indexes do not help in 552 dimensions; scoring cost grows with stored rows | | |
| | Follow-up grid (6 candidates, 265 s) | n_neighbors 51, 101, 151 with both weightings | the original optimum sat at the grid edge | 51, distance (0.764) | 0.763 (101); 0.761 (151) |
| Gaussian naive Bayes (grid, 5; 0.734 ± 0.006; 0.735) | var_smoothing: share of the largest variance added to every variance | 1e-9, 1e-6, 1e-3, 0.1, 1 | log-spaced from the default upwards, since near-constant one-hot columns have tiny variances | 0.1 | 0.702 (1) |
| | Fixed settings | class priors from training frequencies | the priors then equal the 48.6% prevalence | | |
| Decision tree (grid, 24; 0.803 ± 0.004; 0.827) | max_depth: longest root-to-leaf path | 4, 8, 12, None | from a readable tree to unlimited growth | 12 | 0.793 (None) |
| | min_samples_leaf: smallest leaf | 5, 20, 50 | larger leaves give steadier lost-time rates | 50 | 0.794 (20) |
| | criterion: impurity measure | gini, entropy | the two impurity measures taught | gini | 0.802 (entropy) |
| | Fixed settings | best split over all features; no cost-complexity pruning | depth and leaf size already prune the tree | | |
| Random forest (random search, 4 of 24 grid points; 0.821 ± 0.003; 0.946) | n_estimators: number of trees | 100, 200 | the default and double | 100 | not sampled |
| | max_depth: longest path in each tree | 8, 12, None | as for the single tree | None | 0.808 (12) |
| | min_samples_leaf: smallest leaf | 5, 20 | small leaves suit averaged deep trees | 5 | 0.815 (20) |
| | max_features: features tried per split | sqrt, log2 | the two usual subset sizes for decorrelation | sqrt | not sampled |
| | Fixed settings | bootstrap samples, Gini impurity | the design of Breiman (2001) | | |
| | Follow-up grid (all 24 points, 559 s) | every combination of the values above | the random search sampled four points | 200 trees, otherwise as selected (0.822 ± 0.003) | 0.821 (100 trees); 0.810 (log2) |
| HGB (grid, 16; 0.838 ± 0.003; 0.873) | learning_rate: shrinkage applied to each tree | 0.05, 0.1 | the default and half of it | 0.05 | 0.837 (0.1) |
| | max_iter: cap on boosting rounds | 200, 400 | caps only, since early stopping ends training | 400 | 0.837 (200) |
| | max_leaf_nodes: leaves per tree | 15, 31 | the default and about half | 31 | 0.836 (15) |
| | l2_regularization: penalty on leaf values | 0, 1 | no penalty or a unit penalty | 0 | 0.838 (1) |
| | early_stopping | True | stops after 10 rounds without loss improvement on a 10% validation split | True | none |
| | Fixed settings | min_samples_leaf 20, 255 bins, no depth limit | the leaf count already limits tree size | | |
| | Follow-up grid (24 candidates, 1,104 s) | max_leaf_nodes 15, 31, 63; min_samples_leaf 20, 100 in place of the inert max_iter (fixed at 400); learning_rate and l2_regularization as above | 31 leaves sat at the grid edge, and leaf size was not searched | 63 leaves, min_samples_leaf 100, l2 1 (0.840 ± 0.003; training 0.886) | 0.838 (selected settings) |
| SVM, RBF kernel (grid, 4; 0.795 ± 0.014; 0.858) | C: penalty on margin violations | 1, 10 | the default and a weaker penalty; each fit took about three minutes | 1 | 0.786 (10) |
| | gamma: RBF kernel width | scale, 0.01 | the default heuristic and one fixed value | 0.01 | 0.791 (scale) |
| | Fixed settings | Platt scaling over five inner folds; 9,913-row subsample of whole mine-years | probabilities are needed for thresholds; training time is quadratic in rows | | |
| MLP (grid, 9; 0.833 ± 0.004; 0.870) | hidden_layer_sizes: units per hidden layer | (64), (128), (128, 64) | one or two small layers for about 65,000 rows per fold | (128) | 0.832 (64) |
| | alpha: L2 weight penalty | 1e-4, 1e-2, 1e-1 | log-spaced from the default upwards | 0.01 | 0.832 (0.1) |
| | Fixed settings | ReLU units; Adam with initial learning rate 0.001; batches of 200; early stopping on a 10% split; at most 200 epochs | standard settings, with early stopping against overfitting | | |

{fig:validation} plots training and validation PR-AUC against each search's most varied hyperparameter. The decision tree underfits at depth 4 with equal scores, overfits at unlimited depth with a wide gap and peaks at depth 12 with large leaves. The random forest's deep trees open a wider gap still, yet its validation score rises with depth, because averaging cancels much of each tree's variance. The SVM overfits as C grows and has the largest fold-to-fold spread, a cost of its small subsample.

![Validation curves of the eight tuned models, giving mean training-fold and validation-fold PR-AUC (average precision) with one standard deviation bands over the five grouped folds of the training split against the hyperparameter with most searched values, the other hyperparameters held at their selected values (for the random forest, the best sampled candidate per value), with the selected value dashed; curves are drawn because the train-validation gap and its shape show over- and underfitting. Source: author's analysis of MSHA open data.](reports/figures/F14_validation_curves.png){#fig:validation width=5.5}

Logistic regression is flat from C = 0.1 upwards, with training and validation scores close together, so the model has high bias. KNN scores 1.000 on its training folds under distance weighting, because each training row is its own nearest neighbour. Its validation score was still rising at the grid edge of K = 51, yet K = 101 and 151, tried after testing, scored lower. Gaussian naive Bayes peaks sharply in var_smoothing without a gap; one smoothing constant decides its score. HGB and the MLP have flat curves and moderate gaps, so their model class matters more than the selected values.

The threshold was tuned as a further hyperparameter on out-of-fold probabilities, never on test rows, whose labels would then make the reported recall optimistic. The capacity rule takes the lowest threshold whose out-of-fold specificity reaches the target, a second rule flags half the cases, and a cost rule minimises out-of-fold expected cost at each cost ratio ({fig:cost}). For HGB the capacity threshold of 0.210 gave a test specificity of 0.403 against the 0.40 target, so the review workload planned on training rows held on unseen mine-years.

The grids stayed narrow for compute reasons ({tab:decision}). Searches took from about a minute for logistic regression to twenty minutes for four SVM candidates, and the temporal split repeated all of them. Wider grids for KNN, the forest and HGB, searched after the test scoring and not used for selection, raised no validation PR-AUC by more than 0.002 ({tab:grids}), within one fold standard deviation. Under the one-standard-error rule, which picks the simplest candidate within one standard error of the best mean (Hastie et al., 2009), logistic regression would take a tenfold stronger penalty, the MLP its smaller layer and HGB the faster learning rate with fewer rounds. I kept the highest mean, since each alternative lies within fold noise and none changes the ranking of algorithms.

## 4.4 Evaluation metrics

A true positive (TP) is a lost-time case flagged at filing, whose reserve and return-to-work plan then start early. A false negative (FN) is a lost-time case left in routine handling, the miss priced at USD 4,000 in {tab:costs}. A false positive (FP) sends a restricted-duty or no-lost-time case to an unnecessary review costing about USD 200, and a true negative (TN) is such a case left alone. Accuracy, the share of correct cells, was reported but not used, because it prices both errors alike and the majority class reaches 0.514 while flagging nothing ({tab:test}).

Recall, the true positive rate (TPR), is TP / (TP + FN), the share of lost-time cases flagged and the quantity the insurer most needs. Precision, TP / (TP + FP), is the share of flags that are lost-time and sets the review effort per catch. Specificity, the true negative rate (TNR), is TN / (TN + FP), the share of other cases left alone, which fixes the workload. F1, the harmonic mean of precision and recall, weights both errors equally, so it was reported but not optimised; the expected cost states the asymmetry that an F-beta score would only approximate. Balanced accuracy averages TPR and TNR, and the Matthews correlation coefficient (MCC) is zero for any constant prediction, so both expose a model that flags everything.

ROC-AUC is the probability that a random lost-time case scores above a random other case and does not depend on prevalence (Fawcett, 2006). PR-AUC averages precision over all recall levels, and its no-skill level equals the prevalence of 0.486. A model dominates in precision-recall space exactly when it dominates in ROC space (Davis and Goadrich, 2006). At this prevalence the imbalance argument of Saito and Rehmsmeier (2015) does not apply, and the two measures rank the models alike. PR-AUC is primary because precision is the review yield along the adjuster's queue, while S3 measures recall at the planned capacity.

Because the application shows each probability and the cost rule takes it at face value, calibration was checked. The Brier score, the mean squared gap between probability and outcome, is 0.250 for a constant prediction at the prevalence, and {fig:calibration} compares predicted with observed rates. Gaussian naive Bayes places many lost-time cases in its lowest bin, the overconfidence expected in Section 2.4. HGB follows the diagonal. Expected cost per 1,000 cases is 1,000 x (FN x C_FN + FP x C_FP) / N, with C_FP from {tab:costs} and C_FN set by the cost ratio ({fig:cost}). Reviewing everything costs the same at any ratio, and at high ratios the cheapest threshold flags nearly every case at about that cost, so cost separates models only at the low ratios that S4 tests.

![Recall, specificity, precision, F1 and flag rate (top) and expected cost per 1,000 cases at six cost ratios on a log scale (bottom) against the decision threshold of HGB on the 20,320 test reports, with the three out-of-fold thresholds dashed and the cost of reviewing everything dotted; one threshold axis lets each rate and cost be read at the same cut-off. Source: author's analysis of MSHA open data.](reports/figures/F18_threshold_cost_curve_hist_gb.png){#fig:cost width=6.3}

![Calibration curves of the eight tuned models and the lookup baseline on the 20,320 test reports, plotting the observed lost-time rate against the mean predicted probability in ten equal-width bins with marker size proportional to the cases in each bin and Brier scores in the legend; a reliability diagram is used because the application displays these probabilities. Source: author's analysis of MSHA open data.](reports/figures/F19_calibration.png){#fig:calibration width=5.0}

Cross-validated scores are fold means and standard deviations on training rows and served selection only; the test set was scored once with every choice fixed, and the two sit in separate tables. No metric here captures fairness between sectors, whose label partly records employer practice, or drift after the training years, so results are also reported by sector and on a temporal split ({tab:robustness}).

## 4.5 Comparison and final model

HGB leads both the cross-validated scores ({tab:cv}) and the test scores at the default threshold of 0.5 ({tab:test}). The MLP is second, and logistic regression nearly ties the random forest. The curves of the top six models lie close together, while KNN and naive Bayes fall below the lookup baseline ({fig:roc}; {fig:pr}). At each capacity threshold, recall is close among the top six, and the MLP's highest recall comes with the lowest specificity ({tab:operating}).

Table: Cross-validated scores (mean ± SD over five grouped folds of the 81,289-row training split) of each model's selected candidate at the default threshold of 0.5, ordered by PR-AUC, the refit metric; KNN and the SVM were cross-validated on their 39,999-row and 9,913-row subsamples, and TNR is pooled over the out-of-fold predictions of all training rows, so it has no SD. Source: author's analysis of MSHA open data. {#tab:cv}

| Model | PR-AUC | ROC-AUC | F1 | Recall (TPR) | TNR (pooled) | Precision | Accuracy | Brier |
|---|---|---|---|---|---|---|---|---|
| HGB | 0.838 ± 0.003 | 0.839 ± 0.002 | 0.743 ± 0.002 | 0.723 ± 0.007 | 0.789 | 0.764 ± 0.005 | 0.757 ± 0.002 | 0.163 ± 0.001 |
| MLP | 0.833 ± 0.004 | 0.835 ± 0.004 | 0.740 ± 0.007 | 0.719 ± 0.019 | 0.788 | 0.763 ± 0.007 | 0.755 ± 0.002 | 0.166 ± 0.002 |
| Random forest | 0.821 ± 0.003 | 0.828 ± 0.003 | 0.732 ± 0.004 | 0.704 ± 0.009 | 0.793 | 0.762 ± 0.003 | 0.750 ± 0.002 | 0.173 ± 0.001 |
| Logistic regression | 0.820 ± 0.003 | 0.825 ± 0.002 | 0.739 ± 0.002 | 0.740 ± 0.007 | 0.751 | 0.738 ± 0.005 | 0.746 ± 0.002 | 0.171 ± 0.001 |
| Decision tree | 0.803 ± 0.004 | 0.812 ± 0.004 | 0.724 ± 0.015 | 0.715 ± 0.046 | 0.753 | 0.735 ± 0.021 | 0.735 ± 0.004 | 0.176 ± 0.002 |
| SVM, RBF kernel | 0.795 ± 0.014 | 0.807 ± 0.011 | 0.722 ± 0.017 | 0.722 ± 0.020 | 0.752 | 0.723 ± 0.016 | 0.731 ± 0.015 | 0.179 ± 0.005 |
| KNN | 0.764 ± 0.009 | 0.766 ± 0.008 | 0.648 ± 0.013 | 0.585 ± 0.017 | 0.793 | 0.725 ± 0.014 | 0.691 ± 0.010 | 0.197 ± 0.003 |
| Gaussian naive Bayes | 0.734 ± 0.006 | 0.735 ± 0.002 | 0.601 ± 0.003 | 0.529 ± 0.006 | 0.781 | 0.696 ± 0.006 | 0.659 ± 0.003 | 0.305 ± 0.003 |

Table: Test-set scores of the eight tuned models and four baselines on the 20,320 held-out reports at the default threshold of 0.5, in the brief's column order with F1, PR-AUC and accuracy added and ordered by PR-AUC; TPR equals recall, TP / (TP + FN), and both columns are kept because the brief lists both, while TNR is specificity, TN / (TN + FP). Source: author's analysis of MSHA open data. {#tab:test}

| Model | TPR | TNR | AUC | Recall | Precision | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|---|---|---|
| HGB | 0.738 | 0.785 | 0.843 | 0.738 | 0.764 | 0.751 | 0.841 | 0.762 |
| MLP | 0.749 | 0.769 | 0.839 | 0.749 | 0.754 | 0.752 | 0.837 | 0.759 |
| Random forest | 0.707 | 0.799 | 0.833 | 0.707 | 0.769 | 0.737 | 0.827 | 0.755 |
| Logistic regression | 0.744 | 0.758 | 0.829 | 0.744 | 0.744 | 0.744 | 0.825 | 0.751 |
| SVM, RBF kernel | 0.731 | 0.763 | 0.820 | 0.731 | 0.744 | 0.737 | 0.811 | 0.747 |
| Decision tree | 0.676 | 0.799 | 0.817 | 0.676 | 0.761 | 0.716 | 0.805 | 0.739 |
| Lookup baseline (nature x body part x sector) | 0.655 | 0.801 | 0.804 | 0.655 | 0.756 | 0.702 | 0.790 | 0.730 |
| KNN | 0.589 | 0.807 | 0.773 | 0.589 | 0.743 | 0.657 | 0.770 | 0.701 |
| Gaussian naive Bayes | 0.524 | 0.802 | 0.741 | 0.524 | 0.715 | 0.605 | 0.741 | 0.667 |
| Baseline: hand rule (not a cut, dust in the eye or a scratch) | 0.898 | 0.423 | 0.660 | 0.898 | 0.595 | 0.716 | 0.584 | 0.654 |
| Baseline: majority class | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.486 | 0.514 |
| Baseline: review everything | 1.000 | 0.000 | 0.500 | 1.000 | 0.486 | 0.654 | 0.486 | 0.486 |

![ROC curves of the eight tuned models and the lookup baseline on the 20,320 test reports, with test ROC-AUC in the legend and the chance diagonal dotted; ROC space is shown because it does not depend on prevalence. Source: author's analysis of MSHA open data.](reports/figures/F12_roc_curves.png){#fig:roc width=5.0}

![Precision-recall curves of the eight tuned models and the lookup baseline on the 20,320 test reports, with average precision (PR-AUC) in the legend and the prevalence of 0.486 dotted as the no-skill level; precision-recall space is shown because precision along the curve is the review cost of each lost-time case found. Source: author's analysis of MSHA open data.](reports/figures/F13_pr_curves.png){#fig:pr width=5.0}

Table: Test-set scores on the 20,320 held-out reports at each model's capacity threshold, the lowest threshold whose out-of-fold specificity reached 0.40 on the training split, with the resulting flag rate, balanced accuracy, MCC and Brier score; the threshold-free AUC and PR-AUC are in {tab:test}, and the two constant baselines are omitted because no threshold changes a constant score. Source: author's analysis of MSHA open data. {#tab:operating}

| Model | Threshold | Flag rate | TPR (recall) | TNR | Precision | F1 | Balanced accuracy | MCC | Brier |
|---|---|---|---|---|---|---|---|---|---|
| HGB | 0.210 | 0.765 | 0.943 | 0.403 | 0.599 | 0.733 | 0.673 | 0.409 | 0.161 |
| MLP | 0.187 | 0.783 | 0.948 | 0.373 | 0.588 | 0.726 | 0.660 | 0.389 | 0.163 |
| Random forest | 0.290 | 0.759 | 0.935 | 0.408 | 0.599 | 0.730 | 0.672 | 0.401 | 0.170 |
| Logistic regression | 0.222 | 0.760 | 0.934 | 0.405 | 0.597 | 0.728 | 0.669 | 0.396 | 0.169 |
| SVM, RBF kernel | 0.237 | 0.753 | 0.927 | 0.411 | 0.598 | 0.727 | 0.669 | 0.391 | 0.173 |
| Decision tree | 0.239 | 0.774 | 0.932 | 0.376 | 0.585 | 0.719 | 0.654 | 0.368 | 0.174 |
| Lookup baseline (nature x body part x sector) | 0.225 | 0.753 | 0.916 | 0.402 | 0.591 | 0.719 | 0.659 | 0.368 | 0.179 |
| KNN | 0.308 | 0.733 | 0.886 | 0.412 | 0.588 | 0.707 | 0.649 | 0.337 | 0.195 |
| Gaussian naive Bayes | 0.001 | 0.727 | 0.868 | 0.407 | 0.580 | 0.696 | 0.637 | 0.308 | 0.298 |
| Baseline: hand rule (not a cut, dust in the eye or a scratch) | not applicable (fixed rule) | 0.733 | 0.898 | 0.423 | 0.595 | 0.716 | 0.660 | 0.363 | 0.346 |

{tab:stats} tests whether these differences exceed noise (Dietterich, 1998; Demšar, 2006). McNemar's test rejects equal error rates between HGB and the lookup baseline at both thresholds, also when clustered by mine-year (Durkalski et al., 2003). Between the top two, the paired fold test gives p = 0.044 after the Nadeau and Bengio (2003) correction for overlapping training folds, while the Wilcoxon test cannot fall below 0.0625 with five folds. The MLP's test PR-AUC lies inside HGB's mine-year bootstrap interval, so the gap between the top two sits at the edge of noise and the gap to the lookup baseline does not.

Table: Statistical comparisons of HGB, namely McNemar tests against the lookup baseline on the 20,320 test reports at the default and capacity thresholds, row-wise and clustered by the 6,724 test mine-years, paired tests against the MLP on the five cross-validation folds of PR-AUC, and 95% percentile bootstrap intervals (1,000 resamples) of test metrics by rows and by mine-years. Source: author's analysis of MSHA open data. {#tab:stats}

| Comparison | Test | Result | p-value |
|---|---|---|---|
| HGB against lookup baseline, threshold 0.5 | McNemar, rows treated as independent | chi-square 131.7; HGB alone correct on 1,929 reports, lookup baseline alone on 1,278 | below 0.001 |
| HGB against lookup baseline, threshold 0.5 | McNemar clustered by mine-year (Durkalski et al., 2003) | chi-square 78.5 | below 0.001 |
| HGB against lookup baseline, capacity thresholds (0.210 and 0.225) | McNemar, rows treated as independent | chi-square 35.2; 1,289 against 1,004 reports | below 0.001 |
| HGB against lookup baseline, capacity thresholds | McNemar clustered by mine-year (Durkalski et al., 2003) | chi-square 24.4 | below 0.001 |
| HGB against MLP, cross-validated PR-AUC | paired t-test | mean difference 0.0054 (SD 0.0028); HGB higher in all five folds; t = 4.36 | 0.012 |
| HGB against MLP, cross-validated PR-AUC | corrected resampled t-test (Nadeau and Bengio, 2003) | t = 2.90 | 0.044 |
| HGB against MLP, cross-validated PR-AUC | Wilcoxon signed-rank | W = 0 | 0.0625, the smallest value possible with five folds |
| HGB, test PR-AUC 0.841 | bootstrap of rows; bootstrap of mine-years | 0.835 to 0.848; 0.825 to 0.856 | |
| HGB, test ROC-AUC 0.843 | bootstrap of rows; bootstrap of mine-years | 0.838 to 0.848; 0.835 to 0.851 | |
| HGB, recall 0.943 at the capacity threshold | bootstrap of rows; bootstrap of mine-years | 0.939 to 0.947; 0.938 to 0.949 | |

I selected HGB by the rule fixed before any test score was computed, the highest cross-validated PR-AUC, where its lead over the MLP exceeds one fold standard deviation ({tab:cv}). {tab:decision} reports test-set evidence that checks this choice but did not make it. HGB has the lowest expected cost at the three low cost ratios, and its thresholds landed nearer their targets, where the MLP over-flagged. At an equal workload both catch the same share of lost-time cases. On the temporal split HGB kept the higher PR-AUC, and its specificity barely moved when the 103(i) status or the immediate-reporting flag was removed, unlike the MLP's.

Its permutation importance puts nature of injury and body part first ({fig:importance}), as do most of the largest logistic coefficients, and two of its four largest mean SHAP values ({tab:shap}; Lundberg and Lee, 2017), for cuts and finger injuries, match negative coefficients ({fig:coefficients}). The final pipeline uses the settings of {tab:grids}, refitted on all training rows, at its out-of-fold capacity threshold ({tab:confusion}; {fig:confusion}). There it flags about three test reports in four and misses 561 of 9,873 lost-time cases, against 830 misses for the lookup baseline at about the same specificity.

Table: Multi-criteria comparison of the four strongest tuned models and the SVM, compiled after HGB was selected by cross-validated PR-AUC: test-set cost at the out-of-fold cost-optimal thresholds, transfer of out-of-fold threshold targets to the test set, recall at one equal workload, the temporal split (top two only), explanations, run times on eight workers and the size of the saved pipeline. Source: author's analysis of MSHA open data. {#tab:decision}

| Criterion | HGB | MLP | Random forest | Logistic regression | SVM, RBF kernel |
|---|---|---|---|---|---|
| Cross-validated PR-AUC (rank of eight) | 0.838 (1) | 0.833 (2) | 0.821 (3) | 0.820 (4) | 0.795 (6) |
| Test PR-AUC | 0.841 | 0.837 | 0.827 | 0.825 | 0.811 |
| Test cost per 1,000 cases at 2:1 / 3:1 / 5:1 (USD) | 65,896 / 77,697 / 87,638 | 67,274 / 78,031 / 89,085 | 67,648 / 78,839 / 90,463 | 68,760 / 80,079 / 90,945 | 70,955 / 82,234 / 93,317 |
| Test specificity at the out-of-fold threshold for specificity 0.40 | 0.403 | 0.373 | 0.408 | 0.405 | 0.411 |
| Test flag rate at the out-of-fold threshold for a 50% flag rate | 0.506 | 0.525 | 0.497 | 0.498 | 0.498 |
| Test recall at one equal workload, the top 75.9% of test scores (the median flag rate of the eight capacity thresholds) | 0.940 | 0.940 | 0.935 | 0.933 | 0.930 |
| Temporal split: PR-AUC on 2020-2025 | 0.843 | 0.834 | not run | not run | not run |
| Temporal split: specificity at the capacity threshold across three feature variants | 0.376 to 0.378 | 0.313 to 0.425 | not run | not run | not run |
| Explanation available | permutation importance; SHAP values | permutation importance | permutation importance | coefficients as odds ratios | permutation importance |
| Scaler needed | no | yes | no | yes | yes |
| Tuning time (s) | 580 | 188 | 140 | 71 | 1,193 |
| Prediction time for 20,320 reports (s) | 0.27 | 0.12 | 0.64 | 0.09 | 45.36 |
| Saved pipeline (MB) | 0.45 | 0.81 | 20.37 | 0.01 | 1.25 |

![Permutation importance of the 20 most important input features of HGB, measured as the mean drop in test PR-AUC (average precision) over five shuffles on 5,000 test reports with one standard deviation whiskers; importance is computed on input features rather than one-hot columns so that each nominal code counts once. Source: author's analysis of MSHA open data.](reports/figures/F17_feature_importance_hist_gb.png){#fig:importance width=5.5}

![The 20 largest coefficients of the tuned logistic regression fitted on the training split, with numeric inputs standardised and one-hot columns coded 0 or 1 and labelled by their codebook meaning, where positive values (blue) raise and negative values (orange) lower the odds of a lost-time case; signed bars are used because each coefficient converts to an odds ratio. Source: author's analysis of MSHA open data.](reports/figures/F17b_coefficients_logreg.png){#fig:coefficients width=5.5}

Table: Confusion counts on the 20,320 test reports for HGB at its capacity threshold and at 0.5, and for the lookup baseline at its capacity threshold, where a true positive is a lost-time case flagged for early review and a false negative a lost-time case left in routine handling. Source: author's analysis of MSHA open data. {#tab:confusion}

| Model and threshold | TP: lost-time, flagged | FN: lost-time, not flagged | FP: other, flagged | TN: other, not flagged | Recall | Precision | Flag rate |
|---|---|---|---|---|---|---|---|
| HGB, capacity threshold 0.210 | 9,312 | 561 | 6,232 | 4,215 | 0.943 | 0.599 | 0.765 |
| HGB, threshold 0.5 | 7,284 | 2,589 | 2,249 | 8,198 | 0.738 | 0.764 | 0.469 |
| Lookup baseline, capacity threshold 0.225 | 9,043 | 830 | 6,248 | 4,199 | 0.916 | 0.591 | 0.753 |

![Confusion matrices on the 20,320 test reports for HGB (top) and the lookup baseline (bottom) at the default threshold of 0.5 (left) and at each one's out-of-fold capacity threshold (right), with counts and row percentages; matrices are shown because each cell is a distinct outcome for the adjuster. Source: author's analysis of MSHA open data.](reports/figures/F16_confusion_matrices.png){#fig:confusion width=6.0}

Against the success criteria ({tab:criteria_results}), HGB passes S1 to S4 and S6, and {tab:stats} holds the comparisons that S7 requires. It misses S5 at the out-of-fold threshold for a 50% flag rate (0.775 against 0.78), and the miss recurs on the temporal and mine-grouped splits. The MLP passes S1 to S6 only by flagging more than half the reports; among the top half of test scores, the planned form of S5, neither model reaches the target. I kept the pre-fixed rule, since switching after the test results would turn the test score into a selection score. The lookup baseline also passes S1, S2 and S6, so S3 to S5 separate the learned models from it. At half capacity more lost-time cases would go unflagged than planned, so I recommend the capacity threshold, where S3 passes.

Table: Success criteria S1 to S7, fixed before modelling, with S7 a reporting criterion, evaluated on the 20,320 test reports for HGB, the MLP and the lookup baseline with thresholds fixed on out-of-fold probabilities, with S5 repeated on the temporal and mine-grouped splits. Source: author's analysis of MSHA open data. {#tab:criteria_results}

| Criterion | Target | HGB | Result | MLP | Lookup baseline |
|---|---|---|---|---|---|
| S1 ROC-AUC | 0.80 or more | 0.843 | pass | 0.839 | 0.804 |
| S2 PR-AUC | 0.75 or more | 0.841 | pass | 0.837 | 0.790 |
| S3 recall at the out-of-fold threshold for specificity 0.40 | 0.92 or more | 0.943 (specificity 0.403) | pass | 0.948 (0.373) | 0.916 (0.402) |
| S4 cost per 1,000 cases at the out-of-fold cost-optimal threshold, 2:1 / 3:1 / 5:1 (USD) | below the review-everything floor (102,825 on the test set) and below the lookup baseline | 65,896 / 77,697 / 87,638 | pass | 67,274 / 78,031 / 89,085 | 73,553 / 85,472 / 98,061 |
| S5 recall at the out-of-fold threshold for a 50% flag rate | 0.78 or more | 0.775 (flag rate 0.506) | fail | 0.790 (0.525) | 0.753 (0.520) |
| S5 as planned: recall among the 50% of test reports with the highest scores (cut-off from the test scores, labels only count hits) | 0.78 or more | 0.769 (flag rate 0.500) | fail | 0.768 (0.500) | 0.725 (0.490, tied scores) |
| S5 on the temporal split (2020-2025) and the mine-grouped split | 0.78 or more | 0.749 and 0.762 | fail | 0.724 (temporal) | 0.710 (mine-grouped) |
| S6 Brier score on the test set, and by sector on the temporal split | 0.20 or less | test 0.161 (coal 0.135, metal and non-metal 0.179); temporal split: coal 0.125, metal and non-metal 0.187 | pass | test 0.163 (0.135 and 0.182); temporal split: 0.131 and 0.193 | test 0.179 (0.150 and 0.199) |
| S7 statistical comparison | McNemar test against the lookup baseline and a paired fold test between the top two | clustered McNemar p below 0.001; corrected paired t p = 0.044 ({tab:stats}) | reported | | |

No fatality or permanent disability was missed ({tab:errors}). Most misses are cuts and lacerations, above all to fingers, thumbs and hands, where half or more of the lost-time cases score below the threshold; such injuries are usually minor, so the few that keep a worker away resemble the many that do not. Metal and non-metal mines hold most misses and have the lower precision ({tab:modelcard}), consistent with their lower lost-time rate. Most false positives are restricted-duty cases, which differ from lost-time ones mainly in whether restricted work was offered.

Table: Errors of HGB at its capacity threshold of 0.210 on the 20,320 test reports, giving missed lost-time cases by sector, by degree-of-injury code and for the six nature x body part cells with most misses, and flagged non-lost-time cases by degree code. Source: author's analysis of MSHA open data. {#tab:errors}

| Group | Cases | Misclassified | Share |
|---|---|---|---|
| All lost-time cases | 9,873 | 561 missed | 0.057 |
| Lost-time cases, coal | 5,280 | 152 missed | 0.029 |
| Lost-time cases, metal and non-metal | 4,593 | 409 missed | 0.089 |
| Fatality (code 01) | 142 | 0 missed | 0.000 |
| Permanent disability (02) | 224 | 0 missed | 0.000 |
| Days away only (03) | 7,380 | 376 missed | 0.051 |
| Days away and restricted activity (04) | 2,127 | 185 missed | 0.087 |
| Cut, laceration or puncture x finger or thumb | 324 | 170 missed | 0.525 |
| Cut, laceration or puncture x hand | 104 | 65 missed | 0.625 |
| Fracture x finger or thumb | 392 | 51 missed | 0.130 |
| Fracture x mouth, lips, teeth or throat | 19 | 17 missed | 0.895 |
| Dust in the eyes x eye | 36 | 16 missed | 0.444 |
| Burn or chemical injury x eye | 46 | 16 missed | 0.348 |
| Non-lost-time: restricted activity only (05) | 4,212 | 3,297 flagged | 0.783 |
| Non-lost-time: no days away and no restriction (06) | 6,235 | 2,935 flagged | 0.471 |

{tab:modelcard} is the model card of HGB (Mitchell et al., 2019). Its monitoring entries follow Sculley et al. (2015), who show that deployed models accrue hidden debt through shifting inputs and feedback loops. A feedback loop is plausible here, since return-to-work plans prompted by flags could turn lost-time cases into restricted-duty ones and change the label the model learns from.

Table: Model card of the HGB pipeline, after Mitchell et al. (2019), with metrics on the 20,320 test reports at the capacity threshold unless stated otherwise. Source: author's analysis of MSHA open data. {#tab:modelcard}

| Field | Content |
|---|---|
| Model | HistGradientBoostingClassifier from scikit-learn 1.9.1 inside the preprocessing pipeline of {fig:pipeline}; learning rate 0.05, at most 400 rounds with early stopping, 31 leaf nodes, no L2 penalty; refitted on the 81,289 training rows on 14 October 2026; saved pipeline 0.45 MB |
| Intended use | ranking new Form 7000-1 reports at filing for early review by workers'-compensation adjusters at insurers and self-insured operators, flagging reports that score 0.210 or more (about three in four); an adjuster confirms every flag |
| Out-of-scope uses | ranking mines for inspection, which the DART label suits better; setting benefit levels or denying claims; judging individual workers; reports outside 2010-2025 or outside degree codes 01 to 06 |
| Data | 101,609 MSHA reports of 2010-2025 (Section 1.3), split by 32,732 mine-years with 6,724 of them in the test set; secondary temporal split (2010-2019 against 2020-2025) and mine-grouped split |
| Metrics | PR-AUC (primary), ROC-AUC, recall, specificity, precision, Brier score and expected cost per 1,000 cases, with thresholds from out-of-fold probabilities |
| Overall performance | PR-AUC 0.841 (mine-year bootstrap 95% interval 0.825 to 0.856), ROC-AUC 0.843 and Brier score 0.161; recall 0.943, specificity 0.403 and precision 0.599 at threshold 0.210 |
| Subgroup performance | coal (8,270 reports, prevalence 0.638): recall 0.971, specificity 0.397, precision 0.740, PR-AUC 0.912, Brier 0.135; metal and non-metal (12,050 reports, prevalence 0.381): recall 0.911, specificity 0.406, precision 0.486, PR-AUC 0.708, Brier 0.179 |
| Stability checks | temporal split: PR-AUC 0.843 and recall 0.944 at a realised specificity of 0.377; mine-grouped split: PR-AUC 0.836 and recall 0.936; removing immed_reportable or the 103(i) status, counting only resolved earlier cases or blanking early-January exposure changed test PR-AUC by at most 0.001; without the 175 amputation reports, all lost-time, test PR-AUC is 0.834 |
| Limitations | misses S5 (recall 0.775 at a 50% flag rate); the label records whether restricted duty was offered; precision below 0.5 in metal and non-metal mines; self-reported data omit unreported injuries; scores describe association, not cause |
| Ethical considerations | no names or narratives are used, but occupation, experience and state can stand in for groups of workers; a flag must never reduce benefits; the sector gap is disclosed; the application states the intended use and leaves the decision to a person |
| Monitoring | track the flag rate against the planned three in four, and PR-AUC and Brier score by sector on newly closed cases; retrain or recalibrate when PR-AUC falls below the S2 target of 0.75, the Brier score exceeds the S6 limit of 0.20 or the share of pooled infrequent levels grows (Sculley et al., 2015) |
