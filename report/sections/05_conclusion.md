# 5 Conclusion

## 5.1 Key data analysis findings

1. Lost-time cases are 48.6% of the 101,609 in-scope reports ({fig:target}), so a rule that finds none is right about half the time and the insurer needs a ranked review queue.
2. Nature of injury and body part lead the association with lost time (Cramer's V of 0.391 and 0.364, {tab:associations}), with cuts near 18% lost-time ({fig:cat}). The fields the safety officer completes on the form carry most of the signal at filing.
3. The mine's previous-year lost-time share is the strongest numeric signal (point-biserial r = 0.292, {tab:associations}; {fig:box}). Mine history known before the accident shifts each score, though it partly reflects the restricted-duty practice behind the label.
4. Coal reports are lost-time in 63.0% of cases against 38.6% in metal and non-metal mines, while the DART label hardly varies by sector ({tab:label}), so the label records employers' restricted-duty practice as well as severity, and scores must be read within sector.
5. Gaps are mostly structural, but reports without total experience are lost-time in 59.3% of training cases against 48.5% of recorded ones ({fig:missing}; {tab:decisionlog}). Missing-value indicators keep that higher risk visible to the reserve decision.
6. The classes overlap in every two-dimensional view ({fig:pair}; {fig:pca}), and the one-field hand rule reaches a specificity of 0.429 ({tab:criteria}), so sparing needless reviews without losing lost-time cases needs a model that combines many fields.

## 5.2 Interpretation of results

Per 1,000 filed cases at the capacity threshold, HGB misses 27.6 lost-time cases against 49.5 for the one-field hand rule and flags 32 more ({tab:per1000}). Its cost-optimal thresholds undercut reviewing everything at low cost ratios. At the base ratio of {tab:costs} the cheapest threshold flags almost every case ({fig:cost}), leaving the score to order the queue when capacity binds.

Table: Outcomes per 1,000 filed cases on the 20,320-row held-out test set for HGB at three thresholds fixed on out-of-fold probabilities and at two other review capacities, for the lookup baseline at its capacity threshold, for the one-field hand rule built for this study and for reviewing everything, with the expected cost at the 2:1 and 5:1 cost ratios at USD 200 per review; counts are test confusion counts scaled to 1,000 cases, the two capacity rows take their cut-off from the test scores (labels only count the hits), and reviewing everything costs USD 102,825 per 1,000 on the test set against USD 102,829 on all in-scope reports. Source: author's analysis of MSHA open data. {#tab:per1000}

| Policy | Threshold | Flagged | Lost-time cases caught | Lost-time cases missed | False alarms | Expected cost at 2:1 (USD) | Expected cost at 5:1 (USD) |
|---|---|---|---|---|---|---|---|
| HGB, capacity threshold (out-of-fold specificity 0.40) | 0.210 | 765.0 | 458.3 | 27.6 | 306.7 | 72,382 | 88,947 |
| HGB, cost-optimal at 2:1 | 0.335 | 621.9 | 421.4 | 64.5 | 200.5 | 65,896 | 104,577 |
| HGB, cost-optimal at 5:1 | 0.165 | 821.0 | 468.7 | 17.2 | 352.3 | 77,333 | 87,638 |
| HGB, half of the filings flagged (top 50% of test scores) | 0.463 | 500.0 | 373.7 | 112.2 | 126.3 | 70,148 | 137,470 |
| HGB, five in six filings flagged (top 83.3% of test scores) | 0.157 | 833.3 | 470.4 | 15.5 | 362.9 | 78,760 | 88,031 |
| Lookup baseline, capacity threshold | 0.225 | 752.5 | 445.0 | 40.8 | 307.5 | 77,835 | 102,343 |
| Hand rule built for this study: nature is not a cut, dust in the eye or a scratch | not applicable | 733.2 | 436.4 | 49.5 | 296.8 | 79,144 | 108,819 |
| Review everything | not applicable | 1,000.0 | 485.9 | 0.0 | 514.1 | 102,825 | 102,825 |

Nature, body part and sector lead the permutation importance ({fig:importance}), and their joint cells explain about half of HGB's lead over logistic regression. A one-hot block for the joint cell raises the cross-validated PR-AUC of logistic regression from 0.820 to 0.829, against 0.838 for HGB ({tab:robustness}). Logistic regression assumes additive log-odds, so it forms those cells only when they are supplied. Trees also rank skewed inputs, follow curved size effects and isolate each structural "none" level, which fits the rest of the gap and the lead of tree ensembles on tabular benchmarks (Grinsztajn et al., 2022).

The estimates hold under resampling and harder splits ({tab:robustness}). The fold standard deviation of PR-AUC is 0.003, and the clustered bootstrap interval of test PR-AUC clears the S2 target. HGB's lead over the MLP is thin, since the Wilcoxon test does not confirm the corrected fold test. Unseen mines and 2020-2025 reports move PR-AUC by under 0.01, and in 2020-2025 recall holds in both sectors while specificity falls below plan. S5 is missed by 0.005, so a team reviewing half the filings catches fewer cases than planned.

Table: Uncertainty, generalisability and mechanism checks of HGB, covering cross-validation on the training split, 95% percentile bootstrap intervals and paired tests on the held-out test set, results by sector, a temporal split, a mine-grouped split, the reports still open at filing and logistic regression given the lookup baseline's joint cell; the capacity threshold is the out-of-fold threshold for specificity 0.40, and the last two checks were run after the test scoring and changed no choice. Source: author's analysis of MSHA open data. {#tab:robustness}

| Check | Data and protocol | Result for HGB |
|---|---|---|
| Cross-validation | 5-fold mine-year-grouped CV on the 81,289 training rows | PR-AUC 0.838 (SD 0.003, folds 0.834 to 0.841); ROC-AUC 0.839 (SD 0.002) |
| Bootstrap intervals | 1,000 resamples of the 20,320 test rows and of their 6,724 mine-years | PR-AUC 0.841 (rows 0.835 to 0.848; mine-years 0.825 to 0.856); ROC-AUC 0.843 (mine-years 0.835 to 0.851); recall at the capacity threshold 0.943 (mine-years 0.938 to 0.949) |
| Paired fold test against the MLP | PR-AUC on the same five CV folds | mean difference 0.005; corrected t-test p = 0.044; Wilcoxon signed-rank p = 0.0625 |
| McNemar test against the lookup baseline | test set at the capacity threshold, row-wise and clustered by mine-year | p < 0.001 in both forms (clustered chi-square 24.4, 1 df) |
| By sector | test set at the capacity threshold 0.210 | coal (n = 8,270) recall 0.971, specificity 0.397, precision 0.740; metal and non-metal (n = 12,050) recall 0.911, specificity 0.406, precision 0.486 |
| Temporal split | re-tuned on 2010-2019 (71,877 rows) with thresholds from its out-of-fold probabilities, scored on 2020-2025 (29,732 rows) | ROC-AUC 0.834, PR-AUC 0.843; recall 0.944 and specificity 0.377 at the capacity threshold; coal recall 0.973 and Brier 0.125; metal and non-metal recall 0.918 and Brier 0.187 |
| Mine-grouped split | no mine on both sides (8,303 mines; 20,289 test rows), primary hyperparameters | ROC-AUC 0.838, PR-AUC 0.836, recall 0.936 at the capacity threshold, recall 0.762 at a 50% flag rate |
| Feature variants | primary split with fixed hyperparameters: immed_reportable set to 0; status_103i_at_accident constant; 175 amputation test rows (nature code 100) removed | PR-AUC 0.840, 0.841 and 0.834 against 0.841 |
| S5, recall at a 50% flag rate | out-of-fold threshold 0.456, test flag rate 50.6%; and the top 50% of test scores, the planned form | 0.775 against the 0.78 target (missed); MLP 0.790 at a 52.5% flag rate; top half of test scores: HGB 0.769, MLP 0.768 |
| Reports still open at filing | 45,679 in-scope reports (45.0%) were filed once the worker was back at work, lost-time in 20.4% against 71.6% of the rest; test reports with a return-to-work date after the filing date or none (11,290, prevalence 0.712), thresholds unchanged | ROC-AUC 0.812, PR-AUC 0.920; recall 0.950 and specificity 0.212 at the capacity threshold; reports filed after the return (9,030, prevalence 0.203): ROC-AUC 0.828, recall 0.914 |
| Logistic regression with the joint cell | the selected logistic regression refitted with a one-hot block for the nature x body part x sector cell (724 columns), same five folds and test rows | CV PR-AUC 0.829 (SD 0.004) against 0.820 without the block and 0.838 for HGB; test PR-AUC 0.833 against 0.825 |

The question of Section 1.1 is answered yes for an insurer that reviews about three cases in four, the load at which the score outranks both hand-built references. I recommend that insurers and self-insured operators score daily filings and review them in score order down to capacity, from the 0.210 threshold. An adjuster should decide every flagged claim, and the score should not rank mines for inspection.

## 5.3 Limitations

The data are operator self-reports under Part 50, so the findings cover reported injuries and inherit under-reporting that varies by mine and year. The label also records each employer's restricted-duty practice, which the model learns with severity. Exposure is understated at contractor-heavy mines because the production file has no contractor hours, and Mines attributes are a snapshot that can post-date the accident. About one in twenty late-2025 reports may still be re-coded. Test metrics also pool reports filed after the worker's return, and ranking is weaker on the open ones ({tab:robustness}).

Grids were small, though wider ones tried after testing gained at most 0.002; the test set was one fold, and the SVM and KNN used subsamples. No external data set was scored, and the cost comparison rests on an assumed USD 4,000 per missed case ({tab:costs}). SHAP values cover 500 rows, and the application retrains on reduced grids, so its live numbers differ from the full run.

## 5.4 Future work

External validation on closed 2026 reports, with the thresholds fixed here, comes first. Sector-level monitoring of the Brier score and review rate should trigger a refit at the S6 limit. An insurer's closed claims could then price a missed case and replace the assumed ratio. Narrative text with post-filing leakage controls, and a DART model, come last. Uzbekistan publishes only aggregate accident statistics (Kun.uz, 2026), so transfer there would first need case-level reports with a degree-of-injury code.

## 5.5 Ethical considerations

The MSHA files are public-domain US government works (Office of the Law Revision Counsel, 2020), credited to the US Department of Labor as requested (US Department of Labor, no date). No names or narratives enter the model, but occupation, experience, mine and date could identify a worker at a small mine, so scored lists should stay with claims staff. Self-reporting adds representational bias and the label measurement bias (Mehrabi et al., 2021), as the same injury scores lower where restricted duty is offered. Misses fall more often on metal and non-metal workers: recall at the capacity threshold is 0.971 in coal against 0.911 in metal and non-metal mines ({tab:robustness}). The model card ({tab:modelcard}) and the application's decision-support notice rule out using a score to contest or delay a claim. I recommend that an adjuster confirm every flag and that the insurer compare recall by sector each quarter, recalibrating when the gap widens.
