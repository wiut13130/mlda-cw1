# 5 Conclusion

## 5.1 Key data analysis findings

1. Lost-time cases are 48.6% of the 101,609 in-scope reports ({fig:target}); with the classes near balance, review capacity sets the insurer's operating point.
2. Nature of injury and body part, both entered by the safety officer, are most associated with lost time (Cramer's V 0.391 and 0.364, {tab:associations}; cuts near 18% lost-time, {fig:cat}).
3. The mine's previous-year lost-time share is the strongest numeric signal (point-biserial r = 0.292, {tab:associations}; {fig:box}) and partly reflects restricted-duty practice.
4. Coal reports are lost-time in 63.0% of cases against 38.6% in metal and non-metal mines while the DART label hardly varies by sector ({tab:label}), evidence that the label records restricted-duty practice and that scores must be read within sector.
5. Reports without total experience are lost-time in 59.3% of training cases against 48.5% of recorded ones ({fig:missing}; {tab:decisionlog}), a risk the missing-value indicators keep visible.
6. The classes overlap in every two-dimensional view ({fig:pair}; {fig:pca}); combining many fields mainly raises recall, with HGB catching 0.936 of lost-time cases on all filings against 0.906 for the one-field hand rule at the same 75% workload ({tab:policy}).
7. Of all filings, 45.0% record a return to work by the filing date ({tab:robustness}); on the open reports that remain, the workload rule at HGB's pre-fixed cut-off flags 74.3% with recall 0.850 ({tab:policy}).

## 5.2 Interpretation of results

Per 1,000 test reports at the capacity threshold, HGB misses 28.5 lost-time cases against 49.5 for the hand rule while flagging 29 more ({tab:per1000}). At the 20:1 base ratio of {tab:costs} the cost-optimal threshold flags nearly every case ({fig:cost}), and the score's value lies in ordering the queue when review capacity binds.

Table: Outcomes per 1,000 test reports by policy || Results on the 20,320 test reports for HGB at three thresholds fixed on out-of-fold probabilities and at two other workloads, the lookup baseline at its capacity threshold, the one-field hand rule and reviewing everything, with expected cost at 2:1 and 5:1 and USD 200 per unnecessary review; counts are scaled to 1,000 reports, and the half and five-in-six rows take their cut-off from the test scores. Source: author's analysis of MSHA open data. {#tab:per1000}

| Policy | Threshold | Flagged | Lost-time cases caught | Lost-time cases missed | False alarms | Expected cost at 2:1 (USD) | Expected cost at 5:1 (USD) |
|---|---|---|---|---|---|---|---|
| HGB, capacity threshold (out-of-fold specificity 0.40) | 0.211 | 762.7 | 457.4 | 28.5 | 305.3 | 72,461 | 89,557 |
| HGB, cost-optimal at 2:1 | 0.331 | 626.0 | 422.9 | 63.0 | 203.1 | 65,817 | 103,612 |
| HGB, cost-optimal at 5:1 | 0.164 | 820.9 | 469.0 | 16.8 | 351.9 | 77,106 | 87,205 |
| HGB, half of the reports flagged (top 50% of test scores) | 0.464 | 500.0 | 374.3 | 111.6 | 125.7 | 69,793 | 136,762 |
| HGB, five in six reports flagged (top 83.3% of test scores) | 0.157 | 833.3 | 471.0 | 14.9 | 362.3 | 78,406 | 87,323 |
| Lookup baseline, capacity threshold | 0.225 | 752.5 | 445.0 | 40.8 | 307.5 | 77,835 | 102,343 |
| Hand rule built for this study: nature is not a cut, dust in the eye or a scratch | not applicable | 733.2 | 436.4 | 49.5 | 296.8 | 79,144 | 108,819 |
| Review everything | not applicable | 1,000.0 | 485.9 | 0.0 | 514.1 | 102,825 | 102,825 |

Nature, body part and sector lead the permutation importance ({fig:importance}). Their joint cells, which an additive log-odds model cannot form unaided, explain about half of HGB's lead over logistic regression, whose cross-validated PR-AUC rises from 0.820 to 0.829 with the cell, against 0.838 for HGB ({tab:robustness}). The rest was not isolated; one plausible source is that trees need no transformation of the skewed size measures (Grinsztajn et al., 2022).

The estimates hold under resampling and harder splits ({tab:robustness}): the fold SD of PR-AUC is 0.003, its mine-year bootstrap interval clears the S2 target, and unseen mines and 2020-2025 reports move it by under 0.01. HGB's lead over the MLP is thin: the corrected fold test gives p = 0.029, and at equal workload on open reports their recalls are level.

Table: Uncertainty, generalisability and mechanism checks of HGB || Cross-validation, bootstrap intervals and paired tests, results by sector, temporal and mine-grouped splits, open reports, logistic regression given the lookup baseline's joint cell, a refit without mine histories drawn from test-split outcomes and a refit on open training reports; the capacity threshold is the out-of-fold threshold for specificity 0.40, and the open-report, joint-cell, history and open-only checks were run after the test scoring and changed no choice. Source: author's analysis of MSHA open data. {#tab:robustness}

| Check | Data and protocol | Result for HGB |
|---|---|---|
| Cross-validation | 5-fold mine-year-grouped CV on the 81,289 training rows | PR-AUC 0.838 (SD 0.003, folds 0.833 to 0.842); ROC-AUC 0.838 (SD 0.002) |
| Bootstrap intervals | 1,000 resamples of the 20,320 test rows and of their 6,724 mine-years | PR-AUC 0.842 (rows 0.835 to 0.849; mine-years 0.826 to 0.856); ROC-AUC 0.843 (mine-years 0.836 to 0.852); recall at the capacity threshold 0.941 (mine-years 0.936 to 0.947) |
| Paired fold test against the MLP | PR-AUC on the same five CV folds | mean difference 0.006; corrected t-test p = 0.029; Wilcoxon signed-rank W = 0, p = 0.0625 (the minimum with five folds) |
| McNemar test against the lookup baseline | test set at the capacity threshold, row-wise and clustered by mine-year | p < 0.001 in both forms (clustered chi-square 24.5, 1 df) |
| By sector | all 20,320 test reports at the capacity threshold 0.211; the 11,290 open test reports under the workload rule at the out-of-fold cut-off 0.376 | all filings: coal (n = 8,270) recall 0.970, specificity 0.395, precision 0.739; metal and non-metal (n = 12,050) recall 0.909, specificity 0.411, precision 0.487; open reports: coal (n = 4,773, prevalence 0.912) flag rate 0.915, recall 0.930, precision 0.928; metal and non-metal (n = 6,517, prevalence 0.565) flag rate 0.617, recall 0.755, precision 0.692, Brier 0.216 |
| Temporal split | re-tuned on 2010-2019 (71,877 rows) with thresholds from its out-of-fold probabilities, scored on 2020-2025 (29,732 rows) | ROC-AUC 0.834, PR-AUC 0.843; recall 0.945 and specificity 0.373 at the capacity threshold; coal recall 0.974 and Brier 0.125; metal and non-metal recall 0.919 and Brier 0.186 |
| Mine-grouped split | no mine on both sides (8,303 mines; 20,289 test rows), primary hyperparameters | ROC-AUC 0.838, PR-AUC 0.837, recall 0.934 at the capacity threshold, recall 0.762 at a 50% flag rate |
| Feature variants | primary split with fixed hyperparameters: immed_reportable set to 0; status_103i_at_accident constant; 175 amputation test rows (nature code 100) removed | PR-AUC 0.839, 0.842 and 0.835 against 0.842 |
| S5, recall at a 50% flag rate | out-of-fold threshold 0.456, test flag rate 50.7%; and the top 50% of test scores, the planned form | 0.777 against the 0.78 target (missed by 0.003); MLP 0.777 at a 51.1% flag rate; top half of test scores: HGB 0.770, MLP 0.769 |
| Open reports | 45,679 of all filings (45.0%) were filed after the return to work, lost-time in 20.4% against 71.6% of the rest; open test reports have no return-to-work date on or before the filing date (11,290, prevalence 0.712); thresholds unchanged | ROC-AUC 0.812, PR-AUC 0.920; capacity threshold: flag rate 0.900, recall 0.949; exactly 75% reviewed: recall 0.855 against 0.828 for the lookup baseline (paired difference 0.027, 0.021 to 0.033) and 0.854 for the MLP (0.000, -0.003 to 0.004); out-of-fold cut-off 0.376: flag rate 0.743, recall 0.850; mean prediction 0.592 against 0.712 (Brier 0.172); reports filed after the return (9,030, prevalence 0.203): ROC-AUC 0.829, mean prediction 0.354 (Brier 0.147) |
| Logistic regression with the joint cell | the selected logistic regression refitted with a one-hot block for the nature x body part x sector cell (718 columns), same five folds and test rows | CV PR-AUC 0.829 (SD 0.004) against 0.820 without the block and 0.838 for HGB; test PR-AUC 0.833 against 0.825 |
| History audit | prev_incidents, prev_incident_rate_200k, prev_lost_time_rate_200k and prev_lost_time_share are built from the full Accidents file, so with the mine-year grouped split, for 11,426 training rows (14.1%) the previous mine-year is in the test split, and 11,416 of them carry a non-blank history drawn from it; those histories blanked on the training rows only, as for a mine-year without history, and HGB refitted | test PR-AUC 0.842 in both fits (the refit 0.0006 higher); recall at the capacity threshold 0.941 in both |
| Open-only refit | HGB refitted on the 44,640 open training reports only and scored on the 11,290 open test reports | exactly 75% reviewed: recall 0.877, precision 0.833 and 88 lost-time cases missed per 1,000, against 0.855, 0.812 and 103 for the fit on all filings; PR-AUC 0.932 and ROC-AUC 0.844 against 0.920 and 0.812 |

For the insurer's decision of Section 1.1, HGB orders open test reports better than the lookup baseline at equal workload (recall difference 0.027, 95% interval 0.021 to 0.033; {tab:robustness}). That meets the ranking aim of Section 1.4, and the calibration aim is met on all filings; on open reports the probability is a ranking score (Section 4.4). I recommend HGB fitted on all filings, with the workload rule for open reports: at its cut-off of 0.376, fixed from out-of-fold scores before testing, it flags 74.3% of open test reports with recall 0.850. The capacity threshold of 0.211 remains the operating point for all filings, where it meets the 0.92 recall target of {tab:criteria}, but it would flag 90.0% of open reports (recall 0.949). Operational use waits for a prospective trial on reports scored at receipt. Its candidate is HGB refitted on open training reports, which ranked open test reports better (recall 0.877 against 0.855 with exactly 75% reviewed), at a cut-off fixed in the same way from its own out-of-fold scores, with the all-filings model at 0.376 as the comparator. An adjuster decides every flagged claim, and because the label partly records restricted-duty practice, the score must not rank mines for inspection.

## 5.3 Limitations

The data are Part 50 operator self-reports, with under-reporting that varies by mine and year, and contractor hours are missing from exposure. The Mines snapshot can post-date the accident, and its location and commodity fields are assumed unchanged since then; about one in twenty late-2025 reports may still be re-coded.

With no filing-time degree or days values in the extract, whether the filer, writing a median of 8 days after the accident, already knew the outcome cannot be checked. On open reports ranking is weaker (ROC-AUC 0.812 against 0.843). The application shows the workload-rule decision beside the capacity-threshold class (Appendix C), but it holds only the all-filings model, and its cost note assumes probabilities calibrated on all filings (Section 4.4).

For 14.1% of training rows the previous mine-year is in the test split; blanking the histories drawn from it raised test PR-AUC by less than 0.001 ({tab:robustness}). Grids were small (wider ones gained at most 0.0025), the test set was one fold, the SVM and KNN used subsamples, no external data set was scored and the cost comparison assumes USD 4,000 per missed case ({tab:costs}); the application's live numbers come from reduced grids.

## 5.4 Future work

In the prospective trial of Section 5.2, an insurer would score each open Form 7000-1 at receipt for one quarter with the open-only refit and the all-filings model, flag the top three in four at each model's pre-fixed cut-off and compare the flags with its adjusters' decisions, once the refit and its cut-off are added to the application. Closed 2026 reports can test the refit on a later year, and closed claims could replace the assumed cost ratio. Narrative text with post-filing leakage controls, and a DART model, come last. Transfer to Uzbekistan would first need case-level reports with a degree-of-injury code, as it publishes only aggregate accident statistics (Kun.uz, 2026).

## 5.5 Ethical considerations

The MSHA files are public-domain US government works (Office of the Law Revision Counsel, 2020), credited to the US Department of Labor as requested (US Department of Labor, no date). No names or narratives enter the model, but occupation, experience, mine and date could identify a worker at a small mine, so scored lists should stay with claims staff. Self-reporting adds representational bias and the label measurement bias (Mehrabi et al., 2021), as the same injury scores lower where restricted duty is offered. Misses fall more often on metal and non-metal workers: on open test reports under the workload rule their recall is 0.755 against 0.930 in coal ({tab:robustness}). Before use the insurer should recalibrate, since the metal and non-metal Brier score there (0.216) exceeds the S6 limit, and set the cut-off within each sector; recall and the Brier score should then be compared by sector each quarter. The model card ({tab:modelcard}) and the application's decision-support notice rule out using a score to contest or delay a claim.
