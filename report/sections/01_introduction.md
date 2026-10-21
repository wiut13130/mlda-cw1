# 1 Introduction

## 1.1 Business case

The operator's safety officer files every reportable US mine injury on Mine Safety and Health Administration (MSHA) Form 7000-1 within ten working days (Legal Information Institute, no date a). The workers'-compensation insurer or self-insured operator then decides whether to open an indemnity reserve, a return-to-work plan and medical case management. Federal surveillance counted 3,604 non-fatal lost-time injuries at US mines in 2024 (National Institute for Occupational Safety and Health, no date); in-scope reports here averaged about 5,000 a year from 2020. The score is computed on receipt of Form 7000-1, at its filing date, from circumstance, worker and mine fields only. The median form is filed 8 days after the accident (quartiles 5 and 13) and 80.9% within 14 days. At filing, 45.0% of in-scope reports already carry a return-to-work date and are not scored, as the outcome is recorded (20.4% lost-time); the other 55.0% are still open at filing (71.6% lost-time, {tab:robustness}) and are called open reports below. The extract holds only final degree and days values, so whether the filer knew the outcome cannot be reconstructed. The study therefore classifies retrospectively labelled reports from filing-time fields; operational use needs prospective validation on reports scored at receipt.

{tab:costs} prices the two errors. A false negative delays the reserve, return-to-work plan and medical management, and late-reported claims cost more (Business Insurance, 2015); a false positive costs a needless review. Their ratio is assumed and varied. With almost half the reports positive, from 10:1 upwards the selected model's cost-optimal threshold flags more than nine in ten test reports and saves less than 5% on reviewing everything. Review capacity therefore sets the operating point, and I use two. The capacity threshold is the probability at which out-of-fold specificity on all training rows is 0.40, which flags about 76% of all filings. The workload rule reviews the top three in four open reports by score. They differ because the flag rate equals prevalence x recall + (1 - prevalence) x (1 - specificity), so a specificity target flags a larger share of open reports, whose prevalence is higher. As neither capacity has a public source, {tab:per1000} also flags half and five in six of all filings.

Table: Error costs of the claim-flagging decision || Each value is given with the evidence behind it; the false-negative cost and the cost ratio are assumptions varied in the threshold analysis, and the review-everything floor is computed on all 101,609 in-scope reports. Source: author's assumptions from the cited cost sources and analysis of MSHA open data. {#tab:costs}

| Item | Meaning at filing | Value used | Evidence and status |
|---|---|---|---|
| False positive (C_FP) | a case flagged for early review that ends as restricted duty or medical treatment only | USD 200 (range 105 to 260) | 2 to 4 claims-adjuster hours at the median wage of USD 37.51 per hour, loaded by 1.4, plus about 1 hour of safety-officer time (Bureau of Labor Statistics, 2025) |
| False negative (C_FN) | a lost-time case left unflagged, so reserve, return-to-work plan and medical management start late | USD 4,000 (range 3,100 to 6,200), an assumption | 5% to 10% of the National Council on Compensation Insurance (NCCI) accident-year 2025 average lost-time claim of USD 61,900 (indemnity 31,300 plus medical 30,600) (Gen Re, 2026); the share is a judgement, and the NCCI report-lag study, where claims reported in week 3 cost 35% more (Business Insurance, 2015), is an analogy only |
| Cost ratio C_FN : C_FP | weight of one miss against one unnecessary review | 20:1 base; sensitivity set 2:1, 3:1, 5:1, 10:1, 20:1, 30:1 | implied threshold on a calibrated probability p* = 1 / (1 + ratio), 0.333 at 2:1 and 0.048 at 20:1 |
| Review-everything floor | error cost of flagging every case: each of the 514 non-lost-time cases per 1,000 becomes a USD 200 false positive, and correct decisions cost nothing because C_FN is the loss net of a review | USD 102,829 per 1,000 cases at any ratio | within 5% of the cheapest test threshold from 10:1 upwards at 48.6% prevalence, so review capacity sets the operating point: the capacity threshold (specificity 0.40 on all training rows, about 76% of all filings flagged) and the workload rule (three in four open reports), both assumptions without a public source |

A one-field hand rule built for this study, flagging all but cuts, dust in the eye and scratches, catches most cases of both classes but cannot trade recall for specificity as capacity changes ({tab:criteria}). In Mitchell's (1997) terms, task T is labelling each new report lost-time or not, experience E is the labelled training reports, and performance P is the held-out area under the precision-recall curve (PR-AUC). Following Provost and Fawcett (2013), this is supervised binary classification with probability estimates, scored in batch on daily filings and interactively in the application.

## 1.2 Problem framing

One row is one injured person on one Form 7000-1 (Mine Safety and Health Administration, no date). The target lost_time, derived from DEGREE_INJURY_CD ({tab:label}), is positive for fatalities, permanent disabilities and days-away cases, which trigger wage replacement, and negative for restricted-duty-only and no-lost-time cases. A "serious" label (fatal, permanent or 30 or more days lost; 24.8% positive) assumes days lost are final at filing, yet open cases are right-censored. The days away, restricted or transferred (DART) label (69.0% positive) assumes restricted duty matters to the payer, which owes no indemnity for it. I kept the days-away label because it matches the reserve decision and the federal injury-survey definition (Bureau of Labor Statistics, no date).

Table: Degree-of-injury codes mapped to the target || Mapping of MSHA degree-of-injury codes to the lost_time target, with row counts and prevalence on all in-scope reports of 2010-2025 before the train/test split, and the two alternative labels that were assessed. Source: author's analysis of MSHA open data. {#tab:label}

| DEGREE_INJURY_CD | Codebook meaning | Rows | Class |
|---|---|---|---|
| 01 | Fatality | 567 | positive |
| 02 | Permanent total or partial disability | 1,124 | positive |
| 03 | Days away from work only | 36,948 | positive |
| 04 | Days away from work and restricted activity | 10,728 | positive |
| 05 | Days of restricted activity only | 20,788 | negative |
| 06 | No days away from work, no restrictions | 31,454 | negative |
| 00, 07, 08, 09, 10 | Accident only, occupational illness, natural causes, non-employee, all other cases including first aid | 19,932 | excluded |
| All in-scope reports | 49,367 positive and 52,242 negative | 101,609 | prevalence 48.6% |
| Coal mines | sector from COAL_METAL_IND | 41,461 | prevalence 63.0% |
| Metal and non-metal mines | sector from COAL_METAL_IND | 60,148 | prevalence 38.6% |
| Alternative "serious" | 01 or 02, or DAYS_LOST of 30 or more; DAYS_LOST is right-censored for open cases | 25,233 positive | prevalence 24.8%; sensitivity analysis only |
| Alternative DART | 01 to 05 against 06; invariant to restricted-duty practice | 70,155 positive | prevalence 69.0% (coal 69.9%, metal and non-metal 68.5%); Cramer's V with sector 0.015 against 0.240 for lost_time |

Coal reports are lost-time far more often than metal and non-metal ones (63.0% against 38.6%), so metrics are reported by sector. At 48.6% prevalence the classes are near balance, and the cost asymmetry enters through the threshold. I fixed seven success criteria before fitting any model ({tab:criteria}). PR-AUC is primary (S2), ahead of the area under the receiver operating characteristic curve (ROC-AUC, S1), because precision along the ranked queue is the share of reviewed cases that turn out to be lost-time (Section 4.4). Recall is tested on all filings at the capacity threshold (S3), cost only where a threshold can beat reviewing everything (S4), and the Brier score because the application shows probabilities (S6), with calibration judged on the reliability diagram. The model must also beat the four baselines of {tab:criteria}.

Table: Success criteria and baselines || Success criteria S1 to S7, fixed before modelling and evaluated on the 20% mine-year-grouped held-out test set (S6 also by sector on the temporal split), with the reference value each must beat and the four baselines; reference and baseline values are scored on the same 20,320 test reports, except the best one-field rules of S3, which come from the pre-modelling data profile of all in-scope reports. Source: author's analysis of MSHA open data. {#tab:criteria}

| # | Criterion | Target | Reference value |
|---|---|---|---|
| S1 | ROC-AUC (area under the receiver operating characteristic curve) | 0.80 or more | lookup baseline 0.804 |
| S2 | PR-AUC (average precision), the primary metric | 0.75 or more | lookup baseline 0.790; prevalence 0.486 (random ranking) |
| S3 | Recall on all filings at the capacity threshold (set on out-of-fold probabilities of all training rows so that specificity is 0.40, about 76% of all filings flagged); recall under the workload rule (top three in four open reports by score) is reported beside it | 0.92 or more (implied precision about 0.59) | lookup baseline 0.916; best one-field rules from the data profile 0.928 at specificity 0.293 and 0.896 at 0.429 |
| S4 | Expected cost per 1,000 cases at the cost-optimal out-of-fold threshold, ratios 2:1, 3:1 and 5:1 | below the review-everything floor and below the lookup at each ratio | floor USD 102,825; lookup baseline USD 73,553, 85,472 and 98,061 |
| S5 | Recall at the out-of-fold threshold for a 50% flag rate (planned as recall among the 50% of cases with the highest scores; both are reported) | 0.78 or more | lookup baseline 0.753; random 0.50 (0.486 is the expected precision of a random half) |
| S6 | Brier score on the test set, and by sector on the temporal split | 0.20 or less | lookup baseline 0.179; constant prediction at the prevalence 0.250 |
| S7 | Statistical comparison: McNemar test against the lookup baseline and a paired test across cross-validation folds between the two best models | reported | not applicable |
| Baseline | Majority class (predict "not lost-time" for every case) | | accuracy 0.514, recall 0.000 |
| Baseline | Review everything (predict "lost-time" for every case) | | recall 1.000, specificity 0.000, precision 0.486, USD 102,825 per 1,000 cases |
| Baseline | Hand rule built for this study from the data profile: nature of injury is not a cut, dust in the eye or a scratch | | recall 0.898, specificity 0.423, precision 0.595 |
| Baseline | Lookup baseline: lost-time rate of each nature of injury x body part x sector cell, fitted on training rows | | ROC-AUC 0.804, PR-AUC 0.790 |

## 1.3 Dataset origin, provenance, licensing and integration

The data are three open MSHA files (US Department of Labor, Mine Safety and Health Administration, 2026a, 2026b). Under Title 30, Part 50 of the Code of Federal Regulations, operators file Form 7000-1 for each accident, injury and illness and Form 7000-2 for quarterly hours (Legal Information Institute, no date b). Because MSHA refreshes the files weekly, I pinned the version of 9 October 2026 by SHA-256 digest ({tab:sources}). As federal government works they are public domain under 17 U.S.C. 105, free to copy, modify and redistribute (Office of the Law Revision Counsel, 2020).

MSHA collects the forms for enforcement and for incidence rates per 200,000 employee-hours (Mine Safety and Health Administration, 2026); scoring single reports is a reuse with four collection biases. Operators self-report and may omit first-aid cases, which limits the results to reported injuries. The label also records employer practice, since a back sprain becomes days away or restricted duty depending on whether restricted work is offered. Because the Mines table is a snapshot, attributes that may post-date the accident were re-dated or, like the SAFETY_COMMITTEE_IND, PART48_TRAINING and PORTABLE_OPERATION flags, dropped. Contractor hours are absent from the production file, which understates exposure at mines that use contractors. I kept 2010-2025, as older rows have more blank experience fields and 2026 cases are still open.

I combined the tables with left joins validated as many-to-one and asserted the row count after each ({tab:sources}). Production was summed to mine-years and joined on the previous year, since current-year hours are not yet due for first-quarter accidents; unmatched rows kept a missing-value flag. {tab:dictionary} lists the resulting features. The three snapshot flags are excluded because, lacking an effective date, their value at the accident is unknown. The repository's data card follows Gebru et al. (2021), and the dataset was reserved in the module Discussions forum on 10 October 2026.

Table: The three MSHA source tables and their joins || The three source tables with their size, grain, key and licence, and how each was joined to the accident rows; the files were published on 9 October 2026 and pulled on 14 October 2026 at 12:06 UTC from the MSHA Open Government Data page (https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp), each zip file at https://arlweb.msha.gov/OpenGovernmentData/DataSets/ followed by its name, and the full SHA-256 digests are in data/raw/SNAPSHOT.json. Source: author's analysis of MSHA open data. {#tab:sources}

| Source file (SHA-256, first 12 hex digits) | Rows x columns | Grain (one row is) | Key | Licence | Join and rows before -> after | Matched rows |
|---|---|---|---|---|---|---|
| Accidents.txt in Accidents.zip (62d0c861a5c3) | 275,220 x 57 | one injured person on one Form 7000-1, or one accident without injury | DOCUMENT_NO | US public domain (17 U.S.C. 105); attribution to the US Department of Labor requested (US Department of Labor, no date) | base table; CAL_YR from 2010 (275,220 -> 125,718), to 2025 (-> 121,541), degree codes 01 to 06 (-> 101,609) | not applicable |
| Mines.txt in Mines.zip (1a0957ff0049) | 92,060 x 59 | one mine, with attributes as at the snapshot date | MINE_ID (7-digit string) | US public domain (17 U.S.C. 105) | left join on MINE_ID, many-to-one; 101,609 -> 101,609 | 101,609 (100.0%), no orphans |
| MinesProdYearly.txt in MinesProdYearly.zip (c0fdca86d175) | 658,922 x 11 | employee-hours and average employees for one mine, calendar year and subunit (Form 7000-2) | MINE_ID, CALENDAR_YR, SUBUNIT_CD | US public domain (17 U.S.C. 105) | summed to 370,794 mine-years; left join on MINE_ID and CALENDAR_YR = CAL_YR - 1, many-to-one; 101,609 -> 101,609 | 100,403 (98.8%); 1,206 kept with missing exposure and prev_year_matched = 0 |

Table: Data dictionary of the 52 model features || The 52 features (20 categorical, 20 numeric and 12 binary) are grouped by type, with meaning and source table; the definitions apply to every row of the training and test splits, every feature is known when Form 7000-1 is filed, and outcome fields such as DAYS_LOST are excluded as leakage. Source: author's analysis of MSHA open data and codebooks. {#tab:dictionary}

| Type | Feature(s) | Meaning | Source table |
|---|---|---|---|
| Categorical | SUBUNIT_CD | part of the mine where the accident happened (underground, surface at an underground mine, strip or open pit, mill or preparation plant, office and others) | Accidents |
| Categorical | FIPS_STATE_CD | state of the mine | Accidents |
| Categorical | UG_LOCATION_CD, UG_MINING_METHOD_CD | underground location and mining method; level "none" when the accident was not underground | Accidents |
| Categorical | MINING_EQUIP_CD, EQUIP_MFR_NAME | equipment involved and its manufacturer; level "none" when no equipment was involved | Accidents |
| Categorical | CLASSIFICATION_CD, ACCIDENT_TYPE_CD | circumstance class (for example handling materials, slip or fall) and accident type (for example struck by a falling object, over-exertion) | Accidents |
| Categorical | OCCUPATION, ACTIVITY_CD | job title of the injured worker and the activity at the time | Accidents |
| Categorical | INJURY_SOURCE_CD, NATURE_INJURY_CD, INJ_BODY_PART_CD | object or substance that caused the injury, nature of the injury (for example sprain, fracture, cut) and body part | Accidents |
| Categorical | COAL_METAL_IND | sector, coal or metal and non-metal | Accidents |
| Categorical | PRIMARY_CANVASS | commodity group of the mine (coal, metal, non-metal, stone, sand and gravel) | Mines |
| Categorical | status_103i_at_accident | section 103(i) spot-inspection status in force at the accident date (never, in force, formerly, unknown) | Mines, with the accident date |
| Categorical | type_at_time | mine type (underground, surface or facility) from the previous year's production subunits | MinesProdYearly (fallback: Accidents SUBUNIT_CD) |
| Categorical | mine_size_band | previous-year average employees in five bands, from 1-4 to 200+ | MinesProdYearly |
| Categorical | accident_weekday, accident_month | day of the week and month of the accident | Accidents (ACCIDENT_DT) |
| Numeric | TOT_EXPER, MINE_EXPER, JOB_EXPER | years of total mining, current-mine and current-job experience after the consistency rules | Accidents |
| Numeric | log1p_tot_exper | log(1 + TOT_EXPER), which reduces right skew | derived from Accidents |
| Numeric | exper_ratio_mine, exper_ratio_job | mine and job experience as a share of total experience, between 0 and 1 | derived from Accidents |
| Numeric | no_injuries_capped | number of people injured in the occurrence, capped at 10 | Accidents |
| Numeric | accident_hour, shift_begin_hour | clock hour of the accident and of the start of the shift | Accidents |
| Numeric | hours_into_shift | hours from the start of the shift to the accident, missing above 16 | Accidents |
| Numeric | latitude, longitude | mine coordinates, with invalid pairs set to missing | Mines |
| Numeric | prev_hours, log1p_prev_hours | employee-hours worked at the mine in the previous year, raw and log-transformed | MinesProdYearly |
| Numeric | prev_employees, prev_hours_per_employee | previous-year average employees and hours per employee | MinesProdYearly |
| Numeric | prev_incidents | reports at the mine in the previous year that were filed before the accident date | Accidents (full file) |
| Numeric | prev_incident_rate_200k, prev_lost_time_rate_200k | previous-year reports and lost-time cases per 200,000 employee-hours, capped at 50 and log-transformed | Accidents (full file) and MinesProdYearly |
| Numeric | prev_lost_time_share | share of the previous year's valid-code cases (degree 01 to 06) filed before the accident date that were lost-time | Accidents (full file) |
| Binary | is_night, is_weekend | accident between 18:00 and 05:59; accident on a Saturday or Sunday | Accidents |
| Binary | new_to_mine | less than one year of experience at the current mine | Accidents |
| Binary | is_contractor | the report carries a contractor identifier | Accidents |
| Binary | immed_reportable | the accident was one of the types MSHA must be told of within 15 minutes | Accidents |
| Binary | exper_inconsistent | job or mine experience exceeded total experience before correction | Accidents |
| Binary | prev_year_matched | previous-year production was found for the mine | MinesProdYearly |
| Binary | tot_exper_missing, accident_hour_missing, hours_into_shift_missing, prev_share_missing, coords_missing | missing-value indicators for total experience, accident hour, hours into shift, previous-year lost-time share and coordinates | derived |

## 1.4 Aims and objectives

The aim is a lost-time score at filing that ranks cases better than the hand rule and the lookup baseline and is calibrated on all filings, pursued through four objectives. Section 2 describes every model feature on the training rows and ties each skewed, redundant or incomplete feature to a preparation step. Section 3 fits every learned step on training rows only and asserts that no mine-year crosses the split. Section 4 tunes eight algorithms with grouped cross-validation, tests the final model against S1 to S7 and compares models at equal workloads on open reports. Section 5 states the expected cost per 1,000 cases at the 2:1 and 5:1 ratios of {tab:costs}, recall by sector, a mine-history audit and the limits of each.
