# Data profile and modelling decisions (MSHA lost-time injury, snapshot 2026-10-09)

Synthesis of `docs/profile/accidents_profile.md` (Task A), `docs/profile/mines_production_join.md` (Task B) and `docs/profile/domain_research.md` (Task C), with every figure recomputed on the final modelling window (CAL_YR 2010-2025, DEGREE_INJURY_CD 01-06, n = 101,609) on 2026-10-14 with pandas 3.0.6. Figures quoted from the profiles on a different base say so. Machine-readable twin for the builders: `docs/profile/decisions.json`. Project: WIUT 6COSC017C CW1, student ID 13130. Revised on 2026-10-14 after the leakage-and-time, label-and-bias and join-features-feasibility reviews; every accepted finding and the change it caused is listed in section 10 (Review log), and all new figures were recomputed on the same base with pandas 3.0.6.

## 0. Sources, load call and encoding

| File | Rows x cols | Key | Strict UTF-8 | Loaded as | Non-ASCII bytes and where |
|---|---|---|---|---|---|
| `data/raw/Accidents.txt` | 275,220 x 57 | DOCUMENT_NO (unique) | fails at byte 3,626,765 | latin-1 | 123 bytes: NARRATIVE 90 rows, EQUIP_MODEL_NO 2 rows |
| `data/raw/Mines.txt` | 92,060 x 59 | MINE_ID (unique, 7 digits) | fails at byte 1,069,693 | latin-1 | 1,425 bytes in names, directions, town (224 rows) |
| `data/raw/MinesProdYearly.txt` | 658,922 x 11 | MINE_ID + CALENDAR_YR + SUBUNIT_CD (unique) | fails at byte 59,788,956 | latin-1 | 105 bytes, CURR_MINE_NM only |

Load call for all three: `pd.read_csv(path, sep="|", quotechar='"', dtype=str, keep_default_na=False, encoding="latin-1")`, then `str.strip()` on every column (IMMED_NOTIFY_CD stores its placeholder as `? `; OPERATOR_NAME, MINING_EQUIP, ACTIVITY carry trailing spaces), then explicit numeric conversion of the NUMBER columns. No join key, code or numeric column contains a non-ASCII byte, so latin-1 is lossless for modelling. Headers match the Accidents and Mines codebooks exactly; the 2011 production codebook names differ from the file header (`CALENDAR_YR` = CAL_YR, `ANNUAL_HRS` = ANNUAL_HOURS, `AVG_ANNUAL_EMPL` = AVG_EMPLOYEE_CNT, `C_M_IND` = COAL_METAL_IND, `CURR_MINE_NM` = MINE_NAME, `STATE_ABBR` = STATE, `SUBUNIT_DESC` = SUBUNIT). Placeholders counted separately from blanks: `""`, `?`, `? `, `NO VALUE FOUND`. ACCIDENT_DT parses for all rows; CAL_YR, CAL_QTR and FISCAL_YR agree with ACCIDENT_DT on 100% of rows. Snapshot hashes: `data/raw/SNAPSHOT.json` (files dated 2026-10-09, pulled 2026-10-14T12:06Z).

## 1. Target definition

Unit of analysis: one row = one injured person on one Form 7000-1 (one form per person per occurrence, 30 CFR 50.20). Label = MSHA's closed-case degree-of-injury classification, taken from `DEGREE_INJURY_CD`.

| Code | Codebook label | MSHA class | n (2010-2025) | Class | Reason |
|---|---|---|---|---|---|
| 01 | Fatality | FATAL | 567 | positive | death; most costly lost-time case |
| 02 | Permanent total or partial disability | NFDL | 1,124 | positive | indemnity case |
| 03 | Days away from work only | NFDL | 36,948 | positive | days away = wage replacement |
| 04 | Days away from work and restricted activity | NFDL | 10,728 | positive | days away |
| 05 | Days restricted activity only | NFDL | 20,788 | negative | medical cost, no indemnity (BLS DJTR) |
| 06 | No days away, no restrictions | NDL | 31,454 | negative | medical treatment only |
| 00 | Accident only | not an injury | 12,258 | excluded | no injured person (all five injury codes are `?`) |
| 07 | Occupational illness | illness | 5,476 | excluded | diagnosis-dated, not an accident; 1,289 of these rows carry DAYS_LOST > 0 (median 30 days) and are out of scope, see section 2 step 4 |
| 08 | Natural causes | n/a | 1,001 | excluded | not work-caused |
| 09 | Non-employee | n/a | 347 | excluded | outside the workforce |
| 10 | All other (incl. first aid) | n/a | 850 | excluded | not reportable as injury |
| ? | No value found | n/a | 0 | excluded | all 910 `?` rows are pre-2010 |

| Quantity | Value |
|---|---|
| Positive (01-04) | 49,367 |
| Negative (05-06) | 52,242 |
| N | 101,609 |
| Prevalence | 48.585% (neg:pos = 1.058) |
| Majority-class accuracy | 51.4% (predict negative) |
| Prevalence by commodity | coal 63.04% (n 41,461); metal/non-metal 38.62% (n 60,148) |
| Fallback (ii) "serious" = 01/02 or DAYS_LOST >= 30 | 25,233 positive, 24.833% |

Label validity: 99.14% of positives have DAYS_LOST > 0 or SCHEDULE_CHARGE > 0. The 424 positives that fail the check (371 code 03, 53 code 04; 0.86% of positives) split into 183 rows with DAYS_LOST blank and 241 rows with DAYS_LOST = 0; the zero-days rows cluster in 2010-2012 (75 / 40 / 81 = 196 of 241) and are 147 coal / 94 metal. All 424 are kept as positives on the strength of the MSHA degree code. 100.00% of negatives have DAYS_LOST = 0 (1 code-05/06 row in the base has DAYS_LOST > 0). Business meaning: codes 01-04 equal BLS "days away from work" plus fatalities and ILO temporary/permanent incapacity, and days away is the indemnity trigger for workers'-compensation reserving; the definition is narrower than MSHA's published NFDL and lost-time rates, which include code 05 (FY2025 NFDL 1.25 vs lost-time 1.26 per 200,000 h, so the difference is small). Class balance: 48.6% positive, so accuracy, ROC-AUC and PR-AUC are all informative and no resampling is needed; the false-negative cost asymmetry enters through the decision threshold (section 8), not through the label.

What the label measures: codes 03/04 versus 05 depend on whether the operator offers restricted duty, so the positive class is an indemnity-trigger label, not a severity label. Evidence on the final base: in the 43 nature x body-part cells with >= 200 rows in each sector, coal's lost-time rate exceeds metal's in 40 (median +19.0 pp, max +46.9 pp); for sprain/strain of the back (330 | 420) coal is 85.9% lost-time with 8.8% of its DART cases (03/04/05) coded 05, metal 43.7% lost-time with 45.8% coded 05; among the 100 mines with >= 100 DART rows the share coded 05 runs from 0.4% (p5) to 68.4% (p95), among the 57 operators with >= 200 DART rows from 0.3% to 72.6%. COAL_METAL_IND has Cramer's V 0.240 with the lost-time label but 0.015 with the DART label (01-05 vs 06: coal 69.88%, metal 68.47%), so most of the commodity effect is return-to-work coding practice rather than hazard, and the strongest mine feature (prev_lost_time_share, section 4c) is a direct proxy for that practice. Consequence: the label suits the insurer / self-insured operator (indemnity reserving, return-to-work planning); for MSHA inspection targeting the DART sensitivity label (iii-b) below is the appropriate target, and every metric is reported stratified by COAL_METAL_IND.

Per-year class counts and sector split (final window; DART = codes 03/04/05):

| CAL_YR | n | Positive | Negative | Prevalence % | Coal n | Coal share % | Coal prev. % | Metal n | Metal prev. % | Coal 05 share of DART % | Coal 04 share of DART % | Metal 05 share of DART % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010 | 8,586 | 4,167 | 4,419 | 48.53 | 4,339 | 50.5 | 59.74 | 4,247 | 37.09 | 12.17 | 10.80 | 45.40 |
| 2011 | 8,988 | 4,272 | 4,716 | 47.53 | 4,622 | 51.4 | 57.51 | 4,366 | 36.97 | 13.59 | 11.17 | 44.35 |
| 2012 | 8,479 | 4,068 | 4,411 | 47.98 | 4,044 | 47.7 | 59.84 | 4,435 | 37.16 | 11.35 | 10.75 | 45.17 |
| 2013 | 7,965 | 3,877 | 4,088 | 48.68 | 3,591 | 45.1 | 61.68 | 4,374 | 38.00 | 10.42 | 9.46 | 44.90 |
| 2014 | 7,691 | 3,765 | 3,926 | 48.95 | 3,380 | 43.9 | 62.46 | 4,311 | 38.37 | 10.66 | 8.88 | 46.06 |
| 2015 | 6,765 | 3,238 | 3,527 | 47.86 | 2,674 | 39.5 | 62.45 | 4,091 | 38.33 | 10.32 | 8.17 | 44.33 |
| 2016 | 5,805 | 2,743 | 3,062 | 47.25 | 2,032 | 35.0 | 63.93 | 3,773 | 38.27 | 10.51 | 9.60 | 44.27 |
| 2017 | 5,913 | 2,877 | 3,036 | 48.66 | 2,404 | 40.7 | 62.52 | 3,509 | 39.16 | 11.59 | 9.48 | 43.71 |
| 2018 | 5,869 | 2,816 | 3,053 | 47.98 | 2,230 | 38.0 | 63.95 | 3,639 | 38.20 | 8.01 | 7.88 | 45.35 |
| 2019 | 5,816 | 2,816 | 3,000 | 48.42 | 2,217 | 38.1 | 63.24 | 3,599 | 39.29 | 8.63 | 6.49 | 44.17 |
| 2020 | 4,590 | 2,235 | 2,355 | 48.69 | 1,502 | 32.7 | 69.17 | 3,088 | 38.73 | 7.42 | 7.06 | 45.80 |
| 2021 | 4,864 | 2,425 | 2,439 | 49.86 | 1,562 | 32.1 | 69.33 | 3,302 | 40.64 | 8.41 | 8.41 | 44.33 |
| 2022 | 5,094 | 2,492 | 2,602 | 48.92 | 1,759 | 34.5 | 67.25 | 3,335 | 39.25 | 6.91 | 8.03 | 44.44 |
| 2023 | 5,191 | 2,596 | 2,595 | 50.01 | 1,792 | 34.5 | 68.75 | 3,399 | 40.13 | 7.66 | 7.20 | 44.46 |
| 2024 | 5,140 | 2,547 | 2,593 | 49.55 | 1,698 | 33.0 | 68.43 | 3,442 | 40.24 | 7.23 | 6.90 | 43.57 |
| 2025 | 4,853 | 2,433 | 2,420 | 50.13 | 1,615 | 33.3 | 70.71 | 3,238 | 39.87 | 4.61 | 5.46 | 44.55 |
| Total | 101,609 | 49,367 | 52,242 | 48.59 | 41,461 | 40.8 | 63.04 | 60,148 | 38.62 | | | |

The overall prevalence looks flat (47.25% to 50.13%) only because two trends cancel: coal lost-time prevalence rises monotonically from 59.74% (2010) to 70.71% (2025) while the coal share of rows falls from 50.5% to 33.3%; metal moves from 37.09% to 39.87%. r(CAL_YR, y) = 0.011 overall but 0.074 within coal (0.021 within metal). The mechanism is a decline in restricted-duty coding in coal: the share of coal DART cases coded 05 falls from 12.17% to 4.61% and coded 04 from 10.80% to 5.46%, while the metal 05 share stays at 43.6-46.1%. Within the 273 coal mines with rows in both 2010-2012 and 2023-2025 the 05 share falls from 11.82% to 6.22% and prevalence rises from 57.02% to 70.58%; within the 1,376 persistent metal mines the 05 share moves 47.81% to 46.70% and prevalence 35.02% to 38.67%. For the temporal split (section 7) this is a label-drift risk: coal prevalence 61.19% in 2010-2019 vs 68.91% in 2020-2025 (+7.7 pp), metal 38.03% vs 39.83%; calibration and metrics on that split are reported by sector, and CAL_YR (or a coal x year term) is a candidate feature for the temporal variant only (section 4b).

Fallback definition (ii), kept only as a sensitivity label: positive = code 01/02 or DAYS_LOST >= 30 (25,233 rows, 24.8%). It depends on DAYS_LOST, which is right-censored for open cases (2026 prevalence collapses to 10.08% from 24.8%), has 583 values above 365 days in the window, and uses an arbitrary 30-day cut. Definition (iii) (03/04/05 vs 06) was rejected as the primary label: 68.6% positive, it discards the 1,691 fatal/permanent rows, and its negative class is defined by the absence of a return-to-duty form. Definition (iii-b), the DART label (01-05 vs 06; 70,155 positive, 69.04%; coal 69.88%, metal 68.47%; 65.7-72.8% by year), is kept as a second sensitivity label because it is invariant to restricted-duty practice and is therefore the right target for the inspection-targeting use; it is not the primary label because it is not the indemnity trigger.

## 2. Row filters

| Step | Rule | Rows removed | Rows kept | Evidence |
|---|---|---|---|---|
| 0 | all rows | | 275,220 | |
| 1 | CAL_YR >= 2010 | 149,502 | 125,718 | TOT_EXPER blank 20.11% pre-2010 vs 11.39% after (1.20% in the final base); all 910 `?` degree codes are pre-2010; RETURN_TO_WORK_DT 1-May batch default 450-800 rows/yr in 2003-2012, under 60/yr since 2013; 2009 production exists for the 2010 lag features (98.3% coverage); app memory budget (`docs/architecture.md`) |
| 2 | CAL_YR <= 2025 | 4,177 | 121,541 | 2026 is partial (last accident 2026-10-06) and label-incomplete: 46.97% of its code-03/04 rows lack DAYS_LOST (2025: 0.09%), closing document present for 33.45% of rows (2025: 55.1%), degree mix shifted (03 = 46.23% vs 38.51%, 04 = 5.42% vs 9.79%, 05 = 16.97% vs 21.41%); the degree code is provisional at filing and migrates as cases close: 880 of the 939 2026 rows with neither closing document nor RETURN_TO_WORK_DT are coded 03 (93.7%; 01/02 54, 06 4, 05 0), and the September 2026 mix is 03 53.9% / 04 1.7% / 05 8.6% / 06 33.1% (n 360) against 38.5 / 9.8 / 21.4 / 28.5 for 2025, so the 06 surplus (4.6 pp) cannot supply the 05 deficit (12.8 pp) and part of the provisional 03s will become 05 (positive -> negative flips); 2026 production hours = 0.491 x 2025 |
| 3 | drop code 00 | 12,258 | 109,283 | no injured person; 99.99% have all five injury codes `?` |
| 4 | drop code 07 | 5,476 | 103,807 | illness, diagnosis-dated; the exclusion is not clean: 1,289 of the rows have DAYS_LOST > 0 (median 30 days; joint/tendon inflammation 761, heat illness 197, occupational disease NEC 135) and 393 have SCHEDULE_CHARGE > 0, contradicting the codebook definition "Occupational illness not degree 1-6"; these 1,289 compensable illness cases (2.54% of the lost-time population) are out of scope for the insurer use, while the base keeps 816 accident-dated rows with illness natures (poisoning 506, joint/tendon inflammation 267, hearing loss 29, dermatitis 12, frostbite 2; lost-time 36.0%) |
| 5 | drop code 08 | 1,001 | 102,806 | natural causes |
| 6 | drop code 09 | 347 | 102,459 | non-employee |
| 7 | drop code 10 | 850 | 101,609 | first aid / other; 0 rows with DAYS_LOST > 0 or DAYS_RESTRICT > 0, so this exclusion is clean |
| 8 | drop code ? | 0 | 101,609 | none left after step 1 |

Result: N = 101,609 rows, 16 calendar years, 8,303 distinct mines (median 3 rows per mine, p90 24, max 1,705; 37.1% of rows from mines with >= 100 rows). Overall prevalence moves only between 47.25% and 50.13%, so the 2010 lower bound does not change the overall label mix, but this hides the within-coal label drift documented in section 1 (coal 59.74% -> 70.71% while the coal share falls 50.5% -> 33.3%). Reporting rules: no change to 30 CFR Part 50 inside 2010-2025 is recorded in the codebooks (Accidents codebook dated 2014) or in `domain_research.md`, and the 15-minute immediate-notification rule (30 CFR 50.10) predates the window. Code lists did change inside the window: the OCCUPATION and EQUIP_MFR lists were revised around January 2014 (OCCUPATION code 054 is labelled "Discontinued 1/31/2014. Several other codes cover Scoop Car Operator", 533 base rows in 2010-2014 only; seven other titles with >= 100 rows first appear in 2013-2014: Plumber/Pipefitter/Millwright 585, Groundman/Yardman 303, Steel worker 278, Dry screening plant operator 253, Ball/Rod mill operator 183, Bagging operator 122, Lab Technician 111; seven manufacturer names with >= 100 rows first appear in 2013-2014: Volvo 313, Sandvik 202, DeWalt 177, Milwaukee 146, Peterbilt 131, Bobcat 130, JLG 119), and IMMED_NOTIFY_CD 13 "Not marked" (77 rows, 17 / 30 / 30 in 2010-2012, 92.2% positive) does not occur after 2012. All other categorical features (NATURE, BODY_PART, SOURCE, ACCIDENT_TYPE, ACTIVITY, CLASSIFICATION, MINING_EQUIP, UG_*, SUBUNIT, FIPS_STATE) have no level with >= 100 rows that starts after 2012 or ends before 2022. Handling: E28 and E29 in section 5; level drift is part of the temporal-split checks in section 7.

Label completeness inside the window: the DAYS_LOST test used for 2026 (blank DAYS_LOST among code 03/04 rows: 46.97% in 2026, 2 rows in 2024, 2 in 2025) does not detect late re-coding, so the completeness check is the quarter-level degree mix. 2025 Q4 (n 1,097, 1.08% of the window) shows the 2026 signature in weaker form: 03 43.1% vs 37.2% in 2025 Q1-Q3, 04 7.9% vs 10.3%, 05 18.0% vs 22.4%, prevalence 53.2% vs 49.2%; by month the 05 share is 15.0-19.9% in Oct-Dec against 20.0-25.2% in Jan-Sep, whereas for 2018-2024 every Q4 vs Q1-Q3 gap is within 3.5 pp with no consistent sign, and only 1 of the 560 2025 Q4 code-03/04 rows lacks DAYS_LOST. Roughly 5% of the 2025 Q4 rows (about 50-65, 0.05% of the window) are therefore likely to be re-coded 9-12 months after the snapshot, some flipping 05 <-> 03/04. Decision: keep 2025 Q4 (ending the window at 2025-09-30 would remove 1,097 rows to avoid a 0.05% label-churn exposure) and state the exposure in section 9. The 2026 cut remains a judgement call; an alternative is to drop cases filed within about 90 days of the snapshot.

## 3. Join strategy

| # | Left | Right | Keys | how | validate | indicator | Rows before -> after | Matched | Unmatched | Action |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | Accidents (final base) | Mines | MINE_ID (7-char zero-padded str) | left | many_to_one | `_m_mines` | 101,609 -> 101,609 | 101,609 (100.0%) | 0 | assert row count unchanged; assert COAL_METAL_IND agrees (100.0%); keep only the allowed mine columns below |
| J2 | Accidents | Production aggregated to mine-year: `groupby(MINE_ID, CALENDAR_YR).agg(hours=sum ANNUAL_HRS, employees=sum AVG_ANNUAL_EMPL)` plus prior-year case counts from the full Accidents file restricted to cases filed before the row's ACCIDENT_DT (filing date = DOCUMENT_NO Julian date, positions 2-8) | MINE_ID and CAL_YR - 1 = CALENDAR_YR | left | many_to_one | `_m_prev` | 101,609 -> 101,609 | 100,403 (98.813%), all with hours > 0 | 1,206 (lost-time rate 53.65% vs 48.52% matched) | keep rows; exposure features NaN; indicator kept as feature `prev_year_matched` |
| J3 (EDA only) | Accidents | same mine-year aggregate | MINE_ID and CAL_YR = CALENDAR_YR | left | many_to_one | `_m_same` | 101,609 -> 101,609 | 101,477 (99.87%) | 132 | injury rates per 200,000 h by year for EDA; never a model feature (year unfinished at filing) |
| J4 | Accidents | production subunit set per mine-year (a column of the J2 aggregate) | MINE_ID and CAL_YR - 1 = CALENDAR_YR (same key as J2) | left | many_to_one | `_m_prev` | 101,609 -> 101,609 | 100,403 (98.81%) | 1,206 -> fallback to the accident SUBUNIT_CD (01/02 -> Underground; 30/99/17 -> Facility; else Surface) | derive `type_at_time` (01 or 02 in the set -> Underground; set within {30, 99, 17} -> Facility; else Surface); agrees with the same-year derivation for 99.43% of the 100,362 rows that have both and with CURRENT_MINE_TYPE for 87.67% (same-year version 87.73%); the same-year derivation (99.87% coverage) is EDA-only because for the 25.05% of accidents in Q1 no quarter of year t has been reported at filing (Form 7000-2 is quarterly, due 15 days after quarter end, 30 CFR 50.30), whereas Q4 of t-1 is due by 15 January |

Production key uniqueness (MINE_ID, CALENDAR_YR, SUBUNIT_CD): 658,922 of 658,922; 370,794 mine-years; `validate="many_to_one"` holds for every join above. The subunit-level same-year join (99.2% in the 2010+ profile, r = 0.05 with the target) is not used. Previous-year exposure is the time-safe variant: at filing time the current year's hours are unreported (Form 7000-2 is quarterly) and the current year's case count would include the row itself.

Temporal caveat for CURRENT_* mine attributes (Mines.txt is a snapshot of 2026-10-09, not of the accident year):

| Mines column | Evidence on the final base | Decision |
|---|---|---|
| CURRENT_MINE_STATUS, CURRENT_STATUS_DT | status date post-dates the accident for 40.43% of rows; only 71.62% of rows are at mines now Active; lost-time rate 46.5% (Active) vs 51.8-60.7% (abandoned/non-producing) is an accident-year effect | drop (future state) |
| CURRENT_MINE_TYPE | differs from type-at-time for 12.27% of rows | replace by `type_at_time` from the t-1 subunit set (J4) with the accident SUBUNIT_CD as fallback |
| COMPANY_TYPE, CURRENT_CONTROLLER_BEGIN_DT | controller begin date <= ACCIDENT_DT for 73.29%; a dated "Unknown" level would be a training-only artefact that encodes accident age: 41.2% of 2010 rows, 3.0% of 2025 rows (34.4% in 2010-2019 vs 8.0% in 2020-2025), 0% at deployment where the current attribute is always known; median gap between controller-begin date and accident among those rows 1,832 days; its lost-time rate (53.02% vs 46.97%) is commodity composition (within coal 64.9 vs 61.9, within metal 37.1 vs 39.0); raw COMPANY_TYPE has Cramer's V 0.044 (levels Other 51,142, Corporation 48,195, Partnership 2,135, Sole Proprietor 135, blank 2) | drop: neither the raw nor the dated version is both time-safe and informative |
| CURRENT_103I, CURRENT_103I_DT | in force (date <= ACCIDENT_DT) or "Never Had 103I Status" for 77.99%; blank 7.03%; 8,091 rows are "Removed From 103I Status" with a removal date after the accident (median 4.3 years later); the mine may have been in 103I at the accident, but the placement date is not in the file, so this cannot be checked (lost-time 61.5% vs 56.6% when removed before the accident); a date-rule-only "Unknown" level is 26.5% of 2010 rows and 14.3% of 2025 rows (24.1% vs 16.9% across the temporal split) and its lost-time gap (54.34% vs 46.96%) is again composition (within coal 62.8 vs 63.2, within metal 37.8 vs 38.7) | `status_103i_at_accident` with four levels: "Never had 103I" (61,397 rows, 60.42%, lost-time 42.27%); "In 103I at accident" (9,344, 9.20%, 69.03%: 5/10/15-day, Hazard and Ignition statuses entered on or before the accident date); "Formerly 103I" = removed before the accident (8,500, 8.37%, 56.60%); "Unknown" = any status dated after the accident or blank (22,368, 22.01%, 54.34%: 8,091 "Removed" statuses dated after the accident, 7,135 other later-dated statuses, 7,142 blank; 24.1% of 2010-2019 rows vs 16.9% of 2020-2025; at deployment only the blank statuses remain). Implementation review F1 moved the 8,091 removed-after rows from "In 103I at accident" to "Unknown": that level used a fact that happened years after the accident and could not be checked. Cramer's V 0.173; within coal the levels run 56.7-70.9%, within metal 37.8-45.1%. Temporal-split metrics are reported with and without the feature |
| NO_EMPLOYEES, DAYS_PER_WEEK, HOURS_PER_SHIFT, PROD/MAINT_SHIFTS_PER_DAY | 0 or blank for 20.05% (NO_EMPLOYEES) and 26.6% (HOURS_PER_SHIFT) of rows: zeros encode later closure | drop; size comes from previous-year production |
| MILES_FROM_OFFICE | 0 = not recorded (69.6% of abandoned mines); r within commodity -0.03 | drop |
| SAFETY_COMMITTEE_IND, PART48_TRAINING, PORTABLE_OPERATION | undated current flags; lost-time rate Y 61.74% vs N 46.12%; PART48 Y 39.29% vs N 53.71% | keep with documented caveat |
| LATITUDE, LONGITUDE | valid inside lat 17.5-72, lon -180 to -64 (plus AS) for 99.80%; 3 present-but-invalid pairs | keep, invalid -> NaN |
| COAL_METAL_IND, PRIMARY_CANVASS, PRIMARY_SIC, STATE, DISTRICT, FIPS county | stable attributes, 0 blank after join | allowed (keep one of COAL_METAL_IND/PRIMARY_CANVASS as default; PRIMARY_SIC 100 levels and DISTRICT 13 levels as alternatives) |

Allowed mine columns after J1: PRIMARY_CANVASS (5 levels), PRIMARY_SIC (100, optional), DISTRICT (13, optional alternative to FIPS_STATE_CD), LATITUDE, LONGITUDE (validated), PORTABLE_OPERATION, SAFETY_COMMITTEE_IND, PART48_TRAINING, status_103i_at_accident, type_at_time. COAL_METAL_IND and state are taken from Accidents (identical). company_type_at_accident was removed after review (table above).

Inference lookup: the eleven mine-level numeric features and type_at_time are keyed on (MINE_ID, CAL_YR - 1), so the Streamlit inference page cannot compute them from the form alone. `mlda-train` persists two small artefacts: `data/processed/mine_year_features.parquet`, one row per (MINE_ID, CALENDAR_YR) for 2009-2025 with hours, employees, subunit set and derived type (226,182 mine-years across 24,797 mines in the production file; 107,740 when restricted to the 8,303 base mines), and `data/processed/mine_case_history.parquet`, one row per Accidents case at those mines with MINE_ID, CAL_YR, filing date, DEGREE_INJURY_CD, return-to-work date and the filing date of the closing document, so that the FeatureEngineer can rebuild the filed-before-ACCIDENT_DT lagged counts for any accident date. The inference form falls back to manual entry of previous-year hours, employees and case counts when the mine is not in the table.

## 4. Feature inventory

Rare-level grouping: `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001)` (0.1% of rows = 102 on the full window, about 81 on the 80% training split). Counts below use 100 rows on the full window. Description columns replace codes for OCCUPATION (243 codes -> 195 titles, sector-prefix duplicates; the title "Discontinued 1/31/2014. Several other codes cover Scoop Car Operator", code 054, 533 rows in 2010-2014 only, is mapped to the infrequent bucket, E28) and EQUIP_MFR (144 codes -> 138 names, 3-digit and 4-digit series). Both lists were revised around January 2014 (section 2), so a few one-hot columns are zero for the whole temporal test period; this is checked, not corrected.

### 4a. Categorical codes (Accidents)

| Column | Levels | Levels >= 100 rows | Rows in levels < 100 | Top-1 share | Cramer's V (2010+ base) | Placeholder handling | Known at filing? |
|---|---|---|---|---|---|---|---|
| NATURE_INJURY_CD | 27 | 21 | 143 | 32.49% (330 sprain/strain) | 0.391 | none in base; level 100 (amputation, 925 rows) is 100% lost-time: a documented deterministic level, section 9 | yes (Item on 7000-1) |
| INJ_BODY_PART_CD | 46 | 39 | 208 | 18.63% (340 fingers) | 0.366 | none | yes |
| INJURY_SOURCE_CD | 124 | 88 | 915 | 12.72% | 0.314 | none | yes |
| ACCIDENT_TYPE_CD | 41 | 30 | 350 | 13.34% | 0.286 | none | yes |
| FIPS_STATE_CD | 54 | 49 | 84 | 14.16% (54 WV) | 0.243 | none | yes |
| COAL_METAL_IND | 2 | 2 | 0 | 59.20% (M) | 0.238 | none | yes |
| OCCUPATION (title) | 195 | 76 (75 after E28) | 2,696 (3,229 after E28) | 18.43% | 0.232 (code) | none; discontinued code 054 (533 rows) -> infrequent bucket | yes |
| ACTIVITY_CD | 97 | 54 | 1,928 | 20.66% | 0.230 | `?` on 6 rows -> infrequent bucket | yes |
| CLASSIFICATION_CD | 21 | 16 | 165 | 35.13% (09 handling materials) | 0.216 | none | yes |
| MINING_EQUIP_CD | 66 | 41 | 915 | 53.82% (`?` = no equipment) | 0.205 | keep `?` as level "none" (lost-time 50.75% vs 46.07% when coded) | yes |
| UG_MINING_METHOD_CD | 8 | 7 | 22 | 66.93% (`?`) | 0.192 | keep `?` as one level (structural for surface; 927 genuinely missing in subunit 01) | yes |
| SUBUNIT_CD | 10 | 9 | 84 | 33.99% (01 underground) | 0.184 | none | yes |
| UG_LOCATION_CD | 8 | 7 | 19 | 66.01% (`?`) | 0.175 | keep `?` as level (0 `?` in subunit 01, 67,077 = all non-underground rows) | yes |
| EQUIP_MFR_NAME | 139 | 30 | 2,265 | 53.83% (`NO VALUE FOUND`) | 0.155 (code) | keep as level; "Not Reported" 21,478 and "Not on this list" 7,688 are own levels | yes |

Estimated one-hot width at the 100-row threshold: 468 retained levels (469 minus the discontinued occupation title) + 13 infrequent buckets + 23 mine/derived levels (PRIMARY_CANVASS 5, type_at_time 3, status_103i_at_accident 4, three Y/N flags 6, mine_size_band 5) = about 504 columns.

### 4b. Numeric raw (Accidents)

| Column | n present | Missing % | Median | p99 | Max | Skew (2010+ base) | r (point-biserial) | Known at filing? |
|---|---|---|---|---|---|---|---|---|
| TOT_EXPER (years) | 100,387 | 1.20 | 6.0 | 41 | 66 | 1.37 | 0.063 | yes |
| MINE_EXPER | 100,331 | 1.26 | 2.58 | 37.3 | 58 | 2.26 | -0.012 | yes |
| JOB_EXPER | 100,736 | 0.86 | 3.0 | 37.2 | 66 | 2.02 | 0.014 | yes |
| NO_INJURIES | 101,609 | 0.00 | 1 | 1 | 36 | 48.6 | 0.029 | yes (917 rows > 1, 33 rows > 10) |

ACCIDENT_TIME and SHIFT_BEGIN_TIME (HHMM integers with 9999 = not reported and 2400 = midnight) are never used raw; only the derived features below. CAL_YR is not a feature of the primary model (r = 0.011 overall; extrapolation risk at deployment), but the within-coal label drift (r = 0.074 within coal, section 1) makes CAL_YR or a coal x year term a documented candidate for the temporal-split variant, with the extrapolation caveat stated; it is used for filtering and for the temporal split.

### 4c. Engineered features

| Name | Formula | Source columns | Known at filing? | Coverage / signal on final base |
|---|---|---|---|---|
| accident_hour | floor(t / 100) for t in 0000-2359 with minutes < 60; 2400 -> 0; 9990-9999 -> NaN | ACCIDENT_TIME | yes | 98,518 rows (96.96%); r = 0.012; lost-time 54.5% at 00-05 h and 18-23 h vs 45.7% at 06-11 h (2010+ base) |
| is_night | accident_hour in {0..5, 18..23} | ACCIDENT_TIME | yes | 29,425 of 101,656 valid rows in the 2010+ base |
| shift_begin_hour | same parsing as accident_hour | SHIFT_BEGIN_TIME | yes | 101,090 rows (99.49%); r = 0.076 |
| hours_into_shift | ((acc_min - shift_min) mod 1440) / 60, where acc_min = 60*hh + mm; the modulo handles shifts that wrap past midnight (11,538 rows have accident clock time earlier than shift start); values > 16 h -> NaN | ACCIDENT_TIME, SHIFT_BEGIN_TIME | yes | 98,414 rows with both times; median 4.5 h, p95 10.08 h; 662 rows > 16 h set NaN; r = -0.020 |
| accident_weekday | ACCIDENT_DT.weekday (0 = Monday) | ACCIDENT_DT | yes | Mon 19,021 ... Fri 15,344, Sat 7,711, Sun 4,624 |
| is_weekend | weekday in {5, 6} | ACCIDENT_DT | yes | 12,335 rows (12.14%) |
| accident_month | ACCIDENT_DT.month | ACCIDENT_DT | yes | r = -0.009 (2010+ base) |
| accident_quarter | CAL_QTR | CAL_QTR | yes | V = 0.010; redundant with month, optional |
| log1p_tot_exper | log1p(TOT_EXPER) | TOT_EXPER | yes | skew 1.37 -> about 0 |
| exper_ratio_mine | MINE_EXPER / TOT_EXPER after the consistency rule (clip 0-1); NaN when either input is missing or TOT_EXPER <= 0 | MINE_EXPER, TOT_EXPER | yes | 0 zero denominators in the base (TOT_EXPER min 0.02), so the guard matters only for free inference input (E30) |
| exper_ratio_job | JOB_EXPER / TOT_EXPER after the consistency rule (clip 0-1); same guard | JOB_EXPER, TOT_EXPER | yes | |
| new_to_mine | MINE_EXPER < 1 | MINE_EXPER | yes | 28,889 rows (28.43%); lost-time 47.83% vs 48.88% otherwise |
| is_contractor | CONTRACTOR_ID != "" | CONTRACTOR_ID | yes | 12,667 rows (12.47%); lost-time 43.82%; V = 0.036 |
| immed_reportable | IMMED_NOTIFY_CD present (not a placeholder; `notna()` after `load.py` masking) and != "13" ("Not marked", 77 rows in 2010-2012 only, E29) | IMMED_NOTIFY_CD | yes (15-minute notification precedes the form) | 3,665 rows (3.61%); lost-time 74.41%; 560 of them are code 01 death (100% positive), 69.79% positive without them; never use the raw code |
| no_injuries_capped | min(NO_INJURIES, 10) | NO_INJURIES | yes | 33 rows capped |
| underground | SUBUNIT_CD == "01" | SUBUNIT_CD | yes | 34,532 rows (33.99%); lost-time 60.7%; redundant with the one-hot, optional for linear models |
| type_at_time | from the production subunit set at (MINE_ID, CAL_YR - 1) (J4); fallback = accident SUBUNIT_CD (01/02 -> Underground, 30/99/17 -> Facility, else Surface) | MinesProdYearly SUBUNIT_CD, SUBUNIT_CD | yes (Q4 of t-1 is due 15 days after year end) | 98.81% from production + 1,206 fallback rows = 100%; Underground 40,379 rows (lost-time 58.71%), Surface 51,053 (41.98%), Facility 10,177 (41.53%); agrees with the same-year derivation for 99.43% of rows |
| prev_hours | sum of ANNUAL_HRS over subunits at (MINE_ID, CAL_YR - 1) | MinesProdYearly | yes | 100,403 rows (98.81%); r(log1p) = 0.036 |
| prev_employees | sum of AVG_ANNUAL_EMPL at (MINE_ID, CAL_YR - 1) | MinesProdYearly | yes | 98.81% |
| prev_hours_per_employee | prev_hours / prev_employees | derived | yes | r = 0.040 |
| mine_size_band | `pd.cut(prev_employees, [0, 5, 20, 50, 200, inf], right=False, labels=["1-4", "5-19", "20-49", "50-199", "200+"])` (left-closed bins; 1,036 / 733 / 403 / 39 matched rows sit exactly on 5 / 20 / 50 / 200 and would change band under the pandas default right=True) | prev_employees | yes | n 2,285 / 14,180 / 16,654 / 30,483 / 36,801; lost-time 49.8 / 47.8 / 48.2 / 42.7 / 53.7% |
| prev_incidents | count of Accidents rows (any code) at (MINE_ID, CAL_YR - 1) filed before ACCIDENT_DT (filing date = DOCUMENT_NO Julian date); 0 when the mine has production but no such case | Accidents, MinesProdYearly | yes | 100,403 rows; 17,822 true zeros (strict `<` filing rule; a `<=` cut gives 17,820); median 6, p95 66; r = 0.102 |
| prev_incident_rate_200k | prev_incidents / prev_hours x 200,000, winsorised at 50 then log1p | derived | yes | median 4.22, p95 15.35, max 20,000; 364 rows > 50 (764 rows have prev_hours < 2,000); r(log1p) = 0.075 |
| prev_lost_time_rate_200k | lost-time cases (01-04) at t-1 filed before ACCIDENT_DT / prev_hours x 200,000, winsorised at 50 then log1p | derived | yes | r(log1p) = 0.207 |
| prev_lost_time_share | lost-time cases / valid-code cases (01-06) at t-1 filed before ACCIDENT_DT; NaN when 0 such cases | derived | yes | 81,469 rows (80.18%, all with a production match; the earlier draft's 81,525 / 81,471 used `<=`); ROC-AUC 0.671; r = 0.295 (0.240 coal, 0.197 metal within commodity); lost-time 31.5% at share 0, 29.8% at 0-0.25, 41.4% at 0.25-0.5, 59.6% at 0.5-0.75, 77.0% at 0.75-0.999, 61.3% at exactly 1 (small mines) |
| prev_year_matched | `_m_prev == "both"` | J2 indicator | yes | 1,206 zeros |
| ytd_lost_time_share (optional, off by default) | lost-time share of same-mine cases in year t (codes 01-06) filed before ACCIDENT_DT; NaN when none | Accidents | yes | 65,321 rows (64.29%); ROC-AUC 0.664; r = 0.290; makes same-year mine history an explicit time-safe feature instead of letting the model absorb it from future same-year rows (section 7) |
| latitude, longitude | validated coordinates (section 3) | Mines | yes | 99.80%; r(lon) = 0.133, the Appalachian coal gradient |

Strictness of the lagged case features (recomputed): of the 1,600,584 (row, t-1 case) pairs at the same mine (82,781 rows have >= 1 prior case), 1.87% of the prior cases were filed after the row's ACCIDENT_DT, affecting 11.7% of the rows with prior cases and 21.2% of Q1 rows (22.58% of cases occur in Q4); 12.8% of the referenced t-1 lost-time cases were filed, closed or returned to work after the accident date (26.2% for Q1 rows), and the initial degree code is provisional (section 2). The default therefore counts only t-1 cases filed strictly before ACCIDENT_DT (272 (row, case) pairs are filed on the accident date itself and are excluded; every count in this section uses this strict rule): compared with the year-total variant, prev_lost_time_share changes for 7,743 rows (mean |delta| 0.044), is lost for 115 rows and keeps ROC-AUC 0.671 (0.6708 vs 0.6710) and r 0.295; prev_incidents changes for 9,722 rows. The stricter resolved-before variant (filing date, RETURN_TO_WORK_DT and closing document all before ACCIDENT_DT) changes 28,789 rows (mean |delta| 0.080), is available for 80,656 rows and gives ROC-AUC 0.668, r 0.291; it is kept as a sensitivity variant only, because "resolved" has no clean definition for code-06 cases (a closing document exists for 33.5% of negatives). Implementation review F1 built it: the base table carries `prev_incidents_resolved`, `prev_valid_cases_resolved` and `prev_lost_time_cases_resolved` (a date that is absent counts as resolved; 30,505 rows differ from the default), and the refit of the selected model on them is in `reports/metrics/sensitivity.json` under `primary_split_variants`.

### 4d. Leakage DROP list

| Column | Reason |
|---|---|
| DEGREE_INJURY_CD, DEGREE_INJURY | define the target |
| DAYS_LOST | outcome, finalised on the closing form; > 0 for 95.3% of positives and 0.0% of negatives (V = 0.955) |
| DAYS_RESTRICT | outcome, closing form (V = 0.192) |
| SCHEDULE_CHARGE | statutory days charged to death/permanent injury; > 0 only for positives |
| TRANS_TERM | recorded on the closing form; blank rows are 97.5% positive, Y rows 78.7% |
| RETURN_TO_WORK_DT | entered when the worker returns; median lag 63 days for positives vs 1 day for negatives; 1-May batch default on 6,683 rows |
| CLOSED_DOC_NO | exists only after the case closes; present for 72.2% of positives vs 33.5% of negatives (V = 0.387) |
| INVEST_BEGIN_DT | MSHA investigation follows severity and is entered after filing (9.1% vs 4.4%) |
| NARRATIVE | free text written at filing, may contain outcome words (fatality wording 98.4% positive); excluded by project scope |
| CURRENT_MINE_STATUS, CURRENT_STATUS_DT | snapshot state; post-dates the accident for 40.43% of rows |
| NO_EMPLOYEES, DAYS_PER_WEEK, HOURS_PER_SHIFT, PROD_SHIFTS_PER_DAY, MAINT_SHIFTS_PER_DAY, AVG_MINE_HEIGHT, METHANE_LIBERATION, MINE_GAS_CATEGORY_CD, NO_*_PITS, NO_TAILING_PONDS | current MIF values; zeros encode closure after the accident year (20-27% of rows) |
| same-year production hours, employees, counts, rates | not known at filing; include the row itself; 2026 half-year |
| DOCUMENT_NO filing lag | technically known, but late filing happens when the case is still open (median 9 vs 7 days; r = 0.042) |
| CURRENT_MINE_TYPE | drifts from type-at-time for 12.27% of rows (replaced by type_at_time) |
| COMPANY_TYPE, CURRENT_CONTROLLER_BEGIN_DT (and the derived company_type_at_accident) | the dated "Unknown" level encodes accident age (41.2% of 2010 rows, 3.0% of 2025 rows, 0% at deployment); raw COMPANY_TYPE has V = 0.044 (section 3) |
| MILES_FROM_OFFICE | 0 = not recorded; no within-commodity signal |

Admin/ID drop list: MINE_ID (join key only), CONTROLLER_ID, CONTROLLER_NAME, OPERATOR_ID, OPERATOR_NAME, CONTRACTOR_ID (collapsed to is_contractor), DOCUMENT_NO, FISCAL_YR, FISCAL_QTR, CAL_YR (filter only), ACCIDENT_DT (decomposed), description twins of used codes (SUBUNIT, UG_LOCATION, UG_MINING_METHOD, MINING_EQUIP, CLASSIFICATION, ACCIDENT_TYPE, ACTIVITY, INJURY_SOURCE, NATURE_INJURY, INJ_BODY_PART, IMMED_NOTIFY) and the code twins of used descriptions (OCCUPATION_CD, EQUIP_MFR_CD), EQUIP_MODEL_NO (11,356 levels, 76.7% blank); Mines: CURRENT_MINE_NAME, CURRENT_CONTROLLER_ID/NAME, CURRENT_OPERATOR_ID/NAME, ASSESS_CTRL_NO, OFFICE_CD/NAME, CONG_DIST_CD, BOM_STATE_CD, STATE (duplicate of FIPS_STATE_CD), FIPS_CNTY_CD/NM (296 / 1,784 levels), PORTABLE_FIPS_ST_CD, DIRECTIONS_TO_MINE, NEAREST_TOWN, MINERS_REP_IND (constant N), SECONDARY_* (99.1% blank), PRIMARY_SIC_CD_1, PRIMARY_SIC_CD_SFX, PILLAR_RECOVERY_USED, HIGHWALL_MINER_USED, MULTIPLE_PITS (undated, V <= 0.04); Production: CURR_MINE_NM, STATE_ABBR, SUBUNIT_DESC, AVG_EMPLOYEE_HOURS (exact rounded ratio), ANNUAL_COAL_PROD, C_M_IND (duplicate).

## 5. Error-correction rules (counts on the final base, n = 101,609)

| # | Rule | Condition | Rows affected | Action |
|---|---|---|---|---|
| E1 | job experience exceeds total | JOB_EXPER > TOT_EXPER | 3,731 | set JOB_EXPER to NaN and set flag `exper_inconsistent` = 1 (median excess 6.0 years, p90 22.0, 3,129 exceed by > 1 year, so these are not rounding errors; a field swap is likely but unprovable, so clipping would invent values) |
| E2 | mine experience exceeds total | MINE_EXPER > TOT_EXPER | 83 | set MINE_EXPER to NaN, same flag (either rule: 3,771 rows, 3.71%) |
| E3 | implausible tenure | any experience > 60 years | 3 (TOT 3, JOB 2) | set to NaN |
| E4 | negative experience | any experience < 0 | 0 | assert |
| E5 | accident time not reported | ACCIDENT_TIME in 9990-9999 | 3,091 (all 9999) | NaN (accident_hour, hours_into_shift missing) |
| E6 | accident time midnight code | ACCIDENT_TIME = 2400 | 326 | map to 0000 |
| E7 | accident time other invalid | minutes >= 60 or > 2400 outside the codes, or blank | 0 | assert |
| E8 | shift start not reported | SHIFT_BEGIN_TIME in 9990-9999, or blank | 518 + 1 | NaN |
| E9 | shift start midnight code | SHIFT_BEGIN_TIME = 2400 | 1,833 | map to 0000 |
| E10 | lost leading zero | SHIFT_BEGIN_TIME with 3 characters (e.g. 700) | 48 (plus the 1 blank already counted in E8) | parse as integer HHMM (700 -> 07:00), do not string-slice; `load.py`'s comment that times are zero-padded HHMM strings should reference this rule |
| E11 | shift wrap past midnight | accident clock time < shift start | 11,538 | handled by (acc_min - shift_min) mod 1440; no change |
| E12 | implausible time into shift | hours_into_shift > 16 | 662 | NaN (keep accident_hour) |
| E13 | structural placeholder | UG_LOCATION_CD `?` | 67,077 (0 in subunit 01) | keep as level (not imputed) |
| E14 | structural + partial placeholder | UG_MINING_METHOD_CD `?` | 68,004 (927 in subunit 01) | keep as one level |
| E15 | no-equipment placeholder | MINING_EQUIP_CD `?`, EQUIP_MFR_NAME `NO VALUE FOUND` | 54,691 / 54,701 | keep as level "none" |
| E16 | residual placeholder | ACTIVITY_CD `?` | 6 | falls into the infrequent bucket |
| E17 | placeholder with trailing space | IMMED_NOTIFY_CD `? ` | 97,867 | strip, then flag = 0; `src/mlda/data/load.py` already strips and masks every placeholder (`""`, `?`, `? `, `NO VALUE FOUND`, `No Value Found`, `NULL`) to NaN, so every placeholder test in this document and in `decisions.json` is written as `isna()` / `fillna("none")`, never as a comparison with the literal string (a literal test on NaN would set the flag for all 97,867 placeholder rows) |
| E18 | duplicate code series | EQUIP_MFR_CD 3-digit vs 4-digit (28 four-digit codes in base) | 144 codes -> 138 names | use EQUIP_MFR_NAME |
| E19 | sector-prefixed codes | OCCUPATION_CD | 243 codes -> 195 titles | use OCCUPATION title |
| E20 | free-text model number | EQUIP_MODEL_NO blank or `?` | 77,962 (76.7%) | drop column |
| E21 | victim count outliers | NO_INJURIES > 10 | 33 | cap at 10; NO_INJURIES = 0 with an injury code: 0 rows (assert) |
| E22 | dates | ACCIDENT_DT unparsable, after snapshot, or CAL_YR/CAL_QTR inconsistent | 0 / 0 / 0 | assert only |
| E23 | invalid coordinates | outside lat 17.5-72 or lon -180 to -64 (AS exempt) | 3 pairs | NaN |
| E24 | mine attribute not yet in force | CONTROLLER_BEGIN_DT > ACCIDENT_DT; 103I date > ACCIDENT_DT and not "Never Had" | 27,142 (26.71%) / 22,368 (22.01%) | COMPANY_TYPE: feature dropped (the Unknown level tracks accident age, section 3); 103I: every status dated after the accident, "Removed From 103I Status" included (8,091), or blank -> level "Unknown" (22,368, 22.01%; review F1 replaced the earlier rule that sent the 8,091 to "In 103I at accident") |
| E25 | unstable lagged rates | prev_hours < 2,000 (rates up to 20,000 per 200k h) | 764 (369 rows with rate > 50) | winsorise rates at 50, then log1p; keep raw prev_incidents |
| E26 | type coercion | NUMBER columns read as str; codes keep zero padding (`01`, `0200024`) | all | explicit `to_numeric`; codes stay str |
| E27 | whitespace | trailing spaces in codes and descriptions | all columns | `str.strip()` at load |
| E28 | discontinued code list entry | OCCUPATION code 054 / title "Discontinued 1/31/2014. Several other codes cover Scoop Car Operator" | 533 (129 / 164 / 140 / 98 / 2 in 2010-2014) | map to the "missing" level of OCCUPATION (the successor codes are not identified in the codebook). A stateless step cannot reach the one-hot infrequent bucket: any fixed label still has 533 rows (0.66% of training rows, above `min_frequency` 0.001), so it becomes its own one-hot column `OCCUPATION_missing`. That column is non-zero only in 2010-2014 (lost-time 56.7% vs 48.3% for all rows of those years) and is all zero in the temporal test period; `temporal_split.json` reports it under `level_drift` (`e28_retired_occupation_column_all_zero`). Review F1 accepted this realisation instead of a fitted pooling transformer |
| E29 | retired notification code | IMMED_NOTIFY_CD = 13 "Not marked" | 77 (2010-2012 only; 92.2% positive) | immed_reportable = 0 |
| E30 | zero or missing denominator | TOT_EXPER <= 0 or missing in exper_ratio_mine / exper_ratio_job | 0 in the base (TOT_EXPER min 0.02) | ratio = NaN (guard for free inference input; without it x / 0 -> inf -> clip -> 1.0) |

Leak-column oddities (RETURN_TO_WORK_DT before accident 1 row, 1-May batch default 1,736 rows in the 2010+ base, DAYS_LOST > 365 583 rows) need no rule because the columns are dropped; descriptive EDA on them must exclude those values.

## 6. Missing-value plan

All imputers are fitted on the training split inside `ColumnTransformer` with `SimpleImputer(strategy="median", add_indicator=False)`; the missing indicators are the seven explicit flags listed below (TOT_EXPER_missing, exper_inconsistent, accident_hour_missing, hours_into_shift_missing, prev_year_matched, prev_share_missing, coords_missing). `add_indicator=True` was dropped because prev_hours, prev_employees, prev_hours_per_employee, prev_incidents and both rates are missing on exactly the same 1,206 rows (0 matched rows have hours = 0), so it would emit six identical copies of 1 - prev_year_matched plus duplicates of prev_share_missing and TOT_EXPER_missing: harmless for trees, but it inflates KNN distances and adds collinear columns for LogReg. KNNImputer and IterativeImputer were considered and rejected: about 504 one-hot columns on 100k rows for numeric gaps of at most 4%.

| Group | Columns | Missing on final base | Mechanism argument | Imputer | Indicator |
|---|---|---|---|---|---|
| Worker experience | TOT_EXPER, MINE_EXPER, JOB_EXPER (+ 3.7% set NaN by E1/E2) | 1,222 (1.20%) / 1,278 (1.26%) / 873 (0.86%); any 1,877 (1.85%); all three 473 | not MCAR: lost-time rate 58.67% when TOT_EXPER is blank vs 48.46% when present; missingness fell from 2.33% (2010) to < 1% (2017+), so reporting practice drives it (MAR given year/mine, with an MNAR component) | median (skew 1.4-2.3) | yes, `TOT_EXPER_missing`, plus `exper_inconsistent` |
| Time of day | accident_hour, shift_begin_hour, hours_into_shift | 3,091 (3.04%) / 519 (0.51%) / 3,195 + 662 (3.80%) | 9999 = "not reported" by the operator; MAR given mine and year | median | yes, `accident_hour_missing`, `hours_into_shift_missing` (no flag for the 0.51% shift_begin_hour gap) |
| Structural categorical | UG_LOCATION_CD, UG_MINING_METHOD_CD, MINING_EQUIP_CD, EQUIP_MFR_NAME | 66.0% / 66.9% / 53.8% / 53.8% `?` | MNAR by design: not applicable for surface subunits or no-equipment accidents (0 `?` for UG_LOCATION in subunit 01) | none: `?` is its own level | implicit |
| Other categorical | ACTIVITY_CD and any level unseen in training | 6 rows | negligible | `handle_unknown="infrequent_if_exist"` | no |
| Dated mine attribute | status_103i_at_accident | 22,368 (22.01%) "Unknown" (103I status dated after the accident, "Removed" included, or blank); 24.1% of 2010-2019 rows vs 16.9% of 2020-2025; only the blank statuses (7.03%) remain at deployment | unknown-at-time by construction (training-only level) | level "Unknown"; temporal-split metrics reported with and without the feature | implicit |
| Coordinates | latitude, longitude | 199 rows (0.20%) | old records with blank or malformed coordinates (WV, PA, KY) | median | yes, `coords_missing` |
| Previous-year exposure | prev_hours, prev_employees, prev_hours_per_employee, prev_incidents, both rates | 1,206 (1.19%), identical rows for all six columns | mine reported no hours in t-1 (new, idle or contractor-only); lost-time rate 53.65% vs 48.52% (not significant in the 2010+ profile, z = 1.63) | median | yes, the single flag `prev_year_matched` |
| Previous-year lost-time share | prev_lost_time_share | 20,140 (19.82%) with the strict filed-before default (the `<=` cut gave 20,084; 19,969 with year totals) | MNAR: missing exactly when the mine had no valid-code case filed in t-1 before the accident date (quiet or small mines); lost-time rate 47.7% in the missing group vs 29.8-77.0% across share bins | median (0.5) | yes, `prev_share_missing` |
| Dropped for sparsity (> 60% missing) | EQUIP_MODEL_NO 76.7%; AVG_MINE_HEIGHT 60% (coal only); SECONDARY_* 99%; MINE_GAS_CATEGORY_CD 99.6%; METHANE_LIBERATION 89%; pits and ponds 65-91% | | | drop | |

## 7. Duplicates and grouping plan

| Check (final base) | Result | Action |
|---|---|---|
| Exact duplicate rows (all 57 columns) | 0 | none |
| Duplicates ignoring DOCUMENT_NO | 0 | none |
| Duplicates ignoring DOCUMENT_NO, CLOSED_DOC_NO, NARRATIVE | 5 | keep: one Form 7000-1 per injured person, so identical circumstances are distinct victims |
| DOCUMENT_NO unique | 101,609 of 101,609 | assert |
| Group key `MINE_ID + CAL_YR` (mine-year) | 32,732 groups; median 1 row (17,802 single-row mine-years), p90 6, max 182; 16 mine-years with > 100 rows hold 2,067 rows; 83,587 rows (82.26%) have >= 1 same-mine-year sibling on a different date | group id for the split and the CV folds |
| Nested key `MINE_ID + ACCIDENT_DT` (multi-victim) | 98,693 groups; 2,614 multi-row groups holding 5,530 rows (5.44%); max 33 rows; 883 groups with mixed labels; every same-day group lies inside one mine-year | kept together automatically by the mine-year split; used as the finer multi-victim key in tests |
| Finer key `+ ACCIDENT_TIME` | 451 multi-row groups, 1,050 rows (1.03%), 103 mixed | not used: 9999 time codes both over- and under-group |
| NO_INJURIES as victim counter | disagrees with group size in both directions (2010+ profile) | not used |

Split plan: `group_id = MINE_ID + "_" + CAL_YR`. Reason for grouping by mine-year rather than by MINE_ID + ACCIDENT_DT: under the same-day key, rows of the same mine in the same calendar year, including cases that occur after the predicted case, straddle train and test (about 80% of the 82.26% of rows with same-mine-year siblings would have their siblings in train), and a model that can identify the mine-year absorbs their outcomes, which are not available at filing. Recomputed on the final base: the mean label of the later-dated same-mine-year siblings alone ranks a row with ROC-AUC 0.6665 (n 68,524; coal 0.640, metal 0.612), the same as the past-only siblings (0.6688) and as strong as the strongest planned feature (prev_lost_time_share, 0.671); the tuple (latitude, longitude, prev_hours, prev_employees), all planned numeric features, identifies a single mine-year for 99.85% of rows (32,670 distinct tuples vs 32,732 mine-years), so KNN, RF and HGB can memorise it and would be favoured over linear models by inflated held-out metrics. Hold out 20% with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the mine-year groups (one fold = test), then 5-fold `StratifiedGroupKFold` on the remaining 80% as the validation scheme for GridSearchCV; persist indices in `data/processed/split_indices.json`. Stratification is needed and feasible: an unstratified random assignment of mine-years to five folds gives fold prevalence 46.47-50.92%, while the median group holds one row. Tests: no group_id in both train and test; class share within 1 pp of 48.6% in each split; no MINE_ID + ACCIDENT_DT group split across sets (implied by nesting). The temporal split (train 2010-2019, n = 71,877: coal 31,533, metal 40,344; test 2020-2025, n = 29,732: coal 9,928, metal 19,804; 91.88% of test rows are at mines seen in training, so mine history stays available) is reported side by side with the primary split for every criterion S1-S6, so that any remaining inflation is visible; it also carries the level-drift check (section 2) and the coal label drift (coal prevalence 61.19% vs 68.91%, section 1), with metrics and calibration by sector. Grouping by MINE_ID alone (8,303 groups; 37.1% of rows from mines with >= 100 rows) remains the strictest robustness variant. Same-year-to-date mine history, if wanted, enters only through the explicit time-safe feature ytd_lost_time_share (section 4c, ROC-AUC 0.664), never through shared mine-year rows.

## 8. Business framing

| Item | Content |
|---|---|
| Decision | At filing of Form 7000-1 (within 10 working days of the injury, 30 CFR 50.20), flag the case for early intervention: set an indemnity reserve, assign a return-to-work coordinator, start medical case management. The model scores Sections A-C of the form (circumstances, worker, mine) only, independent of the filer's Section D knowledge: for 45,679 rows (44.96%) the worker had already returned to work on or before the filing date (positive rate 20.35%; by degree 06 98.7%, 05 25.6%, 03 22.3%, 04 8.2%, 02 17.5%, 01 0.4%), for the other 55,930 rows (positive rate 71.64%) the outcome was still open, so the filer often holds outcome knowledge the model does not use; metrics are optionally reported for the two groups separately. MSHA inspection targeting (mine-level aggregates) is served by the DART sensitivity label, not by the primary label (section 1) |
| Stakeholders | mine operator safety officer (completes the form); workers'-compensation insurer or self-insured operator (claims adjuster); MSHA district office (DART label only) |
| Positive class | case will be classified 01-04 (fatal, permanent, days away) = the indemnity trigger, which depends on the operator's restricted-duty practice (section 1) |
| C_FP | USD 200: an unnecessary extra case review, 2-4 adjuster hours at USD 37.51/h (BLS, May 2025) loaded x1.4, plus about 1 hour of safety-officer time (range USD 105-260) |
| C_FN | USD 4,000, an assumption: the incremental cost of recognising a lost-time case late, taken as 5-10% (a judgement, not a cited share) of the NCCI AY2025 average lost-time claim of USD 61.9k (indemnity 31.3k + medical 30.6k) = USD 3.1-6.2k. The NCCI report-lag study (USD 13,210 -> 17,785 -> 19,936 when reporting to the insurer slips to week 3-4) is an analogy only: every case here is already filed with MSHA within 10 working days, and the anchor mixes a 2015 all-claim median with a 2025 lost-time mean |
| Cost ratio | 20:1 is the base assumption (the anchors above span 12:1 to 45:1, but they are not measurements of a missed case at filing time); sensitivity set 2:1, 3:1, 5:1, 10:1, 20:1, 30:1, exposed as a parameter in the threshold analysis. At 48.6% prevalence the threshold is non-trivial only at about 5:1 or below: no nature, body-part or source level with >= 100 rows has a lost-time rate under 0.079 (minima 0.158 / 0.096 / 0.079), 0 rows sit in nature x body-part x coal cells (n >= 100) below 0.0476 and only 1,087 rows (1.07%) below 0.0909 (lowest cell with n >= 500: 0.0871), so at >= 10:1 a calibrated model flags every case |
| Implied threshold on a calibrated probability | p* = C_FP / (C_FP + C_FN) = 0.3333 (2:1); 0.25 (3:1); 0.1667 (5:1); 0.0909 (10:1); 0.0476 (20:1); 0.0323 (30:1) |
| Primary selection metric | PR-AUC (average precision), with ROC-AUC, recall/TPR, specificity/TNR, precision, F1, balanced accuracy, MCC, Brier and expected cost per 1,000 cases co-reported; CV mean +/- std kept separate from the held-out test |

Baselines on the final window (n = 101,609; cost per 1,000 cases = 1000 x (FN x C_FN + FP x C_FP) / N with C_FP = 200 and C_FN = 200 x ratio):

| Rule | Flag rate % | Recall % | Specificity % | Precision % | F1 | Balanced acc. % | Cost/1,000 at 2:1 | 3:1 | 5:1 | 10:1 | 20:1 | 30:1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Majority class (all negative) | 0.0 | 0.0 | 100.0 | 0.0 | 0.000 | 50.0 | 194,341 | 291,512 | 485,853 | 971,705 | 1,943,411 | 2,915,116 |
| All positive (review every case) | 100.0 | 100.0 | 0.0 | 48.6 | 0.654 | 50.0 | 102,829 | 102,829 | 102,829 | 102,829 | 102,829 | 102,829 |
| Coal mine | 40.8 | 52.9 | 70.7 | 63.0 | 0.576 | 61.8 | 121,611 | 167,336 | 258,784 | 487,406 | 944,649 | 1,401,892 |
| Classification in {18 slip/fall, 12 powered haulage, 07 roof fall, 19 stepping on object, 06 fall of face} | 35.2 | 43.8 | 72.9 | 60.5 | 0.508 | 58.4 | 136,994 | 191,579 | 300,751 | 573,679 | 1,119,537 | 1,665,394 |
| Nature in {330 sprain/strain, 220 fracture, 160 contusion, 400 unclassified, 170 crushing} | 63.3 | 77.4 | 50.0 | 59.4 | 0.672 | 63.7 | 95,334 | 117,307 | 161,251 | 271,114 | 490,838 | 710,563 |
| Body part not in {340 finger, 330 hand, 130 eye, 320 wrist} | 68.4 | 81.5 | 43.9 | 57.8 | 0.677 | 62.7 | 93,692 | 111,687 | 147,676 | 237,648 | 417,593 | 597,538 |
| Nature level with lost-time rate > 50% (17 of 27 levels, in-sample) | 69.5 | 86.7 | 46.8 | 60.6 | 0.713 | 66.7 | 80,593 | 93,517 | 119,365 | 183,986 | 313,226 | 442,467 |
| Not {180 cut/laceration, 320 dust in eyes, 310 scratch} | 72.9 | 89.6 | 42.9 | 59.7 | 0.717 | 66.3 | 78,869 | 88,943 | 109,091 | 159,460 | 260,199 | 360,938 |
| Nature rule or coal | 81.4 | 92.8 | 29.3 | 55.4 | 0.694 | 61.1 | 86,610 | 93,586 | 107,538 | 142,417 | 212,174 | 281,932 |
| Nature x coal cell rate > 50% (in-sample) | 35.5 | 55.6 | 83.5 | 76.1 | 0.642 | 69.5 | 103,367 | 146,558 | 232,940 | 448,895 | 880,806 | 1,312,716 |
| Nature x body part x coal cell rate > 50% (in-sample) | 41.6 | 65.3 | 80.9 | 76.4 | 0.704 | 73.1 | 87,043 | 120,739 | 188,131 | 356,610 | 693,568 | 1,030,527 |

Ranking baselines (lookup of cell lost-time rates): in-sample ROC-AUC nature 0.701, nature x coal 0.771, nature x body part x coal 0.810 (886 cells), nature x body part x source x coal 0.860 (12,754 cells, overfits). Temporal holdout for the 3-field lookup (fit 2010-2019, score 2020-2025, n 29,732, prevalence 49.54%; unseen cells fall back to nature x coal, then nature): ROC-AUC 0.794; recall 0.750 at a 50% flag rate (specificity 0.668), 0.823 at 60% (0.580), 0.890 at 70% (0.438), 0.906 at 75% (0.403); at specificity 0.40 the threshold is 0.21 (recall 0.9095, flag rate 75.65%, cost per 1,000 at 20:1 = 240,529). At the 20:1 threshold p* = 0.0476 the same lookup flags 99.76% of cases (recall 0.9978, specificity 0.0025, cost 105,112 vs 100,928 for the all-positive rule on the same rows), so the cost-tuned operating point and the capacity operating point are 4.4x apart. Cost-optimal thresholds of the lookup by ratio on the temporal holdout: 2:1 -> 0.2526 (cost 76,860; flag 68.2%, recall 0.867, specificity 0.499); 3:1 -> 0.1993 (87,677; 78.9%, 0.927, 0.347); 5:1 -> 0.1429 (97,437; 84.6%, 0.954, 0.260); 10:1, 20:1 and 30:1 -> flag everything (100,928). Random ranking gives recall = flag rate.

Cost floor: with prevalence 48.6% the "review every case" rule costs USD 102,829 per 1,000 cases at any ratio, and a classifier beats it only if (1 - recall) < ((1 - prevalence) / (prevalence x ratio)) x specificity, i.e. 0.529 x specificity at 2:1, 0.353 at 3:1, 0.212 at 5:1, 0.106 at 10:1, 0.053 at 20:1 and 0.035 at 30:1. At >= 10:1 the cost-optimal policy is therefore to review every case and a cost criterion cannot discriminate between models (the earlier S4, "below the best hand rule at 10/20/30:1", was passed by the all-positive rule: 102,829 < 212,174); the operating point is set by review capacity instead (S3), the cost criterion is tested where the threshold is non-trivial (S4), and the whole 2:1-30:1 cost curve is reported with the floor marked. The model's value is ranking under limited review capacity and beating the rules the stakeholder could apply by hand.

Success criteria (set before modelling; evaluated on the 20% mine-year-grouped held-out test set, with CV mean +/- std and the temporal split (train 2010-2019, test 2020-2025) reported alongside, and every metric also stratified by COAL_METAL_IND):

| # | Criterion | Threshold | Reference |
|---|---|---|---|
| S1 | ROC-AUC | >= 0.80 | 3-field lookup 0.794 on the temporal holdout |
| S2 | PR-AUC (average precision) | >= 0.75 | prevalence floor 0.486 |
| S3 | Capacity operating point: threshold chosen on out-of-fold probabilities so that specificity = 0.40 (flag rate about 75%, the review-capacity assumption); recall at that threshold | recall >= 0.92 (implies precision >= 0.59 at 48.6% prevalence) | 3-field lookup on the temporal holdout 0.9095 at specificity 0.394 (0.906 at a 75% flag rate); best single-field rules 0.928 / 0.293 and 0.896 / 0.429 |
| S4 | Expected cost per 1,000 cases at the cost-optimal out-of-fold threshold, for each ratio in the sensitivity set | at 2:1, 3:1 and 5:1: below the all-positive floor (102,829 on the full window, recomputed on the test set) and below the 3-field lookup's cost-optimal cost at the same ratio (temporal holdout 76,860 / 87,677 / 97,437, recomputed on the test set); at 10:1, 20:1 and 30:1 the floor is the optimum, so only the distance to it is reported | lookup and floor |
| S5 | Capacity: recall among the 50% of cases with the highest predicted probability | >= 0.78 (lift >= 1.6) | lookup 0.750; random 0.486 |
| S6 | Calibration: Brier score | <= 0.20 (overall and within each sector on the temporal split) | constant predictor at prevalence 0.250 |
| S7 | Statistical comparison | McNemar vs the 3-field lookup on the test set; paired test across CV folds between the top two models | |

## 9. Open risks and limitations

| Risk | Evidence | Mitigation / wording for the report |
|---|---|---|
| Economic framing and cost assumptions | at 48.6% prevalence and >= 10:1 costs, "review every case" is cost-optimal (the lookup's cost-optimal threshold is 0 at 10:1 and above) and no calibrated model can pass a recall-and-specificity test at p* = 0.0476; C_FN = USD 4,000 and the 20:1 ratio are assumptions built on an analogy (insurer report-lag study) and an uncited 5-10% share | S3 is a capacity operating point (specificity 0.40) and S4 is tested at 2:1-5:1 where thresholds are non-trivial; the full 2:1-30:1 cost curve is reported with the floor marked; C_FN and the ratio are labelled assumptions in section 8 and `decisions.json` |
| Label completeness | 183 code-03/04 rows lack DAYS_LOST and 241 have DAYS_LOST = 0 (section 1); 2026 excluded (47% missing DAYS_LOST, provisional 03 codes: 93.7% of unresolved 2026 rows are coded 03); 2025 Q4 shows a weaker version of the 2026 degree-mix signature (05 share 18.0% vs 22.4%, about 5% of its 1,097 rows exposed to re-coding); MSHA's case-update lag is undocumented | quarter-level degree-mix test as the completeness check (the DAYS_LOST test misses re-coding); 2025-12-31 cut stated as a judgement with the 0.05%-of-window exposure; weekly refresh changes counts (snapshot 2026-10-09 pinned) |
| Label is not fully future-blind for the filer | code 01 (death, 567 rows) is known at filing (Form 7000-1 Item 18) and IMMED_NOTIFY_CD 01 marks it; `immed_reportable` is 100% positive for those 560 rows; for 44.96% of rows the worker had returned to work on or before the filing date, so the filer already holds the days-away outcome | the model scores Sections A-C circumstances only; metrics with and without the flag are produced (`sensitivity.json` `primary_split_variants.without_immed_reportable` and `temporal_split.json` `without_immed_reportable`), optionally also for RTW-before vs RTW-after filing; never use the raw IMMED_NOTIFY_CD |
| Label measures return-to-work practice | codes 03/04 vs 05 depend on whether the operator offers restricted duty: coal exceeds metal in 40 of 43 matched nature x body-part cells (median +19.0 pp); the 05 share of DART cases ranges 0.4-68.4% across mines (p5-p95); COAL_METAL_IND V = 0.240 with the lost-time label vs 0.015 with the DART label; prev_lost_time_share is a proxy for the practice | state the positive class as an indemnity-trigger label; the insurer use stands; inspection targeting uses the DART sensitivity label (iii-b); all metrics stratified by COAL_METAL_IND |
| Label drift within coal | coal prevalence rises 59.74% (2010) -> 70.71% (2025) as the 05 share of coal DART cases falls 12.17% -> 4.61% (persistent coal mines 11.82% -> 6.22%); the overall series is flat only because the coal share of rows falls 50.5% -> 33.3%; temporal split: coal 61.19% train vs 68.91% test | sector x year table in section 1; calibration and metrics by sector on the temporal split; CAL_YR / coal x year as a documented candidate for the temporal variant with an extrapolation caveat |
| Lagged features and exposure | 1.87% of t-1 cases were filed after the row's ACCIDENT_DT (11.7% of rows with prior cases, 21.2% of Q1 rows); 12.8% of referenced t-1 lost-time cases were still unresolved at the accident date (26.2% for Q1); contractor hours are absent from MinesProdYearly, so per-200k-hour rates understate exposure at contractor-heavy mines (12.5% of rows involve contractors) | default counts only cases filed before ACCIDENT_DT (no signal loss: AUC 0.671 -> 0.671); resolved-before variant implemented as a sensitivity run (`sensitivity.json`); contractor caveat documented |
| Mines snapshot mismatch | 40.43% of status dates, 26.71% of controller dates and 22.01% of 103I dates post-date the accident; dated "Unknown" levels track accident age (company type 41.2% of 2010 rows -> 3.0% of 2025 rows) and are 0% at deployment | company type dropped; 103I four-level rule with every later-dated status in "Unknown" (22.01%; 24.1% train vs 16.9% test period); temporal-split and primary-split metrics with and without the feature; stated as a data-card limitation |
| Mine-year leakage through the split | 82.26% of rows have same-mine-year siblings on other dates; the mean label of later-dated siblings ranks a row with AUC 0.6665; (latitude, longitude, prev_hours, prev_employees) identifies the mine-year for 99.85% of rows; 883 of 2,614 same-day groups have mixed labels | primary split and CV folds grouped by MINE_ID x CAL_YR (nests the same-day multi-victim key); temporal split reported side by side for S1-S6; MINE_ID grouping run as the strictest check (`sensitivity.json` `mine_grouped_split`, S1-S6 also in `success_criteria.json` `robustness_mine_grouped_split`); same-year history only via the explicit ytd_lost_time_share. Latitude and longitude identify the mine for 92.6% of rows, so on the primary split 81.8% of test rows have same-mine training rows from later years |
| Deterministic categorical level (documented exception) | `NATURE_INJURY_CD = 100` (AMPUTATION OR ENUCLEATION) has 925 rows (750 training, 175 test rows of the primary split) and all 925 are lost-time: 923 are degree 02 (permanent disability) and 2 are degree 01 (death). The level is definitional and known at filing, so it is kept, but it is a rule rather than a pattern: no other level with at least 100 rows is 100% one class, and none lies within 2 pp of it | `tests/test_leakage.py` fails if the set of such levels changes from exactly this one; `sensitivity.json` `excluding_pure_levels` gives every model's and baseline's held-out metrics with and without these rows, at the thresholds fixed beforehand, and the report is to quote both |
| Previous-year hours incomplete early in January | 1,643 rows were filed on or before 15 January of CAL_YR, when the previous year's fourth-quarter employment report (due 15 days after quarter end, 30 CFR 50.30) was not yet due; their prev_hours, prev_employees, rates and type_at_time may be incomplete | contract join at CAL_YR - 1 kept; `prev_year_incomplete_at_filing` flag in the base table (not a model input); sensitivity refit with those rows' exposure features blanked (`sensitivity.json`) |
| Temporal evaluation protocol | the primary split trains on 79.5% of the 2020-2025 rows, so a threshold, grid choice or model ranking taken from it carries 2020-2025 labels | `temporal_evaluation` repeats the protocol on 2010-2019 only: every model re-tuned with the same grids and grouped CV, the two best by pre-2020 CV average precision kept, thresholds from pre-2020 out-of-fold probabilities, 2020-2025 labels used for the final scores only |
| Code-list revision inside the window | OCCUPATION and EQUIP_MFR lists revised around January 2014 (discontinued scoop-car title 533 rows; 7 + 7 levels with >= 100 rows first seen 2013-2014); IMMED_NOTIFY_CD 13 absent after 2012 | E28 / E29; level drift included in the temporal-split checks; some one-hot columns are zero for the whole temporal test period |
| Illness scope | 1,289 code-07 cases with DAYS_LOST > 0 (2.54% of the lost-time population) are excluded although the insurer would reserve for them; the base keeps 816 accident-dated illness natures; the codebook definition of 07 ("not degree 1-6") is contradicted by the data | stated as a scope and measurement limitation in section 2; code 10 exclusion is clean (0 rows with days lost or restricted) |
| Inference-time feature availability | the 11 mine-level numeric features and type_at_time need a (MINE_ID, year) lookup that the Streamlit form cannot supply | `data/processed/mine_year_features.parquet` and `mine_case_history.parquet` persisted by `mlda-train` (section 3); manual-entry fallback |
| Loader placeholder handling | `src/mlda/data/load.py` masks every placeholder to NaN, so formulas written against the literal `?` would invert (97,867 ones instead of 3,665 for immed_reportable) | all placeholder tests written as `isna()` / `fillna("none")` in `decisions.json` and section 4c; load.py's time comment should reference E10 (48 three-character values) |
| Related-work wording (outside this document) | `domain_research.md` item 7 says the JEOM (2023) study "includes outcome fields", which its listed predictors do not support; `dataset_research.md` names "relational join" as a differentiator although Manjunatha (2023) already joined Accidents to Mines | open edits for those two files: keep "post-outcome days-lost threshold as label" for item 7 and drop "includes outcome fields"; replace "relational join" with "exposure-hours join with lagged mine-year features" |
| Experience fields | 3.71% of rows have job or mine experience above total; cause unknown | set to NaN with a flag; sensitivity check with clipping; features are weak (abs(r) <= 0.08) |
| High-cardinality codes | OCCUPATION 195 titles, INJURY_SOURCE 124, EQUIP_MFR 139; Cramer's V for PRIMARY_SIC (100) and STATE (54) is inflated | min_frequency grouping; bias-corrected V reported; target encoding only inside CV folds if used |
| Reporting bias in the source | Part 50 is operator self-reporting; under-reporting varies by mine and era; first-aid cases are not reported | state as measurement bias; results describe reported cases |
| Transfer to Uzbekistan | US codes, costs (NCCI, BLS) and reporting rules; no public case-level data in Uzbekistan | present as a transferable template, not a calibrated model for Uzbek mines |
| Statistics tooling | scipy and scikit-learn absent in the system interpreter used for profiling; Cramer's V and point-biserial r computed with numpy | recompute in the project environment; values should match to rounding |
| Unverified domain figures | NSC cost figures read from search snippets (HTTP 403); NIOSH fatality page 503; some author lists unverified | cite with status flags from `domain_research.md`; cost ratio does not depend on them |

## 10. Review log (2026-10-14)

This log records the profile review. The implementation review that followed (section 11) supersedes its 103(i) rule (finding 2) and the "infrequent bucket" wording for E28.

Twenty findings from three review lenses were applied to this document and to `docs/profile/decisions.json`; every number quoted in a decision was recomputed on the final base (n = 101,609) with pandas 3.0.6 before the edit. Accepted 20, rejected 0 (finding 16 duplicates finding 6; finding 14 is accepted but its edits fall in files outside this task's write scope and are listed as open).

| # | Lens | Severity | Claim (short) | Decision | Change |
|---|---|---|---|---|---|
| 1 | leakage-and-time | major | MINE_ID + ACCIDENT_DT split lets same-mine-year rows (incl. later cases) straddle train and test; future-sibling mean label AUC 0.6665; planned numerics identify the mine-year | accepted (recomputed: 32,732 mine-years; 83,587 rows with siblings; AUC 0.6665 / 0.6688; tuple identifies the mine-year for 99.85% of rows; 91.88% of test-period rows at mines seen in training) | section 7: primary split and CV grouped by MINE_ID x CAL_YR, same-day key nested; temporal split reported side by side for S1-S6; optional explicit ytd_lost_time_share (AUC 0.664) added in 4c; risk row in section 9; `decisions.json` group_key, split |
| 2 | leakage-and-time | minor | "Unknown" levels of company_type_at_accident and status_103i_at_accident encode accident age and are 0% at deployment | accepted (recomputed: company Unknown 41.2% -> 3.0% by year, 34.4% vs 8.0% across the temporal split; 103I 26.5% -> 14.3%; 8,091 removed-after rows; raw COMPANY_TYPE V = 0.044) | company_type_at_accident dropped; status_103i_at_accident rebuilt with four levels (Never had / In 103I at accident incl. removed-after / Formerly 103I / Unknown 14.05%); sections 3, 4d, 5 (E24), 6; metrics with and without on the temporal split |
| 3 | leakage-and-time | minor | Lagged case features: "a few t-1 cases may still be open" understates it; strict variant should be the default | accepted (recomputed: 1.87% of prior cases filed after ACCIDENT_DT, 11.7% of rows with prior cases, 21.2% of Q1 rows; filed-before variant changes 7,743 rows, AUC 0.6708 vs 0.6710) | prev_incidents, both rates and prev_lost_time_share count only t-1 cases filed before ACCIDENT_DT; quantified paragraph replaces the "a few" wording in 4c and 9; resolved-before kept as sensitivity variant; J2 note; `decisions.json` formulas |
| 4 | leakage-and-time | minor | Degree code is provisional at filing (03 -> 04/05 migration) and 45% of filers already know the RTW outcome | accepted (recomputed: 880 of 939 unresolved 2026 rows coded 03; Sep 2026 mix 53.9 / 1.7 / 8.6 / 33.1; RTW <= filing for 45,679 rows, 44.96%, positive rate 20.35% vs 71.64%) | section 2 step 2 cites the migration; section 8 Decision row and section 9 state the 45% / 55% split and that the model scores Sections A-C only |
| 5 | leakage-and-time | minor | Code lists changed inside the window (OCCUPATION / EQUIP_MFR 2013-2014, IMMED code 13 only 2010-2012) | accepted (recomputed: discontinued scoop-car title 533 rows; 7 + 7 levels first seen 2013-2014; IMMED 13 = 77 rows, 92.2% positive; all other categoricals stable) | section 2 paragraph on the January 2014 revision; E28 (discontinued title -> infrequent bucket) and E29 (IMMED 13 -> flag 0); immed_reportable = 3,665 rows; level drift added to the temporal checks |
| 6 | leakage-and-time | minor | type_at_time from the accident-year subunit set is not known at filing for Q1 cases | accepted (recomputed: 25.05% of rows in Q1; t-1 derivation agrees 99.43% on 100,362 rows, coverage 98.81%) | J4 now keyed on CAL_YR - 1 with SUBUNIT_CD fallback (1,206 rows); 4c row updated (Underground 58.71%, Surface 41.98%, Facility 41.53%) |
| 7 | label-and-bias | blocker | S3 (recall >= 0.92 and specificity >= 0.40 at p* = 0.0476) is infeasible for a calibrated model; S4 is passed by flagging every case | accepted (recomputed: no level with n >= 100 below 0.079; lookup at p* = 0.0476 flags 99.76%, specificity 0.0025; specificity 0.40 needs threshold 0.21 with recall 0.9095; lookup cost-optimal thresholds: 2:1 0.2526, 3:1 0.1993, 5:1 0.1429, >= 10:1 flag all) | S3 redefined as the capacity operating point (specificity = 0.40 on OOF probabilities, recall >= 0.92); S4 redefined as cost at the cost-optimal threshold at 2:1 / 3:1 / 5:1 below floor and lookup, distance to floor only at >= 10:1; sensitivity set 2, 3, 5, 10, 20, 30; baseline table extended; `decisions.json` cost and success_criteria |
| 8 | label-and-bias | major | "Prevalence is flat" hides a monotonic within-coal drift driven by declining restricted-duty coding | accepted (recomputed: coal 59.74% -> 70.71%, coal share 50.5% -> 33.3%; coal 05 share of DART 12.17% -> 4.61%; r within coal 0.074; temporal split coal 61.19% vs 68.91%) | sector x year table and drift paragraph in section 1; "flat" wording replaced in sections 2 and 4b; CAL_YR as a candidate for the temporal variant; risk row; `decisions.json` label_drift |
| 9 | label-and-bias | major | Label partly measures operator return-to-work practice; undermines the MSHA inspection-targeting claim | accepted (recomputed: coal > metal in 40 of 43 matched cells, median +19.0 pp; 330 / 420 coal 85.9% / 8.8% vs metal 43.7% / 45.8%; 05 share p5-p95 0.4-68.4% across mines; V 0.240 vs 0.015 for the DART label) | section 1 "What the label measures" paragraph; DART label (01-05 vs 06, 69.04%) added as sensitivity label (iii-b); inspection-targeting claim narrowed in section 8; sector-stratified metrics required; `decisions.json` target.sensitivity_labels |
| 10 | label-and-bias | minor | 20:1 ratio rests on an analogy and an uncited 5-10% share; range presented as evidence | accepted | C_FN and ratio labelled assumptions in section 8 and `decisions.json` cost.note; sensitivity set extended downward to 2:1, 3:1, 5:1; p* table completed |
| 11 | label-and-bias | minor | 2025 Q4 shows the 2026 incompleteness signature; the DAYS_LOST test does not detect it | accepted (recomputed: 2025 Q4 n 1,097: 03 43.1% vs 37.2%, 05 18.0% vs 22.4%, prevalence 53.2% vs 49.2%; 2018-2024 Q4 gaps within 3.5 pp; 1 of 560 code-03/04 rows lacks DAYS_LOST) | window kept (documented option); quarter-level degree-mix test adopted as the completeness check; exposure (about 5% of 1,097 rows, 0.05% of the window) stated in sections 2 and 9; `decisions.json` completeness_check |
| 12 | label-and-bias | minor | Code-07 exclusion removes 1,289 lost-time illness cases; base keeps illness natures | accepted (recomputed: 1,289 with DAYS_LOST > 0, median 30; 393 with SCHEDULE_CHARGE; 816 base rows with illness natures, lost-time 36.0%; code 10 clean) | section 1 table, section 2 steps 4 and 7, section 9 Illness scope row |
| 13 | label-and-bias | minor | Only 183 of the 424 positives failing the DAYS_LOST / SCHEDULE_CHARGE check are explained | accepted (recomputed: 424 = 183 blank + 241 zero; 371 code 03, 53 code 04; 196 of the 241 in 2010-2012; 147 coal / 94 metal) | section 1 label-validity paragraph states both components and the 2010-2012 clustering; rows kept as positives |
| 14 | label-and-bias | minor | domain_research.md item 7 ("includes outcome fields") unsupported; dataset_research.md "relational join" overstated | accepted; the two files are outside this task's write scope | recorded as an open edit in section 9 (Related-work wording row) and in the task summary |
| 15 | join-features-feasibility | minor | mine_size_band formula (default right=True) gives different bands from the documented counts | accepted (recomputed: right=True 3,321 / 13,877 / 16,324 / 30,119 / 36,762 vs right=False 2,285 / 14,180 / 16,654 / 30,483 / 36,801; 2,211 rows on boundaries) | formula written as `pd.cut(..., right=False, labels=[...])` in 4c and `decisions.json` |
| 16 | join-features-feasibility | minor | type_at_time (J4) not known at filing for Q1 rows | accepted as a duplicate of finding 6 | same change as finding 6 |
| 17 | join-features-feasibility | minor | Formulas written against literal placeholder strings invert after load.py masks them to NaN; load.py time comment wrong | accepted (recomputed: 48 three-character SHIFT_BEGIN_TIME values + 1 blank) | `decisions.json` load section states that placeholders arrive as NaN and every affected formula uses `isna()` / `fillna("none")`; E10 and E17 updated; load.py comment fix listed as an open item (file outside this task's scope) |
| 18 | join-features-feasibility | minor | add_indicator=True duplicates prev_year_matched six times and other explicit flags | accepted (recomputed: the six prev_* columns are missing on exactly the same 1,206 rows) | `add_indicator=False`; seven explicit indicator columns enumerated in section 6 and `decisions.json` |
| 19 | join-features-feasibility | minor | Inference-time feasibility of the mine-level features is not addressed | accepted (recomputed: 226,182 mine-years 2009-2025, 24,797 mines; 107,740 mine-years at the 8,303 base mines) | section 3 Inference lookup paragraph: `mine_year_features.parquet` + `mine_case_history.parquet`, manual fallback; `decisions.json` inference_lookup |
| 20 | join-features-feasibility | minor | Experience ratios have no zero-denominator guard | accepted (recomputed: 0 zeros, TOT_EXPER min 0.02) | ratios NaN when TOT_EXPER <= 0 or either input missing (4c, E30, `decisions.json`) |

Net effect on the modelling frame: N = 101,609 and prevalence 48.585% unchanged; one engineered feature removed (company_type_at_accident), one added as optional (ytd_lost_time_share); status_103i_at_accident, type_at_time, immed_reportable, mine_size_band and the four lagged case features redefined; split key, S3, S4 and the cost sensitivity set redefined.

## 11. Implementation review F1 (2026-10-14)

Thirty-one findings from four reviewers (tester 7, leakage 7, cv-stats 5, contract 12) on the built pipeline were applied to `src/mlda`, `tests/` and the documents. Numbers below were recomputed on the final base (n = 101,609).

| # | Reviewer | Finding (short) | Decision | Change |
|---|---|---|---|---|
| 1 | tester | `NATURE_INJURY_CD = 100` is 100% lost-time (925 rows) and undocumented | accepted | documented in section 9; `mlda.sensitivity.pure_levels`, `DOCUMENTED_PURE_LEVELS`; `tests/test_leakage.py`; `sensitivity.json` `excluding_pure_levels` |
| 2 | tester | stored float32 predictions move rows across stored thresholds | accepted | `write_predictions` writes float64; the quick-run test recomputes the thresholds from the stored scores exactly |
| 3 | tester | quick-run fixture exceeds 90 s | accepted | 20% mine-year sample in the test, sampled heavy figures and smaller explanation samples in quick mode |
| 4 | tester | counts in sections 4c and 6 used `<=` | accepted | strict `<` counts (81,469 / 20,140 / 17,822) here, in `decisions.json` and in `data/README.md` |
| 5 | tester | `mlda-zip` imports a missing module | accepted in part | the command exits with one line instead of a traceback; `mlda.submission` is outside interfaces.md sections 1-11 and stays an open issue |
| 6 | tester | `clean.parquet` and `feature_schema.json` are not produced | accepted | `docs/architecture.md` lists the files the pipeline writes |
| 7 | tester | no unit tests for `mlda.explain` and `mlda.cli` | accepted | `tests/test_explain.py`, `tests/test_cli.py` |
| 8 | leakage | temporal evaluation used primary-split thresholds, grids and ranking | accepted | protocol repeated on 2010-2019 rows (section 9) |
| 9 | leakage | row-level sample cap leaves sibling rows on both sides | accepted | `stratified_group_subsample`; rows outside the cap share no mine-year with the fitted rows |
| 10 | leakage | the 103(i) status used post-accident information | accepted | later-dated statuses, "Removed" included, become "Unknown" (section 3); the feature stays, with and without runs on both splits |
| 11 | leakage | lagged counts include cases still open at the accident | accepted | `*_resolved` base columns and the `resolved_lag_counts` refit |
| 12 | leakage | MINE_ID-grouped robustness run missing | accepted | `mine_grouped_robustness` |
| 13 | leakage | metrics without `immed_reportable` missing | accepted | primary-split and temporal variants |
| 14 | leakage | early-January rows use an incomplete previous year | accepted as a sensitivity run | flag column and `exposure_blank_if_q4_incomplete` refit; the contract join is unchanged |
| 15 | cv-stats | sample cap drawn by row (same defect as 9) | accepted | same change as 9; the docstring now says row- and group-disjoint |
| 16 | cv-stats | bootstrap and McNemar treat rows as independent | accepted | `groups` arguments (cluster bootstrap, Durkalski McNemar); both versions reported |
| 17 | cv-stats | the paired t-test understates variance | accepted | Nadeau-Bengio corrected t; NaN for zero spread; no p-value for models fitted on different rows |
| 18 | cv-stats | no test of out-of-fold probabilities | accepted | hand-rebuilt folds in `tests/test_models.py`; a mutation check confirmed the old tests passed with in-sample scores |
| 19 | cv-stats | two tests always pass | accepted | hand-computed t values; `os.cpu_count` monkeypatched |
| 20 | contract | E28 not realised as the infrequent bucket | accepted by documentation | section 5, E28 row |
| 21 | contract | `iqr_clip` flattens the victim count | accepted | open fences when the IQR is zero |
| 22 | contract | deprecated `np.testing.suppress_warnings` | accepted | `warnings.catch_warnings` |
| 23 | contract | skew range quoted for the exposure columns | accepted | the text names both ranges |
| 24 | contract | KNN imputer on unscaled columns | accepted | `ScaledKNNImputer` |
| 25 | contract | rules duplicated in `features.py` and `cleaning.py` | accepted | one definition, in `cleaning.py` |
| 26 | contract | placeholders not handled for raw rows | accepted | `PLACEHOLDERS` masked in `features._text` |
| 27 | contract | README figures differ from `decisions.json` | accepted | strict counts everywhere |
| 28 | contract | type rule wording ignores zero-hour rows | accepted | wording in `join.py` and `data/README.md` |
| 29 | contract | y-axis labels missing on grid panels | accepted | `histograms_grid` and `outlier_fences` label every panel |
| 30 | contract | no real-data test of the engineered features | accepted | `test_real_feature_counts_match_decisions` |
| 31 | contract | five modules not `ruff format` clean | accepted | `ruff format` on `plots`, `cleaning`, `features`, `eda`, `preprocessing` |
