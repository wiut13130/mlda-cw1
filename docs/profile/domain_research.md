# Domain research: MSHA lost-time injury prediction (Task C)

Compiled 14 October 2026 from web sources for module 6COSC017C CW1 (student ID 13130). Every fact carries a URL in the
table and a full entry in the reference list. Verification legend: **verified** = read directly from the cited page or PDF;
**snippet** = the cited page refused automated fetch (HTTP 403/503/captcha) and the figure was read from the search
engine's extract of that same page; **not verified** = could not be checked against a primary source. Data snapshot used by
the project: MSHA files dated 9 October 2026, pulled 14 October 2026 (`data/raw/SNAPSHOT.json`).

## 1. MSHA reporting rules and injury-severity definitions

### 1.1 30 CFR Part 50 and Form 7000-1

| Rule | Content (quoted or paraphrased from the regulation / form) | Status | Source |
|---|---|---|---|
| 30 CFR 50.2 "accident" | 12 enumerated events: death at a mine; injury "which has a reasonable potential to cause death"; entrapment > 30 min; unplanned inundation; unplanned ignition/explosion of gas or dust; unplanned fire not extinguished within 10 min (underground) / 30 min (surface); unplanned ignition/explosion of explosives; unplanned roof fall at or above anchorage zone or impairing ventilation/passage; coal or rock outburst disrupting mining > 1 h; unstable impoundment/refuse pile; hoisting damage endangering a person or > 30 min outage; event causing death or injury to a person not at the mine | verified | https://www.law.cornell.edu/cfr/text/30/50.2 |
| 30 CFR 50.2 "occupational injury" | "Any injury to a miner which occurs at a mine for which medical treatment is administered, or which results in death or loss of consciousness, inability to perform all job duties on any day after an injury, temporary assignment to other duties, or transfer to another job" | verified | https://www.law.cornell.edu/cfr/text/30/50.2 |
| 30 CFR 50.2 "occupational illness" | "An illness or disease of a miner which may have resulted from work at a mine or for which an award of compensation is made" | verified | https://www.law.cornell.edu/cfr/text/30/50.2 |
| 30 CFR 50.2 "first aid" | "One-time treatment, and any follow-up visit for observational purposes, of a minor injury" | verified | https://www.law.cornell.edu/cfr/text/30/50.2 |
| 30 CFR 50.10 immediate notification | Operator "shall immediately contact MSHA at once without delay and within 15 minutes" for death, injury with reasonable potential to cause death, entrapment with such potential, or any other 50.2 accident | verified | https://www.law.cornell.edu/cfr/text/30/50.10 |
| 30 CFR 50.20 who reports, deadline | "Each operator shall report each accident, occupational injury, or occupational illness at the mine" on Form 7000-1; "within ten working days after an accident or occupational injury occurs or an occupational illness is diagnosed" | verified | https://www.law.cornell.edu/cfr/text/30/50.20 |
| 30 CFR 50.20-3 medical treatment vs first aid | Medical treatment includes "suturing of any wound, treatment of fractures, application of a cast or other professional means of immobilizing an injured part"; first aid is "any one-time treatment, and follow-up visit for the purpose of observation, of minor injuries such as cuts, scratches, first degree burns"; eight injury-type criteria (abrasions, bruises, burns, cuts, eye, inhalation, splinters, sprains/strains) | verified | https://www.law.cornell.edu/cfr/text/30/50.20-3 |
| 30 CFR 50.30 employment report | "Each operator of a mine in which an individual worked during any day of a calendar quarter" files Form 7000-2 (employee-hours, employees, coal production) "within 15 days after the end of each calendar quarter"; copy kept 5 years. This is the source of `MinesProdYearly` hours | verified | https://www.law.cornell.edu/cfr/text/30/50.30 |
| Form 7000-1 scope | OMB No. 1219-0007; required by 30 U.S.C. 813 and 30 CFR Part 50; "A form is required on each individual becoming injured or ill, even when several were injured or made ill in a single occurrence"; "a Form 7000-1 is required on each accident as defined in 30 CFR Part 50 whether any person was injured or not"; covers operator and contractor employees | verified | https://www.msha.gov/sites/default/files/Support_Resources/Forms/7000-1.pdf |
| Form 7000-1 who completes it | "The principal officer in charge of health and safety at the mine or the supervisor of the mine area in which the accident, injury, or illness occurred shall complete or review the Form 7000-1"; burden estimate 30 minutes per response; record kept at the mine 5 years | verified | https://www.msha.gov/sites/default/files/Support_Resources/Forms/7000-1.pdf |
| Form 7000-1 first-aid exclusion | "First aid cases (those for which no medical treatment was received, no time was lost, and no restriction of work, motion, or loss of consciousness occurred) need not be reported" | verified | https://www.msha.gov/sites/default/files/Support_Resources/Forms/7000-1.pdf |
| Form 7000-1 Item 18 (death) | "Check this box if the injury or illness resulted in death" | verified | same |
| Form 7000-1 Item 19 (permanent disability) | "any injury or occupational illness other than death which results in the loss (or complete loss of use) of any member (or part of a member) of the body, or a permanent impairment of functions of the body, or which permanently and totally incapacitates the injured person from following any gainful occupation" | verified | same |
| Form 7000-1 Item 30 (days away) | "the number of workdays the injured person did not report to his/her place of employment, i.e., number of days away from work"; footnote: "should not include the day of injury or onset of illness, or any days on which the employee was not previously scheduled to work" | verified | same |
| Form 7000-1 Item 31 (days restricted) | "the number of workdays the injured person was on restricted work activity; do not include days away from work reported in Item 30" | verified | same |
| Form 7000-1 Section D timing | If return-to-duty information "is not available within ten working days after a reportable occurrence, then the first two pages are sent to MSHA without Section D being completed; PAGE 3 is then mailed ... when the data is available. Until all the items are answered ... the occurrence remains an open case" | verified | same |

Implication for the business question: at filing time (<= 10 working days) Items 29-31 (return date, days away, days restricted)
are often still blank, which is why `DAYS_LOST`, `DAYS_RESTRICT`, `RETURN_TO_WORK_DT`, `SCHEDULE_CHARGE` and `CLOSED_DOC_NO`
are post-outcome fields and must be excluded from features (consistent with `docs/dataset_research.md`).

### 1.2 Degree-of-injury codes and how MSHA, OSHA/BLS and ILO classify severity

MSHA's own severity classes (used in all MSHA rate publications):

| MSHA class | Definition (MSHA, Mine Injury and Worktime Quarterly, CY2025 preliminary) | Source |
|---|---|---|
| FATAL | "Occurrences resulting in death" | https://arlweb.msha.gov/Stats/Part50/WQ/2025/MIWQ%20-%20CY2025Q4.pdf |
| NFDL | "Non fatal occurrences with Days Lost (lost workdays). That is, non fatal injury occurrences that result in days away from work or days of restricted work activity" | same |
| NDL | "Occurrences with No Days Lost. That is, non fatal injury occurrences resulting only in loss of consciousness or medical treatment other than first aid" | same |
| Incidence rate | "IR = (Number of Injury Occurrences / Number of Employee Hours) x 200,000" | same |

Mapping of the dataset's `DEGREE_INJURY_CD` (codebook `Accidents_Definition_File.txt`, line 18) to the external definitions:

| Code | Codebook description | MSHA class | OSHA/BLS analogue (29 CFR 1904.7; BLS SOII) | ILO 16th ICLS analogue | Project target |
|---|---|---|---|---|---|
| 00 | Accident only | not an injury | not recordable as injury | not an injury | exclude |
| 01 | Fatality | FATAL | Death | Fatal occupational injury (death within 1 year) | positive (lost-time) |
| 02 | Permanent total or permanent partial disability | NFDL | Days away / restricted (also "significant injury") | Permanent incapacity | positive |
| 03 | Days away from work only | NFDL | DAFW case | Temporary incapacity (absence >= 1 day excluding day of accident) | positive |
| 04 | Days away from work and restricted activity | NFDL | DAFW case (days away counted first) | Temporary incapacity | positive |
| 05 | Days restricted activity only | NFDL | DJTR case (DART but not DAFW) | "Restricted activity" case, noted but not covered by the resolution | negative |
| 06 | No days away from work, no restrictions | NDL | "Other recordable case" (medical treatment beyond first aid or loss of consciousness) | Not covered (no absence) | negative |
| 07 | Occupational illness not degree 1-6 | illness | Illness (recordable) | Occupational disease (separate statistics) | exclude |
| 08 | Injuries due to natural causes | n/a | not work-related | n/a | exclude |
| 09 | Injuries involving non-employees | n/a | not an employee case | n/a | exclude |
| 10 | All other cases (incl. 1st aid) | n/a | First aid, not recordable | n/a | exclude |
| ? | No value found | n/a | n/a | n/a | exclude |

Consequences of the mapping: (a) MSHA's published "NFDL" and "Lost-Time" rates include code 05 (restricted-only), so the
project's positive class is narrower than MSHA's NFDL and corresponds to BLS "DAFW" plus fatalities; (b) MSHA counts
*workdays* away (Item 30), OSHA 1904.7 counts *calendar* days (capped at 180), ILO counts calendar days up to one year, so
day counts are not directly comparable across regimes; (c) the FY2025 gap between MSHA's NFDL rate (1.25) and Lost-Time
rate (1.26) shows restricted-only-plus-transfer cases are a small share (MSHA, 2026b).

External definitions used above:

| Regime | Definition | Status | Source |
|---|---|---|---|
| OSHA 29 CFR 1904.7(b)(1) | Record if the injury results in "Death, Days away from work, Restricted work or transfer to another job, Medical treatment beyond first aid, Loss of consciousness, A significant injury or illness diagnosed by a physician" | verified | https://www.law.cornell.edu/cfr/text/29/1904.7 |
| OSHA 1904.7(b)(3) days away | "count the number of calendar days the employee was unable to work"; "begin counting days away on the day after the injury occurred"; "you may 'cap' the total days away at 180 calendar days" | verified | same |
| OSHA 1904.7(b)(4) restricted work | employee kept "from performing one or more of the routine functions of his or her job, or from working the full workday" | verified | same |
| BLS SOII DAFW | "Cases which result in days away from work (beyond the day of injury or onset of illness)" | verified | https://www.bls.gov/opub/hom/soii/concepts.htm |
| BLS SOII DJTR | "Those which result only in job transfer or restricted work activity" | verified | same |
| BLS SOII DART | "Those which involve days away from work beyond the day of injury or onset of illness, days of job transfer or restricted work activity, or both" | verified | same |
| DART / incidence rate formula | (N / EH) x 200,000, where 200,000 = 100 full-time workers x 40 h x 50 weeks | snippet (NSC page 403); formula also verified in MSHA MIWQ above | https://injuryfacts.nsc.org/definitions/dart-days-away-restricted-or-transferred-cases/ |
| ILO occupational injury | "Any personal injury, disease or death resulting from an occupational accident" | verified (ILO Dept. of Statistics slides on the 1998 resolution) | https://www.ilo.org/media/289311/download |
| ILO occupational accident | "An unexpected and unplanned occurrence, including acts of violence, arising out of or in connection with work, which results in one or more workers incurring a personal injury, disease or death" | verified | same |
| ILO coverage threshold | Statistics should cover fatal injuries and "non-fatal injuries causing an absence from work of at least one day" (excluding the day of the accident) | verified | same |
| ILO temporary incapacity | Workers "unable to work from the day after the day of the accident, but were later able to perform again the normal duties ... within a period of one year" | verified | same |
| ILO permanent incapacity | "unable to work from the day of the accident, and were never able to perform again the normal duties of work in the job or post" | verified | same |
| ILO fatal injury | Death "within one year of the day of the accident" | verified | same |
| ILO days lost | "calendar days during which those persons temporarily incapacitated were unable to work, excluding the day of the accident, up to a maximum of one year" | verified | same |
| ILO rates | Frequency rate per 1,000,000 hours worked; incidence rate per 100,000 workers; severity rate = days lost per 1,000,000 hours | verified | same |

Note: the ILO slide deck header says "17th ICLS"; the resolution landing page (https://www.ilo.org/resource/resolution-concerning-statistics-occupational-injuries-resulting) dates it to the 16th ICLS, October 1998, which is the correct attribution.

## 2. US mining safety statistics 2022-2025

### 2.1 MSHA "Mine Safety and Health At A Glance" (fiscal years; operator plus contractor employees)

| Metric | FY2022 | FY2023 | FY2024 | FY2025 | Source |
|---|---|---|---|---|---|
| All mines: number of mines | 12,702 | 12,740 | 12,789 | 12,568 | MSHA (2026a) |
| All mines: number of miners | 313,624 | 324,903 | 330,122 | 327,465 | MSHA (2026a) |
| All mines: fatalities | 33 | 41 | 31 | 28 | MSHA (2026a) |
| All mines: fatal injury rate per 200,000 h | 0.0123 | 0.0146 | 0.0110 | 0.0101 | MSHA (2026a) |
| All mines: all-injury rate per 200,000 h | 1.89 | 1.85 | 1.82 | 1.77 | MSHA (2026a) |
| All mines: NFDL rate per 200,000 h | 1.32 | 1.31 | 1.29 | 1.25 | MSHA (2026b) |
| All mines: lost-time rate per 200,000 h | 1.34 | 1.32 | 1.30 | 1.26 | MSHA (2026b) |
| Coal: mines / miners | 989 / 65,836 | 1,006 / 68,499 | 952 / 68,462 | 886 / 62,901 | MSHA (2026a) |
| Coal: fatalities | 12 | 10 | 11 | 5 | MSHA (2026a) |
| Coal: fatal rate / all-injury rate | 0.0201 / 2.85 | 0.0155 / 2.76 | 0.0174 / 2.79 | 0.0085 / 2.73 | MSHA (2026a) |
| Coal: NFDL rate | 2.07 | 2.01 | 2.05 | 2.01 | MSHA (2026b) |
| Coal production (million short tons) | 596 | 583 | 525 | 523 | MSHA (2026a) |
| Metal/nonmetal: mines / miners | 11,713 / 247,788 | 11,734 / 256,404 | 11,837 / 261,660 | 11,682 / 264,564 | MSHA (2026a) |
| Metal/nonmetal: fatalities | 21 | 31 | 20 | 23 | MSHA (2026a) |
| Metal/nonmetal: fatal rate / all-injury rate | 0.0101 / 1.61 | 0.0143 / 1.58 | 0.0091 / 1.54 | 0.0105 / 1.52 | MSHA (2026a) |
| Metal/nonmetal: NFDL rate | 1.11 | 1.10 | 1.08 | 1.05 | MSHA (2026b) |
| Contractor fatalities (coal / MNM / total) | 2 / 4 / 6 | 4 / 5 / 9 | 1 / 8 / 9 | 1 / 6 / 7 | MSHA (2026a) |
| Citations and orders issued (all) | 85,716 | 94,045 | 94,352 | 87,372 | MSHA (2026a) |

Sources: https://www.msha.gov/sites/default/files/MSHA-Glance/FY2025-MSHA-At-A-Glance.pdf (dated 27 Feb 2026) and
https://www.msha.gov/sites/default/files/MSHA-Glance/By%20The%20Numbers%20FY%201978%20-%202025.pdf. The FY2024 edition (31 Jan
2025, https://www.msha.gov/sites/default/files/Data_Reports/Charts/FY2024-MSHA-ataGlance-1-31-2025.pdf) reports slightly
different figures for the same years because citation counts and hours are re-pulled (e.g. FY2024 all-injury rate 1.81 vs
1.82; FY2024 miners 328,732 vs 330,122). Long-run context: all-injury rate 5.07 in FY2000 and 2.86 in FY2010 (MSHA, 2026b).

### 2.2 Counts of injuries and fatalities by calendar year

| Metric | 2021 | 2022 | 2023 | 2024 | 2025 | Status | Source |
|---|---|---|---|---|---|---|---|
| Nonfatal lost-time injuries (NIOSH, excludes office employees) | 3,421 | 3,491 | 3,664 | 3,604 | n/a | verified | https://wwwn.cdc.gov/niosh-mining/MMWC/Injuries/NumberAndRate |
| Lost-time injury rate per 100 FTE (2,000 h = 1 FTE) | 1.49 | 1.42 | 1.43 | 1.40 | n/a | verified | same |
| Mining fatalities, calendar year (MSHA chargeable) | n/a | not verified | not verified | 28 | 33 (25 metal/nonmetal, 8 coal) | trade press citing MSHA | https://www.pitandquarry.com/msha-industry-finishes-2025-with-33-miner-fatalities/ ; https://northamericanmining.com/index.php/2026/01/09/msha-reported-33-fatalities-in-2025/ |
| 2025 leading causes | | | | | powered haulage 13; machinery 6; fall of face/rib/highwall 2; roof/back 2; falling/rolling material 2; electrical 2 | verified (Pit & Quarry) | same |
| Record-low year | 25 fatalities in 2016 | | | | | verified (Pit & Quarry, 9 Oct 2024) | https://www.pitandquarry.com/?p=97926 |

Total reportable injuries per fiscal year are not printed in At A Glance; they can be derived from the published rates
(hours = fatalities / fatal rate x 200,000; injuries = all-injury rate x hours / 200,000; rounding of the 4-decimal fatal
rate gives about +/-2 %): FY2022 about 5,070; FY2023 about 5,200; FY2024 about 5,130; FY2025 about 4,900 (coal about 1,600,
metal/nonmetal about 3,330). These derived figures are for orientation only; the exact per-year counts for the project's
positive and negative classes will be computed from `Accidents.txt` in the data profile. MSHA's quarterly operator and
contractor tables (CY2025 Q4 preliminary, updated 10 Apr 2026) exist only as image PDFs
(https://arlweb.msha.gov/ACCINJ/ALLMINES.HTM), and the NIOSH fatality chart page returned HTTP 503 during this research.

## 3. Cost evidence for the false-negative vs false-positive asymmetry

### 3.1 Cost benchmarks

| Item | Value | Year basis | Status | Source |
|---|---|---|---|---|
| NSC total cost of work injuries | USD 181.4 billion (wage/productivity 54.9 bn; medical 36.8 bn; administrative 64.5 bn) | 2024 | snippet (page returns 403) | https://injuryfacts.nsc.org/work/costs/work-injury-costs/ |
| NSC cost per medically consulted injury | USD 48,000 | 2024 | snippet | same |
| NSC cost per work death | USD 1,540,000 | 2024 | snippet | same |
| NSC 2023 cost per medically consulted injury | USD 43,000 | 2023 | snippet | same |
| NCCI average lost-time indemnity claim severity | USD 31.3 thousand (+4 % on AY2024) | accident year 2025 (preliminary) | verified via Gen Re summary of NCCI 2026 State of the Line | https://www.genre.com/us/knowledge/publications/2026/may/workers-comp-healthy-amid-change-en |
| NCCI average lost-time medical claim severity | USD 30.6 thousand (+4 %) | AY2025 (preliminary) | verified (Gen Re); +4 % / +4 % also verified in NCCI NCOIL slides | same ; https://ncoil.org/wp-content/uploads/2026/07/2026-NCCI-SOTL-NCOIL.pdf |
| NCCI implied average total lost-time claim | about USD 61.9 thousand (31.3 + 30.6) | AY2025 | derived | as above |
| NCCI lost-time claim frequency change | -1 % (wage-adjusted, NCOIL slides) / -2 % (Gen Re) for 2025; -5 % for AY2024 | 2024-2025 | verified | NCCI (2026); Risk & Insurance (2025a) |
| NCCI 2025 State of the Line (AY2024) | indemnity severity +6 %, medical severity +6 %; calendar-year combined ratio 86; AY2022-2024 increases above long-run 2 %/yr | AY2024 | verified | https://riskandinsurance.com/workers-comp-stays-profitable-in-2024-despite-lower-premiums-and-rising-claim-severity/ |
| NCCI 2026 State of the Line (CY2025) | combined ratio 91 (CY) / 102 (AY); USD 14 bn reserve redundancy; cumulative indemnity severity +71 % vs wage inflation +96 % (2005-2025) | 2025 | verified | https://ncoil.org/wp-content/uploads/2026/07/2026-NCCI-SOTL-NCOIL.pdf |
| Medical-only vs lost-time claims | medical-only 69.8 % of claim counts but 5.4 % of paid losses; lost-time 30.2 % of counts and 94.6 % of losses; mean paid USD 382 vs USD 15,518 per claim | NCCI-sourced data 1988-2020, single-jurisdiction study | not verified (PMC page captcha-blocked; from search extract) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8478314/table/T2 |
| OSHA $afety Pays direct costs (NCCI lost-time claims, policy years 2015-2017) | sprain 30,487; strain 32,023; fracture 54,856; contusion 27,630; laceration 21,872; carpal tunnel 30,930; amputation 96,003; asphyxiation 202,995 (USD) | 2015-2017 | verified | https://www.osha.gov/safetypays/estimator-text |
| OSHA $afety Pays indirect-cost multiplier | 4.5 for direct cost USD 0-2,999; 1.6 for 3,000-4,999; 1.2 for 5,000-9,999; 1.1 for >= 10,000; based on a Stanford Dept. of Civil Engineering study via Business Roundtable; "general estimates based on the limited research" | n/a | verified | https://www.osha.gov/safetypays/background |
| Liberty Mutual Workplace Safety Index 2025 | direct cost of the most disabling injuries (> 5 days away) USD 58.78 bn; top 10 causes USD 50.87 bn (86.55 %): overexertion 13.70 bn (23.31 %), falls same level 10.52 bn (17.89 %), struck by 5.81 bn, falls to lower level 5.78 bn, other exertions 3.90 bn, roadway 2.84 bn, slip/trip no fall 2.60 bn, caught in 2.16 bn, repetitive motion 1.83 bn, struck against 1.73 bn | 2022 data (index published 2025) | verified | https://www.riskeducation.org/wp-content/uploads/2025/11/Work-Safety-Index.pdf ; https://riskandinsurance.com/two-injury-types-drive-40-of-americas-58-78b-workplace-safety-bill/ |
| Cost of delayed claim handling (NCCI report-lag study, 44 states) | median claim cost USD 13,210 if reported within 2 weeks; +35 % (USD 17,785) if reported in week 3; USD 19,936 (+51 %) in week 4; late claims show more attorney involvement, lump sums, longer return-to-work | 2015 study | verified via Business Insurance summary | https://www.businessinsurance.com/delayed-reporting-of-workplace-injuries-can-raise-comp-claim-costs-up-to-51/ |
| Cost of an internal case review (labour) | claims adjusters, examiners and investigators: median USD 37.51 per hour, USD 78,000 per year, 376,100 employed | May 2025 | verified | https://www.bls.gov/ooh/business-and-financial/claims-adjusters-appraisers-examiners-and-investigators.htm |

### 3.2 Derivation of the FN:FP cost ratio

| Step | Assumption | Value |
|---|---|---|
| C_FP: extra review of a case flagged lost-time that is not | 2-4 adjuster hours at USD 37.51/h, loaded x1.4 for benefits and overhead, plus about 1 hour of mine safety-officer time (about USD 50) | USD 105 (2 h unloaded) to USD 260 (4 h loaded + 1 h); base USD 200 |
| C_FN: avoidable cost of missing a lost-time case at filing | Not the whole claim (the injury has already happened) but the increment caused by late recognition: no early reserve, no early return-to-work plan, no early medical management. Two anchors: (a) NCCI report-lag study: USD 4,575 to USD 6,726 extra per claim when handling slips by 1-2 weeks; (b) 5-10 % of the NCCI AY2025 average lost-time claim (USD 61.9 k) = USD 3,100 to USD 6,200 | USD 3,100 to USD 6,700; base USD 4,000 |
| Ratio C_FN / C_FP | base 4,000 / 200 | **20:1**; range 12:1 (3,100 / 260) to 45:1 (6,700 / 150) |
| Sensitivity set | conservative, base, aggressive | 10:1, 20:1, 30:1 |
| Decision threshold implied for a calibrated probability p | p* = C_FP / (C_FP + C_FN) = 1 / (1 + ratio) | 0.091 (10:1), 0.048 (20:1), 0.032 (30:1) |

Assumptions stated: (1) US cost levels (NCCI, BLS, OSHA) are used because the data are US; (2) societal costs (NSC USD 48,000
per medically consulted injury) are not used for C_FN because they include wage loss already borne regardless of prediction;
(3) indirect-cost multipliers (OSHA 1.1 for claims >= USD 10,000) are applied implicitly by using total claim cost rather
than medical cost alone; (4) C_FP excludes any benefit of the review (an extra review sometimes finds something), which makes
the ratio conservative; (5) with the project's positive class expected to be a large share of injury reports (to be
confirmed in the data profile), a threshold near 0.05 will trade specificity for recall, so the ratio should be treated as a
parameter in the threshold analysis rather than a fixed constant.

## 4. Uzbekistan relevance

| Indicator | Value | Year | Status | Source |
|---|---|---|---|---|
| Nominal GDP | 836.8 trillion soums (USD 72.65 bn) | 2022 | verified | USGS Minerals Yearbook 2022 (Safirova), https://pubs.usgs.gov/myb/vol3/2022/myb3-2022-uzbekistan.pdf |
| Mining and quarrying share (within industrial production) of GDP | 14.8 % (15.3 % in 2021) | 2022 | verified | same |
| Mining and quarrying contribution to industrial production in GDP | 12.2 % | 2024 | verified (USGS country page summarising the 2024 yearbook) | https://www.usgs.gov/centers/national-minerals-information-center/uzbekistan |
| Gold production | 106,700 kg (2022); 113,200 kg, 9th in the world, 3.9 % of world output (2024) | 2022 / 2024 | verified | USGS 2022 PDF; USGS country page |
| Copper production (content in concentrate) | 138,400 t (2022); 154,200 t (2024) | 2022 / 2024 | verified | same |
| Uranium production | 4,000 t | 2024 | verified | USGS country page |
| Gold share of export revenue | 21.3 % (2022, USGS); USD 8.1 bn = 33.4 % of USD 24.4 bn exports (2023); USD 7.48 bn of USD 26.94 bn = 27.7 % (2024) | 2022-2024 | verified | USGS 2022; https://timesca.com/?p=17256 ; https://kun.uz/en/news/2025/01/21/uzbekistans-foreign-trade-turnover-hit-66-billion-in-2024 |
| Mining and quarrying share of industrial output | 7.6 % of 85,057.6 bn soums (Jan 2026); 20.3 of 289.3 trn soums = 7.0 % (Jan-Mar 2026); 27.5 of about 394 trn soums = 7.0 % (Jan-Apr 2026) | 2026 | verified | National Statistics Committee press releases (2026a, 2026b, 2026c) |
| Navoiy region share of national industrial output | 27,126.6 of 85,057.6 bn soums = 31.9 % (Jan 2026); per-capita output 245.9 m soums vs national 22.4 m | Jan 2026 | verified | https://stat.uz/img/news/press-reliz-sanoat-25_02_2026-j_-eng_p17783.pdf |
| Largest employers (Ministry of Employment and Poverty Reduction list) | 1. Navoi Mining and Metallurgical Combinat JSC 48,308; 2. Almalyk Mining and Metallurgical Combinat JSC 36,629; 3. Temiryulinfratuzilma JSC 34,816 | 29 Apr 2024 | verified | https://kun.uz/en/news/2024/05/02/employment-ministry-names-uzbekistans-largest-employment-providers |
| NMMC 2024 results | revenue USD 7.4 bn (+29.5 % vs 5.7 bn); net profit USD 2.14 bn (+44.6 %); adjusted EBITDA USD 4.6 bn (62 % margin); gold 3.10 Moz (96.6 t, +5.4 %); AISC USD 979/oz; USD 1 bn debut Eurobond; USD 914 m development investment | 2024 | verified | https://kun.uz/en/news/2025/04/17/uzbekistans-gold-giant-nmmc-reports-446-profit-growth-in-2024 |
| NMMC share of GDP / budget | 6.4 % of GDP; 20.9 % of budget revenues | 2024 | not verified (mining.com op-ed blocked, HTTP 403) | https://www.mining.com/op-ed-how-nmmc-became-the-golden-asset-of-uzbekistans-economy/ |
| AMMC 2024 results | output 38.14 trn soums (+20.5 % vs 31.66 trn); exports USD 610.8 m (-8 % vs 661.9 m); taxes 10.79 trn soums; 51.69 kt copper products shipped; 130.0 kt copper from own raw materials | 2024 | verified (kun.uz); agmk.uz figure from search extract | https://kun.uz/en/news/2025/01/14/ammc-increases-production-by-20-in-2024-while-exports-decline-by-8 ; https://agmk.uz/en/news/okmkda-2024-yilda-barcha-ishlab-chiqarish-rejalari-bajarildi |
| Privatisation | Presidential Decree PP-145 of 21 Apr 2025 schedules IPOs of 12 large SOEs incl. partial offerings of NMMC, Navoiyuran and AMMC in 2025-2028; "mining and quarrying are the leading industries of Uzbekistan's economy" | 2025 | verified | https://www.trade.gov/country-commercial-guides/uzbekistan-mining-and-quarrying-sectors |
| OSH legal base | Law "On Occupational Safety" No. LRU-410 of 22 Sep 2016; Labour Code in new edition, Law No. ZRU-798 of 28 Oct 2022, in force 30 Apr 2023; Resolution No. 286 of 6 Jun 1997 on investigation and accounting of occupational accidents (employer files an accident report with investigation materials to the State Labour Inspectorate) | 2016-2023 | verified (GRATA 14 Nov 2025); Labour Code dates from GRATA/Mondaq search extract | https://gratanet.com/publications/occupational-health-and-safety-regulations-in-uzbekistan ; https://gratanet.com/news/the-labor-code-of-the-republic-of-uzbekistan-has-been-adopted-in-a-new-edition |
| OSH regulator | Ministry of Employment and Poverty Reduction, acting through the State Labour Inspectorate (investigates and records workplace accidents) | current | verified | https://kun.uz/en/news/2026/01/13/workplace-accidents-kill-248-people-in-uzbekistan-in-2025over-240-people |
| Industrial-safety regulator for hazardous production facilities (mines) | Committee for Industrial, Radiation and Nuclear Safety under the Cabinet of Ministers; draft law "On industrial safety at hazardous production facilities" (10 Feb 2026) introduces mandatory safety audits, declarations, employer insurance and fines; 2021-2025: 9,500 inspections, 72,000 violations, 7,800 fines (6.1 bn soums), 27 accidents causing > 51 bn soums damage, 317 occupational incidents, 81 fatalities, 187 seriously injured | 2021-2026 | verified | https://gov.uz/en/cirns/news/view/130368 |
| ILO conventions and projects | C155 (OSH) ratification law signed 3 Oct 2024; C167 (construction) ratified 9 Jun 2022, in force 9 Jun 2023; ILO project "Improving labour inspection and OSH systems in Uzbekistan" (UZB/24/01/DEU), 20 Dec 2023 - 31 Dec 2025, EUR 1,232,000, funded by BMZ, aims to strengthen the State Labour Inspectorate | 2022-2025 | verified | https://kun.uz/en/news/2024/10/03/uzbekistan-ratifies-ilo-convention-on-occupational-safety-and-health ; https://www.ilo.org/projects-and-partnerships/projects/improving-labour-inspection-and-occupational-safety-and-health-systems |
| National workplace-accident statistics 2025 (State Labour Inspectorate) | 849 accidents (50 group, 583 severe, 216 fatal); 248 deaths, 680 serious injuries, 33 minor; construction 234, light industry 47, transport 37; Tashkent city 153, Kashkadarya 93, Bukhara 92, Andijan 75; 432 criminal cases, 241 administrative fines | 2025 | verified | https://kun.uz/en/news/2026/01/13/workplace-accidents-kill-248-people-in-uzbekistan-in-2025over-240-people |
| 2026 to date | Q1: 198 accidents, 62 deaths, 146 serious, 15 minor (industry 71, construction 54; Navoi region 18). H1: 447 accidents, 137 deaths, 337 serious, 20 minor, 10.376 bn soums compensation. Jan-Jul: 529 accidents, 146 deaths, 402 serious, 22 minor (industry 168, construction 159; Navoi 31), 11.379 bn soums compensation | 2026 | verified | https://anhor.uz/news/work-accidents/ ; https://zamin.uz/en/society/212013-137-people-died-in-workplace-accidents-in-uzbekistan-in-6-months.html ; https://kun.uz/en/news/2026/08/11/workplace-accidents-kill-146-people-in-uzbekistan-over-seven-months |
| Fatal occupational injury rate | 3.60 per 100,000 workers (2024, -10 % vs 2023) | 2024 | not verified (aggregator site; ILOSTAT page not reachable) | https://statbase.ru/data/uzb-fatal-occupational-injuries/ |
| Comparator: US mining fatal rate | 0.0101 per 200,000 h = 10.1 per 100,000 FTE (FY2025) | 2025 | derived from MSHA (2026a) | see section 2 |
| Public row-level injury data for Uzbekistan | none found on data.gov.uz (see `docs/dataset_research.md`); the State Labour Inspectorate publishes aggregates only | 2026 | verified (project log) | https://kun.uz/en/news/2026/01/13/workplace-accidents-kill-248-people-in-uzbekistan-in-2025over-240-people |

Relevance in one line: the two largest employers in Uzbekistan are mining combinats (about 85,000 workers), mining supplies
roughly a quarter to a third of export revenue, the country has no public case-level injury dataset, and the regulator is
mid-reform (new Labour Code, C155, draft industrial-safety law), so a case-level lost-time model built on the MSHA form
fields (which the Resolution 286 accident report broadly mirrors: date, time, activity, injury nature, body part, experience)
is a transferable template.

## 5. Related work and how this project differs

| # | Citation (short) | Data | Target | Features | Headline result | Difference from this project |
|---|---|---|---|---|---|---|
| 1 | Yedla, Kakhki and Jannesari (2020), IJERPH 17(19):7054, doi 10.3390/ijerph17197054 | MSHA accident records | (a) injury outcome classes; (b) days away from work (regression) | tabular fields and narratives; word-embedding synthetic augmentation for imbalance | narratives beat tabular for outcome; tabular gave lower MSE for days away; key predictors shift start time, accident time, mining experience; augmentation raised F1 for minority classes (verified via https://ideas.repec.org/a/gam/jijerp/v17y2020i19p7054-d420263.html and https://scholarworks.sjsu.edu/faculty_rsca/1008) | This project is tabular-only (no narrative), uses a binary lost-time target with explicit exclusions, joins mine context and hours, and tunes a cost-based threshold |
| 2 | Dinesh, Brown, Burgess and Cui (2023), University of Arizona manuscript (NIOSH award, SMMR seed grant) | MSHA Accident Injuries, > 245,000 records since 2000; 50,000 training rows, 5,000 test rows | injured body part (ankle, back, eye, hand, knee, shoulder, other) from narrative text | TF-IDF/embeddings of narrative; SVM, RF, FastText, BERT; association-rule mining | F1 0.899 SVM, 0.898 RF, 0.884 BERT, 0.854 FastText within MSHA; transfer to 500 operator safety-interaction reports F1 0.61 best; 149 rules, 39 judged useful (verified, https://miningsh.arizona.edu/sites/default/files/documents/Brown-2023-MultiClass-Classifiers-Leading-Indicators.pdf) | Predicts body part, not severity; text-based; no join; no deployment |
| 3 | Manjunatha (2023), MSc thesis, National College of Ireland | Accidents left-joined to Mines: 252,085 rows x 39 columns (29 accident + 6 mine columns), 2000-2022 | degree of injury: 11 codes reduced to 5, then binary (days off vs not), then 3-class | categorical mine and accident fields; PySpark; 70:30 split; RF, XGBoost, DT, KNN, SGD, ANN | CS1 multiclass train 92 % / test 74 % (overfit); CS2 binary about 92 % accuracy (XGBoost, DT, ANN); CS3 3-class 91 % XGBoost (verified, https://norma.ncirl.ie/6559/) | Closest prior work. Differences: hours join (MinesProdYearly) giving exposure features, filing-time leakage drop-list, 2010+ window, imbalance handling inside pipelines, PR-AUC/TPR/TNR and cost-based threshold instead of accuracy, calibration, persisted models and a deployed app |
| 4 | Brown, Khademian, Mohamed, Rashed and Xue (2024), SME preprint 24-009 (NIOSH) | MSHA narratives 1983-2021 (711,960 events, 60 variables) | ground-fall accident cause category | narrative text classification | average F1 96 % (verified, https://stacks.cdc.gov/view/cdc/215666) | Text classification of cause, not outcome prediction |
| 5 | Shahsavar, Gomez and Sattarvand (2023), SME | MSHA 2020 records, 60 fields | narrative class | CNN and naive Bayes on narratives | results not shown on abstract page (verified abstract, https://onemine.org/documents/natural-language-processing-for-classification-of-narratives-from-msha-data) | Text; no severity target |
| 6 | Ganguli, Miller and Pothina (2021), Minerals 11(7):776, doi 10.3390/min11070776 | MSHA narratives, applied to a single mine's own narratives (University of Utah and Teck Red Dog) | nine accident types | nine random-forest text models | published 17 Jul 2021; author order not verified (MDPI page HTTP 403) | Text; operational narrative triage, not lost-time prediction |
| 7 | Journal of Environmental and Occupational Medicine (2023) 40, pp. 1115-1120 (authors not verified) | MSHA 2001-2021, metal miners, non-fatal injuries | binary: lost workdays >= 105 (serious) vs < 105 | Lasso / stepwise feature selection; logistic regression, GBDT, XGBoost (12 models) | GBDT best: sensitivity 0.949, specificity 0.753, Youden 0.702; predictors age, accident time, service length, cause, activity, body part, nature of injury (from search extract, https://pesquisa.bvsalud.org/gim/resource/en/wpr-998764) | Uses post-outcome days-lost threshold as label and includes outcome fields; metal only; no deployment |
| 8 | Milam (2023), MS thesis, Marshall University | MSHA violations paired with accidents within 35 days | whether a violation leads to an accident, and fatal vs non-fatal | violation fields; random forest with resampling | qualitative "high confidence" claims (verified abstract, https://mds.marshall.edu/etd/1813) | Violation-level, not injury-report-level |
| 9 | Gernand (2016), ASCE-ASME J. Risk Uncertainty Eng. Syst. B 2(4):041002, doi 10.1115/1.4032929 | MSHA inspection and injury data, mine-level, 12-month windows | lost-days rate; fatal or serious disabling injuries | inspection findings; CART and random forest | targeted oversight of highest-risk mines predicted to beat blanket penalty increases (verified, https://pure.psu.edu/en/publications/evaluating-the-effectiveness-of-mine-safety-enforcement-actions-i/) | Mine-year aggregate forecasting, not case-level |
| 10 | Santa Clara University engineering thesis (RF on narratives, F1 0.94 for injury outcome; most important predictors nature of injury and body part) | MSHA | injury outcome; days away | narratives + tabular | F1 0.94 | not verified (https://scholarcommons.scu.edu/eng_grad/17 returned HTTP 403); appears to be the thesis version of item 1 |

Project differentiation summary: (1) tabular features only, narrative deliberately excluded; (2) relational join of three MSHA
tables (Accidents, Mines, MinesProdYearly) adding mine type, commodity, size and exposure hours; (3) binary lost-time target
aligned to BLS DAFW plus fatality, with codes 00, 07-10 and "?" excluded; (4) filing-time feature set with a documented
leakage drop-list; (5) cost-sensitive threshold from the 20:1 ratio in section 3 and calibration; (6) deployed Streamlit app.
None of the ten works above combines items 2, 5 and 6.

## 6. Licence and terms of use

| Item | Text / fact | Status | Source |
|---|---|---|---|
| 17 U.S.C. 105(a) | "Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise." | verified (2020 edition) | https://www.govinfo.gov/content/pkg/USCODE-2020-title17/html/USCODE-2020-title17-chap1-sec105.htm |
| 17 U.S.C. 101 "work of the United States Government" | "a work prepared by an officer or employee of the United States Government as part of that person's official duties" | verified (quoted in the 105 revision notes) | same |
| data.gov licensing guidance | "Government works are by default in the U.S. Public Domain" under 17 U.S.C. 105; recommended labels http://www.usa.gov/publicdomain/label/1.0/ (legacy) or CC0 https://creativecommons.org/publicdomain/zero/1.0/ for international use; open licences must permit "reproductions, modifications and derivative works" without royalties | verified | https://resources.data.gov/open-licenses/ |
| DOL copyright statement | "Materials created by the federal government are generally part of the public domain and may be used, reproduced and distributed without permission"; attribution to "U.S. Department of Labor" and/or dol.gov requested; no suggestion of endorsement; third-party content on DOL sites may be copyrighted | verified | https://www.dol.gov/general/aboutdol/copyright |
| DOL disclaimer | "We do not control or guarantee the accuracy, relevance, timeliness, or completeness of this outside information"; delays between official publication and web appearance | verified | https://www.dol.gov/general/disclaim |
| data.gov catalogue entry | "MSHA Accident Injuries Data Set", publisher MSHA / Department of Labor; access level public; licence field not populated; dataset last updated 12 Mar 2026; description: "All accidents, injuries and illnesses reported by mine operators and contractors. Obtained from the Mine Accident, Injury and Illness Report form (MSHA Form 7000-1)" | verified | https://catalog.data.gov/dataset/msha-accident-injuries-data-set |
| MSHA Open Government portal | 20+ datasets with definition files; "These files are updated every Friday afternoon unless otherwise noted here"; links only to the DOL privacy statement and disclaimer, no separate licence text | verified | https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp |
| Form 7000-1 privacy notice | Personal identifiers (name, last four SSN digits) are collected under 30 U.S.C. 813 but are not released in the public extract; the public file contains no names | verified | https://www.msha.gov/sites/default/files/Support_Resources/Forms/7000-1.pdf |

Licence statement for `data/README.md` (factual, derived from the above): "The Accidents, Mines and MinesProdYearly files are
works of the United States Government (U.S. Department of Labor, Mine Safety and Health Administration) and are in the public
domain under 17 U.S.C. 105. They were downloaded from https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp on 14 October
2026 (file dates 9 October 2026; SHA-256 in data/raw/SNAPSHOT.json). MSHA refreshes the files weekly, so figures in this
report may differ from the live files. Source attribution: U.S. Department of Labor, MSHA. Use of the data does not imply
endorsement by MSHA or DOL."

## Reference list (Harvard)

Anhor.uz (2026) *V Uzbekistane bolee 200 chelovek postradali pri neschastnykh sluchayakh na rabote* [More than 200 people injured in workplace accidents in Uzbekistan, Q1 2026]. 29 April. Available at: https://anhor.uz/news/work-accidents/ (Accessed: 14 October 2026).

Brown, C., Khademian, Z., Mohamed, K., Rashed, G. and Xue, Y. (2024) 'An integrated method to classify ground-fall accidents and to estimate ground-fall trends in U.S. mines using machine learning algorithms', *SME Annual Conference preprint 24-009*. Society for Mining, Metallurgy and Exploration, pp. 1-11. Available at: https://stacks.cdc.gov/view/cdc/215666 (Accessed: 14 October 2026).

Bureau of Labor Statistics (2025) *Occupational Outlook Handbook: Claims adjusters, appraisers, examiners, and investigators*. Washington, DC: U.S. Department of Labor. Available at: https://www.bls.gov/ooh/business-and-financial/claims-adjusters-appraisers-examiners-and-investigators.htm (Accessed: 14 October 2026).

Bureau of Labor Statistics (no date) *Handbook of Methods: Survey of Occupational Injuries and Illnesses - Concepts*. Available at: https://www.bls.gov/opub/hom/soii/concepts.htm (Accessed: 14 October 2026).

Business Insurance (2015) 'Delayed reporting of workplace injuries can raise comp claim costs up to 51%', 7 June. Available at: https://www.businessinsurance.com/delayed-reporting-of-workplace-injuries-can-raise-comp-claim-costs-up-to-51/ (Accessed: 14 October 2026).

Committee for Industrial, Radiation and Nuclear Safety under the Cabinet of Ministers of the Republic of Uzbekistan (2026) *New law draft on industrial safety at hazardous production facilities*. 10 February. Available at: https://gov.uz/en/cirns/news/view/130368 (Accessed: 14 October 2026).

Data.gov (2026) *MSHA Accident Injuries Data Set* [dataset catalogue entry]. Available at: https://catalog.data.gov/dataset/msha-accident-injuries-data-set (Accessed: 14 October 2026).

Data.gov (no date) *Open licenses*. Available at: https://resources.data.gov/open-licenses/ (Accessed: 14 October 2026).

Dinesh, A., Brown, L.D., Burgess, J.L. and Cui, H. (2023) *Using multi-class classifiers to predict injuries and identify leading indicators in mine industry safety report corpus*. Manuscript draft, University of Arizona Mining Safety and Health Research Center. Available at: https://miningsh.arizona.edu/sites/default/files/documents/Brown-2023-MultiClass-Classifiers-Leading-Indicators.pdf (Accessed: 14 October 2026).

Ganguli, R., Miller, P. and Pothina, R. (2021) 'Effectiveness of natural language processing based machine learning in analyzing incident narratives at a mine', *Minerals*, 11(7), 776. doi: 10.3390/min11070776. Available at: https://www.mdpi.com/2075-163X/11/7/776 (Accessed: 14 October 2026; author list not verified).

Gen Re (2026) *Workers' comp healthy amid change*. 26 May. Available at: https://www.genre.com/us/knowledge/publications/2026/may/workers-comp-healthy-amid-change-en (Accessed: 14 October 2026).

Gernand, J.M. (2016) 'Evaluating the effectiveness of mine safety enforcement actions in forecasting the lost-days rate at specific worksites', *ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part B: Mechanical Engineering*, 2(4), 041002. doi: 10.1115/1.4032929. Available at: https://pure.psu.edu/en/publications/evaluating-the-effectiveness-of-mine-safety-enforcement-actions-i/ (Accessed: 14 October 2026).

GRATA International (2025) *Occupational health and safety regulations in Uzbekistan*. 14 November. Available at: https://gratanet.com/publications/occupational-health-and-safety-regulations-in-uzbekistan (Accessed: 14 October 2026).

GRATA International (no date) *The Labor Code of the Republic of Uzbekistan has been adopted in a new edition*. Available at: https://gratanet.com/news/the-labor-code-of-the-republic-of-uzbekistan-has-been-adopted-in-a-new-edition (Accessed: 14 October 2026).

International Labour Organization (1998) *Resolution concerning statistics of occupational injuries (resulting from occupational accidents)*, adopted by the Sixteenth International Conference of Labour Statisticians, Geneva, October 1998. Available at: https://www.ilo.org/resource/resolution-concerning-statistics-occupational-injuries-resulting (Accessed: 14 October 2026).

International Labour Organization, Department of Statistics (no date) *Statistics of occupational injuries: resolution concerning statistics of occupational injuries resulting from occupational accidents* [presentation]. Available at: https://www.ilo.org/media/289311/download (Accessed: 14 October 2026).

International Labour Organization (no date) *Improving labour inspection and occupational safety and health systems in Uzbekistan* [project UZB/24/01/DEU]. Available at: https://www.ilo.org/projects-and-partnerships/projects/improving-labour-inspection-and-occupational-safety-and-health-systems (Accessed: 14 October 2026).

Journal of Environmental and Occupational Medicine (2023) 'Applicability of feature selection combined with Boosting algorithm in severity prediction of non-fatal occupational injuries in miners', *Journal of Environmental and Occupational Medicine*, 40, pp. 1115-1120. Available at: https://pesquisa.bvsalud.org/gim/resource/en/wpr-998764 (Accessed: 14 October 2026; authors, issue and DOI not verified).

Kun.uz (2024a) 'Employment ministry names Uzbekistan's largest employment providers', 2 May. Available at: https://kun.uz/en/news/2024/05/02/employment-ministry-names-uzbekistans-largest-employment-providers (Accessed: 14 October 2026).

Kun.uz (2024b) 'Uzbekistan ratifies ILO convention on occupational safety and health', 3 October. Available at: https://kun.uz/en/news/2024/10/03/uzbekistan-ratifies-ilo-convention-on-occupational-safety-and-health (Accessed: 14 October 2026).

Kun.uz (2025a) 'AMMC increases production by 20% in 2024, while exports decline by 8%', 14 January. Available at: https://kun.uz/en/news/2025/01/14/ammc-increases-production-by-20-in-2024-while-exports-decline-by-8 (Accessed: 14 October 2026).

Kun.uz (2025b) 'Uzbekistan's foreign trade turnover hit $66 billion in 2024', 21 January. Available at: https://kun.uz/en/news/2025/01/21/uzbekistans-foreign-trade-turnover-hit-66-billion-in-2024 (Accessed: 14 October 2026).

Kun.uz (2025c) 'Uzbekistan's gold giant NMMC reports 44.6% profit growth in 2024', 17 April. Available at: https://kun.uz/en/news/2025/04/17/uzbekistans-gold-giant-nmmc-reports-446-profit-growth-in-2024 (Accessed: 14 October 2026).

Kun.uz (2026a) 'Workplace accidents kill 248 people in Uzbekistan in 2025', 13 January. Available at: https://kun.uz/en/news/2026/01/13/workplace-accidents-kill-248-people-in-uzbekistan-in-2025over-240-people (Accessed: 14 October 2026).

Kun.uz (2026b) 'Workplace accidents kill 146 people in Uzbekistan over seven months', 11 August. Available at: https://kun.uz/en/news/2026/08/11/workplace-accidents-kill-146-people-in-uzbekistan-over-seven-months (Accessed: 14 October 2026).

Legal Information Institute (no date a) *30 CFR 50.2 - Definitions*. Cornell Law School. Available at: https://www.law.cornell.edu/cfr/text/30/50.2 (Accessed: 14 October 2026).

Legal Information Institute (no date b) *30 CFR 50.10 - Immediate notification*. Available at: https://www.law.cornell.edu/cfr/text/30/50.10 (Accessed: 14 October 2026).

Legal Information Institute (no date c) *30 CFR 50.20 - Preparation and submission of MSHA Report Form 7000-1*. Available at: https://www.law.cornell.edu/cfr/text/30/50.20 (Accessed: 14 October 2026).

Legal Information Institute (no date d) *30 CFR 50.20-3 - Criteria - Differences between medical treatment and first aid*. Available at: https://www.law.cornell.edu/cfr/text/30/50.20-3 (Accessed: 14 October 2026).

Legal Information Institute (no date e) *30 CFR 50.30 - Preparation and submission of MSHA Form 7000-2*. Available at: https://www.law.cornell.edu/cfr/text/30/50.30 (Accessed: 14 October 2026).

Legal Information Institute (no date f) *29 CFR 1904.7 - General recording criteria*. Available at: https://www.law.cornell.edu/cfr/text/29/1904.7 (Accessed: 14 October 2026).

Liberty Mutual Insurance (2025) *2025 Workplace Safety Index: annual report from Risk Control Services*. Boston, MA: Liberty Mutual. Available at: https://www.riskeducation.org/wp-content/uploads/2025/11/Work-Safety-Index.pdf (Accessed: 14 October 2026).

Manjunatha, A. (2023) *Injury prediction in mining industry through applied machine learning approaches*. MSc thesis. National College of Ireland, Dublin. Available at: https://norma.ncirl.ie/6559/ (Accessed: 14 October 2026).

Milam, O. (2023) *Identifying hazardous patterns in MSHA data using random forests*. MS thesis. Marshall University. Available at: https://mds.marshall.edu/etd/1813 (Accessed: 14 October 2026).

Mine Safety and Health Administration (2025) *Mine Safety and Health At a Glance, FY2018-FY2024*, 31 January. Arlington, VA: U.S. Department of Labor. Available at: https://www.msha.gov/sites/default/files/Data_Reports/Charts/FY2024-MSHA-ataGlance-1-31-2025.pdf (Accessed: 14 October 2026).

Mine Safety and Health Administration (2026a) *Mine Safety and Health At A Glance, FY2019-FY2025*, 27 February. Available at: https://www.msha.gov/sites/default/files/MSHA-Glance/FY2025-MSHA-At-A-Glance.pdf (Accessed: 14 October 2026).

Mine Safety and Health Administration (2026b) *MSHA By the Numbers FY1978-2025: includes fatal and injury rates*. Available at: https://www.msha.gov/sites/default/files/MSHA-Glance/By%20The%20Numbers%20FY%201978%20-%202025.pdf (Accessed: 14 October 2026).

Mine Safety and Health Administration (2026c) *Mine Injury and Worktime, Quarterly: January-December 2025, preliminary*. Available at: https://arlweb.msha.gov/Stats/Part50/WQ/2025/MIWQ%20-%20CY2025Q4.pdf (Accessed: 14 October 2026).

Mine Safety and Health Administration (no date a) *Mine Accident, Injury and Illness Report, MSHA Form 7000-1 (OMB No. 1219-0007), with instructions*. Available at: https://www.msha.gov/sites/default/files/Support_Resources/Forms/7000-1.pdf (Accessed: 14 October 2026).

Mine Safety and Health Administration (no date b) *Open Government Initiative Portal: MSHA data sets*. Available at: https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp (Accessed: 14 October 2026).

Mine Safety and Health Administration (no date c) *Accidents definition file*. Available at: https://arlweb.msha.gov/OpenGovernmentData/DataSets/Accidents_Definition_File.txt (Accessed: 14 October 2026).

Mine Safety and Health Administration (no date d) *Fatality reports*. Available at: https://www.msha.gov/data-and-reports/fatality-reports/search (Accessed: 14 October 2026).

National Council on Compensation Insurance (2026) *State of the Line: workers compensation* [presentation to NCOIL, 17 July 2026]. Available at: https://ncoil.org/wp-content/uploads/2026/07/2026-NCCI-SOTL-NCOIL.pdf (Accessed: 14 October 2026).

National Institute for Occupational Safety and Health (no date) *NIOSH Mine and Mine Worker Charts: number and rate of nonfatal lost-time injuries, 2021-2024*. Available at: https://wwwn.cdc.gov/niosh-mining/MMWC/Injuries/NumberAndRate (Accessed: 14 October 2026).

National Safety Council (no date a) *Injury Facts: work injury costs*. Available at: https://injuryfacts.nsc.org/work/costs/work-injury-costs/ (Accessed: 14 October 2026; page refused automated fetch, figures taken from the search engine extract).

National Safety Council (no date b) *Injury Facts: DART (days away, restricted, or transferred cases)*. Available at: https://injuryfacts.nsc.org/definitions/dart-days-away-restricted-or-transferred-cases/ (Accessed: 14 October 2026; page refused automated fetch).

National Statistics Committee of the Republic of Uzbekistan (2026a) *Press release: industrial production in the Republic of Uzbekistan for January 2026*, 28 February. Available at: https://stat.uz/img/news/press-reliz-sanoat-25_02_2026-j_-eng_p17783.pdf (Accessed: 14 October 2026).

National Statistics Committee of the Republic of Uzbekistan (2026b) *Press release: industrial production, January-March 2026*. Available at: https://stat.uz/img/press-relizlar/press-reliz-sanoat-30_03_2025-j_-eng1_p88969.pdf (Accessed: 14 October 2026).

National Statistics Committee of the Republic of Uzbekistan (2026c) *Press release: industrial production, January-April 2026*. Available at: https://stat.uz/img/news/press_release_industrial_production_p62140.pdf (Accessed: 14 October 2026).

North American Mining (2026) 'MSHA reported 33 fatalities in 2025', 9 January. Available at: https://northamericanmining.com/index.php/2026/01/09/msha-reported-33-fatalities-in-2025/ (Accessed: 14 October 2026).

Occupational Safety and Health Administration (no date a) *$afety Pays: individual injury estimator - background of cost estimates*. Available at: https://www.osha.gov/safetypays/background (Accessed: 14 October 2026).

Occupational Safety and Health Administration (no date b) *$afety Pays: estimator (text version)*. Available at: https://www.osha.gov/safetypays/estimator-text (Accessed: 14 October 2026).

Office of the Law Revision Counsel (2020) *17 U.S.C. 105 - Subject matter of copyright: United States Government works*. United States Code, 2020 edition. Available at: https://www.govinfo.gov/content/pkg/USCODE-2020-title17/html/USCODE-2020-title17-chap1-sec105.htm (Accessed: 14 October 2026).

Pit & Quarry (2024) 'Mining fatalities skyrocket in August, September', 9 October. Available at: https://www.pitandquarry.com/?p=97926 (Accessed: 14 October 2026).

Pit & Quarry (2026) 'MSHA: industry finishes 2025 with 33 miner fatalities', 5 January. Available at: https://www.pitandquarry.com/msha-industry-finishes-2025-with-33-miner-fatalities/ (Accessed: 14 October 2026).

PMC (2021) *Table 2: distribution of medical-only and lost-time claims, NCCI data 1988-2020* [in an open-access article, PMC8478314]. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC8478314/table/T2 (Accessed: 14 October 2026; captcha-blocked, not verified).

Risk & Insurance (2025a) 'Workers' comp stays profitable in 2024 despite lower premiums and rising claim severity', 19 May. Available at: https://riskandinsurance.com/workers-comp-stays-profitable-in-2024-despite-lower-premiums-and-rising-claim-severity/ (Accessed: 14 October 2026).

Risk & Insurance (2025b) 'Two injury types drive 40% of America's $58.78B workplace safety bill', 18 July. Available at: https://riskandinsurance.com/two-injury-types-drive-40-of-americas-58-78b-workplace-safety-bill/ (Accessed: 14 October 2026).

Safirova, E. (2024) '2022 Minerals Yearbook: Uzbekistan [advance release]', *U.S. Geological Survey Minerals Yearbook*, vol. III. Reston, VA: U.S. Geological Survey. Available at: https://pubs.usgs.gov/myb/vol3/2022/myb3-2022-uzbekistan.pdf (Accessed: 14 October 2026).

Shahsavar, M., Gomez, J. and Sattarvand, J. (2023) 'Natural language processing for classification of narratives from MSHA data', *SME Annual Conference*. Society for Mining, Metallurgy and Exploration. Available at: https://onemine.org/documents/natural-language-processing-for-classification-of-narratives-from-msha-data (Accessed: 14 October 2026).

Statbase (no date) *Fatal occupational injuries: Uzbekistan*. Available at: https://statbase.ru/data/uzb-fatal-occupational-injuries/ (Accessed: 14 October 2026; not verified).

The Times of Central Asia (2024) 'Uzbekistan's first quarter 2024 gold exports rise to $2.66 billion', 25 April. Available at: https://timesca.com/?p=17256 (Accessed: 14 October 2026).

U.S. Department of Commerce, International Trade Administration (2025) *Uzbekistan country commercial guide: mining and quarrying sectors*, 9 December. Available at: https://www.trade.gov/country-commercial-guides/uzbekistan-mining-and-quarrying-sectors (Accessed: 14 October 2026).

U.S. Department of Labor (no date a) *Copyright information*. Available at: https://www.dol.gov/general/aboutdol/copyright (Accessed: 14 October 2026).

U.S. Department of Labor (no date b) *Disclaimers*. Available at: https://www.dol.gov/general/disclaim (Accessed: 14 October 2026).

U.S. Geological Survey (no date) *Uzbekistan* [National Minerals Information Center country page]. Available at: https://www.usgs.gov/centers/national-minerals-information-center/uzbekistan (Accessed: 14 October 2026).

Yedla, A., Kakhki, F.D. and Jannesari, A. (2020) 'Predictive modeling for occupational safety outcomes and days away from work analysis in mining operations', *International Journal of Environmental Research and Public Health*, 17(19), 7054. doi: 10.3390/ijerph17197054. Available at: https://ideas.repec.org/a/gam/jijerp/v17y2020i19p7054-d420263.html (Accessed: 14 October 2026).

Zamin.uz (2026) '137 people died in workplace accidents in Uzbekistan in 6 months', 12 July. Available at: https://zamin.uz/en/society/212013-137-people-died-in-workplace-accidents-in-uzbekistan-in-6-months.html (Accessed: 14 October 2026).
