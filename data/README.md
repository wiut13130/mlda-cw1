# Data card: MSHA accidents, mines and annual production (student ID 13130)

The analytic table joins three open-data extracts from the US Mine Safety and Health
Administration (MSHA). It has one row per injured person on a Form 7000-1 from 2010 to 2025,
with the mine's attributes and its previous-year exposure and case history. The numbers below
match `docs/profile/decisions.json`, where the strict filing-date counts were corrected on
14 October 2026, and `mlda.data.join.load_base_table()` reproduced them on the same day.

## 1. Sources and snapshot

Publisher: MSHA Open Government Data, https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp.
All files were downloaded on 2026-10-14 at 12:06:01 UTC. The three data archives are the
versions MSHA published on Friday 9 October 2026. MSHA refreshes them every week, so the hashes
pin the exact version analysed. The SHA-256 digests below are also recorded in
`data/raw/SNAPSHOT.json` and in `mlda.config.SOURCES`, and `uv run mlda-download` checks them
before extracting.

| File | Bytes | Last modified (server) | SHA-256 |
|---|---|---|---|
| `Accidents.zip` (member `Accidents.txt`) | 52,269,752 | 2026-10-09 11:12:32 GMT | `62d0c861a5c3a8e76bd06de79ac60e4dcc643bca148501cb3fc752fa76c6687d` |
| `Mines.zip` (member `Mines.txt`) | 7,308,890 | 2026-10-09 11:12:35 GMT | `1a0957ff00496855289725e351797d1737faad53c48ef7bf5ac51d725e997e02` |
| `MinesProdYearly.zip` (member `MinesProdYearly.txt`) | 7,072,831 | 2026-10-09 11:12:36 GMT | `c0fdca86d1756fc03b3f22dbc9430a02cefae9a8afac5d8e9e8503f590a3c342` |
| `Accidents_Definition_File.txt` | 10,214 | 2014-10-27 21:51:09 GMT | `c7681808dd372b6c5bd95fdf70b01e856e0d2501b7e5d7cc9185f4c4e1f876e5` |
| `Mines_Definition_File.txt` | 10,285 | 2023-05-16 13:47:39 GMT | `efd2bc2c493207a464be7c4e4795ffd08a221588b76b716ee334a75dfc14f1f9` |
| `MinesProdYearly_Definition_File.txt` | 2,023 | 2011-02-25 19:30:17 GMT | `2931d666cf837770bdb12889e1f6fd13b024eed87f52c264a95b4e37c290ab65` |

URLs: `https://arlweb.msha.gov/OpenGovernmentData/DataSets/<file>` for each file name above.

Licence: the files are works of the United States federal government, published as open
government data without a licence text. Under 17 U.S.C. 105 such works are not subject to
copyright in the United States, so they may be copied, redistributed and analysed freely. The
archives are committed to this repository as downloaded. The extracted `.txt` files are
git-ignored and recreated by `uv run mlda-download`.

Encoding: the files are `|`-delimited, with double-quoted strings and bytes in the latin-1
range (a few Spanish narratives and some mojibake in free-text directions). They are read with
`encoding="latin-1"`. No join key, code or numeric column contains a non-ASCII byte, so this
encoding loses nothing the models use. `mlda.data.load` turns the placeholders `""`, `?`, `? `,
`NO VALUE FOUND`, `No Value Found` and `NULL` into missing values.

## 2. Table grain and keys

| Table | Rows x columns | Grain (one row is) | Primary key |
|---|---|---|---|
| `Accidents.txt` | 275,220 x 57 | one injured person on one accident/injury/illness form (Form 7000-1), or one accident without injury (degree code 00) | `DOCUMENT_NO` (12 characters: type digit, Julian filing date YYYYDDD, sequence) |
| `Mines.txt` | 92,060 x 59 | one mine, with its attributes as of the snapshot date | `MINE_ID` (7-digit zero-padded string) |
| `MinesProdYearly.txt` | 658,922 x 11 | employee-hours and average employment for one mine, calendar year and subunit (Form 7000-2) | `MINE_ID`, `CALENDAR_YR`, `SUBUNIT_CD` |

The header names match the Accidents and Mines codebooks. The 2011 production codebook uses
names that differ from the file header (`CALENDAR_YR` = CAL_YR, `ANNUAL_HRS` = ANNUAL_HOURS,
`AVG_ANNUAL_EMPL` = AVG_EMPLOYEE_CNT, `C_M_IND` = COAL_METAL_IND, `CURR_MINE_NM` = MINE_NAME,
`STATE_ABBR` = STATE, `SUBUNIT_DESC` = SUBUNIT); the file header names are used throughout.

## 3. Joins

All joins are left joins from the filtered accident rows, with `validate="many_to_one"` and an
indicator column. After each join the code asserts that the row count has not changed.

| Join | Right table | Keys | Matched | Unmatched | Action |
|---|---|---|---|---|---|
| J1 | Mines | `MINE_ID` | 101,609 (100.0 %) | 0 | keep `PRIMARY_CANVASS`, `PRIMARY_SIC`, `DISTRICT`, `LATITUDE`, `LONGITUDE`, `PORTABLE_OPERATION`, `SAFETY_COMMITTEE_IND`, `PART48_TRAINING`, `CURRENT_103I`, `CURRENT_103I_DT`, `STATE`, `CURRENT_MINE_TYPE` (EDA only) and `COAL_METAL_IND` as `mine_coal_metal_ind`; the two commodity indicators agree on 100.0 % of rows |
| J2 | Production aggregated to mine-year (`hours` = sum of `ANNUAL_HRS`, `employees` = sum of `AVG_ANNUAL_EMPL`, set of `SUBUNIT_CD`) | `MINE_ID`, `CAL_YR - 1` = `CALENDAR_YR` | 100,403 (98.81 %) | 1,206 | `prev_hours`, `prev_employees`, `prev_subunits`, `prev_year_matched`; unmatched rows kept with NaN |
| J3 | the same aggregate | `MINE_ID`, `CAL_YR` = `CALENDAR_YR` | 101,477 (99.87 %) | 132 | `same_year_hours`, `same_year_employees`: EDA only, never features (the year is unfinished at filing) |
| J4 | previous-year subunit set (a column of J2) | as J2 | 100,403 | 1,206 | `type_at_time`: Underground if the mine-year has a production row for subunit 01 or 02 (hours may be zero), Facility if every subunit is in {30, 99, 17}, else Surface; fallback from the accident's own `SUBUNIT_CD` for the 1,206 rows without previous-year production. Underground 40,379, Surface 51,053, Facility 10,177 |
| Lagged cases | the full Accidents file, all degree codes, 2009-2024 | `MINE_ID`, `CAL_YR - 1`, and filing date (`DOCUMENT_NO` positions 2-8) strictly before the row's `ACCIDENT_DT` | 100,403 rows (17,822 true zeros; 81,469 rows have at least one valid-code case, so `prev_lost_time_share` is missing on 20,140 rows, 19.82 %) | NaN where J2 is unmatched | `prev_incidents` (any code), `prev_valid_cases` (codes 01-06), `prev_lost_time_cases` (codes 01-04); sensitivity-only copies `*_resolved` also require the return-to-work date and the closing document, when present, to be earlier than `ACCIDENT_DT`; `prev_year_incomplete_at_filing` marks 1,643 forms filed by 15 January |

Production is taken from the previous year because Form 7000-2 is filed quarterly, 15 days
after the quarter ends. For the 25 % of accidents in the first quarter, no part of the current
year has been reported at filing, while the fourth quarter of the previous year is due by
15 January.

The lagged cases are counted by filing date, not accident date, because 1.87 % of previous-year
cases were filed after the row's accident date (11.7 % of rows with any prior case, 21.2 % of
first-quarter rows). Counting by accident date would use information that did not exist when
the form was filed.

## 4. Target

`lost_time` = 1 when `DEGREE_INJURY_CD` is 01 (fatality), 02 (permanent disability), 03 (days
away from work only) or 04 (days away and restricted activity); 0 when it is 05 (restricted
activity only) or 06 (no days away, no restrictions). Codes 01-04 trigger indemnity in workers'
compensation, because wage replacement starts when the worker is away. That is why 05 counts
as negative even though MSHA's published lost-time rate includes it. The label is MSHA's
closed-case classification, and 99.14 % of positives have `DAYS_LOST > 0` or
`SCHEDULE_CHARGE > 0`.

| Quantity | Value |
|---|---|
| Rows (N) | 101,609 |
| Positive (01-04) | 49,367 |
| Negative (05-06) | 52,242 |
| Prevalence | 48.585 % |
| Prevalence by commodity | coal 63.04 % (n 41,461), metal/non-metal 38.62 % (n 60,148) |

The label partly measures return-to-work practice, that is, whether the operator offers
restricted duty. It is therefore reported by `COAL_METAL_IND` (see `label_drift` in
`docs/profile/decisions.json`).

## 5. Row filters (applied in this order)

| Step | Rule | Rows removed | Rows kept | Reason |
|---|---|---|---|---|
| 0 | all rows | | 275,220 | |
| 1 | `CAL_YR >= 2010` | 149,502 | 125,718 | older code lists and reporting practice; experience fields blank for 20 % of pre-2010 rows; 2009 production remains available for the lag features |
| 2 | `CAL_YR <= 2025` | 4,177 | 121,541 | 2026 is partial and its degree codes are provisional at filing (47 % of 2026 code-03/04 rows lack `DAYS_LOST`) |
| 3 | drop degree code 00 | 12,258 | 109,283 | accident without an injured person |
| 4 | drop degree code 07 | 5,476 | 103,807 | occupational illness, diagnosis-dated rather than accident-dated |
| 5 | drop degree code 08 | 1,001 | 102,806 | natural causes |
| 6 | drop degree code 09 | 347 | 102,459 | non-employee |
| 7 | drop degree code 10 | 850 | 101,609 | all other cases including first aid (0 rows with days lost or restricted) |
| 8 | drop missing degree code | 0 | 101,609 | all 910 missing codes are pre-2010 |

The filtered table has 101,609 rows from 16 calendar years and 8,303 mines. They fall into
32,732 mine-years (`group_id` = `MINE_ID_CAL_YR`, the grouping key of the train/test split) and
98,693 same-day accident groups (`accident_group` = `MINE_ID_ACCIDENT_DT`, the multi-victim key).
Every same-day group lies inside one mine-year.

## 6. Leakage drop-list

The base table keeps these columns for EDA and label-validity checks, but they are never model
inputs (`mlda.config.LEAKAGE_COLUMNS`, `MINES_CURRENT_STATE_COLUMNS` and the same-year
production columns).

| Column(s) | Reason |
|---|---|
| `DEGREE_INJURY_CD`, `DEGREE_INJURY` | define the target |
| `DAYS_LOST` | outcome, finalised on the closing form; > 0 for 95.3 % of positives and 0.0 % of negatives |
| `DAYS_RESTRICT` | outcome, closing form |
| `SCHEDULE_CHARGE` | statutory days charged to death or permanent injury; > 0 only for positives |
| `TRANS_TERM` | recorded on the closing form; blank rows are 97.5 % positive |
| `RETURN_TO_WORK_DT` | entered when the worker returns; median lag 63 days for positives vs 1 day for negatives |
| `CLOSED_DOC_NO` | exists only after the case closes; present for 72.2 % of positives vs 33.5 % of negatives |
| `INVEST_BEGIN_DT` | an MSHA investigation follows severity and is entered after filing |
| `NARRATIVE` | free text written at filing that may contain outcome words (fatality wording is 98.4 % positive); excluded by project scope and not persisted |
| `CURRENT_MINE_STATUS`, `CURRENT_STATUS_DT` | snapshot state; post-dates the accident for 40.4 % of rows |
| `CURRENT_MINE_TYPE` | differs from the type at the time for 12.3 % of rows; replaced by `type_at_time` |
| `COMPANY_TYPE`, `CURRENT_CONTROLLER_BEGIN_DT` | a dated "unknown" level would encode accident age (41 % of 2010 rows, 3 % of 2025 rows, 0 % at deployment); raw `COMPANY_TYPE` carries no signal |
| `NO_EMPLOYEES`, `DAYS_PER_WEEK`, `HOURS_PER_SHIFT`, `PROD_SHIFTS_PER_DAY`, `MAINT_SHIFTS_PER_DAY`, `AVG_MINE_HEIGHT`, `METHANE_LIBERATION`, `MINE_GAS_CATEGORY_CD`, `NO_PRODUCING_PITS`, `NO_NONPRODUCING_PITS`, `NO_TAILING_PONDS` | current Mine Information Form values; zeros encode closure after the accident year (20-27 % of rows) |
| `MILES_FROM_OFFICE` | 0 means not recorded; no within-commodity signal |
| `same_year_hours`, `same_year_employees` and any same-year rate | not known at filing; the current year's counts would include the row itself |
| filing lag (`DOCUMENT_NO` date minus `ACCIDENT_DT`) | technically known, but late filing happens when the case is still open |

Identifier and description columns (`MINE_ID`, `CONTROLLER_*`, `OPERATOR_*`, `CONTRACTOR_ID`,
`DOCUMENT_NO`, `CAL_YR`, `ACCIDENT_DT`, the description twins of every code, `EQUIP_MODEL_NO`,
and the Mines name, office, county and SIC detail columns) are dropped at feature engineering.
The full list is `drop_admin` in `docs/profile/decisions.json`.

## 7. Processed artefacts (`data/processed/`, written by `load_base_table()`)

| File | Grain | Rows | Notes |
|---|---|---|---|
| `base.parquet` | one injured person, 2010-2025, degree 01-06 | 101,609 x 88 | every raw accident column except `NARRATIVE`, `CONTROLLER_NAME` and `OPERATOR_NAME`, the J1 mine columns and the derived columns (including the four sensitivity-only columns); zstd, 7.3 MB |
| `mine_year_features.parquet` | `MINE_ID` x `CALENDAR_YR`, 2009-2025, all mines | 226,182 | `hours`, `employees`, `subunits` (`|`-joined), `type`; the inference page looks up `(MINE_ID, year - 1)` here |
| `mine_case_history.parquet` | one Accidents case, any degree code, 2009-2025 | 132,509 | `MINE_ID`, `CAL_YR`, `filing_date`, `DEGREE_INJURY_CD`, `return_to_work_dt`, `closed_date`; lets the inference page rebuild the filed-before counts for any accident date |

## 8. Known limitations of the data

Details are in `target.label_validity` of `docs/profile/decisions.json`.

- Part 50 reports are filed by the operators themselves.
- The Mines table is a snapshot, so its dated attributes may post-date the accident.
- The production file has no contractor hours, so rates per 200,000 hours understate exposure
  at mines that rely heavily on contractors.
- MSHA may still re-code about 5 % of the 2025 fourth-quarter rows after the snapshot.
