# Accidents.txt profile (MSHA Accident/Injury/Illness file, snapshot 2026-10-09)

Generated 2026-10-14T12:21:53Z with pandas 3.0.6 / numpy 2.5.3 on the system interpreter; runtime 43.1 s. Machine-readable twin: `docs/profile/accidents_profile.json` (same keys as the section names below). Unless stated, **base** = `CAL_YR >= 2010` and `DEGREE_INJURY_CD in {01..06}` (n = 105,191); **lost-time** = codes 01-04 vs 05-06.

## 1. File, encoding and structure

| Item | Value |
|---|---|
| Path | C:\ML\mlda-cw1\data\raw\Accidents.txt |
| Bytes | 228,804,642 |
| utf-8 strict decode | UnicodeDecodeError: first undecodable byte at absolute offset 3,626,765 (latin-1 0xF3 'ó' inside a Spanish NARRATIVE); 123 non-ASCII bytes in total, all in NARRATIVE (90 rows) and EQUIP_MODEL_NO (2 rows) |
| Encoding used | **latin-1** (decodes all bytes; non-ASCII bytes: 123, in NARRATIVE 90 rows and EQUIP_MODEL_NO 2 rows) |
| Line terminator | CRLF (275,221 lines incl. header; lone LF: 0; no embedded newlines inside quoted fields) |
| Rows x columns | 275,220 x 57 |
| Header vs codebook | exact match, same order: True; missing/extra columns: [] / [] |
| Rows with wrong field count | 0 (csv-module pass: every row has 57 fields) |
| Quoting | VARCHAR2/DATE fields double-quoted, NUMBER fields unquoted; empty strings appear as \|\| (blank); NUMBER fields are unquoted so leading zeros are lost (SHIFT_BEGIN_TIME has both `0700` and `700`); the `?` placeholder in IMMED_NOTIFY_CD is stored as `? ` with a trailing space |
| Load call | `pd.read_csv(path, sep='\|', quotechar='"', dtype=str, keep_default_na=False, encoding='latin-1')` |

## 2. Per-column profile (all 57 columns)

`% blank` = empty string; `% placeholder` = `?`, `? ` or `NO VALUE FOUND` (case-insensitive, counted separately from blanks); `n_unique` excludes blanks and placeholders. Numerics show min / median / max / skew after conversion and the number of non-missing values; others show the top-5 values with counts.

| Column | Codebook | Inferred | % blank | % placeholder | n_unique | Summary |
|---|---|---|---|---|---|---|
| `MINE_ID` | VARCHAR2 | identifier | 0.00% | 0.00% | 13,541 | `0200024`=2,936, `4601437`=2,387, `1102752`=1,839, `3607230`=1,559, `0102901`=1,539 |
| `CONTROLLER_ID` | VARCHAR2 | identifier | 0.26% | 0.00% | 6,675 | `C15455`=8,254, `C13408`=7,836, `C00692`=7,548, `C15833`=7,157, `0041211`=6,829 |
| `CONTROLLER_NAME` | VARCHAR2 | high_cardinality_string | 0.26% | 0.00% | 6,681 | `(name redacted #1)`=8,254, `(name redacted #2)`=7,836, `(name redacted #3)`=7,548, `(name redacted #4)`=7,157, `(name redacted #5)`=6,829 |
| `OPERATOR_ID` | VARCHAR2 | identifier | 0.26% | 0.00% | 8,030 | `P00812`=3,150, `L12058`=2,936, `L16168`=2,467, `P01155`=2,258, `P23854`=2,083 |
| `OPERATOR_NAME` | VARCHAR2 | high_cardinality_string | 0.26% | 0.00% | 10,229 | `(name redacted #1)`=3,150, `(name redacted #2)`=2,936, `(name redacted #3)`=2,270, `(name redacted #4)`=2,258, `(name redacted #5)`=1,997 |
| `CONTRACTOR_ID` | VARCHAR2 | identifier | 89.57% | 0.00% | 5,183 | `(blank)`=246,526, `MVK`=713, `E24`=323, `B96`=315, `A01`=312 |
| `DOCUMENT_NO` | VARCHAR2 | identifier | 0.00% | 0.00% | 275,220 | `220152230015`=1, `220112690038`=1, `220020710005`=1, `220101040010`=1, `220102300014`=1 |
| `SUBUNIT_CD` | VARCHAR2 | categorical_code | 0.00% | 0.00% | 10 | `01`=107,309, `03`=82,185, `30`=67,850, `02`=10,680, `06`=3,943 |
| `SUBUNIT` | VARCHAR2 | categorical_description | 0.00% | 0.00% | 10 | `UNDERGROUND`=107,309, `STRIP, QUARY, OPEN PIT`=82,185, `MILL OPERATION/PREPARATION P`=67,850, `SURFACE AT UNDERGROUND`=10,680, `DREDGE`=3,943 |
| `ACCIDENT_DT` | DATE | date | 0.00% | 0.00% | 9,759 | 2000-01-01 .. 2026-10-06; unparsable non-blank 0; top: `07/24/2000`=96, `09/05/2000`=93, `09/10/2001`=93 |
| `CAL_YR` | NUMBER | numeric | 0.00% | 0.00% | 27 | n=275,220; min 2000 / med 2008 / max 2026; mean 2009.90; skew 0.51; zeros 0 |
| `CAL_QTR` | NUMBER | numeric | 0.00% | 0.00% | 4 | n=275,220; min 1 / med 2 / max 4; mean 2.46; skew 0.02; zeros 0 |
| `FISCAL_YR` | NUMBER | numeric | 0.00% | 0.00% | 28 | n=275,220; min 2000 / med 2009 / max 2027; mean 2010.12; skew 0.50; zeros 0 |
| `FISCAL_QTR` | NUMBER | numeric | 0.00% | 0.00% | 4 | n=275,220; min 1 / med 3 / max 4; mean 2.59; skew -0.09; zeros 0 |
| `ACCIDENT_TIME` | NUMBER | numeric | 0.00% | 0.00% | 1,443 | n=275,219; min 0 / med 1,215 / max 9,999; mean 1855.67; skew 2.94; zeros 14 |
| `DEGREE_INJURY_CD` | VARCHAR2 | categorical_code | 0.00% | 0.33% (`?`=910) | 11 | `03`=89,011, `06`=71,700, `05`=43,613, `00`=30,604, `04`=20,834 |
| `DEGREE_INJURY` | VARCHAR2 | categorical_description | 0.00% | 0.33% (`NO VALUE FOUND`=910) | 11 | `DAYS AWAY FROM WORK ONLY`=89,011, `NO DYS AWY FRM WRK,NO RSTR A`=71,700, `DAYS RESTRICTED ACTIVITY ONL`=43,613, `ACCIDENT ONLY`=30,604, `DYS AWY FRM WRK & RESTRCTD A`=20,834 |
| `FIPS_STATE_CD` | VARCHAR2 | categorical_code | 0.00% | 0.00% | 54 | `54`=42,031, `21`=28,830, `42`=20,536, `17`=12,801, `48`=11,980 |
| `UG_LOCATION_CD` | VARCHAR2 | categorical_code | 0.00% | 61.01% (`?`=167,912) | 8 | `?`=167,912, `03`=32,526, `06`=31,698, `04`=19,810, `98`=7,924 |
| `UG_LOCATION` | VARCHAR2 | categorical_description | 0.00% | 61.01% (`NO VALUE FOUND`=167,912) | 8 | `NO VALUE FOUND`=167,912, `FACE`=32,526, `LAST OPEN CROSSCUT`=31,698, `INTERSECTION`=19,810, `OTHER`=7,924 |
| `UG_MINING_METHOD_CD` | VARCHAR2 | categorical_code | 0.00% | 65.38% (`?`=179,947) | 7 | `?`=179,947, `05`=67,084, `01`=13,715, `03`=7,180, `08`=6,468 |
| `UG_MINING_METHOD` | VARCHAR2 | categorical_description | 0.00% | 65.38% (`NO VALUE FOUND`=179,947) | 7 | `NO VALUE FOUND`=179,947, `Continuous Mining`=67,084, `Longwall`=13,715, `Conventional Stoping`=7,180, `Other`=6,468 |
| `MINING_EQUIP_CD` | VARCHAR2 | categorical_code | 0.00% | 55.18% (`?`=151,878) | 69 | `?`=151,878, `28`=28,351, `54`=8,937, `44`=8,915, `24`=7,891 |
| `MINING_EQUIP` | VARCHAR2 | categorical_description | 0.00% | 55.18% (`NO VALUE FOUND`=151,878) | 69 | `NO VALUE FOUND`=151,878, `Hand tools (not powered)`=28,351, `Rock or roof bolting machine`=8,937, `Ore haulage trucks - off hig`=8,915, `Front-end loader, Tractor-sh`=7,891 |
| `EQUIP_MFR_CD` | VARCHAR2 | categorical_code | 0.00% | 55.20% (`?`=151,931) | 245 | `?`=151,931, `121`=41,123, `119`=16,828, `0000`=16,208, `018`=11,195 |
| `EQUIP_MFR_NAME` | VARCHAR2 | categorical_description | 0.00% | 55.20% (`NO VALUE FOUND`=151,931) | 193 | `NO VALUE FOUND`=151,931, `Not Reported`=57,331, `Not on this list`=16,828, `Caterpillar`=14,588, `Fletcher`=4,952 |
| `EQUIP_MODEL_NO` | VARCHAR2 | high_cardinality_string | 60.27% | 20.09% (`?`=55,294, `? `=1) | 21,090 | `(blank)`=165,887, `?`=55,294, `RRII`=643, `777`=346, `488`=339 |
| `SHIFT_BEGIN_TIME` | NUMBER | numeric | 0.36% | 0.00% | 819 | n=274,230; min 0 / med 700 / max 9,999; mean 1306.99; skew 4.45; zeros 135 |
| `CLASSIFICATION_CD` | VARCHAR2 | categorical_code | 0.00% | 0.77% (`?`=2,106) | 28 | `09`=79,090, `18`=46,930, `10`=28,960, `17`=28,050, `07`=27,472 |
| `CLASSIFICATION` | VARCHAR2 | categorical_description | 0.00% | 0.77% (`NO VALUE FOUND`=2,106) | 28 | `HANDLING OF MATERIALS`=79,090, `SLIP OR FALL OF PERSON`=46,930, `HANDTOOLS (NONPOWERED)`=28,960, `MACHINERY`=28,050, `FALL OF ROOF OR BACK`=27,472 |
| `ACCIDENT_TYPE_CD` | VARCHAR2 | categorical_code | 0.00% | 0.77% (`?`=2,110) | 44 | `30`=33,217, `08`=32,046, `44`=30,604, `04`=24,087, `27`=18,048 |
| `ACCIDENT_TYPE` | VARCHAR2 | categorical_description | 0.00% | 0.77% (`No Value Found`=2,110) | 44 | `Over-exertion NEC`=33,217, `Struck by... NEC`=32,046, `Accident type, without injur`=30,604, `Struck by falling object`=24,087, `Over-exertion in lifting obj`=18,048 |
| `NO_INJURIES` | NUMBER | numeric | 0.00% | 0.00% | 16 | n=275,220; min 0 / med 1 / max 36; mean 0.91; skew 31.46; zeros 30,614 |
| `TOT_EXPER` | NUMBER | numeric | 16.12% | 0.00% | 3,349 | n=230,840; min 0.01 / med 7 / max 66; mean 10.94; skew 1.01; zeros 0 |
| `MINE_EXPER` | NUMBER | numeric | 15.08% | 0.00% | 3,230 | n=233,718; min 0.01 / med 2.76 / max 65; mean 6.45; skew 1.88; zeros 0 |
| `JOB_EXPER` | NUMBER | numeric | 14.78% | 0.00% | 3,048 | n=234,550; min 0.01 / med 3.17 / max 66; mean 6.88; skew 1.76; zeros 0 |
| `OCCUPATION_CD` | VARCHAR2 | categorical_code | 0.00% | 11.78% (`?`=32,406) | 255 | `304`=37,208, `?`=32,406, `316`=20,407, `374`=20,324, `046`=13,941 |
| `OCCUPATION` | VARCHAR2 | categorical_description | 0.00% | 11.78% (`NO VALUE FOUND`=32,406) | 203 | `Maintenance man, Mechanic,  `=44,507, `NO VALUE FOUND`=32,406, `Laborer, Blacksmith, Bull ga`=20,407, `Warehouseman, Bagger, Pallet`=20,324, `Roof bolter, Rock bolter,  P`=14,668 |
| `ACTIVITY_CD` | VARCHAR2 | categorical_code | 0.00% | 11.89% (`?`=32,718) | 99 | `028`=46,191, `039`=35,865, `?`=32,718, `030`=25,954, `092`=22,692 |
| `ACTIVITY` | VARCHAR2 | categorical_description | 0.00% | 11.89% (`NO VALUE FOUND`=32,718) | 99 | `Handling supplies or materia`=46,191, `Machine maintenance`=35,865, `NO VALUE FOUND`=32,718, `Hand tools (not powered)`=25,954, `Walking, Running`=22,692 |
| `INJURY_SOURCE_CD` | VARCHAR2 | categorical_code | 0.00% | 11.91% (`?`=32,776) | 127 | `?`=32,776, `088`=29,381, `117`=18,342, `086`=16,644, `089`=13,036 |
| `INJURY_SOURCE` | VARCHAR2 | categorical_description | 0.00% | 11.91% (`NO VALUE FOUND`=32,776) | 127 | `NO VALUE FOUND`=32,776, `METAL,NEC(PIPE,WIRE,NAIL)`=29,381, `GROUND`=18,342, `METAL COVERS & GUARDS`=16,644, `BROKEN ROCK,COAL,ORE,WSTE`=13,036 |
| `NATURE_INJURY_CD` | VARCHAR2 | categorical_code | 0.00% | 11.99% (`?`=33,003) | 38 | `330`=76,325, `180`=53,331, `220`=35,003, `?`=33,003, `160`=18,716 |
| `NATURE_INJURY` | VARCHAR2 | categorical_description | 0.00% | 11.99% (`NO VALUE FOUND`=33,003) | 38 | `SPRAIN,STRAIN RUPT DISC`=76,325, `CUT,LACER,PUNCT-OPN WOUND`=53,331, `FRACTURE,CHIP`=35,003, `NO VALUE FOUND`=33,003, `CONTUSN,BRUISE,INTAC SKIN`=18,716 |
| `INJ_BODY_PART_CD` | VARCHAR2 | categorical_code | 0.00% | 11.99% (`?`=33,004) | 46 | `340`=40,327, `?`=33,004, `420`=31,916, `512`=18,756, `450`=15,070 |
| `INJ_BODY_PART` | VARCHAR2 | categorical_description | 0.00% | 11.99% (`NO VALUE FOUND`=33,004) | 46 | `FINGER(S)/THUMB`=40,327, `NO VALUE FOUND`=33,004, `BACK (MUSCLES/SPINE/S-CORD/T`=31,916, `KNEE/PATELLA`=18,756, `SHOULDERS (COLLARBONE/CLAVIC`=15,070 |
| `SCHEDULE_CHARGE` | NUMBER | numeric | 23.62% | 0.00% | 227 | n=210,199; min 0 / med 0 / max 6,000; mean 69.10; skew 9.32; zeros 204,824 |
| `DAYS_RESTRICT` | NUMBER | numeric | 22.61% | 0.00% | 448 | n=212,979; min 0 / med 0 / max 3,310; mean 9.09; skew 17.31; zeros 146,761 |
| `DAYS_LOST` | NUMBER | numeric | 17.00% | 0.00% | 668 | n=228,447; min 0 / med 0 / max 3,616; mean 27.85; skew 5.69; zeros 116,074 |
| `TRANS_TERM` | VARCHAR2 | categorical_code | 13.29% | 0.00% | 2 | `N`=226,730, `(blank)`=36,587, `Y`=11,903 |
| `RETURN_TO_WORK_DT` | DATE | date | 15.68% | 0.00% | 9,708 | 1969-12-31 .. 2026-10-05; unparsable non-blank 0; top: `(blank)`=43,143, `05/01/2006`=784, `05/01/2007`=769 |
| `IMMED_NOTIFY_CD` | VARCHAR2 | categorical_code | 0.00% | 66.52% (`? `=183,073) | 13 | `? `=183,073, `13`=55,851, `08`=17,340, `02`=8,535, `11`=4,430 |
| `IMMED_NOTIFY` | VARCHAR2 | categorical_description | 0.00% | 66.52% (`NO VALUE FOUND`=183,073) | 13 | `NO VALUE FOUND`=183,073, `NOT MARKED`=55,851, `ROOF FALL`=17,340, `SERIOUS INJURY`=8,535, `HOISTING`=4,430 |
| `INVEST_BEGIN_DT` | DATE | date | 80.07% | 0.00% | 8,739 | 2000-01-01 .. 2026-09-29; unparsable non-blank 0; top: `(blank)`=220,363, `08/27/2001`=36, `09/05/2000`=33 |
| `NARRATIVE` | VARCHAR2 | free_text | 0.00% | 0.00% | 273,665 | len min 0 / med 175 / max 610; empty 2; `Employee has been diagnosed `=41, `Exposure to Coal Mine Dust`=39, `HEARING LOSS.`=34, `CARPAL TUNNEL SYNDROME.`=27, `EE sustained noise-induced h`=26 |
| `CLOSED_DOC_NO` | VARCHAR2 | identifier | 55.62% | 0.00% | 122,139 | `(blank)`=153,064, `320110810007`=2, `320052840030`=2, `320091030022`=2, `320090050016`=2 |
| `COAL_METAL_IND` | VARCHAR2 | categorical_code | 0.00% | 0.00% | 2 | `M`=146,763, `C`=128,457 |

### 2a. Code -> description consistency

| Pair | n codes | n descriptions | codes -> >1 desc | desc -> >1 code | 1:1 | placeholder mismatch | Inconsistency |
|---|---|---|---|---|---|---|---|
| SUBUNIT_CD -> SUBUNIT | 10 | 10 | 0 | 0 | yes | 0 |  |
| DEGREE_INJURY_CD -> DEGREE_INJURY | 12 | 12 | 0 | 0 | yes | 0 |  |
| UG_LOCATION_CD -> UG_LOCATION | 9 | 9 | 0 | 0 | yes | 0 |  |
| UG_MINING_METHOD_CD -> UG_MINING_METHOD | 8 | 8 | 0 | 0 | yes | 0 |  |
| MINING_EQUIP_CD -> MINING_EQUIP | 70 | 70 | 0 | 0 | yes | 0 |  |
| EQUIP_MFR_CD -> EQUIP_MFR_NAME | 246 | 194 | 0 | 52 | **no** | 0 | two code series for one manufacturer (3-digit legacy and 4-digit), e.g. Black & Decker: 0219=24, 014=22; Buffalo-American: 0232=3, 016=3; Case: 017=184, 0308=58 |
| CLASSIFICATION_CD -> CLASSIFICATION | 29 | 29 | 0 | 0 | yes | 0 |  |
| ACCIDENT_TYPE_CD -> ACCIDENT_TYPE | 45 | 45 | 0 | 0 | yes | 0 |  |
| OCCUPATION_CD -> OCCUPATION | 256 | 204 | 0 | 39 | **no** | 0 | first digit is a sector series (0/1/2 coal-heavy, 3/4/6 metal-heavy) for the same job title, e.g. Apprentice, Trainee: 124=347, 024=113; Backhoe operator, Trackhoe ope: 078=947, 178=11 |
| ACTIVITY_CD -> ACTIVITY | 100 | 100 | 0 | 0 | yes | 0 |  |
| INJURY_SOURCE_CD -> INJURY_SOURCE | 128 | 128 | 0 | 0 | yes | 0 |  |
| NATURE_INJURY_CD -> NATURE_INJURY | 39 | 39 | 0 | 0 | yes | 0 |  |
| INJ_BODY_PART_CD -> INJ_BODY_PART | 47 | 47 | 0 | 0 | yes | 0 |  |
| IMMED_NOTIFY_CD -> IMMED_NOTIFY | 14 | 14 | 0 | 0 | yes | 0 |  |
| CONTROLLER_ID -> CONTROLLER_NAME | 6,676 | 6,682 | 25 | 19 | **no** | 0 | name variants over time (order of partners, punctuation); 19 names map to >1 id (e.g. legacy numeric id and new C/M id) |
| OPERATOR_ID -> OPERATOR_NAME | 8,031 | 10,230 | 1,442 | 137 | **no** | 0 | 1,442 ids with several spellings of the same name; 137 names with >1 id; names change over time, not a code table |

Every code placeholder (`?`) pairs with `NO VALUE FOUND` (or `No Value Found` for ACCIDENT_TYPE) in its description column, with zero mismatches. Within the base, OCCUPATION_CD has 244 codes but only 196 descriptions and EQUIP_MFR_CD has 145 codes for 139 names, so the description is the de-duplicated categorical. EQUIP_MFR_CD lengths: 1-char=151,931, 3-char=91,798, 4-char=31,491.

### 2b. Formatting quirks

| Quirk | Rows |
|---|---|
| `IMMED_NOTIFY_CD` placeholder stored as `? ` (trailing space) | 183,073 |
| `EQUIP_MODEL_NO` = `?` / `? ` | 55,294 / 1 |
| Leading/trailing whitespace: OPERATOR_NAME / MINING_EQUIP / NARRATIVE / ACTIVITY / EQUIP_MODEL_NO / CONTROLLER_NAME | 6,705 / 10,898 / 16,887 / 255 / 253 / 30 |
| `ACCIDENT_TIME` not-reported codes 9990-9999 (9999 alone) | 20,661 (20,223); other value > 2400: 1 |
| `SHIFT_BEGIN_TIME` codes 9990-9999 (9999 alone) | 8,443 (8,415) |
| All five injury-detail codes (`OCCUPATION, ACTIVITY, INJURY_SOURCE, NATURE_INJURY, INJ_BODY_PART`) = `?` | 31,784 all rows; 0 in base (96.28% of them are DEGREE 00 'accident only'; 99.99% of DEGREE-00 rows have all five as `?`) |
| `TOT_EXPER` missing by degree (all years) | 00=100.0%, 01=3.7%, 02=5.71%, 03=5.92%, 04=4.12%, 05=3.91%, 06=4.96%, 07=5.83%, 08=8.79%, 09=80.52%, 10=9.02%, ?=87.25% |
| `TOT_EXPER` missing < 2010 vs >= 2010 | 20.11% vs 11.39% (base: 1.18%) |

## 3. Calendar years and DEGREE_INJURY_CD

| CAL_YR | Rows | Distinct mines | Months present | Max ACCIDENT_DT | Note |
|---|---|---|---|---|---|
| 2000 | 18,704 | 3,862 | 12 | 2000-12-31 |  |
| 2001 | 17,558 | 3,767 | 12 | 2001-12-31 |  |
| 2002 | 16,009 | 3,583 | 12 | 2002-12-31 |  |
| 2003 | 14,481 | 3,386 | 12 | 2003-12-31 |  |
| 2004 | 14,505 | 3,438 | 12 | 2004-12-31 |  |
| 2005 | 14,787 | 3,583 | 12 | 2005-12-31 |  |
| 2006 | 14,498 | 3,498 | 12 | 2006-12-31 |  |
| 2007 | 14,383 | 3,287 | 12 | 2007-12-31 |  |
| 2008 | 13,609 | 3,166 | 12 | 2008-12-31 |  |
| 2009 | 10,968 | 2,774 | 12 | 2009-12-31 |  |
| 2010 | 10,756 | 2,673 | 12 | 2010-12-31 |  |
| 2011 | 11,038 | 2,777 | 12 | 2011-12-31 |  |
| 2012 | 10,036 | 2,559 | 12 | 2012-12-31 |  |
| 2013 | 9,581 | 2,463 | 12 | 2013-12-31 |  |
| 2014 | 9,420 | 2,409 | 12 | 2014-12-31 |  |
| 2015 | 8,239 | 2,285 | 12 | 2015-12-31 |  |
| 2016 | 7,067 | 2,124 | 12 | 2016-12-31 |  |
| 2017 | 6,984 | 2,088 | 12 | 2017-12-31 |  |
| 2018 | 6,863 | 2,091 | 12 | 2018-12-31 |  |
| 2019 | 6,824 | 2,058 | 12 | 2019-12-31 |  |
| 2020 | 5,391 | 1,798 | 12 | 2020-12-31 |  |
| 2021 | 5,676 | 1,852 | 12 | 2021-12-31 |  |
| 2022 | 5,980 | 1,902 | 12 | 2022-12-31 |  |
| 2023 | 6,093 | 1,911 | 12 | 2023-12-31 |  |
| 2024 | 5,991 | 1,894 | 12 | 2024-12-31 |  |
| 2025 | 5,602 | 1,871 | 12 | 2025-12-31 |  |
| 2026 | 4,177 | 1,527 | 10 | 2026-10-06 | **partial** |

Rows outside 2000-2026: none; CAL_YR missing: 0. ACCIDENT_DT spans 2000-01-01 .. 2026-10-06. Only **2026** is partial (10 months, last accident 2026-10-06). 2026 is also *label-incomplete*: 46.97% of its 03/04 cases have no DAYS_LOST yet (2025: 0.09%), a closing document exists for 33.45% of base rows (2025: 55.1%), and the degree mix is shifted (code 03 = 46.23% vs 38.51% in 2025; code 04 = 5.42% vs 9.79%; code 05 = 16.97% vs 21.41%), consistent with cases whose restricted-activity days have not yet been added by the return-to-duty form.

### 3a. DEGREE_INJURY_CD distribution

| Code | Label | n all | % all | n 2010+ | % 2010+ |
|---|---|---|---|---|---|
| `00` | Accident only | 30,604 | 11.12% | 12,610 | 10.03% |
| `01` | Fatality | 1,215 | 0.44% | 591 | 0.47% |
| `02` | Perm total/partial disability | 2,487 | 0.90% | 1,167 | 0.93% |
| `03` | Days away only | 89,011 | 32.34% | 38,604 | 30.71% |
| `04` | Days away + restricted | 20,834 | 7.57% | 10,922 | 8.69% |
| `05` | Restricted only | 43,613 | 15.85% | 21,396 | 17.02% |
| `06` | No days away, no restriction | 71,700 | 26.05% | 32,511 | 25.86% |
| `07` | Occupational illness | 10,627 | 3.86% | 5,655 | 4.50% |
| `08` | Natural causes | 1,548 | 0.56% | 1,038 | 0.83% |
| `09` | Non-employee | 575 | 0.21% | 356 | 0.28% |
| `10` | Other incl. first aid | 2,096 | 0.76% | 868 | 0.69% |
| `?` | No value found | 910 | 0.33% | 0 | 0.00% |
|  | **Total** | 275,220 | 100% | 125,718 | 100% |

### 3b. DEGREE_INJURY_CD x COAL_METAL_IND

| Code | Label | C all | M all | % coal all | C 2010+ | M 2010+ | % coal 2010+ |
|---|---|---|---|---|---|---|---|
| `00` | Accident only | 25,425 | 5,179 | 83.08% | 9,311 | 3,299 | 73.84% |
| `01` | Fatality | 562 | 653 | 46.26% | 244 | 347 | 41.29% |
| `02` | Perm total/partial disability | 1,019 | 1,468 | 40.97% | 446 | 721 | 38.22% |
| `03` | Days away only | 55,472 | 33,539 | 62.32% | 23,669 | 14,935 | 61.31% |
| `04` | Days away + restricted | 5,165 | 15,669 | 24.79% | 2,556 | 8,366 | 23.40% |
| `05` | Restricted only | 6,003 | 37,610 | 13.76% | 2,872 | 18,524 | 13.42% |
| `06` | No days away, no restriction | 27,835 | 43,865 | 38.82% | 12,794 | 19,717 | 39.35% |
| `07` | Occupational illness | 5,614 | 5,013 | 52.83% | 2,966 | 2,689 | 52.45% |
| `08` | Natural causes | 386 | 1,162 | 24.94% | 220 | 818 | 21.19% |
| `09` | Non-employee | 75 | 500 | 13.04% | 39 | 317 | 10.96% |
| `10` | Other incl. first aid | 507 | 1,589 | 24.19% | 187 | 681 | 21.54% |
| `?` | No value found | 394 | 516 | 43.30% | 0 | 0 | - |

Coal rows concentrate in 00 (accident only) and 03 (days away only); metal/non-metal rows dominate 04 and 05 (restricted activity), which is why coal has a much higher lost-time rate (see section 4).

## 4. Candidate target definitions (CAL_YR >= 2010)

### 4a. Exclusions

| Rule (applied in order) | Excluded | Kept |
|---|---|---|
| start: CAL_YR >= 2010 |  | 125,718 |
| drop DEGREE_INJURY_CD=00 (Accident only) | 12,610 | 113,108 |
| drop DEGREE_INJURY_CD=07 (Occupational illness) | 5,655 | 107,453 |
| drop DEGREE_INJURY_CD=08 (Natural causes) | 1,038 | 106,415 |
| drop DEGREE_INJURY_CD=09 (Non-employee) | 356 | 106,059 |
| drop DEGREE_INJURY_CD=10 (Other incl. first aid) | 868 | 105,191 |
| drop DEGREE_INJURY_CD=? (No value found) | 0 | 105,191 |

Codes remaining: 01, 02, 03, 04, 05, 06; unexpected codes: none. All 910 `?` degree rows are pre-2010, so that rule removes nothing after the year filter. Independent counts on 2010+: 00=12,610, 07=5,655, 08=1,038, 09=356, 10=868, ?=0.

### 4b. Definitions, prevalence and label validity

| Definition | n | Pos | Neg | Prevalence | Neg:Pos | Pos w/ DAYS_LOST>0 or SCHED>0 | Pos w/ any of lost/sched/restrict>0 | Neg w/ DAYS_LOST=0 | Neg w/ DAYS_RESTRICT>0 | Median DAYS_LOST pos/neg | Prevalence by sector |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **(i)** DEGREE_INJURY_CD in {01,02,03,04} = 1 vs {05,06} = 0; base CAL_YR>=2010 and code in 01-06 | 105,191 | 51,284 | 53,907 | 48.75% | 1.051 | 97.44% | 97.45% | 100.00% | 39.62% | 28 / 0 | C 63.21% / M 38.92% |
| **(ii)** DEGREE_INJURY_CD in {01,02} or DAYS_LOST>=30 = 1 else 0; same base | 105,191 | 25,594 | 79,597 | 24.33% | 3.110 | 100.00% | 100.00% | 69.37% | 35.48% | 84 / 0 | C 35.79% / M 16.54% |
| **(iii)** DEGREE_INJURY_CD in {03,04,05} = 1 vs {06} = 0; base CAL_YR>=2010 and code in 03-06 (01,02 excluded) | 103,433 | 70,922 | 32,511 | 68.57% | 0.458 | 67.99% | 98.10% | 100.00% | 0.00% | 7 / 0 | C 69.46% / M 67.96% |
| **(iii-b)** DEGREE_INJURY_CD in {01,02,03,04,05} = 1 vs {06} = 0; base CAL_YR>=2010 and code in 01-06 | 105,191 | 72,680 | 32,511 | 69.09% | 0.447 | 68.76% | 98.15% | 100.00% | 0.00% | 7 / 0 | C 69.95% / M 68.51% |
| **(i) 2015+** (i) restricted to CAL_YR>=2015 | 63,482 | 31,135 | 32,347 | 49.05% | 1.039 | 96.48% | 96.49% | 100.00% | 41.41% | 30 / 0 | C 66.0% / M 39.67% |
| **(i) all years** same as (i) but no year filter | 228,860 | 113,547 | 115,313 | 49.61% | 1.016 | 98.13% | 98.13% | 100.00% | 37.74% | 22 / 0 | C 64.77% / M 38.65% |

(ii) components: 1,758 rows are 01/02, 23,836 reach DAYS_LOST >= 30 without being 01/02, and only 1 05/06 row has DAYS_LOST >= 30. (iii) drops 1,758 fatal/permanent rows from its base. Excluding 2026: (i) prevalence 48.585% on 101,609 rows; (ii) 24.833%.

### 4c. Prevalence by year

| CAL_YR | (i) lost-time | (ii) serious | (iii) any DART | DAYS_LOST missing among 03/04 |
|---|---|---|---|---|
| 2010 | 48.53% | 23.34% | 66.49% | 0.0% |
| 2011 | 47.53% | 22.43% | 65.16% | 0.0% |
| 2012 | 47.98% | 22.82% | 66.59% | 0.03% |
| 2013 | 48.68% | 24.85% | 67.56% | 0.29% |
| 2014 | 48.95% | 25.55% | 69.02% | 0.03% |
| 2015 | 47.86% | 25.20% | 67.71% | 0.03% |
| 2016 | 47.25% | 24.07% | 68.37% | 0.08% |
| 2017 | 48.66% | 24.93% | 68.91% | 0.21% |
| 2018 | 47.98% | 24.43% | 68.27% | 0.44% |
| 2019 | 48.42% | 25.70% | 68.55% | 0.48% |
| 2020 | 48.69% | 26.30% | 70.96% | 0.51% |
| 2021 | 49.86% | 26.21% | 72.24% | 0.56% |
| 2022 | 48.92% | 25.30% | 69.86% | 1.45% |
| 2023 | 50.01% | 24.97% | 71.25% | 2.93% |
| 2024 | 49.55% | 27.63% | 70.65% | 0.08% |
| 2025 | 50.13% | 27.65% | 71.01% | 0.09% |
| 2026 | 53.52% | 10.08% | 69.93% | 46.97% |

### 4d. DAYS_LOST / DAYS_RESTRICT / SCHEDULE_CHARGE by degree code (base)

| Code | Label | n | DAYS_LOST missing | =0 | >0 | >=30 | median | DAYS_RESTRICT>0 | SCHED>0 | median SCHED |
|---|---|---|---|---|---|---|---|---|---|---|
| `01` | Fatality | 591 | 387 | 204 | 0 | 0 | 0 | 0 | 591 | 6,000 |
| `02` | Perm total/partial disability | 1,167 | 190 | 326 | 651 | 348 | 11 | 348 | 1,166 | 100 |
| `03` | Days away only | 38,604 | 1,013 | 243 | 37,348 | 19,848 | 34 | 4 | 0 | 0 |
| `04` | Days away + restricted | 10,922 | 39 | 15 | 10,868 | 3,987 | 15 | 10,866 | 0 | 0 |
| `05` | Restricted only | 21,396 | 37 | 21,358 | 1 | 1 | 0 | 21,358 | 0 | 0 |
| `06` | No days away, no restriction | 32,511 | 75 | 32,436 | 0 | 0 | 0 | 0 | 0 | 0 |

### 4e. Recommendation

Use **(i) lost-time** (01-04 vs 05-06) on **CAL_YR 2010-2025** (n = 101,609, prevalence 48.585%, i.e. about 1 : 1.06). Reasons:

1. Class balance: 48.75% positive, so no resampling is needed and accuracy, ROC-AUC and PR-AUC are all informative; the cost-weighted threshold analysis (false negative costlier) remains the place where the business asymmetry enters, rather than the label.
2. Label validity: 97.44% of positives have DAYS_LOST > 0 or SCHEDULE_CHARGE > 0 (the remainder are mostly open 2026 cases: 869 of the 1,052 base 03/04 rows without DAYS_LOST are from 2026), and 100.0% of negatives have DAYS_LOST = 0. The label is MSHA's own closed-case classification, not a threshold the project invents.
3. Business meaning: days away from work is what triggers wage-replacement (indemnity) reserving and return-to-work planning; restricted-duty-only cases (05, 21,396 rows 2010+) carry medical cost but no indemnity, so grouping them with 06 is the operationally meaningful cut. Fatalities and permanent disabilities (01/02, 1,758 rows) are the most expensive lost-time cases and belong with the positives.
4. Against (ii): it depends on DAYS_LOST, which is right-censored for recent cases (2026 prevalence collapses to 10.08% vs 24.833% for 2010-2025), has 583 base values above 365 days, and the 30-day cut is arbitrary; it is a reasonable *secondary* severity label (24.8% positive) for a sensitivity analysis.
5. Against (iii): 68.57% positive (2.2 : 1 the other way), it must discard the 01/02 rows or treat death as 'any days away', and the negative class (06) is defined by the absence of a return-to-duty form rather than by an outcome the insurer prices.
6. Year window: drop 2026 (3,582 base rows) because it is both partial and label-incomplete (section 3); keep 2010 as the lower bound (prevalence is flat at 47-50% across 2010-2025, and TOT_EXPER missingness falls from 2.33% in 2010 to under 1% from 2017).

## 5. Duplicates and multi-victim accidents

| Check | Result |
|---|---|
| Exact duplicate rows (all 57 columns) | 0 |
| Duplicate rows ignoring DOCUMENT_NO | 19 |
| Duplicate rows ignoring DOCUMENT_NO, CLOSED_DOC_NO, NARRATIVE | 77 |
| DOCUMENT_NO | 275,220 unique of 275,220 rows (blank 0, duplicated 0); all 12 chars, all start with `2` (injury form); Julian date parse failures 18 |
| Filing lag (DOCUMENT_NO date - ACCIDENT_DT), days | median 9, p75 14, p95 83, p99 291, max 9,332; negative 91; > 365 d: 1,806 |
| CLOSED_DOC_NO | present 122,156 (44.38%), 122,139 unique (so 17 values reused); first char: `3`=122,153, `2`=3 |

| Grouping key | Groups | Multi-row groups | Rows in multi-row groups | % rows | Max rows/group | Rows-per-group distribution | Multi-row groups with mixed lost-time label |
|---|---|---|---|---|---|---|---|
| key1 = MINE_ID+ACCIDENT_DT+ACCIDENT_TIME, all rows | 272,781 | 1,743 | 4,182 | 1.52% | 34 | 1:271,038, 2:1,415, 3:202, 4:58, 5:24, 6:13, 7:7, 8:6, 9:5, 10:5 | - |
| key1, base | 104,562 | 475 | 1,104 | 1.05% | 33 | 1:104,087, 2:409, 3:39, 4:13, 5:2, 6:6, 7:3, 8:1, 9:1 | 112 |
| key2 = key1 + CLASSIFICATION_CD + ACCIDENT_TYPE_CD, all rows | 273,368 | 1,303 | 3,155 | 1.15% | 29 | 1:272,065, 2:1,056, 3:146, 4:48, 5:13, 6:12, 7:7, 8:7, 9:4, 10:4 | - |
| key2, base | 104,676 | 374 | 889 | 0.84% | 29 | 1:104,302, 2:314, 3:36, 4:10, 5:2, 6:6, 7:3, 8:1, 9:1 | - |
| key0 = MINE_ID+ACCIDENT_DT, all rows | 263,335 | 10,152 | 22,037 | 8.01% | 36 | 1:253,183, 2:9,086, 3:803, 4:151, 5:40, 6:20 | - |
| key0, base | 102,145 | 2,723 | 5,769 | 5.48% | 33 | 1:99,422, 2:2,516, 3:160, 4:27, 5:7, 6:7 | 923 |

key1 multi-row groups: 293 share an identical NARRATIVE (true multi-victim rows) and 1,450 have differing narratives; 1,204 of the 4,182 key1 multi rows carry an invalid ACCIDENT_TIME (9999 = not reported), so key1 both over-groups (same mine, same day, time unknown) and under-groups (victims recorded with slightly different times). 559 key1 groups have differing DEGREE codes. NO_INJURIES > 1 on 2,156 rows (932 base); distribution: 0:30,614, 1:242,450, 2:1,422, 3:265, 4:132, 5:58, 6:82, 7:35, 8:16, 9:45, 10:21, 12:1, 13:13, 14:14, 16:16, 36:36. NO_INJURIES exceeds the key1 group size on 132 rows and is smaller than it on 2,021 rows, so it is not a reliable victim counter.

Implication for the split: rows of one accident share every circumstance feature and often the label (112 of 475 base key1 groups are mixed), so they must not straddle train/test. Use the coarse key **MINE_ID + ACCIDENT_DT** as the group id (GroupShuffleSplit / StratifiedGroupKFold): it covers every key1/key2 group, is immune to the 9999 time code, and only 5.48% of base rows (5,769) sit in multi-row day groups, so stratification is barely affected. Base rows per mine: 8,455 mines, median 3, p90 24, max 1,801; 37.9% of rows come from mines with >= 100 rows, so a mine-level group split is a stricter optional variant.

## 6. Impossible and suspicious values

| Check | n all rows | n base |
|---|---|---|
| MINE_EXPER > TOT_EXPER | 495 | 83 |
| JOB_EXPER > TOT_EXPER | 7,793 | 3,881 |
| either MINE or JOB > TOT | 8,033 | 3,921 |
| any negative experience | 0 | 0 |
| TOT_EXPER > 60 years | 5 | 3 |
| MINE_EXPER > 60 | 1 | 0 |
| JOB_EXPER > 60 | 3 | 2 |
| TOT_EXPER > 50 | 74 | 41 |
| TOT_EXPER missing | 44,380 | 1,236 |
| MINE_EXPER missing | 41,502 | 1,302 |
| JOB_EXPER missing | 40,670 | 888 |
| TOT_EXPER = 0 | 0 | 0 |
| ACCIDENT_TIME blank | 1 | 0 |
| ACCIDENT_TIME invalid (not 0000-2359 or minutes >= 60) | 21,604 | 3,535 |
| ACCIDENT_TIME > 2359 (9990-9999 codes, 2400) | 21,604 | 3,535 |
| ACCIDENT_TIME minutes >= 60 | 20,661 | 3,200 |
| ACCIDENT_TIME = 2400 | 942 | 335 |
| ACCIDENT_TIME = 0000 | 14 | 13 |
| SHIFT_BEGIN_TIME blank | 990 | 1 |
| SHIFT_BEGIN_TIME invalid | 13,730 | 2,432 |
| SHIFT_BEGIN_TIME minutes >= 60 | 8,443 | 544 |
| SHIFT_BEGIN_TIME = 0000 | 135 | 102 |
| shift-elapsed hours > 12 | 4,222 | 1,236 |
| shift-elapsed hours > 16 | 2,792 | 724 |
| DAYS_LOST < 0 | 0 | 0 |
| DAYS_LOST > 365 | 705 | 583 |
| DAYS_LOST missing | 46,773 | 1,741 |
| DAYS_LOST = 999 sentinel | 0 | 0 |
| DAYS_RESTRICT < 0 | 0 | 0 |
| DAYS_RESTRICT > 365 | 107 | 93 |
| DAYS_RESTRICT > 180 | 1,194 | 735 |
| DAYS_RESTRICT missing | 62,241 | 2,077 |
| DAYS_RESTRICT > 0 but degree 03 or 06 | 6 | 4 |
| DAYS_RESTRICT = 0 but degree 04 or 05 | 171 | 30 |
| DAYS_LOST > 0 but degree 05 or 06 | 1 | 1 |
| DAYS_LOST = 0 but degree 03 or 04 | 1,072 | 258 |
| DAYS_LOST missing but degree 03 or 04 | 1,053 | 1,052 |
| SCHEDULE_CHARGE > 0 but degree not 01/02 | 1,674 | 0 |
| degree 01 but SCHEDULE_CHARGE != 6000 | 0 | 0 |
| degree 02 but SCHEDULE_CHARGE 0/missing | 1 | 1 |
| NO_INJURIES = 0 with injury degree 01-06 | 0 | 0 |
| NO_INJURIES >= 1 with degree 00 | 1 | 0 |
| NO_INJURIES > 10 | 80 | 33 |
| RETURN_TO_WORK_DT before ACCIDENT_DT | 494 | 1 |
| RETURN_TO_WORK_DT > 365 d after accident | 5,469 | 3,458 |
| RETURN_TO_WORK_DT present | 232,077 | 101,542 |
| INVEST_BEGIN_DT before ACCIDENT_DT | 0 | 0 |
| INVEST_BEGIN_DT present | 54,857 | 7,051 |
| CAL_YR != year(ACCIDENT_DT) | 0 | 0 |
| CAL_QTR != quarter(ACCIDENT_DT) | 0 | 0 |
| FISCAL_YR != derived (Oct-Sep) | 0 | 0 |
| ACCIDENT_DT blank or unparsable | 0 | 0 |
| ACCIDENT_DT after snapshot 2026-10-09 | 0 | 0 |
| DOCUMENT_NO filing date before ACCIDENT_DT | 91 | 67 |
| filing lag > 365 d | 1,806 | 921 |

Notes. ACCIDENT_TIME/SHIFT_BEGIN_TIME use 9999 (and 9990-9998) as not-reported codes and 2400 for midnight; after dropping those, shift-elapsed hours (accident minus shift start, mod 24 h) has median 4.5, p95 10.1, p99 18.3 on 244,838 rows. Examples of MINE_EXPER > TOT_EXPER: TOT .1 / MINE .12 / JOB 3; TOT 13 / MINE 13.25 / JOB 8; TOT .86 / MINE 1.48 / JOB .86; TOT 8 / MINE 22 / JOB 8. DAYS_LOST > 365 in base by degree: 02=19, 03=495, 04=69 (max DAYS_LOST 3,616, max DAYS_RESTRICT 3,310). RETURN_TO_WORK_DT: 6 values before 2000 (one `1969-12-31` epoch sentinel; by degree 07=5, 01=1); **6,683 rows fall on 1 May** of some year (1,736 in base; by degree 03=5,242, 05=675, 04=412, 06=218, 07=85, 02=38, 10=5, 09=4, 08=3, ?=1; 450-800 per year in 2003-2012, under 60 per year since 2013), a batch default used when closing cases, so the field is unreliable as a duration; RTW = ACCIDENT_DT on 12,721 degree-06 base rows (same-day return). RTW present by degree in base: 01=0.34%, 02=79.61%, 03=92.98%, 04=99.51%, 05=99.82%, 06=99.94%.

Hour-of-day distribution of valid ACCIDENT_TIME (all rows): 0h:4,213, 1h:5,839, 2h:5,721, 3h:5,323, 4h:4,930, 5h:5,950, 6h:8,448, 7h:12,838, 8h:18,036, 9h:19,321, 10h:22,480, 11h:19,150, 12h:12,148, 13h:16,483, 14h:16,575, 15h:13,473, 16h:10,880, 17h:9,067, 18h:8,177, 19h:7,440, 20h:7,354, 21h:7,019, 22h:6,633, 23h:6,117.

## 7. Leakage audit

### 7a. Column classification

| Column | Class | Note |
|---|---|---|
| `MINE_ID` | ID/ADMIN | join key to Mines/MinesProdYearly; 7-digit id; not a feature itself (high cardinality), use joined mine attributes |
| `CONTROLLER_ID` | ID/ADMIN | legal-entity id; could support a 'controller size' feature via aggregation but is an identifier |
| `CONTROLLER_NAME` | ID/ADMIN | name string |
| `OPERATOR_ID` | ID/ADMIN | legal-entity id |
| `OPERATOR_NAME` | ID/ADMIN | name string |
| `CONTRACTOR_ID` | FEATURE | known at filing; use as binary is_contractor flag (blank = operator employee), not raw id |
| `DOCUMENT_NO` | ID/ADMIN | form id; digits 2-8 encode filing Julian date -> filing lag is a possible derived feature, see quantification |
| `SUBUNIT_CD` | FEATURE | location within mine |
| `SUBUNIT` | FEATURE | description of SUBUNIT_CD (redundant) |
| `ACCIDENT_DT` | FEATURE | known; derive month, weekday; not raw date |
| `CAL_YR` | FEATURE | known; use as filter / optional numeric trend |
| `CAL_QTR` | FEATURE | derived from date |
| `FISCAL_YR` | FEATURE | redundant with CAL_YR |
| `FISCAL_QTR` | FEATURE | redundant with CAL_QTR |
| `ACCIDENT_TIME` | FEATURE | known; derive hour |
| `DEGREE_INJURY_CD` | LEAK | defines the target |
| `DEGREE_INJURY` | LEAK | target description |
| `FIPS_STATE_CD` | FEATURE | state |
| `UG_LOCATION_CD` | FEATURE | underground location; '?' for surface |
| `UG_LOCATION` | FEATURE | description |
| `UG_MINING_METHOD_CD` | FEATURE | underground method |
| `UG_MINING_METHOD` | FEATURE | description |
| `MINING_EQUIP_CD` | FEATURE | equipment type |
| `MINING_EQUIP` | FEATURE | description |
| `EQUIP_MFR_CD` | FEATURE | manufacturer code (high cardinality) |
| `EQUIP_MFR_NAME` | FEATURE | description |
| `EQUIP_MODEL_NO` | FEATURE | free-text model; very high cardinality; recommend drop or presence flag |
| `SHIFT_BEGIN_TIME` | FEATURE | known; derive shift-elapsed hours |
| `CLASSIFICATION_CD` | FEATURE | accident classification |
| `CLASSIFICATION` | FEATURE | description |
| `ACCIDENT_TYPE_CD` | FEATURE | accident type |
| `ACCIDENT_TYPE` | FEATURE | description |
| `NO_INJURIES` | FEATURE | number injured in accident; known at filing |
| `TOT_EXPER` | FEATURE | numeric |
| `MINE_EXPER` | FEATURE | numeric |
| `JOB_EXPER` | FEATURE | numeric |
| `OCCUPATION_CD` | FEATURE | occupation |
| `OCCUPATION` | FEATURE | description |
| `ACTIVITY_CD` | FEATURE | activity |
| `ACTIVITY` | FEATURE | description |
| `INJURY_SOURCE_CD` | FEATURE | injury source |
| `INJURY_SOURCE` | FEATURE | description |
| `NATURE_INJURY_CD` | FEATURE | nature of injury (known from initial medical assessment at filing) |
| `NATURE_INJURY` | FEATURE | description |
| `INJ_BODY_PART_CD` | FEATURE | body part |
| `INJ_BODY_PART` | FEATURE | description |
| `SCHEDULE_CHARGE` | LEAK | statutory days charged for permanent injury/death; deterministic function of outcome |
| `DAYS_RESTRICT` | LEAK | outcome: restricted days, finalised at closing |
| `DAYS_LOST` | LEAK | outcome: days away, finalised at closing |
| `TRANS_TERM` | LEAK | permanently transferred/terminated: known only after outcome |
| `RETURN_TO_WORK_DT` | LEAK | recorded when worker returns (closing form) |
| `IMMED_NOTIFY_CD` | FEATURE (caveat) | 15-minute immediate notification category is decided at accident time, before the 7000-1 form is filed; but code 01 Death is itself the fatality outcome -> strongly associated; keep only if collapsed to 'immediately reportable yes/no' or drop |
| `IMMED_NOTIFY` | FEATURE (caveat) | description of IMMED_NOTIFY_CD |
| `INVEST_BEGIN_DT` | LEAK | MSHA investigation start date; investigations follow serious events, recorded after filing |
| `NARRATIVE` | TEXT | free text written at filing; may mention outcome words; excluded from tabular model |
| `CLOSED_DOC_NO` | LEAK | closing (return to duty) form number; exists only after case closes |
| `COAL_METAL_IND` | FEATURE | coal vs metal/non-metal |

Counts: ID/ADMIN: 6; FEATURE: 39; LEAK: 9; FEATURE (caveat): 2; TEXT: 1.

### 7b. Association of LEAK candidates (and ambiguous columns) with the lost-time label, base

| Indicator | Share of positives | Share of negatives | Cramer's V | Positive rate when indicator true |
|---|---|---|---|---|
| `DAYS_LOST_gt0` | 95.29% | 0.00% | 0.955 | 100.00% |
| `DAYS_LOST_missing` | 3.18% | 0.21% | 0.116 | 93.57% |
| `DAYS_RESTRICT_gt0` | 21.87% | 39.62% | 0.192 | 34.44% |
| `SCHEDULE_CHARGE_gt0` | 3.43% | 0.00% | 0.134 | 100.00% |
| `CLOSED_DOC_NO_present` | 72.16% | 33.47% | 0.387 | 67.22% |
| `RETURN_TO_WORK_DT_present` | 93.00% | 99.89% | 0.188 | 46.97% |
| `TRANS_TERM_present` | 95.58% | 99.89% | 0.147 | 47.65% |
| `TRANS_TERM_eq_Y` | 8.89% | 2.29% | 0.145 | 78.73% |
| `INVEST_BEGIN_DT_present` | 9.12% | 4.40% | 0.094 | 66.35% |
| `IMMED_NOTIFY_CD_not_placeholder` | 5.59% | 1.76% | 0.102 | 75.10% |
| `IMMED_NOTIFY_CD_eq_01_death` | 1.14% | 0.00% | 0.076 | 100.00% |
| `IMMED_NOTIFY_CD_eq_02_serious` | 3.84% | 1.48% | 0.074 | 71.16% |
| `IMMED_NOTIFY_CD_eq_13_not_marked` | 0.14% | 0.01% | 0.024 | 92.21% |
| `NARRATIVE_mentions_days_or_lost_time` | 4.52% | 6.91% | 0.051 | 38.37% |
| `NARRATIVE_mentions_fatal_death` | 0.36% | 0.01% | 0.042 | 98.41% |
| `NARRATIVE_blank` | 0.00% | 0.00% | - | - |
| `NO_INJURIES_gt1` | 1.36% | 0.43% | 0.050 | 75.11% |
| `CONTRACTOR_ID_present` | 11.33% | 13.74% | 0.036 | 43.96% |
| `filing_lag_gt_30d` | 12.49% | 7.73% | 0.079 | 60.58% |

| Numeric | Point-biserial r | n | Median pos | Median neg | Mean pos | Mean neg |
|---|---|---|---|---|---|---|
| `DAYS_LOST` | 0.445 | 103,450 | 28 | 0 | 64.1 | 0.0 |
| `DAYS_RESTRICT` | -0.011 | 103,114 | 0 | 0 | 10.1 | 10.8 |
| `SCHEDULE_CHARGE` | 0.087 | 105,129 | 0 | 0 | 81.1 | 0 |
| `filing_lag_days(DOCUMENT_NO - ACCIDENT_DT)` | 0.042 | 105,182 | 9 | 7 | 26.1 | 19.2 |

Cramer's V with the label: DEGREE_INJURY_CD 1.0 (defines it), TRANS_TERM 0.2099, IMMED_NOTIFY_CD 0.1117.

IMMED_NOTIFY_CD x lost-time (base):

| IMMED_NOTIFY_CD | Neg | Pos | Positive rate |
|---|---|---|---|
| `01` | 0 | 583 | 100.00% |
| `02` | 797 | 1,967 | 71.16% |
| `03` | 17 | 37 | 68.52% |
| `04` | 11 | 12 | 52.17% |
| `05` | 8 | 32 | 80.00% |
| `06` | 15 | 10 | 40.00% |
| `07` | 2 | 7 | 77.78% |
| `08` | 13 | 21 | 61.76% |
| `09` | 5 | 13 | 72.22% |
| `11` | 35 | 40 | 53.33% |
| `12` | 41 | 72 | 63.72% |
| `13` | 6 | 71 | 92.21% |
| `?` | 52,957 | 48,419 | 47.76% |

TRANS_TERM x lost-time (base):

| TRANS_TERM | Neg | Pos | Positive rate |
|---|---|---|---|
| `(blank)` | 58 | 2,268 | 97.51% |
| `N` | 52,617 | 44,457 | 45.80% |
| `Y` | 1,232 | 4,559 | 78.73% |

| Lag (days after ACCIDENT_DT) | n pos | median pos | p75 pos | n neg | median neg | p75 neg |
|---|---|---|---|---|---|---|
| RETURN_TO_WORK_DT | 47,693 | 63 | 161 | 53,849 | 1 | 14 |
| INVEST_BEGIN_DT | 4,678 | 0 | 0 | 2,373 | 0 | 0 |

### 7c. Verdicts on the three ambiguous columns

- **IMMED_NOTIFY_CD**: the 15-minute immediate-notification category is decided at the scene, before the 7000-1 form is filed, so it *is* known at filing time. But it is a placeholder on 96.37% of base rows, code 01 'Death' is the fatality outcome itself (583 rows, 100% positive), code 02 'Serious injury' is 71% positive, and any non-placeholder value is 75.1% positive versus 48.8% overall (Cramer's V 0.1117). Keep only as a collapsed flag `immediately_reportable` with code 01 mapped to the flag (never as the raw code), or drop it; either choice moves the headline metric very little.
- **INVEST_BEGIN_DT**: present on 9.12% of positives vs 4.4% of negatives; median lag 0 days (MSHA investigators usually arrive the same day), but the decision to investigate is MSHA's response to severity and the date is entered by MSHA after the fact -> LEAK, drop.
- **TRANS_TERM**: blank on 2,326 base rows that are 97.51% positive (fatalities/permanent cases have no return-to-duty form), `Y` is 78.73% positive; it is recorded on the closing form -> LEAK, drop.
- **DOCUMENT_NO filing lag** (derived): median 9 vs 7 days, p90 43 vs 22, r = 0.0424. Technically known at filing, but operators file late precisely when the case is still open, so it encodes outcome; classified ID/ADMIN and excluded.
- **NARRATIVE**: regex for lost-time wording matches 4.52% of positives and 6.91% of negatives; fatality wording 0.36% vs 0.01% (98.41% positive when present). Median length 193 chars in base. TEXT, excluded from the tabular model.

## 8. FEATURE columns (base)

### 8a. Categorical cardinality and association

| Column | Levels | Share top-20 | Top-1 level | Levels < 100 rows | Levels < 10 rows | % blank/placeholder | Cramer's V | V (bias-corrected) |
|---|---|---|---|---|---|---|---|---|
| `SUBUNIT_CD` | 10 | 100.00% | `01` 33.79% | 1 | 0 | 0.00% | 0.184 | 0.184 |
| `FIPS_STATE_CD` | 54 | 78.77% | `54` 14.15% | 5 | 2 | 0.00% | 0.243 | 0.242 |
| `UG_LOCATION_CD` | 8 | 100.00% | `?` 66.21% | 1 | 0 | 66.21% | 0.175 | 0.175 |
| `UG_MINING_METHOD_CD` | 8 | 100.00% | `?` 67.09% | 1 | 0 | 67.09% | 0.192 | 0.192 |
| `MINING_EQUIP_CD` | 66 | 94.09% | `?` 53.76% | 24 | 4 | 53.76% | 0.205 | 0.203 |
| `EQUIP_MFR_CD` | 145 | 96.44% | `?` 53.77% | 115 | 58 | 53.77% | 0.155 | 0.151 |
| `EQUIP_MODEL_NO` | 11,672 | 79.00% | `(blank)` 76.67% | 11,659 | 11,389 | 76.73% | 0.349 | 0.106 |
| `CLASSIFICATION_CD` | 21 | 100.00% | `09` 35.06% | 5 | 2 | 0.00% | 0.216 | 0.215 |
| `ACCIDENT_TYPE_CD` | 41 | 96.70% | `08` 13.29% | 11 | 3 | 0.00% | 0.286 | 0.285 |
| `OCCUPATION_CD` | 244 | 71.42% | `304` 15.43% | 156 | 71 | 0.00% | 0.232 | 0.227 |
| `ACTIVITY_CD` | 97 | 86.54% | `028` 20.71% | 41 | 13 | 0.01% | 0.230 | 0.228 |
| `INJURY_SOURCE_CD` | 124 | 70.04% | `088` 12.68% | 36 | 15 | 0.00% | 0.314 | 0.312 |
| `NATURE_INJURY_CD` | 27 | 99.74% | `330` 32.36% | 6 | 3 | 0.00% | 0.391 | 0.391 |
| `INJ_BODY_PART_CD` | 46 | 89.68% | `340` 18.65% | 7 | 1 | 0.00% | 0.366 | 0.366 |
| `COAL_METAL_IND` | 2 | 100.00% | `M` 59.52% | 0 | 0 | 0.00% | 0.238 | 0.238 |
| `IMMED_NOTIFY_CD` | 13 | 100.00% | `?` 96.37% | 9 | 1 | 96.37% | 0.112 | 0.111 |
| `CAL_QTR` | 4 | 100.00% | `3` 27.2% | 0 | 0 | 0.00% | 0.010 | 0.009 |
| `CAL_YR` | 17 | 100.00% | `2011` 8.54% | 0 | 0 | 0.00% | 0.024 | 0.021 |
| `CONTRACTOR_ID` | 3,103 | 89.66% | `(blank)` 87.43% | 3,092 | 2,842 | 87.43% | 0.225 | 0.144 |
| `MINE_ID` | 8,455 | 12.42% | `0200024` 1.71% | 8,282 | 6,589 | 0.00% | 0.480 | 0.388 |
| `CONTROLLER_ID` | 3,935 | 32.41% | `0067299` 3.84% | 3,748 | 2,962 | 0.02% | 0.417 | 0.369 |
| `OPERATOR_ID` | 4,572 | 14.70% | `L12058` 1.71% | 4,328 | 3,170 | 0.02% | 0.444 | 0.392 |

MINE_ID, OPERATOR_ID, CONTROLLER_ID, CONTRACTOR_ID and EQUIP_MODEL_NO are listed for reference only: their V is inflated by thousands of sparse levels (EQUIP_MODEL_NO falls from 0.350 to 0.106 after bias correction) and they are ID/ADMIN or near-unique strings. UG_LOCATION_CD, UG_MINING_METHOD_CD, MINING_EQUIP_CD and EQUIP_MFR_CD are structurally missing for surface subunits / no-equipment accidents, so `?` is an informative level, not a gap to impute.

### 8b. Derived binary flags

| Flag | Share of base | Positive rate when true | Positive rate when false | Cramer's V |
|---|---|---|---|---|
| is_contractor | 12.57% | 43.96% | 49.44% | 0.036 |
| coal | 40.48% | 63.21% | 38.92% | 0.238 |
| underground_subunit_01 | 33.79% | 60.71% | 42.65% | 0.171 |

### 8c. Numeric features: distribution and point-biserial correlation with the label

| Feature | n | min | p25 | median | p75 | p99 | max | skew | r (point-biserial) | mean pos | mean neg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `TOT_EXPER` | 103,955 | 0.02 | 1.85 | 6 | 14.31 | 41 | 66 | 1.37 | 0.0625 | 10.42 | 9.12 |
| `MINE_EXPER` | 103,889 | 0.02 | 0.69 | 2.54 | 7.38 | 37.27 | 58 | 2.26 | -0.0116 | 5.72 | 5.90 |
| `JOB_EXPER` | 104,303 | 0.02 | 1 | 3 | 8.92 | 37.23 | 66 | 2.02 | 0.0141 | 6.59 | 6.35 |
| `accident_hour` | 101,656 | 0 | 8 | 11 | 15 | 23 | 23 | 0.07 | 0.0120 | 11.75 | 11.62 |
| `shift_elapsed_hours` | 99,685 | 0 | 2.50 | 4.50 | 7 | 12.71 | 23.98 | 1.38 | -0.0188 | 4.92 | 5.04 |
| `NO_INJURIES` | 105,191 | 1 | 1 | 1 | 1 | 1 | 36 | 48.59 | 0.0290 | 1.05 | 1.01 |
| `SHIFT_BEGIN_hour` | 102,758 | 0 | 6 | 7 | 15 | 23 | 23 | 1.04 | 0.0760 | 10.31 | 9.46 |
| `CAL_YR` | 105,191 | 2010 | 2013 | 2016 | 2021 | 2026 | 2026 | 0.29 | 0.0165 | 2016.93 | 2016.77 |
| `filing_lag_days` | 105,182 | -5 | 5 | 8 | 12 | 336 | 3,661 | 12.12 | 0.0424 | 26.06 | 19.25 |
| `accident_month` | 105,191 | 1 | 3 | 6 | 9 | 12 | 12 | 0.01 | -0.0086 | 6.33 | 6.38 |
| `accident_weekday_0mon` | 105,191 | 0 | 1 | 2 | 4 | 6 | 6 | 0.32 | -0.0195 | 2.29 | 2.36 |

### 8d. Positive rate by bin

`TOT_EXPER_years`: <1 -> 44.31% (n 17,070), 1-2 -> 44.0% (n 9,532), 2-5 -> 45.9% (n 19,728), 5-10 -> 49.86% (n 19,440), 10-20 -> 51.66% (n 20,540), 20-30 -> 52.36% (n 9,565), 30+ -> 54.88% (n 8,080)

`JOB_EXPER_years`: <1 -> 48.04% (n 25,970), 1-2 -> 47.1% (n 13,636), 2-5 -> 48.57% (n 23,421), 5-10 -> 49.09% (n 17,371), 10-20 -> 49.54% (n 14,424), 20-30 -> 50.57% (n 5,929), 30+ -> 50.93% (n 3,552)

`MINE_EXPER_years`: <1 -> 48.04% (n 30,022), 1-2 -> 48.67% (n 13,835), 2-5 -> 48.49% (n 23,307), 5-10 -> 50.99% (n 16,792), 10-20 -> 49.27% (n 12,360), 20-30 -> 44.26% (n 4,413), 30+ -> 45.92% (n 3,160)

`accident_hour`: 00-05 -> 54.46% (n 12,650), 06-11 -> 45.7% (n 40,574), 12-17 -> 47.02% (n 31,657), 18-23 -> 54.56% (n 16,775)

`shift_elapsed_hours`: 0-2 -> 49.57% (n 16,674), 2-4 -> 48.94% (n 24,797), 4-6 -> 48.64% (n 21,820), 6-8 -> 48.08% (n 18,372), 8-10 -> 46.14% (n 11,704), 10-12 -> 43.58% (n 4,812), 12+ -> 49.27% (n 1,506)

### 8e. Cramer's V ranking (top 15, base)

| Rank | Column | Cramer's V | Levels |
|---|---|---|---|
| 1 | `MINE_ID` | 0.480 | 8,455 |
| 2 | `OPERATOR_ID` | 0.444 | 4,572 |
| 3 | `CONTROLLER_ID` | 0.417 | 3,935 |
| 4 | `NATURE_INJURY_CD` | 0.391 | 27 |
| 5 | `INJ_BODY_PART_CD` | 0.366 | 46 |
| 6 | `EQUIP_MODEL_NO` | 0.349 | 11,672 |
| 7 | `INJURY_SOURCE_CD` | 0.314 | 124 |
| 8 | `ACCIDENT_TYPE_CD` | 0.286 | 41 |
| 9 | `FIPS_STATE_CD` | 0.243 | 54 |
| 10 | `COAL_METAL_IND` | 0.238 | 2 |
| 11 | `OCCUPATION_CD` | 0.232 | 244 |
| 12 | `ACTIVITY_CD` | 0.230 | 97 |
| 13 | `CONTRACTOR_ID` | 0.225 | 3,103 |
| 14 | `CLASSIFICATION_CD` | 0.216 | 21 |
| 15 | `MINING_EQUIP_CD` | 0.205 | 66 |

Excluding identifiers, the strongest circumstance codes are NATURE_INJURY_CD (0.391), INJ_BODY_PART_CD (0.366), INJURY_SOURCE_CD (0.314), ACCIDENT_TYPE_CD (0.286), FIPS_STATE_CD (0.243), COAL_METAL_IND (0.239), OCCUPATION_CD (0.232), ACTIVITY_CD (0.230), CLASSIFICATION_CD (0.216) and MINING_EQUIP_CD (0.205). Numeric experience and time-of-day features are weak (|r| <= 0.08). Overall base positive rate: 48.753%.

### 8f. Positive rate by most frequent levels

`NATURE_INJURY_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `330` | SPRAIN,STRAIN RUPT DISC | 34,039 | 62.10% |
| `180` | CUT,LACER,PUNCT-OPN WOUND | 25,256 | 18.43% |
| `220` | FRACTURE,CHIP | 18,284 | 55.19% |
| `160` | CONTUSN,BRUISE,INTAC SKIN | 7,780 | 58.34% |
| `400` | UNCLASSIFIED,NOT DETERMED | 4,555 | 63.12% |
| `170` | CRUSHING | 1,873 | 53.87% |
| `320` | DUST IN EYES | 1,751 | 16.16% |
| `310` | SCRATCH,ABRASION,SUPERFCL | 1,587 | 29.17% |
| `130` | BURN,CHEMICL-FUME,COMPOUN | 1,446 | 38.73% |
| `120` | BURN OR SCALD (HEAT) | 1,303 | 48.96% |

`INJ_BODY_PART_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `340` | FINGER(S)/THUMB | 19,618 | 28.27% |
| `420` | BACK (MUSCLES/SPINE/S-CORD/TAILBONE) | 12,377 | 62.92% |
| `512` | KNEE/PATELLA | 8,479 | 64.16% |
| `450` | SHOULDERS (COLLARBONE/CLAVICLE/SCAPULA) | 7,909 | 64.31% |
| `330` | HAND (NOT WRIST OR FINGERS) | 5,958 | 29.98% |
| `130` | EYE(S) OPTIC NERVE/VISON | 5,187 | 23.85% |
| `700` | MULTIPLE PARTS (MORE THAN ONE MAJOR) | 5,077 | 73.05% |
| `520` | ANKLE | 4,756 | 61.00% |
| `530` | FOOT(NOT ANKLE/TOE)/TARSUS/METATARSUS | 3,091 | 65.25% |
| `320` | WRIST | 2,479 | 40.30% |

`CLASSIFICATION_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `09` | HANDLING OF MATERIALS | 36,885 | 45.34% |
| `18` | SLIP OR FALL OF PERSON | 21,881 | 60.07% |
| `10` | HANDTOOLS (NONPOWERED) | 13,096 | 28.47% |
| `17` | MACHINERY | 11,674 | 44.64% |
| `12` | POWERED HAULAGE | 8,794 | 61.59% |
| `07` | FALL OF ROOF OR BACK | 3,465 | 57.46% |
| `21` | OTHER | 2,823 | 43.89% |
| `19` | STEPPING OR KNEELING ON OBJECT | 1,833 | 61.54% |
| `20` | STRIKING OR BUMPING | 1,279 | 37.14% |
| `06` | FALL OF FACE/RIB/PILLAR/SIDE/HIGHWALL | 956 | 74.58% |

`ACCIDENT_TYPE_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `08` | Struck by... NEC | 13,981 | 26.00% |
| `30` | Over-exertion NEC | 12,436 | 60.45% |
| `04` | Struck by falling object | 10,802 | 51.26% |
| `21` | Caught in, under or between a moving and | 10,736 | 38.36% |
| `27` | Over-exertion in lifting objects | 8,318 | 64.94% |
| `01` | Struck against stationary object | 6,953 | 36.83% |
| `17` | Fall to the walkway or working surface | 6,555 | 61.30% |
| `02` | Struck against a moving object | 5,547 | 61.87% |
| `05` | Struck by flying object | 5,473 | 31.54% |
| `28` | Over-exertion in pulling or pushing obje | 3,888 | 62.47% |

`SUBUNIT_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `01` | UNDERGROUND | 35,545 | 60.71% |
| `03` | STRIP, QUARY, OPEN PIT | 35,139 | 45.58% |
| `30` | MILL OPERATION/PREPARATION PLANT | 27,595 | 37.83% |
| `02` | SURFACE AT UNDERGROUND | 4,144 | 48.67% |
| `06` | DREDGE | 1,500 | 42.80% |
| `99` | OFFICE WORKERS AT MINE SITE | 673 | 43.68% |
| `17` | INDEPENDENT SHOPS OR YARDS | 213 | 48.36% |
| `12` | OTHER MINING | 160 | 30.62% |
| `04` | AUGER | 137 | 59.12% |
| `05` | CULM BANK/REFUSE PILE | 85 | 72.94% |

`COAL_METAL_IND`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `M` |  | 62,610 | 38.92% |
| `C` |  | 42,581 | 63.21% |

`INJURY_SOURCE_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `088` | METAL,NEC(PIPE,WIRE,NAIL) | 13,341 | 33.12% |
| `117` | GROUND | 8,987 | 60.23% |
| `086` | METAL COVERS & GUARDS | 8,443 | 38.79% |
| `089` | BROKEN ROCK,COAL,ORE,WSTE | 5,634 | 53.34% |
| `090` | CAVING ROCK,COAL,ORE,WSTE | 4,278 | 61.41% |
| `123` | MINE FLOOR,BOTTOM,FOOTWAL | 4,188 | 80.56% |
| `076` | SURFACE MINING MACHINES | 3,465 | 58.76% |
| `049` | KNIFE | 2,904 | 7.99% |
| `104` | HGHWY ORE CARIER,LRGE TRK | 2,583 | 59.31% |
| `035` | BELT CONVEYORS | 2,441 | 60.26% |

`ACTIVITY_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `028` | Handling supplies or material | 21,781 | 47.96% |
| `039` | Machine maintenance | 15,965 | 43.63% |
| `030` | Hand tools (not powered) | 11,722 | 27.97% |
| `092` | Walking, Running | 11,420 | 56.93% |
| `023` | Getting on or off equipment | 7,315 | 60.55% |
| `080` | Roof bolter, NEC | 2,515 | 58.29% |
| `031` | Hand tools (powered) | 2,354 | 29.99% |
| `026` | Handling coal, Rock, Ore | 2,038 | 49.36% |
| `055` | Haulage or Dump truck | 2,013 | 59.36% |
| `040` | Move power cable | 1,624 | 72.35% |

`OCCUPATION_CD`:

| Code | Description | n | Positive rate |
|---|---|---|---|
| `304` | Maintenance man, Mechanic,  Repair/Servi | 16,227 | 37.41% |
| `316` | Laborer, Blacksmith, Bull gang, Parts ru | 8,380 | 44.45% |
| `374` | Warehouseman, Bagger, Palletizer/Stacker | 7,284 | 38.48% |
| `046` | Roof bolter, Rock bolter,  Pinner, Mobil | 5,846 | 59.24% |
| `116` | Laborer, Bull gang,  Parts runner, Roust | 5,833 | 63.95% |
| `368` | Bulldozer operator, Universal operator,  | 5,334 | 51.87% |
| `376` | Haul/Off road/Coal/Ore/Pit/Quarry/Rock/R | 4,768 | 57.24% |
| `104` | Maintenance man, Mechanic,  Repair/Servi | 2,537 | 53.72% |
| `382` | Front-end loader, Scraper-loader operato | 2,102 | 53.76% |
| `399` | Miner, Prospector, NEC | 2,068 | 39.65% |
