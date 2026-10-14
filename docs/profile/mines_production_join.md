# Mines, production and join profile (Task B)

Source: MSHA open data snapshot 2026-10-14 (`data/raw/SNAPSHOT.json`). Files: `Mines.txt` (92,060 rows x 59 cols), `MinesProdYearly.txt` (658,922 rows x 11 cols), `Accidents.txt` (275,220 rows; 7 columns loaded, 125,718 rows with CAL_YR >= 2010). Loader: `pd.read_csv(sep='|', quotechar='"', dtype=str, keep_default_na=False)`. Placeholders counted separately: blank `""`, whitespace-only, `?`, `NO VALUE FOUND`. Target used for association checks: lost-time = DEGREE_INJURY_CD in {01,02,03,04} vs {05,06}; other codes excluded.

## 0. Encoding

| File | Strict UTF-8 | First error | Non-ASCII bytes | Lines failing UTF-8 | Loaded as | Rows loaded = newlines-1 | Columns with non-ASCII |
|---|---|---|---|---|---|---|---|
| Mines.txt | fail | byte 1069693: invalid start byte | 1425 | 162 of 92,061 | latin-1 | True | CURRENT_MINE_NAME (2 rows), CURRENT_OPERATOR_NAME (3), DIRECTIONS_TO_MINE (219), NEAREST_TOWN (1); 224 rows total |
| MinesProdYearly.txt | fail | byte 59788956: invalid continuation byte | 105 | 105 of 658,923 | latin-1 | True | CURR_MINE_NM only (105 rows; bytes 0xF1 n-tilde, 0xF3 o-acute) |
| Accidents.txt (7 cols) | fail | byte 3626765: invalid continuation byte | 123 | n/a | latin-1 | True | none of the 7 loaded columns |

Mines.txt is mixed: 162 lines carry raw latin-1 bytes (strict UTF-8 fails) and 62 lines carry valid UTF-8 multi-byte sequences (read as mojibake under latin-1). No join key, code or numeric column contains a non-ASCII byte, so latin-1 (lossless byte-to-char) is safe for modelling; only free-text name/direction fields are affected. pandas row counts equal newline count minus header for all three files (no embedded newlines inside quoted fields).

## 1. Mines.txt

### 1.1 Key

| Check | Value |
|---|---|
| Rows | 92,060 |
| Distinct MINE_ID | 92,060 |
| Duplicate rows on MINE_ID | 0 |
| All 7 characters, all digits | True / True |
| Blank MINE_ID | 0 |
| MINE_IDs in production not in Mines | 0 |
| Mines with no production row 2000-2026 | 58,189 (63.21 %) |

MINE_ID is a clean primary key; `validate='many_to_one'` holds for Accidents -> Mines.

### 1.2 Per-column missingness (all 59 columns) and temporal class

| Column | Blank | WS-only | ? | NO VALUE FOUND | Missing % | Distinct | Temporal class |
|---|---|---|---|---|---|---|---|
| MINE_ID | 0 | 0 | 0 | 0 | 0.00 | 92,060 | Key |
| CURRENT_MINE_NAME | 0 | 0 | 0 | 0 | 0.00 | 67,880 | Current, identifier |
| COAL_METAL_IND | 0 | 0 | 0 | 0 | 0.00 | 2 | Stable |
| CURRENT_MINE_TYPE | 192 | 0 | 0 | 0 | 0.21 | 3 | Current (drifts; see 1.6) |
| CURRENT_MINE_STATUS | 0 | 0 | 0 | 0 | 0.00 | 7 | Current, dated |
| CURRENT_STATUS_DT | 0 | 0 | 0 | 0 | 0.00 | 14,307 | Current, date |
| CURRENT_CONTROLLER_ID | 1,031 | 0 | 0 | 0 | 1.12 | 41,097 | Current, identifier |
| CURRENT_CONTROLLER_NAME | 1,031 | 0 | 0 | 0 | 1.12 | 40,770 | Current, identifier |
| CURRENT_OPERATOR_ID | 975 | 0 | 0 | 0 | 1.06 | 49,851 | Current, identifier |
| CURRENT_OPERATOR_NAME | 0 | 0 | 0 | 0 | 0.00 | 53,279 | Current, identifier |
| STATE | 0 | 0 | 0 | 0 | 0.00 | 55 | Stable |
| BOM_STATE_CD | 0 | 0 | 0 | 0 | 0.00 | 55 | Stable |
| FIPS_CNTY_CD | 0 | 0 | 0 | 0 | 0.00 | 296 | Stable |
| FIPS_CNTY_NM | 0 | 0 | 0 | 0 | 0.00 | 1,784 | Stable |
| CONG_DIST_CD | 24,657 | 0 | 0 | 0 | 26.78 | 36 | Stable (redistricting) |
| COMPANY_TYPE | 975 | 0 | 0 | 0 | 1.06 | 5 | Current (controller), dated via CONTROLLER_BEGIN_DT |
| CURRENT_CONTROLLER_BEGIN_DT | 975 | 0 | 0 | 0 | 1.06 | 12,549 | Current, date |
| DISTRICT | 0 | 0 | 0 | 0 | 0.00 | 13 | Stable (admin) |
| OFFICE_CD | 0 | 0 | 0 | 0 | 0.00 | 89 | Current (admin) |
| OFFICE_NAME | 0 | 0 | 0 | 0 | 0.00 | 89 | Current (admin) |
| ASSESS_CTRL_NO | 54,133 | 0 | 0 | 0 | 58.80 | 37,927 | Current, identifier (post-accident) |
| PRIMARY_SIC_CD | 260 | 0 | 0 | 0 | 0.28 | 120 | Stable |
| PRIMARY_SIC | 260 | 0 | 0 | 0 | 0.28 | 120 | Stable |
| PRIMARY_SIC_CD_1 | 260 | 0 | 0 | 0 | 0.28 | 28 | Stable |
| PRIMARY_SIC_CD_SFX | 260 | 0 | 0 | 0 | 0.28 | 31 | Stable |
| SECONDARY_SIC_CD | 91,201 | 0 | 0 | 0 | 99.07 | 64 | Stable, sparse |
| SECONDARY_SIC | 91,201 | 0 | 0 | 0 | 99.07 | 64 | Stable, sparse |
| SECONDARY_SIC_CD_1 | 91,201 | 0 | 0 | 0 | 99.07 | 22 | Stable, sparse |
| SECONDARY_SIC_CD_SFX | 91,201 | 0 | 0 | 0 | 99.07 | 19 | Stable, sparse |
| PRIMARY_CANVASS_CD | 260 | 0 | 0 | 0 | 0.28 | 6 | Stable |
| PRIMARY_CANVASS | 260 | 0 | 0 | 0 | 0.28 | 5 | Stable |
| SECONDARY_CANVASS_CD | 91,201 | 0 | 0 | 0 | 99.07 | 5 | Stable, sparse |
| SECONDARY_CANVASS | 91,201 | 0 | 0 | 0 | 99.07 | 5 | Stable, sparse |
| CURRENT_103I | 45,494 | 0 | 0 | 0 | 49.42 | 7 | Current, dated via CURRENT_103I_DT |
| CURRENT_103I_DT | 72,230 | 0 | 0 | 0 | 78.46 | 4,224 | Current, date |
| PORTABLE_OPERATION | 0 | 0 | 0 | 0 | 0.00 | 2 | Stable (LID) |
| PORTABLE_FIPS_ST_CD | 77,465 | 0 | 0 | 0 | 84.15 | 54 | Current |
| DAYS_PER_WEEK | 0 | 0 | 0 | 0 | 0.00 | 8 | Current (MIF), 0 = not recorded |
| HOURS_PER_SHIFT | 56,025 | 0 | 0 | 0 | 60.86 | 21 | Current (MIF) |
| PROD_SHIFTS_PER_DAY | 54,112 | 0 | 0 | 0 | 58.78 | 10 | Current (MIF) |
| MAINT_SHIFTS_PER_DAY | 58,344 | 0 | 0 | 0 | 63.38 | 13 | Current (MIF) |
| NO_EMPLOYEES | 39,723 | 0 | 0 | 0 | 43.15 | 292 | Current (MIF), 0 for closed mines |
| PART48_TRAINING | 0 | 0 | 0 | 0 | 0.00 | 2 | Current, undated |
| LONGITUDE | 44,296 | 0 | 0 | 0 | 48.12 | 34,633 | Stable |
| LATITUDE | 44,292 | 0 | 0 | 0 | 48.11 | 30,584 | Stable |
| AVG_MINE_HEIGHT | 76,506 | 0 | 0 | 0 | 83.10 | 190 | Current (MIF), coal |
| MINE_GAS_CATEGORY_CD | 91,718 | 0 | 0 | 0 | 99.63 | 8 | Current (MIF) |
| METHANE_LIBERATION | 82,198 | 0 | 0 | 0 | 89.29 | 248 | Current (MIF) |
| NO_PRODUCING_PITS | 83,736 | 0 | 0 | 0 | 90.96 | 11 | Current (MIF) |
| NO_NONPRODUCING_PITS | 77,450 | 0 | 0 | 0 | 84.13 | 10 | Current (MIF) |
| NO_TAILING_PONDS | 60,120 | 0 | 0 | 0 | 65.31 | 14 | Current (MIF) |
| PILLAR_RECOVERY_USED | 0 | 0 | 0 | 0 | 0.00 | 2 | Current, undated |
| HIGHWALL_MINER_USED | 0 | 0 | 0 | 0 | 0.00 | 2 | Current, undated |
| MULTIPLE_PITS | 0 | 0 | 0 | 0 | 0.00 | 2 | Current, undated |
| MINERS_REP_IND | 0 | 0 | 0 | 0 | 0.00 | 1 | Current, undated (constant N) |
| SAFETY_COMMITTEE_IND | 0 | 0 | 0 | 0 | 0.00 | 2 | Current, undated |
| MILES_FROM_OFFICE | 0 | 0 | 0 | 0 | 0.00 | 466 | Current (office), 0 = not recorded |
| DIRECTIONS_TO_MINE | 54,820 | 6 | 1448 | 0 | 61.13 | 31,564 | Free text |
| NEAREST_TOWN | 36,135 | 0 | 1431 | 0 | 40.81 | 13,230 | Free text |

`?` appears only in the two free-text columns (DIRECTIONS_TO_MINE 1,448, NEAREST_TOWN 1,431); `NO VALUE FOUND` never appears in Mines. Numeric MIF fields use 0 as an additional not-recorded value (see 1.4-1.5).

### 1.3 Categorical distributions

**CURRENT_MINE_TYPE**

| Value | n | % |
|---|---|---|
| Surface | 68,903 | 74.85 |
| Underground | 17,407 | 18.91 |
| Facility | 5,558 | 6.04 |
| <blank> | 192 | 0.21 |

**CURRENT_MINE_STATUS**

| Value | n | % |
|---|---|---|
| Abandoned | 69,585 | 75.59 |
| Abandoned and Sealed | 8,820 | 9.58 |
| Active | 6,614 | 7.18 |
| Intermittent | 5,465 | 5.94 |
| Temporarily Idled | 909 | 0.99 |
| NonProducing | 373 | 0.41 |
| New Mine | 294 | 0.32 |

**COAL_METAL_IND**

| Value | n | % |
|---|---|---|
| M | 56,326 | 61.18 |
| C | 35,734 | 38.82 |

**CURRENT_MINE_TYPE x COAL_METAL_IND** (n)

| Type | C | M |
|---|---|---|
| Surface | 17,624 | 51,279 |
| Underground | 14,589 | 2,818 |
| Facility | 3,413 | 2,145 |
| <blank> | 108 | 84 |

**CURRENT_MINE_STATUS x COAL_METAL_IND** (n)

| Status | C | M |
|---|---|---|
| Abandoned | 25,623 | 43,962 |
| Abandoned and Sealed | 8,811 | 9 |
| Active | 574 | 6,040 |
| Intermittent | 0 | 5,465 |
| Temporarily Idled | 280 | 629 |
| NonProducing | 314 | 59 |
| New Mine | 132 | 162 |

**STATE** (top 15 of 55 distinct, incl. PR, VI, GU, AS)

| State | n | % |
|---|---|---|
| KY | 14,084 | 15.3 |
| WV | 7,157 | 7.77 |
| PA | 6,398 | 6.95 |
| VA | 4,607 | 5.0 |
| TX | 3,660 | 3.98 |
| AL | 2,602 | 2.83 |
| NY | 2,577 | 2.8 |
| CO | 2,542 | 2.76 |
| CA | 2,443 | 2.65 |
| TN | 2,274 | 2.47 |
| OH | 2,265 | 2.46 |
| MN | 2,076 | 2.26 |
| WA | 2,060 | 2.24 |
| AZ | 2,015 | 2.19 |
| WI | 1,991 | 2.16 |
| <other 40 values> | 33,309 | 36.18 |

**COMPANY_TYPE**

| Value | n | % |
|---|---|---|
| Other | 63,418 | 68.89 |
| Corporation | 24,075 | 26.15 |
| Sole Proprietor | 2,541 | 2.76 |
| Partnership | 1,049 | 1.14 |
| <blank> | 975 | 1.06 |
| Limited Liability Corporation | 2 | 0.0 |

**PRIMARY_CANVASS**

| Value | n | % |
|---|---|---|
| Coal | 35,619 | 38.69 |
| SandAndGravel | 30,496 | 33.13 |
| Stone | 16,043 | 17.43 |
| Metal | 5,019 | 5.45 |
| Nonmetal | 4,623 | 5.02 |
| <blank> | 260 | 0.28 |

**PRIMARY_CANVASS_CD**

| Value | n | % |
|---|---|---|
| 2 | 34,047 | 36.98 |
| 5 | 30,496 | 33.13 |
| 6 | 16,043 | 17.43 |
| 8 | 5,019 | 5.45 |
| 7 | 4,623 | 5.02 |
| 1 | 1,572 | 1.71 |
| <blank> | 260 | 0.28 |

**PRIMARY_SIC** (top 15 of 121 distinct)

| PRIMARY_SIC | n | % |
|---|---|---|
| Coal (Bituminous) | 33,911 | 36.84 |
| Construction Sand and Gravel | 29,314 | 31.84 |
| Crushed, Broken Limestone NEC | 6,707 | 7.29 |
| Gold Ore | 3,024 | 3.28 |
| Crushed, Broken Stone NEC | 3,023 | 3.28 |
| Dimension Stone NEC | 1,623 | 1.76 |
| Coal (Anthracite) | 1,572 | 1.71 |
| Sand, Common | 1,176 | 1.28 |
| Crushed, Broken Traprock | 1,142 | 1.24 |
| Common Clays NEC | 747 | 0.81 |
| Sand, Industrial NEC | 742 | 0.81 |
| Dimension Sandstone | 667 | 0.72 |
| Crushed, Broken Granite | 666 | 0.72 |
| Crushed, Broken Sandstone | 605 | 0.66 |
| Silver Ore | 514 | 0.56 |
| <other 106 values> | 6,627 | 7.2 |

**CURRENT_103I**

| Value | n | % |
|---|---|---|
| <blank> | 45,494 | 49.42 |
| Never Had 103I Status | 26,739 | 29.05 |
| Removed From 103I Status | 19,587 | 21.28 |
| Inspect Once Every 5-days | 114 | 0.12 |
| Inspect Once Every 15-days | 58 | 0.06 |
| Inspect Once Every 10-days | 32 | 0.03 |
| Hazard | 24 | 0.03 |
| Ignition or Explosion | 12 | 0.01 |

**Y/N indicators**

| Column | Y | Y % | N | N % | Values not Y/N |
|---|---|---|---|---|---|
| PORTABLE_OPERATION | 14,595 | 15.85 | 77,465 | 84.15 | 0 |
| MINERS_REP_IND | 0 | 0.0 | 92,060 | 100.0 | 0 |
| SAFETY_COMMITTEE_IND | 2,087 | 2.27 | 89,973 | 97.73 | 0 |
| PART48_TRAINING | 48,745 | 52.95 | 43,315 | 47.05 | 0 |

MINERS_REP_IND is the constant `N` for all 92,060 mines (zero information; drop). Other Y/N columns (PILLAR_RECOVERY_USED, HIGHWALL_MINER_USED, MULTIPLE_PITS) contain only Y/N as well.

**Shift pattern fields (value counts)**

DAYS_PER_WEEK: 0=75,500 (82.01%); 5=13,539 (14.71%); 6=793 (0.86%); 7=689 (0.75%); 4=573 (0.62%); 1=422 (0.46%); 3=305 (0.33%); 2=239 (0.26%)

HOURS_PER_SHIFT: <blank>=56,025 (60.86%); 0=20,074 (21.81%); 8=8,319 (9.04%); 10=4,129 (4.49%); 12=1,272 (1.38%); 9=1,206 (1.31%); 11=372 (0.4%); 4=254 (0.28%); 1=103 (0.11%); 6=86 (0.09%); 5=70 (0.08%); 7=43 (0.05%); 2=26 (0.03%); 13=24 (0.03%); 14=21 (0.02%); 3=17 (0.02%); 24=9 (0.01%); 16=3 (0.0%); 15=3 (0.0%); 19=2 (0.0%); 18=1 (0.0%); 20=1 (0.0%)

PROD_SHIFTS_PER_DAY: <blank>=54,112 (58.78%); 0=20,276 (22.02%); 1=15,657 (17.01%); 2=1,452 (1.58%); 3=396 (0.43%); 5=105 (0.11%); 4=23 (0.02%); 8=20 (0.02%); 10=9 (0.01%); 13=6 (0.01%); 6=4 (0.0%)

MAINT_SHIFTS_PER_DAY: <blank>=58,344 (63.38%); 0=30,679 (33.33%); 1=2,711 (2.94%); 2=219 (0.24%); 3=52 (0.06%); 10=18 (0.02%); 5=12 (0.01%); 8=6 (0.01%); 9=6 (0.01%); 6=4 (0.0%); 4=3 (0.0%); 12=3 (0.0%); 11=2 (0.0%); 15=1 (0.0%)

### 1.4 Numeric columns

| Column | n | Missing % | Zeros | Neg | min | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NO_EMPLOYEES | 92,060 | 43.15 | 38,131 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 13 | 60 | 5,870 | 4.04 |
| NO_EMPLOYEES_active_only | 6,614 | 0.06 | 0 | 0 | 1 | 1 | 2 | 4 | 8 | 17 | 100 | 408 | 5,870 | 26.37 |
| DAYS_PER_WEEK | 92,060 | 0.00 | 75,500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 6 | 7 | 0.88 |
| HOURS_PER_SHIFT | 92,060 | 60.86 | 20,074 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 10 | 12 | 24 | 3.92 |
| PROD_SHIFTS_PER_DAY | 92,060 | 58.78 | 20,276 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 13 | 0.55 |
| MAINT_SHIFTS_PER_DAY | 92,060 | 63.38 | 30,679 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 15 | 0.11 |
| AVG_MINE_HEIGHT | 92,060 | 83.10 | 328 | 0 | 0 | 0 | 17 | 30 | 40 | 52 | 86 | 144 | 7,200 | 47.53 |
| AVG_MINE_HEIGHT_coal_only | 35,734 | 56.58 | 289 | 0 | 0 | 0 | 18 | 30 | 40 | 52 | 87 | 145 | 7,200 | 47.65 |
| MILES_FROM_OFFICE | 92,060 | 0.00 | 55,527 | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 230 | 350 | 14,593 | 50.66 |
| METHANE_LIBERATION | 92,060 | 89.29 | 9,612 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 146,281 | 13,248,458 | 20,401.54 |
| NO_PRODUCING_PITS | 92,060 | 90.96 | 7,825 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 11 | 0.08 |
| NO_NONPRODUCING_PITS | 92,060 | 84.13 | 11,745 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 9 | 0.22 |
| NO_TAILING_PONDS | 92,060 | 65.31 | 30,775 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 20 | 0.06 |

NO_EMPLOYEES = 0 for 38,131 mines, none of them Active (zeros by status: Abandoned 29,241, Abandoned and Sealed 8,285, Temporarily Idled 499, NonProducing 74, New Mine 30, Intermittent 2). Among Active mines the median is 8 employees, p95 100, max 5,870 (AZ, surface, metal; plausible outlier, not impossible). AVG_MINE_HEIGHT is populated for 15,515 coal and 39 metal/non-metal mines (all 39 metal values are 0).

**Coordinates**

| Check | LONGITUDE | LATITUDE |
|---|---|---|
| Blank | 44,296 (48.12 %) | 44,292 (48.11 %) |
| Non-numeric non-blank | 0 | 0 |
| Zero | 0 | 0 |
| min / max | -180.9990 / -0.0003 | -14.4144 / 90.9997 |
| p5 / p50 / p95 | -118.15 / -83.23 / -73.53 | 31.31 / 37.84 / 46.23 |
| abs > 180 / abs > 90 | 62 | 63 |
| Positive longitude / negative latitude | 0 | 5 (all STATE = AS, American Samoa: legitimate) |

Both coordinates present: 47,760 (51.88 %). Inside the US bounding box (lat 17.5-72, lon -180 to -64): 47,073 (51.13 % of all mines, 99.65 % of Active mines). Outside the box but non-missing: 687 rows (394 with latitude < 17.5 such as `.1694`, 162 with lat/lon apparently swapped, 63 with |lat| > 90, 62 with |lon| > 180, 5 American Samoa); by state WV 179, PA 121, KY 88; only 11 of them Active. Rule: keep coordinates only when inside the box (plus AS), else NaN.

### 1.5 Impossible or suspect values

| Rule | Count | Note |
|---|---|---|
| DAYS_PER_WEEK > 7 | 0 |  |
| DAYS_PER_WEEK = 0 | 75,500 | 6 Active (0.1 % of Active); 0 = not recorded for closed mines |
| HOURS_PER_SHIFT > 24 | 0 | max is 24 |
| HOURS_PER_SHIFT in (12, 24] | 64 | 13-24 h shifts: suspect |
| HOURS_PER_SHIFT = 0 | 20,074 | 61 Active; blank-or-0 is 1.0 % of Active mines |
| PROD_SHIFTS_PER_DAY > 3 | 167 | values 4-13 per 24 h |
| PROD_SHIFTS_PER_DAY = 0 and Active | 88 |  |
| MAINT_SHIFTS_PER_DAY > 3 | 55 | values 4-15 |
| PROD_SHIFTS x HOURS_PER_SHIFT > 24 | 190 | 110 Active |
| (PROD + MAINT shifts) x HOURS_PER_SHIFT > 24 | 735 |  |
| NO_EMPLOYEES < 0 | 0 |  |
| NO_EMPLOYEES > 2,000 | 1 | top 5: [1123, 1200, 1207, 1386, 5870] |
| AVG_MINE_HEIGHT > 1,200 in (100 ft) | 5 | top 5 (inches): [1320, 1800, 2237, 5160, 7200]; impossible seam heights |
| AVG_MINE_HEIGHT in (0, 24) in | 1,187 | commonest: 18=216, 20=203, 22=166, 1=99, 16=71, 12=69, 15=60, 23=57; value 1 (99 rows) is a placeholder, others likely feet |
| AVG_MINE_HEIGHT = 0 | 328 | incl. all 39 metal/non-metal values |
| MILES_FROM_OFFICE > 1,000 | 1 | max 14593 miles (ID); impossible |
| MILES_FROM_OFFICE > 500 | 112 | next largest 600 (x4) |
| MILES_FROM_OFFICE = 0 | 55,527 | 69.6 % of Abandoned vs 3.3 % of Active: 0 = not recorded |
| Coordinates outside US box (non-missing) | 687 | see 1.4 |
| COAL_METAL_IND not C/M | 0 |  |
| Dates unparseable (3 date columns) | 0 | all nonblank dates parse as mm/dd/yyyy; none after the snapshot date |

Date columns: CURRENT_STATUS_DT 1925-01-01 to 2026-10-08 (100 % populated); CURRENT_CONTROLLER_BEGIN_DT 1950-01-01 to 2026-10-01 (91,085 populated); CURRENT_103I_DT 1901-01-01 to 2026-09-29 (19,830 populated; blank for every 'Never Had 103I Status' mine).

### 1.6 Current-state versus stable columns (temporal mismatch)

Mines.txt is a snapshot of each mine's state on 2026-10-09, not its state in the accident year. Evidence from the 125,718 accident rows (2010+) joined to Mines:

| Accident year | Rows | Mine now Active % | Now Abandoned/Sealed % | Now NonProducing % | Now Temp. Idled % | NO_EMPLOYEES within +/-50 % of that year's avg employees % | corr(log NO_EMPLOYEES, log year avg employees) |
|---|---|---|---|---|---|---|---|
| 2010 | 10,756 | 51.3 | 37.0 | 7.5 | 2.5 | 66.4 | 0.672 |
| 2011 | 11,038 | 51.3 | 35.6 | 7.6 | 3.4 | 67.2 | 0.681 |
| 2012 | 10,036 | 54.6 | 32.3 | 8.0 | 3.1 | 66.3 | 0.68 |
| 2013 | 9,581 | 58.5 | 28.2 | 8.3 | 3.1 | 68.4 | 0.724 |
| 2014 | 9,420 | 61.3 | 26.5 | 7.9 | 2.3 | 69.8 | 0.746 |
| 2015 | 8,239 | 66.1 | 22.5 | 7.2 | 2.0 | 70.8 | 0.785 |
| 2016 | 7,067 | 72.1 | 17.5 | 5.8 | 2.2 | 71.9 | 0.808 |
| 2017 | 6,984 | 73.4 | 15.8 | 5.8 | 2.3 | 73.8 | 0.816 |
| 2018 | 6,863 | 77.3 | 12.8 | 5.9 | 1.8 | 74.1 | 0.843 |
| 2019 | 6,824 | 78.4 | 11.8 | 5.8 | 1.6 | 74.3 | 0.86 |
| 2020 | 5,391 | 83.8 | 7.5 | 4.7 | 1.5 | 77.9 | 0.907 |
| 2021 | 5,676 | 86.3 | 5.4 | 3.8 | 1.5 | 79.3 | 0.927 |
| 2022 | 5,980 | 87.5 | 4.1 | 3.5 | 1.6 | 78.9 | 0.936 |
| 2023 | 6,093 | 88.9 | 2.9 | 3.9 | 1.6 | 79.6 | 0.934 |
| 2024 | 5,991 | 92.0 | 2.1 | 2.6 | 0.9 | 83.6 | 0.943 |
| 2025 | 5,602 | 93.2 | 1.0 | 2.1 | 0.4 | 85.3 | 0.95 |
| 2026 | 4,177 | 94.5 | 0.2 | 1.8 | 0.3 | 86.1 | 0.96 |

- CURRENT_STATUS_DT is later than the accident date for 51,210 rows (40.7 %): the recorded status post-dates the accident. CURRENT_CONTROLLER_BEGIN_DT post-dates the accident for 26.3 % of rows (so COMPANY_TYPE describes a later owner), CURRENT_103I_DT for 16.6 %.
- NO_EMPLOYEES (current) agrees within +/-50 % with the production table's average employees in the accident year for 74.4 % of comparable rows overall, rising from 66 % (2010) to 86 % (2026); 26,100 rows (20.76 %) have NO_EMPLOYEES 0 or blank because the mine has since closed. A zero therefore encodes future closure, which is correlated with accident year, not with the mine's state at the time.
- CURRENT_MINE_TYPE also drifts: 6,799 of 53,577 accidents at underground subunits (01/02) are at mines now typed Surface or Facility, and 99.5 % of those mines reported underground hours in that year. A type derived from the production subunits reported in the accident year (01/02 -> Underground; only 30/99/17 -> Facility; else Surface) is available for 99.77 % of rows and equals CURRENT_MINE_TYPE for 87.54 %. Cramer's V with the target: current type 0.1501, type-at-time 0.1645, accident SUBUNIT_CD 0.1842.
- A current attribute is known to have been in force at the accident only when its date column is <= ACCIDENT_DT: status 59.3 % of rows, controller/COMPANY_TYPE 73.7 %, 103I 19.2 % plus 'Never Had 103I Status' (no date by construction).

**Classification used for feature eligibility**

| Class | Columns | Use |
|---|---|---|
| Stable (safe) | COAL_METAL_IND, PRIMARY_CANVASS(_CD), PRIMARY_SIC(_CD, _CD_1, _SFX), STATE, BOM_STATE_CD, FIPS_CNTY_CD/NM, DISTRICT, LATITUDE, LONGITUDE, PORTABLE_OPERATION; SECONDARY_* (99 % blank) | Allowed as mine context |
| Semi-stable, drifts | CURRENT_MINE_TYPE (87.5 % equal to type-at-time) | Prefer type derived from production subunits of the accident year, or accident SUBUNIT_CD |
| Current, dated | CURRENT_MINE_STATUS (+STATUS_DT), COMPANY_TYPE (+CONTROLLER_BEGIN_DT), CURRENT_103I (+103I_DT) | STATUS: exclude (future information, 40.7 % post-date the accident). COMPANY_TYPE / 103I: only where date <= ACCIDENT_DT or 'Never Had', else unknown |
| Current, undated MIF fields | NO_EMPLOYEES, DAYS_PER_WEEK, HOURS_PER_SHIFT, PROD/MAINT_SHIFTS_PER_DAY, AVG_MINE_HEIGHT, METHANE_LIBERATION, pits, ponds, PART48_TRAINING, SAFETY_COMMITTEE_IND, PILLAR/HIGHWALL/MULTIPLE_PITS, MILES_FROM_OFFICE, MINE_GAS_CATEGORY_CD | Limitation to document; zeros encode closure (treat 0 as NaN). Replace size fields by production-table employees/hours of the previous year |
| Identifiers / admin / free text | CURRENT_*_ID, *_NAME, ASSESS_CTRL_NO, OFFICE_CD/NAME, CONG_DIST_CD, PORTABLE_FIPS_ST_CD, DIRECTIONS_TO_MINE, NEAREST_TOWN, MINERS_REP_IND (constant) | Drop |

## 2. MinesProdYearly.txt

### 2.1 Schema and key

The 2011 codebook names differ from the file header: file `CURR_MINE_NM, STATE_ABBR, SUBUNIT_DESC, CALENDAR_YR, ANNUAL_HRS, ANNUAL_COAL_PROD, AVG_ANNUAL_EMPL, C_M_IND` = codebook `MINE_NAME, STATE, SUBUNIT, CAL_YR, ANNUAL_HOURS, ANNUAL_COAL_PRODUCTION, AVG_EMPLOYEE_CNT, COAL_METAL_IND`. Definitions carry over one-to-one.

| Check | Value |
|---|---|
| Rows | 658,922 |
| Distinct (MINE_ID, CALENDAR_YR, SUBUNIT_CD) | 658,922 |
| Rows in duplicated keys | 0 |
| Fully duplicated rows | 0 |
| Distinct (MINE_ID, CALENDAR_YR) | 370,794 |
| Distinct MINE_ID | 33,871 |
| MINE_ID all 7 digits | True |
| Non-numeric non-blank values in the 5 numeric columns | 0 |

Per-column missingness: only ANNUAL_COAL_PROD has blanks (72,210, 10.96 %); no `?` or `NO VALUE FOUND` anywhere. C_M_IND: M 588,571 (89.32 %), C 70,351 (10.68 %).

### 2.2 CALENDAR_YR

| Year | Rows | Total hours (millions) |
|---|---|---|
| 2000 | 25,696 | 534.49 |
| 2001 | 25,816 | 530.78 |
| 2002 | 25,498 | 506.91 |
| 2003 | 25,417 | 494.18 |
| 2004 | 25,760 | 512.68 |
| 2005 | 26,143 | 537.78 |
| 2006 | 26,456 | 554.1 |
| 2007 | 26,420 | 551.61 |
| 2008 | 26,511 | 548.25 |
| 2009 | 25,957 | 485.37 |
| 2010 | 25,338 | 495.31 |
| 2011 | 25,305 | 523.89 |
| 2012 | 25,256 | 518.68 |
| 2013 | 24,793 | 504.34 |
| 2014 | 24,424 | 499.99 |
| 2015 | 23,824 | 473.62 |
| 2016 | 23,363 | 433.45 |
| 2017 | 23,335 | 447.18 |
| 2018 | 23,254 | 462.15 |
| 2019 | 23,089 | 462.83 |
| 2020 | 22,613 | 410.8 |
| 2021 | 22,235 | 413.8 |
| 2022 | 22,169 | 437.41 |
| 2023 | 22,192 | 452.32 |
| 2024 | 22,127 | 453.16 |
| 2025 | 21,882 | 446.06 |
| 2026 | 24,049 | 218.81 |

Range 2000-2026, no missing. 2026 is partial: total hours are 0.491 x the 2025 total and the median mine reports 0.484 x its 2025 hours (11,453 mines in both years), i.e. about two quarters, while accidents run to 2026-10-06 (4,177 rows). Same-year rates for 2026 would be inflated about 2x.

### 2.3 Subunits

| SUBUNIT_CD | Rows | % |
|---|---|---|
| 03 STRIP, QUARY, OPEN PIT | 303,645 | 46.08 |
| 99 OFFICE WORKERS AT MINE SITE | 227,507 | 34.53 |
| 30 MILL OPERATION/PREPARATION PLANT | 68,055 | 10.33 |
| 06 DREDGE | 20,221 | 3.07 |
| 01 UNDERGROUND | 18,318 | 2.78 |
| 02 SURFACE AT UNDERGROUND | 15,456 | 2.35 |
| 04 AUGER | 2,291 | 0.35 |
| 05 CULM BANK/REFUSE PILE | 1,842 | 0.28 |
| 17 INDEPENDENT SHOPS OR YARDS | 1,219 | 0.18 |
| 12 OTHER MINING | 368 | 0.06 |

### 2.4 Numeric distributions

| Column | n | Blank | Zeros | Neg | min | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANNUAL_HRS | 658,922 | 0 | 3,319 | 0 | 0 | 2 | 12 | 1,223 | 4,203 | 13,108 | 70,061 | 279,082 | 4,642,869 | 19,592.5 |
| AVG_ANNUAL_EMPL | 658,922 | 0 | 3,319 | 0 | 0 | 1 | 1 | 1 | 3 | 7 | 33 | 128 | 2,107 | 10.0 |
| AVG_EMPLOYEE_HOURS | 658,922 | 0 | 3,330 | 0 | 0 | 2 | 10 | 631 | 1,647 | 2,143 | 2,698 | 3,156 | 5,760 | 1,450.5 |
| ANNUAL_COAL_PROD | 658,922 | 72,210 | 559,018 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 463,461 | 117,965,515 | 40,565.4 |

ANNUAL_HRS = 0 and AVG_ANNUAL_EMPL = 0 coincide exactly (3,319 rows; hours > 0 with employees = 0: 0, the reverse: 0). Extremes: ANNUAL_HRS top 5 [3716272, 3767642, 3895003, 3899068, 4642869], AVG_ANNUAL_EMPL top 5 [1805, 1908, 1934, 1978, 2107] (large open-pit metal operations; plausible). AVG_EMPLOYEE_HOURS > 4,000 h/yr for 753 rows and > 3,000 for 11,227 (implausible annual hours per head; none exceed 8,784). Values below 100 h for 68,059 rows reflect intermittent subunits (p5 of ANNUAL_HRS is 12 hours).

### 2.5 ANNUAL_COAL_PROD blanks by C_M_IND

| C_M_IND | Rows | Blank | Blank % | Zero | Positive |
|---|---|---|---|---|---|
| C | 70,351 | 5,768 | 8.2 | 37,002 | 27,581 |
| M | 588,571 | 66,442 | 11.29 | 522,016 | 113 |

Coal rows: blanks only at non-producing subunits (99: 2,168/14,183, 30: 1,530/11,488, 02: 1,912/10,776, 17: 158/837); producing subunits 01/03/04/05 have no blanks. 113 metal/non-metal rows (12 mines, subunit 03) report positive coal production: inconsistent C_M_IND, immaterial for the join. ANNUAL_COAL_PROD is usable only for coal; for M it is 0 or blank by definition.

### 2.6 Consistency AVG_EMPLOYEE_HOURS = ANNUAL_HRS / AVG_ANNUAL_EMPL

| Check | Value |
|---|---|
| Rows checkable (employees > 0) | 655,603 |
| abs(stored - computed) <= 1 | 655,603 (100.0 %) |
| Equal after round-half-even | 620,692 |
| Max absolute difference | 0.5 |
| Differences > 1 | 0 |
| Employees = 0 rows (stored value) | 3,319 (all stored as 0) |
| AVG_EMPLOYEE_HOURS = 0 with hours > 0 | 11 |

The stored column is exactly the rounded ratio (the remaining differences are 0.5 rounding ties), so it carries no extra information and can be recomputed after aggregation.

### 2.7 Mine-year aggregate (sum of hours and of average employees over subunits)

| Measure | n | Zeros | min | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mine-year hours | 370,794 | 916 | 0 | 4 | 139 | 1,952 | 7,529 | 24,092 | 128,940 | 502,520 | 8,490,584 |
| Mine-year employees | 370,794 | 916 | 0 | 1 | 1 | 3 | 5 | 13 | 60 | 234 | 4,251 |
| Hours per employee | 369,878 | 0 | 0 | 6 | 72 | 668 | 1,541 | 2,092 | 2,610 | 3,010 | 5,760 |

370,794 mine-years; subunits per mine-year: 2: 181,078, 1: 137,782, 3: 48,765, 4: 3,156, 5: 13. C_M_IND never mixes within a mine-year (0 cases).

## 3. Join coverage (Accidents, CAL_YR >= 2010)

Accident rows 2010+: 125,718 (DOCUMENT_NO unique: 275,220 of 275,220 overall, 0 duplicates; CAL_YR equals year(ACCIDENT_DT) in all rows; MINE_ID 7 digits in all rows). Target among them: lost-time 51,284, not lost-time 53,907, excluded codes 20,527 (16.33 %); prevalence 48.75 % among valid rows. 8,791 distinct mines.

| Step | Keys | How | Matched rows | Matched % | Unmatched | Matched with hours > 0 % | Note |
|---|---|---|---|---|---|---|---|
| (a) Accidents -> Mines | MINE_ID | left, many_to_one | 125,718 | 100.0 | 0 | n/a | all rows match |
| (b) Accidents -> Production (subunit) | MINE_ID, CAL_YR=CALENDAR_YR, SUBUNIT_CD | left, many_to_one | 124,735 | 99.218 | 983 | 99.185 | 42 matched with 0 hours |
| (c) Accidents -> Production (mine-year, summed) | MINE_ID, CAL_YR=CALENDAR_YR | left, many_to_one | 125,432 | 99.773 | 286 | 99.769 | 697 rows match the mine-year but not the subunit |
| (c') Accidents -> Production (mine, previous year) | MINE_ID, CAL_YR-1=CALENDAR_YR | left, many_to_one | 124,291 | 98.865 | 1427 | 98.86 | time-safe exposure; 2009 production covers 2010 rows |

All merges preserved the left row count (asserted). The production key (MINE_ID, CALENDAR_YR, SUBUNIT_CD) is unique, so `validate='many_to_one'` passes for (b) without aggregation.

### 3.1 (d) COAL_METAL_IND agreement

| Comparison | Rows | Agree | Agree % |
|---|---|---|---|
| Accidents vs Mines | 125,718 | 125,718 | 100.0 |
| Accidents vs Production (subunit match) | 124,735 | 124,735 | 100.0 |
| Accidents vs Production (mine-year match) | 125,432 | 125,432 | 100.0 |
| Mines vs Production (mine-year) |  |  | 100.0 |

Perfect agreement; the three COAL_METAL_IND columns are redundant and any one can be kept.

### 3.2 (e) Unmatched rows

| Group (mine-year production join) | Rows | Lost-time | Not lost-time | Excluded codes | Excluded % | Lost-time rate % (valid rows) |
|---|---|---|---|---|---|---|
| Matched | 125,432 | 51,204 | 53,843 | 20,385 | 16.25 | 48.74 |
| Unmatched | 286 | 80 | 64 | 142 | 49.65 | 55.56 |
| Matched, hours = 0 | 5 | 2 | 3 | 0 | 0.0 | 40.0 |

Lost-time rate differs by 6.81 pp (unmatched higher), two-proportion z = 1.63 (not significant at 5 %, n = 144 valid rows). The unmatched rows are, however, skewed to excluded codes: 35.66 % are code 07 occupational illness versus 4.4 % among matched rows (illness claims filed against mines that reported no hours that year). Subunit-level join: 983 unmatched, lost-time rate 45.58 % vs 48.77 % (z = -1.66).

**Unmatched (mine-year) by year**

| Year | Unmatched | % of year's rows |
|---|---|---|
| 2010 | 27 | 0.25 |
| 2011 | 16 | 0.14 |
| 2012 | 16 | 0.16 |
| 2013 | 19 | 0.2 |
| 2014 | 36 | 0.38 |
| 2015 | 50 | 0.61 |
| 2016 | 32 | 0.45 |
| 2017 | 17 | 0.24 |
| 2018 | 12 | 0.17 |
| 2019 | 9 | 0.13 |
| 2020 | 6 | 0.11 |
| 2021 | 5 | 0.09 |
| 2022 | 10 | 0.17 |
| 2023 | 8 | 0.13 |
| 2024 | 3 | 0.05 |
| 2025 | 8 | 0.14 |
| 2026 | 12 | 0.29 |

**Unmatched (mine-year) by accident subunit and commodity**

| SUBUNIT_CD | Unmatched | % of unmatched | % of that subunit's rows |
|---|---|---|---|
| 01 | 96 | 33.57 | 0.2 |
| 03 | 82 | 28.67 | 0.21 |
| 30 | 61 | 21.33 | 0.2 |
| 02 | 32 | 11.19 | 0.57 |
| 05 | 10 | 3.5 | 9.52 |
| 99 | 4 | 1.4 | 0.48 |
| 04 | 1 | 0.35 | 0.69 |

Commodity: C 246 (86.01 %), M 40 (13.99 %) (coal is 44 % of all rows but 86 % of unmatched). Distinct mines: 172. Current status of those mines: Abandoned 101, Abandoned and Sealed 72, Active 53, Temporarily Idled 31, NonProducing 22, Intermittent 7. 125 of the 286 rows have production for the previous year and 65 for the next year; 20 are at mines that never appear in the production table.

**Degree of injury, unmatched vs matched (mine-year join)**

| DEGREE_INJURY_CD | Unmatched n | Unmatched % | Matched % |
|---|---|---|---|
| 07 | 102 | 35.66 | 4.43 |
| 03 | 61 | 21.33 | 30.73 |
| 06 | 47 | 16.43 | 25.88 |
| 00 | 23 | 8.04 | 10.03 |
| 05 | 17 | 5.94 | 17.04 |
| 04 | 13 | 4.55 | 8.7 |
| 10 | 7 | 2.45 | 0.69 |
| 09 | 6 | 2.1 | 0.28 |
| 02 | 4 | 1.4 | 0.93 |
| 08 | 4 | 1.4 | 0.82 |
| 01 | 2 | 0.7 | 0.47 |

### 3.3 Recommended join strategy

1. `Accidents LEFT JOIN Mines ON MINE_ID` with `validate='many_to_one'` and `indicator='_merge_mines'` (100.0 % match; assert row count unchanged). Keep only the stable columns plus dated current columns with their dates (section 1.6).
2. Aggregate production to mine-year (`groupby(['MINE_ID','CALENDAR_YR']).agg(hours=sum, employees=sum)`), then `LEFT JOIN ON MINE_ID, CAL_YR-1 = CALENDAR_YR` with `validate='many_to_one'`, `indicator='_merge_prod_prev'` (98.86 % of rows get previous-year hours > 0). Previous-year exposure is what is known when the report is filed; same-year totals are only known after year end and are half-complete for 2026.
3. Optionally also join the same-year mine-year total (99.773 %) for descriptive EDA (injury rates per 200,000 hours by year), not as a model feature. The subunit-level join adds little (hours at the accident subunit: r = 0.05) and loses 983 more rows; use the production subunit set only to derive mine type at the time.
4. Unmatched rows: keep them (left join), set exposure features to NaN and keep the indicator as a feature. Dropping them would remove 144 valid-target rows (0.14 %) with a lost-time rate of 55.56 % vs 48.74 %: no measurable bias either way, but keeping them matches the row-count assertions in `data/join.py`. Rows with previous-year hours < 2,000 (902 rows) produce unstable rates (max 20,000 per 200k hours): winsorise or log1p the rate and add the raw count.
5. COAL_METAL_IND agrees 100 % across the three tables; keep the Accidents copy and drop the others after the assertion.

## 4. Candidate mine and mine-year features

### 4.1 Missingness after the join (all 125,718 rows, 2010+)

| Feature | Missing % |
|---|---|
| CURRENT_MINE_TYPE | 0.0 |
| PRIMARY_CANVASS | 0.0 |
| PRIMARY_SIC | 0.0 |
| COAL_METAL_IND (Mines) | 0.0 |
| STATE | 0.0 |
| COMPANY_TYPE | 0.0 |
| NO_EMPLOYEES | 1.48 |
| NO_EMPLOYEES (>0) | 20.76 |
| DAYS_PER_WEEK | 0.0 |
| HOURS_PER_SHIFT | 1.56 |
| HOURS_PER_SHIFT (>0) | 27.65 |
| PROD_SHIFTS_PER_DAY | 1.59 |
| MAINT_SHIFTS_PER_DAY | 0.93 |
| CURRENT_103I | 6.35 |
| PORTABLE_OPERATION | 0.0 |
| MINERS_REP_IND | 0.0 |
| SAFETY_COMMITTEE_IND | 0.0 |
| PART48_TRAINING | 0.0 |
| AVG_MINE_HEIGHT | 59.95 |
| AVG_MINE_HEIGHT (coal rows only) | 9.36 |
| LATITUDE/LONGITUDE valid US | 0.17 |
| MILES_FROM_OFFICE | 0.0 |
| my_hours (same-year mine total) | 0.23 |
| my_hours (>0) | 0.23 |
| my_employees (same-year) | 0.23 |
| my_hours_per_employee (same-year) | 0.23 |
| su_hours (same-year subunit) | 0.78 |
| su_hours (>0) | 0.82 |
| prev_hours (previous-year mine total) | 1.14 |
| prev_hours (>0) | 1.14 |
| prev_employees | 1.14 |
| prev_hours_per_employee | 1.14 |
| prev_incident_rate_200k | 1.14 |
| prev_lost_time_rate_200k | 1.14 |
| prev_lost_time_share | 18.81 |

`(>0)` rows treat 0 as missing. NO_EMPLOYEES, HOURS_PER_SHIFT, DAYS_PER_WEEK and MILES_FROM_OFFICE have near-zero blank rates but 21-28 % zeros that mean 'closed since'. CURRENT_103I is blank for 6.35 % after the join (49 % in Mines, because blanks are mostly abandoned mines); only 19.2 % of rows have a 103I date at or before the accident, 82,448 valid-target rows are demonstrably in force (date <= accident or 'Never Had'), Cramer's V on those 0.1877.

### 4.2 Lagged exposure feature (time-safe)

Definition: for an accident at mine m in year t, `prev_incident_rate_200k = incidents(m, t-1) / hours(m, t-1) x 200,000`, with incidents counted from the full Accidents file (all codes) and hours from the mine-year production aggregate; `prev_lost_time_rate_200k` uses only lost-time cases (codes 01-04) in t-1; `prev_lost_time_share = lost-time cases / valid-code cases in t-1`. Where production exists for t-1 but no accident was recorded, the count is a true 0. Same-year counts are not usable: at filing time the year is unfinished, the count would include the row being predicted and every later case, and the year's hours are unknown until the fourth quarter is reported (and for 2026 only two quarters exist). Residual caveat: 22.0 % of cases occur in Q4, so a few t-1 cases may still be open when a Q1 report is filed; a strict variant would use cases closed before ACCIDENT_DT.

| Measure | Value |
|---|---|
| Rows with previous-year production | 124,291 (98.86 %) |
| Rows with previous-year hours > 0 | 124,291 (98.86 %) |
| Rows with no previous-year production row | 1,427 |
| Coverage by year (min / max) | 98.3 % / 99.4 % (2010 uses 2009 production: 98.3 %) |
| Previous-year incidents = 0 | 20,520 (16.51 % of covered rows) |
| Rows with rate > 50 per 200k h | 472 (309 of them with < 10,000 previous-year hours) |

| Feature | n | Zeros | min | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prev_incidents (count) | 124,291 | 20,520 | 0.00 | 0.00 | 1.00 | 6.00 | 20.00 | 68.00 | 137.00 | 190.00 | 16.76 |
| prev_incident_rate_200k | 124,291 | 20,520 | 0.00 | 0.00 | 1.80 | 4.51 | 8.34 | 15.97 | 31.90 | 20,000.00 | 7.96 |
| prev_lost_time_rate_200k | 124,291 | 38,894 | 0.00 | 0.00 | 0.00 | 1.29 | 3.46 | 8.80 | 16.74 | 20,000.00 | 2.98 |
| prev_lost_time_share | 102,076 | 16,641 | 0.00 | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.00 | 1.00 | 0.49 |
| same_year_incident_rate_200k (leaky, reference) | 125,427 | 0 | 0.13 | 1.68 | 3.88 | 6.74 | 11.00 | 26.72 | 78.11 | 200,000.00 | 26.75 |

**Lost-time rate by previous-year lost-time share** (valid-target rows)

| prev_lost_time_share bin | n | Lost-time rate % |
|---|---|---|
| (-0.01, 0.0] | 14,451 | 31.8 |
| (0.0, 0.25] | 9,450 | 30.1 |
| (0.25, 0.5] | 22,464 | 41.4 |
| (0.5, 0.75] | 18,186 | 59.8 |
| (0.75, 0.999] | 10,155 | 76.9 |
| (0.999, 1.0] | 9,653 | 61.4 |
| missing (no valid-code case in t-1) | 20,832 | 47.7 |

**Lost-time rate by previous-year incident rate per 200k hours**

| prev_incident_rate_200k bin | n | Lost-time rate % |
|---|---|---|
| (-0.01, 0.0] | 18,467 | 47.4 |
| (0.0, 2.0] | 11,459 | 40.2 |
| (2.0, 5.0] | 28,885 | 44.2 |
| (5.0, 10.0] | 27,921 | 51.8 |
| (10.0, 20.0] | 14,378 | 59.0 |
| (20.0, 50.0] | 2,468 | 55.3 |
| (50.0, 1000000000.0] | 373 | 43.7 |

### 4.3 Association with the lost-time target (valid-target rows, n = 105,191, prevalence 48.75 %)

Point-biserial r (Pearson between the 0/1 target and the numeric feature, rows with the feature present):

| Numeric feature | r | n | Missing % |
|---|---|---|---|
| NO_EMPLOYEES (Mines, current) | -0.0335 | 103,566 | 1.54 |
| log1p NO_EMPLOYEES | -0.0457 | 103,566 | 1.54 |
| DAYS_PER_WEEK (Mines, current) | -0.0926 | 105,191 | 0.0 |
| HOURS_PER_SHIFT (Mines, current) | -0.1099 | 103,456 | 1.65 |
| PROD_SHIFTS_PER_DAY (Mines, current) | -0.0514 | 103,420 | 1.68 |
| MAINT_SHIFTS_PER_DAY (Mines, current) | -0.0754 | 104,115 | 1.02 |
| AVG_MINE_HEIGHT (Mines, coal, current) | -0.0375 | 38,339 | 63.55 |
| MILES_FROM_OFFICE (Mines) | -0.1334 | 105,191 | 0.0 |
| LATITUDE | -0.0038 | 104,989 | 0.19 |
| LONGITUDE | 0.1333 | 104,993 | 0.19 |
| my_hours (mine-year, same year) | 0.0045 | 105,047 | 0.14 |
| log1p my_hours | 0.0325 | 105,047 | 0.14 |
| my_employees (mine-year, same year) | -0.0042 | 105,047 | 0.14 |
| log1p my_employees | 0.0327 | 105,047 | 0.14 |
| my_hours_per_employee (same year) | 0.0279 | 105,042 | 0.14 |
| su_hours (subunit-year, same year) | 0.0481 | 104,513 | 0.64 |
| log1p su_hours | 0.0538 | 104,513 | 0.64 |
| prev_hours (mine, previous year) | 0.0091 | 103,951 | 1.18 |
| log1p prev_hours | 0.0369 | 103,951 | 1.18 |
| prev_employees (previous year) | -0.0003 | 103,951 | 1.18 |
| log1p prev_employees | 0.036 | 103,951 | 1.18 |
| prev_hours_per_employee | 0.0409 | 103,951 | 1.18 |
| prev_incidents (count, previous year) | 0.1033 | 104,013 | 1.12 |
| prev_incident_rate_200k (lag exposure) | 0.0033 | 103,951 | 1.18 |
| log1p prev_incident_rate_200k | 0.0746 | 103,951 | 1.18 |
| prev_lost_time_rate_200k (lag) | 0.0093 | 103,951 | 1.18 |
| log1p prev_lost_time_rate_200k | 0.2047 | 103,951 | 1.18 |
| prev_lost_time_share (lag) | 0.2936 | 84,359 | 19.8 |
| same_year_incident_rate_200k (LEAKY, reference only) | -0.0003 | 105,042 | 0.14 |
| same_year_lt_rate_200k (LEAKY, reference only) | 0.013 | 105,042 | 0.14 |

Within commodity (coal n = 42,581, lost-time rate 63.2 %; metal/non-metal n = 62,610, rate 38.9 %):

| Feature | r within coal | r within metal/non-metal |
|---|---|---|
| prev_lost_time_share | 0.2399 | 0.1959 |
| log1p_prev_lost_time_rate_200k | 0.1472 | 0.0892 |
| log1p_prev_incident_rate_200k | 0.0415 | -0.0376 |
| prev_incidents_count | 0.1307 | -0.0356 |
| MILES_FROM_OFFICE | -0.0351 | -0.0123 |
| MILES_FROM_OFFICE_nonzero_only | -0.0331 | -0.0099 |
| HOURS_PER_SHIFT_nonzero_only | -0.1321 | -0.031 |
| DAYS_PER_WEEK_nonzero_only | -0.0033 | -0.0855 |

The lagged lost-time share keeps r = 0.24 / 0.20 inside each commodity, so it is not a coal proxy. MILES_FROM_OFFICE's overall r = -0.13 collapses to -0.03 / -0.01 within commodity: it is confounded by commodity and by the 0 placeholder. Exposure size (hours, employees, same-year or previous-year) is nearly uncorrelated with the target (|r| <= 0.05).

Cramer's V (bias-uncorrected; inflated for high-cardinality columns such as PRIMARY_SIC and STATE):

| Categorical feature | Cramer's V | Levels | n | Missing % |
|---|---|---|---|---|
| CURRENT_MINE_TYPE | 0.1501 | 3 | 105,191 | 0.0 |
| COAL_METAL_IND (Mines) | 0.2385 | 2 | 105,191 | 0.0 |
| COAL_METAL_IND (Accidents) | 0.2385 | 2 | 105,191 | 0.0 |
| PRIMARY_CANVASS | 0.2404 | 5 | 105,191 | 0.0 |
| PRIMARY_SIC | 0.2709 | 100 | 105,191 | 0.0 |
| STATE | 0.2429 | 54 | 105,191 | 0.0 |
| COMPANY_TYPE | 0.0428 | 4 | 105,189 | 0.0 |
| CURRENT_MINE_STATUS (current, leaky-ish) | 0.0878 | 6 | 105,191 | 0.0 |
| CURRENT_103I | 0.209 | 7 | 97,816 | 7.01 |
| PORTABLE_OPERATION | 0.0109 | 2 | 105,191 | 0.0 |
| MINERS_REP_IND | n/a (constant) | 1 | 105,191 | 0.0 |
| SAFETY_COMMITTEE_IND | 0.1152 | 2 | 105,191 | 0.0 |
| PART48_TRAINING | 0.1378 | 2 | 105,191 | 0.0 |
| DISTRICT | 0.2067 | 13 | 105,191 | 0.0 |
| SUBUNIT_CD (Accidents) | 0.1842 | 10 | 105,191 | 0.0 |
| MINE_GAS_CATEGORY_CD | 0.0848 | 7 | 8,631 | 91.79 |
| HIGHWALL_MINER_USED | 0.026 | 2 | 105,191 | 0.0 |
| MULTIPLE_PITS | 0.0176 | 2 | 105,191 | 0.0 |
| PILLAR_RECOVERY_USED | 0.0395 | 2 | 105,191 | 0.0 |
| _m_mines indicator | n/a (constant) | 1 | 105,191 | 0.0 |
| _m_prodmy indicator | 0.005 | 2 | 105,191 | 0.0 |
| _m_prodsu indicator | 0.0051 | 2 | 105,191 | 0.0 |

Lost-time rate by category (valid-target rows):

COAL_METAL_IND: M 38.9 % (n 62,610); C 63.2 % (n 42,581)

PRIMARY_CANVASS: Coal 63.2 % (n 42,581); Stone 38.5 % (n 27,351); Metal 36.7 % (n 17,457); Nonmetal 40.8 % (n 9,146); SandAndGravel 42.7 % (n 8,656)

CURRENT_MINE_TYPE: Surface 43.8 % (n 51,634); Underground 58.9 % (n 36,816); Facility 41.8 % (n 16,741)

CURRENT_103I: Never Had 103I Status 42.5 % (n 63,764); Removed From 103I Status 59.0 % (n 16,773); Inspect Once Every 5-days 71.0 % (n 12,149); Inspect Once Every 10-days 58.7 % (n 2,508); Inspect Once Every 15-days 56.6 % (n 2,355); Hazard 28.2 % (n 149); Ignition or Explosion 55.1 % (n 118)

SAFETY_COMMITTEE_IND: N 46.3 % (n 88,496); Y 62.0 % (n 16,695)

COMPANY_TYPE: Other 50.7 % (n 52,860); Corporation 46.6 % (n 49,979); Partnership 49.5 % (n 2,212); Sole Proprietor 66.7 % (n 138)

PORTABLE_OPERATION: N 48.9 % (n 99,499); Y 46.5 % (n 5,692)

CURRENT_MINE_STATUS: Active 46.5 % (n 76,168); Abandoned 51.8 % (n 12,621); Abandoned and Sealed 57.0 % (n 5,852); NonProducing 60.7 % (n 5,795); Intermittent 46.2 % (n 2,744); Temporarily Idled 60.5 % (n 2,011)

STATE_top12: WV 68.1 % (n 14,888); KY 59.3 % (n 8,396); PA 55.9 % (n 7,900); TX 34.5 % (n 5,596); AZ 36.6 % (n 5,391); IL 55.4 % (n 4,532); NV 33.6 % (n 4,200); AL 60.6 % (n 3,792); IN 45.2 % (n 3,151); CA 38.2 % (n 3,037); VA 54.5 % (n 3,002); WY 44.2 % (n 2,796)

Type at the time of the accident (from production subunits) gives rates Underground 58.9 %, Surface 42.2 %, Facility 41.6 % versus current type 58.9 / 43.8 / 41.8 %. CURRENT_MINE_STATUS shows higher lost-time rates for mines now abandoned (51.8-60.7 %) than active (46.5 %): this is the accident-year effect (older cases, more coal) leaking through a future attribute, not a mine property.

### 4.4 Feature eligibility summary

| Feature | Verdict | Reason (numbers above) |
|---|---|---|
| COAL_METAL_IND / PRIMARY_CANVASS / PRIMARY_SIC | Allowed | Stable; V = 0.24 / 0.24 / 0.27; 0 % missing |
| STATE (or DISTRICT) | Allowed | Stable; V = 0.24 (54 levels) / 0.21 (13 levels) |
| Mine type | Allowed as type-at-time or SUBUNIT_CD | CURRENT_MINE_TYPE differs from type-at-time for 12.5 % of rows; V 0.150 vs 0.165 vs 0.184 |
| LATITUDE / LONGITUDE (validated) | Allowed, 0.17 % missing | Stable; r(lon) = 0.13 is the Appalachian coal gradient, redundant with STATE |
| PORTABLE_OPERATION | Allowed, low value | V = 0.011 |
| COMPANY_TYPE | Allowed only where CONTROLLER_BEGIN_DT <= ACCIDENT_DT (73.7 %) | V = 0.043 |
| CURRENT_103I | Allowed only where in force (82,448 valid rows) | V = 0.209 (all) / 0.188 (in force); 16.6 % of dates post-date the accident |
| SAFETY_COMMITTEE_IND, PART48_TRAINING | Allowed with documented caveat (undated current flags) | V = 0.115 / 0.138 |
| NO_EMPLOYEES, DAYS_PER_WEEK, HOURS_PER_SHIFT, PROD/MAINT shifts | Not recommended; if used, 0 -> NaN | Current MIF values; zeros encode closure (21-28 % of rows); abs(r) <= 0.11 |
| MILES_FROM_OFFICE | Drop | 0 = not recorded for 69.6 % of abandoned mines; r within commodity -0.03 |
| CURRENT_MINE_STATUS | Drop (leak of future state) | 40.7 % of status dates post-date the accident |
| MINERS_REP_IND, IDs, names, ASSESS_CTRL_NO, free text | Drop | constant / identifiers / text |
| prev_hours, prev_employees, prev_hours_per_employee | Allowed (98.9 % coverage) | time-safe size; abs(r) <= 0.04 |
| prev_incidents, prev_incident_rate_200k (log1p) | Allowed | r = 0.10 / 0.07; non-monotone by bin |
| prev_lost_time_rate_200k (log1p), prev_lost_time_share | Allowed, strongest mine-level signals | r = 0.20 / 0.29 (0.24 and 0.20 within commodity); share missing for 18.8 % (no t-1 case) |
| Same-year hours / counts / rates | EDA only | not known at filing time; 2026 half-year |

## 5. Open issues

- CONTRACTOR_ID was not loaded (outside the 7-column brief); contractor employees' hours are not in MinesProdYearly, so rates per 200k hours understate exposure at mines with heavy contractor use. The 286 unmatched rows could not be explained by contractor status here.
- The production codebook (2011) is out of date relative to the file header; the data card should list the file names.
- 2026 production covers about two quarters; any same-year exposure feature or rate for 2026 must be excluded or scaled.
- Illness cases (code 07) are over-represented among rows without same-year hours (102 of 286); they are already outside the target definition.
- Coordinates: 687 non-missing pairs are outside the US box (decimal-point and lat/lon swaps in old WV/PA/KY records); validate before any map or distance feature.
