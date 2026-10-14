# Dataset research log (14 Oct 2026)

Three research passes (government/civic, health/science, business/finance) screened about 60 public datasets
against the brief: not from Kaggle, open licence, stable direct download, tabular with a supervised target,
realistic messiness, a second source to join, a clear business question, and few existing online solutions
(the brief awards 0 for a solution similar to one found on the internet).

## Chosen: MSHA Mine Accident, Injury and Illness data (US Department of Labor)

| Item | Value |
|---|---|
| Landing page | https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp (also catalogued at https://catalog.data.gov/dataset/msha-accident-injuries-data-set) |
| Files used | `Accidents.zip` (about 52 MB, pipe-delimited, 57 columns), `Mines.zip` (59 columns), `MinesProdYearly.zip`, plus the three `*_Definition_File.txt` codebooks |
| Licence | Work of the US federal government, public domain (17 U.S.C. 105); data.gov lists access level "public" |
| Refresh | Weekly. The snapshot date, byte sizes and SHA-256 of every file are recorded in `data/raw/SNAPSHOT.json` and `data/README.md` |
| Rows | About 263k injury records since 2000; the project filters to 2010 onward (roughly 60k to 120k rows) to keep the Streamlit app inside its memory budget |
| Target | Binary lost-time injury: DEGREE_INJURY in {fatal, permanent total or partial disability, days away from work with or without restricted activity} versus {no days away or restricted, first aid, other}. Prevalence to be verified at load time; fallback target "serious" (fatal, permanent, or 30+ days lost) if the balance is uninformative |
| Joins | Accidents -> Mines on MINE_ID (many-to-one: mine type, commodity/SIC, current employees, state and county, coordinates, safety committee flag). Accidents -> MinesProdYearly on MINE_ID + CAL_YR (hours worked and average employees, so injury rates per 200,000 hours) |
| Leakage columns to drop | DAYS_LOST, DAYS_RESTRICT, SCHEDULE_CHARGE, RETURN_TO_WORK_DT, TRANS_TERM, CLOSED_DOC_NO, INVEST_BEGIN_DT, DEGREE_INJURY_CD (defines the target), NARRATIVE (free text; excluded to keep the solution tabular and original) |
| Known messiness | Code plus description pairs with placeholder codes ("?", "NO VALUE FOUND"); three experience fields where mine or job experience can exceed total experience; accident time and shift start stored as HHMM integers; free-text equipment manufacturer and model; repeated rows for multi-victim accidents; blank calendar fields |
| Business question | When an injury report is filed, can the operator, insurer or regulator predict whether the case becomes a lost-time injury, using the circumstances (activity, equipment, body part, nature of injury, worker experience, time into shift) and the mine's context (commodity, size, hours worked)? Supports workers'-compensation reserving, return-to-work planning and inspection targeting. Mining is one of Uzbekistan's largest industrial employers (Navoi MMC, Almalyk MMC) |
| Online-solution density | No Kaggle mirror of the accident file (Kaggle API search "MSHA", "mine accidents", "mining injuries"). GitHub "MSHA accident": 6 repositories, none a finished tabular model. Academic work: NIOSH narrative-text studies; Brown (2023) multiclass classifiers on leading indicators; a Santa Clara thesis (random forest on narratives); a 2023 National College of Ireland MSc thesis. This project differs: tabular features, relational join, cost-based threshold, deployed app |
| Risks | Weekly refresh (pinned snapshot); several hundred MB unzipped (filter years, drop narrative, store zstd parquet); encoding may be latin-1 |

## Runners-up (kept as fallbacks)

| Dataset | Licence | Size / target | Join | Density | Fit |
|---|---|---|---|---|---|
| UK Charity Commission register (10 tables on organisation_number) | OGL v3 | ~170k charities per cycle; late or missing annual return (~21 % positive) | 8+ official tables, postcode -> region, Companies House | No Kaggle mirror, 0 ML repos | 8.5 |
| CDC NHIS 2025 Sample Adult + imputed income + paradata | Public domain | ~25k x 577 columns; PHQ-8 depression or medical-bill hardship (~10 %) | HHX (1:10 imputations) | Released Aug 2026, no notebooks | 9.0 (health framing) |
| FHWA National Bridge Inventory, Iowa, multi-year | Public domain | ~23.7k bridges x 123; Good/Fair/Poor, deterioration label across years | STRUCTURE_NUMBER across years, county FIPS | 2 Kaggle mirrors, ~4 ML repos | 8.5 |
| NHANES 2021-2023 (DEMO, DPQ, BMX, BPXO, labs, questionnaires) | Public domain | ~6-7k adults after joins; PHQ-9 >= 10 (13 %) or hypertension | SEQN across 10+ files | Low-moderate | 8.5 |
| DOL PERM FY2025 | Public domain | 148k x 137; denied (2.4 %) | BLS wages, H-1B by employer | Low for new schema | 8.0 (downloads blocked to scripts) |

## Rejected (saturated, Kaggle-origin, or unusable)

Titanic, Iris, Wine, Adult, Heart Disease, Pima, Breast Cancer, Bank Marketing, Student Performance, Diabetes 130, Online Shoppers,
Students' Dropout, Stroke, OULAD (664 repos), SUPPORT2 (Kaggle mirror, Kazakh Streamlit app exists), SBA 7(a) (40+ Kaggle notebooks),
HMDA (fairness benchmark), Inside Airbnb (2,999 repos), BTS flights (3,739 repos), Stack Overflow survey (79 repos), STATS19, Victoria crashes,
NHTSA CRSS, OSHA Severe Injury Reports (published RF papers), Open Food Facts (1.3 GB, Nutri-Score is a formula), WHO/OWID aggregates,
Eurostat and datahub.io (aggregates only, usable as join sources), data.gov.uz (no row-level labelled data), World Bank L2CU (login).
