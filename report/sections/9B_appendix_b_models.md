# Appendix B Model explanations

{tab:shap} lists the encoded columns with the largest mean absolute SHAP values for HGB, the values that Section 4.5 compares with the logistic coefficients of {fig:coefficients}. Cuts and finger or thumb injuries rank first and fourth, and both carry negative logistic coefficients.

Table: Mean absolute SHAP value of the 15 encoded columns with the largest values for HGB, computed by TreeExplainer on 500 seeded test reports in log-odds units, beside the coefficient of the same column in the tuned logistic regression fitted on the training split (numeric inputs standardised); one-hot columns are labelled with their MSHA codebook meaning. Source: author's analysis of MSHA open data. {#tab:shap}

| Rank | Encoded column | Meaning | Mean absolute SHAP value (log-odds) | Logistic regression coefficient |
|---|---|---|---|---|
| 1 | NATURE_INJURY_CD_180 | Nature of the injury: cut, lacer, punct-opn wound | 0.575 | -1.555 |
| 2 | prev_lost_time_share | Lost-time share of the mine's previous-year cases (ratio) | 0.365 | 0.356 |
| 3 | COAL_METAL_IND_C | Sector (coal or metal/non-metal): coal | 0.350 | 0.258 |
| 4 | INJ_BODY_PART_CD_340 | Body part injured: finger(s)/thumb | 0.206 | -0.882 |
| 5 | longitude | Longitude (degrees east) | 0.082 | 0.103 |
| 6 | NATURE_INJURY_CD_100 | Nature of the injury: amputation or enucleation | 0.073 | 5.605 |
| 7 | NATURE_INJURY_CD_220 | Nature of the injury: fracture, chip | 0.068 | 0.061 |
| 8 | INJ_BODY_PART_CD_130 | Body part injured: eye(s) optic nerve/vison | 0.067 | -0.452 |
| 9 | INJ_BODY_PART_CD_142 | Body part injured: mouth/lip/teeth/tongue/throat/taste | 0.059 | -2.421 |
| 10 | prev_incidents | Incidents reported at the mine in the previous year (count) | 0.050 | 0.179 |
| 11 | prev_employees | Average employees at the mine in the previous year (persons) | 0.048 | -0.166 |
| 12 | immed_reportable | Immediately reportable accident type (flag) | 0.048 | 0.873 |
| 13 | prev_lost_time_rate_200k | log(1 + lost-time cases per 200,000 employee-hours, previous year) | 0.043 | 0.081 |
| 14 | INJ_BODY_PART_CD_330 | Body part injured: hand (not wrist or fingers) | 0.038 | -0.502 |
| 15 | NATURE_INJURY_CD_330 | Nature of the injury: sprain, strain rupt disc | 0.038 | -0.319 |
