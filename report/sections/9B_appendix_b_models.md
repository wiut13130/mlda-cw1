# Appendix B Model explanations

{tab:shap} lists the encoded columns with the largest mean absolute SHAP values for HGB, the values that Section 4.5 compares with the logistic coefficients of {fig:coefficients}. Cuts and finger or thumb injuries rank first and fourth, and both carry negative logistic coefficients.

Table: SHAP values for HGB and logistic coefficients || Mean absolute SHAP value of the 15 encoded columns with the largest values for HGB, computed by TreeExplainer on 500 seeded test reports in log-odds units, beside the coefficient of the same column in the tuned logistic regression fitted on the training split (numeric inputs standardised); one-hot columns are labelled with their MSHA codebook meaning. Source: author's analysis of MSHA open data. {#tab:shap}

| Rank | Encoded column | Meaning | Mean absolute SHAP value (log-odds) | Logistic regression coefficient |
|---|---|---|---|---|
| 1 | NATURE_INJURY_CD_180 | Nature of the injury: cut, lacer, punct-opn wound | 0.570 | -1.549 |
| 2 | prev_lost_time_share | Lost-time share of the mine's previous-year cases (ratio) | 0.361 | 0.359 |
| 3 | COAL_METAL_IND_C | Sector (coal or metal/non-metal): coal | 0.344 | 0.271 |
| 4 | INJ_BODY_PART_CD_340 | Body part injured: finger(s)/thumb | 0.205 | -0.877 |
| 5 | longitude | Longitude (degrees east) | 0.083 | 0.108 |
| 6 | NATURE_INJURY_CD_100 | Nature of the injury: amputation or enucleation | 0.073 | 5.589 |
| 7 | NATURE_INJURY_CD_220 | Nature of the injury: fracture, chip | 0.069 | 0.066 |
| 8 | INJ_BODY_PART_CD_130 | Body part injured: eye(s) optic nerve/vison | 0.065 | -0.451 |
| 9 | INJ_BODY_PART_CD_142 | Body part injured: mouth/lip/teeth/tongue/throat/taste | 0.060 | -2.402 |
| 10 | prev_employees | Average employees at the mine in the previous year (persons) | 0.054 | -0.192 |
| 11 | immed_reportable | Immediately reportable accident type (flag) | 0.048 | 0.870 |
| 12 | prev_incidents | Incidents reported at the mine in the previous year (count) | 0.047 | 0.189 |
| 13 | prev_lost_time_rate_200k | log(1 + lost-time cases per 200,000 employee-hours, previous year) | 0.046 | 0.079 |
| 14 | NATURE_INJURY_CD_330 | Nature of the injury: sprain, strain rupt disc | 0.039 | -0.313 |
| 15 | INJ_BODY_PART_CD_330 | Body part injured: hand (not wrist or fingers) | 0.038 | -0.497 |
