\landscape

# Appendix A Additional descriptive statistics

{tab:descriptives_rest} gives the statistics of {tab:descriptives} for the eight numeric features that Section 2.3 leaves out of the main table, so that all 20 numeric features are described.

Table: Descriptive statistics of eight further numeric features || Sample statistics of the remaining eight of the 20 numeric features on the 81,289 training rows, with the same definitions as the main descriptive table of Section 2.3; the trimmed mean cuts 10% from each tail, variance and SD (sample standard deviation) use the n - 1 denominator, and skewness is read against the bands below 0.5 symmetric, 0.5 to 1 moderate and above 1 high. Source: author's analysis of MSHA open data. {#tab:descriptives_rest}

| Feature (unit) | Missing % | Mean | Median | Trimmed mean | Variance | SD | Min | Q1 | Q3 | Max | IQR | Skewness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exper_ratio_job (ratio) | 5.2 | 0.71 | 0.97 | 0.75 | 0.13 | 0.36 | 0.00 | 0.38 | 1.00 | 1.00 | 0.62 | -0.72 |
| no_injuries_capped (persons) | 0.0 | 1.02 | 1.00 | 1.00 | 0.05 | 0.23 | 1 | 1 | 1 | 9 | 0 | 21.94 |
| accident_hour (hour, 0-23) | 3.0 | 11.7 | 11.0 | 11.6 | 30.7 | 5.5 | 0 | 8 | 15 | 23 | 7 | 0.06 |
| shift_begin_hour (hour, 0-23) | 0.5 | 9.7 | 7.0 | 9.0 | 32.5 | 5.7 | 0 | 6 | 15 | 23 | 9 | 0.96 |
| latitude (degrees) | 0.2 | 38.4 | 38.2 | 38.2 | 26.2 | 5.1 | 18.0 | 35.9 | 40.5 | 68.0 | 4.6 | 0.96 |
| log1p_prev_hours (log1p of hours) | 1.2 | 12.27 | 12.42 | 12.36 | 2.79 | 1.67 | 0.69 | 11.15 | 13.63 | 15.86 | 2.48 | -0.55 |
| prev_hours_per_employee (hours per person) | 1.2 | 2,140.9 | 2,171.3 | 2,171.4 | 171,488 | 414.1 | 1.0 | 1,986.9 | 2,364.2 | 4,786.7 | 377.2 | -1.19 |
| prev_incident_rate_200k (log1p) | 1.2 | 1.51 | 1.66 | 1.53 | 0.83 | 0.91 | 0.00 | 0.97 | 2.18 | 3.93 | 1.20 | -0.27 |

\portrait
