# Polish list (items noticed during review, to be fixed before the report is assembled)

- plots.histograms_grid: long x-axis labels overlap between adjacent panels (F03). Wrap labels with textwrap at ~28
  characters, increase `wspace`, and consider two figures (accident-level and mine-level variables) for the report.
- comparison_operating_point.csv: the `all_positive` baseline row shows TPR 0 and threshold 1.0 because its constant
  score equals the operating threshold search upper bound; report the baselines at their natural operating point
  (all_positive = flag everything) or exclude baselines from the operating-point table.
- Temporal-split variant reuses the primary out-of-fold thresholds (integration note); state this in the report or tune
  thresholds on the pre-2020 folds.
- Paired fold test between the top two models is only exact when both were fitted on the same rows (capped models such
  as svc_rbf use a subsample); report `same_fit_rows` next to the p-value.
