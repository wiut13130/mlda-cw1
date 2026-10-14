# Executive summary

The business question is whether a mine injury report can be classified at filing as a lost-time case, one needing an indemnity reserve and a return-to-work plan. The study used 101,609 injury reports filed with the US Mine Safety and Health Administration in 2010-2025, joined to mine attributes and previous-year hours and case history. Eight algorithms were tuned with mine-year-grouped cross-validation, and histogram-based gradient boosting was selected.

It reaches a held-out test recall of 0.943 and a precision of 0.599 at the capacity threshold, which flags about three cases in four, with a PR-AUC of 0.841. I recommend that workers'-compensation insurers and self-insured operators rank new claims by this score for early review, with an adjuster confirming each flag. The label also records whether the operator offered restricted duty. Results are therefore reported by sector, and the score should not rank mines for inspection.
