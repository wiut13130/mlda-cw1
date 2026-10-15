# Appendix C The deployed application

The application at https://mlda-cw1-8han4medy3o7d8izzcywbm.streamlit.app serves the committed artefacts of the study through seven pages. {fig:app1} to {fig:app7} were captured from the live deployment on 15 October 2026 at a 1,400-pixel viewport. Every page imports the same `mlda` package as the notebook, so the numbers shown are the ones in Sections 2 to 5; the Preprocessing and Model training pages refit on samples of the training rows only, and the Inference page scores a new report through the saved pipeline.

![Data explorer page of the deployed application, with the three source tables, their keys and licence, the joined base table size and tabs for the joins, the target, a filterable sample and the data dictionary. Source: screenshot of the deployed application.](reports/figures/app_1_data_explorer.png){#fig:app1 width=6.3}

![Exploratory analysis page, with tabs for the summary statistics, distributions by class, correlation heatmaps, associations with the target and missingness, drawn from the training rows only. Source: screenshot of the deployed application.](reports/figures/app_2_eda.png){#fig:app2 width=6.3}

![Preprocessing page, where the imputer, scaler, encoder and outlier rule can be changed and the ColumnTransformer is refitted on a sample of training rows, showing the transformed shape, column count and timing. Source: screenshot of the deployed application.](reports/figures/app_3_preprocessing.png){#fig:app3 width=6.3}

![Model training page, which tunes a chosen algorithm live on a sample of training rows with grouped cross-validation and compares the result with the full-run search tables. Source: screenshot of the deployed application.](reports/figures/app_4_model_training.png){#fig:app4 width=6.3}

![Inference page, where a new Form 7000-1 report is entered field by field, the mine's previous-year exposure is looked up, and the saved pipeline returns a probability, the class at the capacity threshold and an explanation; a batch CSV can be scored in the same way. Source: screenshot of the deployed application.](reports/figures/app_5_inference.png){#fig:app5 width=6.3}

![Evaluation page, the dashboard of held-out results: metric cards against the success criteria, the comparison tables, curves, a threshold explorer with expected cost, the statistical tests, the sensitivity checks and feature importance. Source: screenshot of the deployed application.](reports/figures/app_6_evaluation.png){#fig:app6 width=6.3}

![About page, with the business question, data provenance and licence, the reproduction commands, the run manifest and the limits and ethics notes. Source: screenshot of the deployed application.](reports/figures/app_7_about.png){#fig:app7 width=6.3}
