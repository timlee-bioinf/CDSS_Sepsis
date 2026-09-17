# Pediatric Sepsis Mortality Prediction Models

Two baseline models, a logistic regression and a gradient boosted tree (LightGBM), that predict in hospital mortality for pediatric patients presenting with suspected sepsis. Data is sourced from the 2024 Pediatric Sepsis Data Challenge.

## What's in this folder

```
Data/
  train_2025-11-12.csv         # 2,138 patients, 89 deaths (4.2%)
  test_2025-11-12.csv          # 540 patients, 30 deaths (5.6%)
  validate_2025-11-12.csv      # 2,000 patients, labels withheld (used for external scoring)
  DataDictionary_simplified.docx

r_modelBuilding/
  load_data.R                  # reads a CSV and converts variables to proper factors
  training_logreg.R            # fits the logistic regression model
  training_gbm.R               # fits the LightGBM model

scoring/
  get_sepsis_score_LR.R        # loads and serves the trained LR model and threshold
  get_sepsis_score_gbm.R       # loads and serves the trained GBM model and threshold
  Example_LR.RData             # saved LR model object
  Example_lightgbm.model       # saved LightGBM model file
  evaluate_performance.R       # shared metrics: AUC, AUPRC, F1, net benefit, calibration (ECE)
  score_submission.R           # end to end script that loads data, scores, and evaluates
```

Each folder has its own `.Rproj` file, so open `r_modelBuilding/r_modelBuilding.Rproj` to build or retrain, and `scoring/scoring.Rproj` to score or evaluate.

## Data

Each row is one pediatric admission with roughly 70 features recorded at admission. These include vitals (HR, BP, temperature, SpO2, respiratory rate), anthropometrics, exam findings (Blantyre Coma Score components, capillary refill, respiratory distress), symptom history, comorbidities, prior care seeking, and household or socioeconomic context. The label is `inhospital_mortality` (0/1). See `DataDictionary_simplified.docx` for full variable definitions.

`load_data.R` handles the boilerplate. It converts Yes/No columns to logical or factor variables, sets clinically sensible reference levels for ordinal factors (for example, healthiest category first), and drops the patient ID column before modeling.

## Models

Both scripts follow the same pipeline:

1. Load train and test data via `load_data()`.
2. Impute missing values with single imputation predictive mean matching (`mice`, `m=1`).
3. Engineer one derived feature, the shock index (`SI = heart rate / systolic BP`).
4. Fit the model on the full training set, with no feature selection.
5. Pick a decision threshold on the training ROC curve using the Youden index.
6. Save the model object and print train and test performance.

Logistic regression (`training_logreg.R`) is a single `glm()` with every available predictor plus SI. `tobacco_adm` is dropped because sparse levels cause fitting issues.

LightGBM (`training_gbm.R`) uses default settings, `nrounds = 10`, with no hyperparameter tuning.

## Scoring interface

Each `get_sepsis_score_<model>.R` file defines two functions:

`load_sepsis_model()` loads the saved model and its stored threshold.

`get_sepsis_score(data, myModel)` applies the same preprocessing (imputation, SI) used at training time and returns a `data.frame(probSepsis, label)`.

`score_submission.R` runs this pipeline on train, test, and validate, times inference, and calls `evaluate_performance.R` to produce a metrics table (also written to `Results_<script>.csv`). Metrics reported include AUC, AUPRC, F1, sensitivity, specificity, confusion matrix counts, net benefit at the chosen threshold, expected calibration error (ECE), and a weighted composite score.

## Current baseline performance

Everything here is an intentionally minimal baseline. It uses a single imputation pass, one hand picked derived feature, no tuning or feature selection, and thresholds for both models were chosen automatically off the training ROC curve rather than optimized for the actual scoring metric.

## Ideas for improvement

**Feature engineering.** More clinical scores beyond shock index, such as quick SOFA style combinations, MUAC for age z scores, or temperature adjusted heart rate. Interaction terms. Better handling of the many Yes/No/NA symptom fields, since NA may itself be informative (for example "not asked" versus "no").

**Missing data.** The current approach imputes train and test separately, which risks inconsistent imputations at deployment time. It would be better to fit the imputation model on training data only and apply it to test and validate.

**Class imbalance.** Mortality is rare, around 4 to 6 percent. Class weighting, resampling, or a metric driven threshold would likely work better than Youden's J.

**Model selection and tuning.** Cross validation for hyperparameters, especially for the GBM since `nrounds=10` with defaults is very under tuned. Regularization (LASSO or ridge) or variable selection for the logistic model to reduce the long coefficient list. Trying other model families such as random forest, XGBoost, CatBoost, or calibrated ensembles.

**Calibration.** ECE is tracked but not optimized against. Platt scaling or isotonic regression on a held out set could help.

**Threshold selection.** Pick the operating threshold based on the actual deployment cost trade off, or the competition's weighted score, rather than Youden's index on training data.

**Consistent preprocessing pipeline.** Wrap imputation and feature engineering into a single reusable function so training and scoring can't drift apart. Right now the same code is duplicated across scripts.

## Requirements

R packages: `mice`, `pROC`, `lightgbm`, `MLmetrics`.
