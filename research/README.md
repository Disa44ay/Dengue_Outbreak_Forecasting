# Research

Earlier model-selection work. Nothing here is imported by the shipped pipeline in `ml/`.

## Files

- `dengue_model_selection_colab.ipynb`: the model-selection notebook. This copy has no saved cell outputs.
- `model_selection_report.json`: results of the completed full run of that notebook.
- `ml_src/`: the earlier 57-feature pipeline and its tests, kept for reference. Its imports still
  use the old `ml.src` path, so it does not run from this folder without restoring that path.

## How the shipped model was chosen

1. **Exploratory comparison, 57 features** (lags 0-8 across 5 weather columns, rolling windows,
   cyclic week encoding). Walk-forward CV over 2019-2022 with Poisson deviance as the primary
   metric, then a one-time 2023 confirmation. Winner: **Extra Trees (tuned)** with
   `min_samples_leaf=4`, `max_features=0.88013`, `max_samples=0.87011`.
2. **Production comparison, 15 features** (lags 1-4 of temperature, rainfall and humidity, plus
   3 static ward features). Poisson LightGBM had the lowest average MAE at 1.497, against 1.498
   for Poisson HistGradientBoosting and 1.502 for Poisson XGBoost.
3. **Head-to-head on the 15 features.** The tuned Extra Trees architecture with
   `criterion="poisson"` scored 1.532 average MAE, worse than Poisson LightGBM at 1.497.
   **Poisson LightGBM is therefore the shipped model.**

The results in steps 2 and 3 come from a teammate's separate notebook, which is not in this folder.

## Limitation

Steps 2 and 3 compared models on the same 2025 split that is reported as the test result, so the
shipped test MAE is optimistic and is not an untouched holdout score.
