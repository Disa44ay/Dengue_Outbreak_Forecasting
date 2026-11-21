# Requirement — AI for Urban Dengue Outbreak (Preliminary Round)

## Challenge

Dengue outbreaks in Bangladeshi cities lag 2–4 weeks behind the rain that causes them (7–14 days
for mosquito breeding + viral incubation). Build a working prototype that uses weather data to
predict ward-level dengue outbreak risk, giving a 2–4-week early-warning window.

## Functional requirements

1. **Weather Data Processing** — ingest `avg_temp_c`, `rainfall_mm`, `relative_humidity_pct` and
   compute multi-week rainfall/temperature/humidity lags, grouped per `ward_id`, to match the
   mosquito breeding + incubation cycle.
2. **Risk Prediction Engine** — a trained model dynamically generates a risk score/tier (Low,
   Medium, High, Critical) for a selected ward and time window.
3. **Cost and Resource Efficiency** — run efficiently: minimize API calls, keep inference cheap
   enough for a public health office's budget.
4. **User Interface** — a working web dashboard displaying predictions and the main weather
   drivers behind each one, clearly.

## Dataset

Weekly weather + case data, 2016-01-04 to 2025-12-29, 60 wards across 5 cities (Dhaka North,
Dhaka South, Chattogram, Gazipur, Narayanganj). Columns: `date, city, ward_id,
population_density_sqkm, water_logging_risk_index, sanitation_score, avg_temp_c, max_temp_c,
min_temp_c, rainfall_mm, relative_humidity_pct, dengue_cases`. No missing values. Not tracked in
this repo (see `plan.md` for how to obtain it).

## Design decisions

- **Model:** LightGBM regression with a **Poisson objective** (`objective="poisson"` — matches the
  count-data, zero-inflated shape of `dengue_cases` better than squared-error loss) forecasts
  `dengue_cases` 2–4 weeks ahead per ward from lagged weather + static ward vulnerability features
  (`population_density_sqkm`, `water_logging_risk_index`, `sanitation_score`). Chosen after
  comparing 11 candidates (squared-error and Poisson-loss variants of LightGBM, random forest,
  gradient boosting/HistGradientBoosting, XGBoost, plus linear/ridge) on the identical
  chronological test split, verified 2026-11-22: lowest average MAE (1.497), narrowly ahead of
  Poisson HistGradientBoosting (1.498 — effectively tied) and clearly ahead of its own
  squared-error version (1.555) and a naive "always predict the train median" baseline (2.578).
  Poisson HistGradientBoosting was **not** picked despite the statistical tie, because
  `HistGradientBoostingRegressor` has no `feature_importances_` attribute — it would silently
  break the driver-breakdown requirement (criterion 4) rather than erroring. Every Poisson-loss
  variant beat its squared-error counterpart, confirming (independently, on a different feature
  set) the same finding from a teammate's separate model-selection notebook. Predicted rate maps
  to a tier via **train-split-only**
  quantile thresholds (Low = 0, Medium = 1–4, High = 5–10, Critical = 11+ — `medium_max=4.0`,
  `high_max=10.0`, verified 2026-11-22; computed on train rows only, not the full dataset, to
  avoid leaking val/test distribution into the tier boundaries), documented with exact numbers in
  `ARCHITECTURE.md`.
- **Lag features:** 1–4 week lagged rainfall/temperature/humidity, computed per `ward_id`
  (`groupby` + `shift`/rolling window).
- **Validation:** chronological split — train on 2016–2023, validate on 2024, test on 2025. No
  shuffling, no k-fold.
- **Inference:** model trained offline, serialized (`joblib`); the running app only does local
  inference. The brief only requires *minimizing* API calls (criteria 3 and 6), not eliminating
  them — the team chose zero external API calls and zero LLM tokens per prediction as the
  simplest way to satisfy that, not because the brief mandates it.
- **Stack:** FastAPI backend (serves inference + driver breakdown) + React frontend (ward/time
  window selection, risk display).

## Non-goals

- Live weather API integration, alerting/notification delivery, authentication.
- Cities/wards outside the provided dataset.
- Hyperparameter tuning or ensembling beyond a solid chronological-split baseline.

## Scored deliverables

Reproducible local setup (≤3 min clone-and-run), the multi-week lag pipeline, the trained risk
engine with tiers, the interactive dashboard, a root `ARCHITECTURE.md` (data-flow diagram,
validation-split rationale, model rationale), documented compute/API cost controls, a 3-minute
**uncut** demo video (**no slides allowed** — hard 0 if violated), and distributed git history
with an AI/external-API disclosure.

## Bonus (up to +5 pts, only after all core requirements are met)

Per the brief, high-impact domain-relevant extensions only — cosmetic UI additions don't qualify:

- Interactive ward-level risk heatmaps.
- Multi-week forecast confidence intervals.
- Chronological validation error dashboard (MAE/RMSE) — cheapest to add since `metrics.json`
  already exists from `ml/train.py`.
