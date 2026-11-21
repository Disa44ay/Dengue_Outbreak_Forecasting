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

- **Model:** regression model forecasts `dengue_cases` 2–4 weeks ahead per ward from lagged
  weather + static ward vulnerability features (`population_density_sqkm`,
  `water_logging_risk_index`, `sanitation_score`). Predicted rate maps to a tier via data-driven
  quantile thresholds (Low = 0, Medium = 1–4, High = 5–9, Critical = 10+), documented with exact
  numbers in `ARCHITECTURE.md`.
- **Lag features:** 1–4 week lagged rainfall/temperature/humidity, computed per `ward_id`
  (`groupby` + `shift`/rolling window).
- **Validation:** chronological split — train on 2016–2023, validate on 2024, test on 2025. No
  shuffling, no k-fold.
- **Inference:** model trained offline, serialized (`joblib`); the running app only does local
  inference — zero external API calls, zero LLM tokens per prediction.
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
uncut demo video, and distributed git history with an AI/external-API disclosure.
