# AI for Urban Dengue Outbreak — Ward-Level Risk Prediction

**Team:** Arman, Sayeed, Ahnaf — EDU AI Engineering Hackathon 2026, Preliminary Round (24-hour qualifier).

## What this is

Dengue outbreaks in Bangladeshi cities lag 2–4 weeks behind the rain that causes them: stagnant
water takes 7–14 days to breed adult Aedes aegypti mosquitoes, plus viral incubation before
symptoms show. By the time hospital admissions spike, mosquito-control teams are already weeks
behind. This project ingests weekly ward-level weather data, builds multi-week rainfall/temperature/
humidity lag features per ward, and trains an ML model that forecasts dengue case-rate 2–4 weeks
ahead per ward — converted into a Low / Medium / High / Critical risk tier and shown in a web
dashboard with the main weather drivers behind each prediction.

Full rationale and design decisions: [requirement.md](requirement.md), [plan.md](plan.md),
[ARCHITECTURE.md](ARCHITECTURE.md) (root architecture doc, to be added).

## Dataset

`data/dengue_dataset.csv` — 31,320 rows, weekly cadence, **2016-01-04 to 2025-12-29**, covering
**60 wards across 5 cities** (Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj). No
missing values.

| Column | Meaning |
|---|---|
| `date` | Week start date |
| `city`, `ward_id` | Location |
| `population_density_sqkm` | Static ward crowding |
| `water_logging_risk_index` | Static ward drainage vulnerability (0–1) |
| `sanitation_score` | Static ward waste-collection quality (0–1) |
| `avg_temp_c`, `max_temp_c`, `min_temp_c` | Weekly temperature |
| `rainfall_mm` | Weekly rainfall |
| `relative_humidity_pct` | Weekly humidity |
| `dengue_cases` | Ground-truth weekly case count per ward (target) |

## Environment / versions

| Tool | Required (verified 2026-11-22 across two teammates' machines) |
|---|---|
| Python | **3.11 or newer** (verified working on 3.11.9 and 3.12 — exact package versions are pinned in `requirements.txt`, so any Python new enough to install them works) |
| Node.js | **26.13.1 LTS** (`winget upgrade --id OpenJS.NodeJS.LTS`) |
| npm | bundled with Node |
| git | 2.55+ |

(Earlier draft of this table claimed Python 3.15.2 was required — untested at the time. Don't
install a specific bleeding-edge version just to match that; any 3.11+ install works.)

## Setup

Run every command below from the **repo root** — `backend/app/inference.py` imports the sibling
`ml` package (`from ml.risk import compute_risk_tier`), which only resolves if Python is launched
from the root, not from inside `backend/`.

```powershell
# Backend (FastAPI) — from repo root
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload

# Frontend (React) — in a second terminal
cd frontend
npm install
npm run dev
```

## AI assistants and external APIs used

- Claude Code (Anthropic) — planning, scaffolding, and pair-programming assistance. Claude never
  commits to this repo (see `CLAUDE.md`); all commits are authored by the human teammates.
- No external/paid APIs are called at inference time — the risk model runs from a serialized
  local file (see `ARCHITECTURE.md` for the cost/efficiency rationale).
