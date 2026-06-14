# 🏆 FIFA World Cup 2026 Predictions — V8.0 Dynamic Engine

> **Quantitative Football Prediction System**: Dual-Engine Poisson Integration + 9D Intel Radar + LLM Synthesis

[**🇨🇳 中文**](README.md) | **English**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/Engine-V8.0-orange.svg)](v8_engine_whitepaper.md)

---

> **⚠️ Disclaimer**
>
> This project is built **purely out of personal interest** — I'm curious whether quantitative algorithms can meaningfully predict football match outcomes. I treat football as an interesting **data modeling and algorithm validation problem**, not a betting tool.
>
> - 🚫 **This project does NOT constitute any betting advice**, nor does it encourage gambling in any form
> - 🚫 **I do NOT recommend using this system for actual wagering or betting decisions**
> - ✅ If you're interested in the algorithms themselves, feel free to study, discuss, and improve them
> - ✅ I will continue to conduct **prediction vs actual result comparisons** to validate whether the algorithm truly has predictive power
>
> Football is inherently uncertain. No model can guarantee outcomes. Please be rational.

---

## 📖 About

This is a **quantitative football prediction system built out of personal interest**, designed for the 2026 FIFA World Cup. I want to explore whether **programming + mathematical modeling + data analysis** can produce meaningful predictions for match outcomes.

The system uses a three-layer architecture: **"Quantitative Engine + Intel Radar + LLM Synthesis"**:

1. **V8 Poisson Math Engine**: Dual-engine ML model trained on 49,000+ historical matches
2. **5D Intel Harvesters**: Real-time crawling of injuries, fatigue, weather, odds, and referee data
3. **LLM Synthesis Layer**: Fuses quantitative data with qualitative intel to generate structured analysis reports

> 📊 I will continuously track prediction accuracy and let the data speak — whether the algorithm actually works, the backtest results will tell.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    v8_agent_orchestrator.py                  │
│                    (Orchestrator / Scheduler)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Intel Harvesting          Phase 2: Math Engine    │
│  ┌─────────────────┐             ┌─────────────────┐        │
│  │ 🔍 News Harvester│             │  predict_v8.py  │        │
│  │ 🏃 Fatigue       │──impact──→  │  ┌───────────┐  │        │
│  │ 🎯 Game Theory   │   dict      │  │ Stacking   │  │        │
│  │ 💹 Market Odds   │             │  │ Classifier │  │        │
│  │ ⚖️ Micro Referee │             │  │ (30% wt.)  │  │        │
│  └─────────────────┘             │  ├───────────┤  │        │
│                                   │  │ Poisson   │  │        │
│  Phase 3: Context Building        │  │ Regressor │  │        │
│  ┌─────────────────┐             │  │ (70% wt.)  │  │        │
│  │ mega_context.md │◀────────────│  └───────────┘  │        │
│  │ (LLM Input)      │  results    └─────────────────┘        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  LLM Synthesis   │ → Final Report (pdata/)                │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
worldcup/
├── results.csv                        # 49,000+ historical matches (1872-2026)
├── requirements.txt                   # Python dependencies
├── v8_engine_whitepaper.md            # Algorithm whitepaper
├── .gitignore
│
├── v8_engine/                         # 🔧 Core Engine
│   ├── v8_shared.py                   # Shared module (caching, translation, model loading)
│   ├── core_model.py                  # Model training (Elo + Stacking + Poisson)
│   ├── predict_v8.py                  # Prediction engine main logic
│   ├── v8_agent_orchestrator.py       # LLM orchestrator (3-phase pipeline)
│   ├── backtest_v8.py                 # Historical blind backtest
│   ├── evaluate_v8.py                 # WC 2022 full evaluation
│   ├── robustness_test.py             # Parameter grid search (rho / fusion weights)
│   ├── train_model_v8.py              # Standalone training script
│   │
│   ├── orchestrator/                  # 📡 5D Intel Harvesters
│   │   ├── harvester_news_weather.py  # Injury/morale news (DuckDuckGo)
│   │   ├── harvester_fatigue.py       # Rest days / fatigue
│   │   ├── harvester_game_theory.py   # Motivation / biscotto risk
│   │   ├── harvester_market_odds.py   # Market odds / money flow (The-Odds-API)
│   │   └── harvester_micro_referee.py # Referee-team style interaction
│   │
│   └── data_scrapers/                 # 📊 Static Data
│       ├── squad_values.csv           # Squad market values (244 teams)
│       ├── tactical_styles.csv        # Tactical data (possession/PPDA/aerial)
│       ├── referee_stats.csv          # Referee strictness index (12 refs)
│       ├── xg_history.csv             # Historical xG data (49K matches)
│       ├── tactical_data_builder.py   # Tactical data generator
│       ├── referee_data_builder.py    # Referee data generator
│       ├── fbref_scraper.py           # FBRef data scraper
│       ├── transfermarkt_scraper.py   # Transfermarkt value scraper
│       └── update_pipeline.bat        # Data update pipeline
│
├── pdata/                             # 📈 Prediction Report Archive
│   ├── 2026_06_15_V8_Ultimate_Report.md
│   ├── 2026-06-16_V8_Hedge_Fund_Report.md
│   └── ...
│
└── Visualization/                     # 🎨 Visualization Frontend
    ├── backend/main.py                # FastAPI backend
    ├── frontend/                      # React + Vite frontend
    └── start_server.bat               # One-click launch script
```

---

## 🧮 Core Algorithm

### Dual-Engine Fusion Architecture

| Engine | Model | Output | Weight |
|--------|-------|--------|--------|
| **Engine A (Classifier)** | XGBoost + RandomForest + MLP → LogisticRegression → CalibratedClassifierCV | Win/Draw/Loss probabilities | 30% |
| **Engine B (Poisson Flow)** | XGBoost Poisson Regressor → Dixon-Coles correction (ρ=-0.05) → 15×15 score matrix | xG → Score probabilities | 70% |

Final probability = `P_engine_A × 0.3 + P_engine_B × 0.7`

### 5D Input Features

| Feature | Description | Source |
|---------|-------------|--------|
| `elo_diff` | Elo rating difference | Dynamic (K=60/40/30/20) |
| `sv_diff` | Squad value difference (€M) | Transfermarkt |
| `aerial_diff` | Aerial win rate difference | FBRef |
| `ppda_diff` | Pressing intensity difference | FBRef |
| `t_weight` | Tournament weight (WC=60) | Schedule |

### Dynamic xG Correction (V8 New)

External intel modifies xG via a multiplier chain:

```
xG_final = xG_base × mult_home × mult_away

Multipliers:
├── Fatigue (<5 days rest + high press): ×0.90
├── Rain + possession-based team:        ×0.85
├── Low motivation / biscotto risk:      ×0.60
└── Injury (per unit):                   ×(1 - 0.10)

Hard cap: total multiplier ≥ 0.60 (max 40% reduction)
```

### Dixon-Coles Low-Score Correction

Standard Poisson distribution has systematic bias on low scores (0-0, 1-0, 0-1, 1-1). Dixon-Coles correction factor ρ=-0.05 specifically addresses this:

```
τ(0,0) = 1 - λ₁λ₂ρ     # Correct 0-0
τ(0,1) = 1 + λ₁ρ        # Correct 0-1
τ(1,0) = 1 + λ₂ρ        # Correct 1-0
τ(1,1) = 1 - ρ           # Correct 1-1
```

---

## 🚀 Quick Start

### Requirements

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/LuckyOneTwoThree/fifa-worldcup-predictions.git
cd fifa-worldcup-predictions
pip install -r requirements.txt
```

### Dependencies

```
duckduckgo_search    # News harvesting
xgboost              # Gradient boosting models
scikit-learn         # ML pipeline
pandas               # Data processing
numpy                # Numerical computation
requests             # HTTP requests (odds API)
```

### Run Predictions

```bash
# Option 1: Math engine only (no LLM required)
cd v8_engine
python predict_v8.py 2026-06-15

# Option 2: Full pipeline (math engine + intel harvesting + LLM synthesis)
python v8_agent_orchestrator.py 2026-06-15
```

### Run Backtests

```bash
cd v8_engine

# WC 2022 full blind backtest (with Brier Score, calibration checks)
python evaluate_v8.py

# Parameter grid search (rho values / fusion weights)
python robustness_test.py

# Custom date blind backtest
python backtest_v8.py
```

---

## 📡 5D Intel Harvesters

| Harvester | Data Source | Collected | xG Impact |
|-----------|------------|-----------|-----------|
| **News Harvester** | DuckDuckGo News | Injuries, morale, breaking news | Injury ×0.90 per unit |
| **Fatigue Harvester** | results.csv history | Days since last match | <5 days ×0.90 (high-press teams) |
| **Game Theory Harvester** | Rule engine | Tournament stage, biscotto risk | LOW ×0.60 |
| **Market Odds Harvester** | The-Odds-API | Real odds / theoretical odds | For LLM reference |
| **Micro Referee Harvester** | Referee data | Referee-team style interaction | For LLM reference |

### Injury Detection Keywords (EN + CN)

```
injury, out, doubtful, sidelined, acl, knee, hamstring, ruled out,
缺阵, 受伤, 膝盖, 韧带, 报销, 伤退
```

---

## 📊 Data Description

### results.csv (Core Data)

- **Source**: International football match historical data
- **Time span**: 1872 - 2026
- **Records**: ~49,400 matches
- **Fields**: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

### Static Feature Data

| File | Entries | Description |
|------|---------|-------------|
| `squad_values.csv` | 244 teams | Squad market values (Top 31 precise, rest default 50M) |
| `tactical_styles.csv` | 243 teams | Possession / PPDA / Aerial win rate |
| `referee_stats.csv` | 12 refs | Cards per game / Penalties / Strictness index |
| `xg_history.csv` | 49,400 matches | Historical expected goals data |

---

## 🧪 Backtesting

### WC 2022 Blind Test Method

```python
# For each WC 2022 match day:
# 1. Train model on ALL data before that date (absolute blind test)
# 2. Predict all matches on that day
# 3. Compare with actual results
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **W/D/L Accuracy** | Win/Draw/Loss prediction accuracy |
| **Brier Score** | Probability calibration (lower is better) |
| **Exact Score Hit** | Exact score prediction hit rate |
| **Top 3 Coverage** | Top 3 score prediction coverage |
| **Calibration Check** | Actual hit rate for high-confidence predictions |

---

## 🤖 LLM Report Structure

The orchestrator generates `mega_context.md` containing full quantitative data and external intel, which the LLM uses to produce reports with the following structure:

### 5 Mandatory Sections

1. **📋 Global Overview** — One-page summary of all matches (30-second decision)
2. **🎯 Per-Match Analysis**
   - 🧮 Core Data Radar (7-metric comparison table)
   - 🎲 Poisson Score Matrix (Top 5)
   - 📡 External Intel (injury/fatigue/referee/motivation)
   - 💹 Market Odds & Expected Value
   - 📌 Execution Directive (resonance/divergence + Devil's Advocate)
3. **💰 Portfolio Bankroll Management** — Total exposure + circuit breaker
4. **🔍 Model Self-Check & Disclaimer** — Backtest results + data blind spots

### Risk Control Principles

- All positions based on Kelly Criterion, max 5% per match
- No "all-in" or emotional language
- When quant and tactical signals diverge → force No Bet
- Previous losses → auto-halve subsequent positions

---

## 🔧 Configuration & Extension

### Adding New Team Data

1. Add market value to `v8_engine/data_scrapers/squad_values.csv`
2. Add tactical data to `v8_engine/data_scrapers/tactical_styles.csv`
3. Add Chinese name to `v8_engine/v8_shared.py`'s `zh_translation`

### Connecting Real Odds API

Set environment variable `ODDS_API_KEY` (get from [The-Odds-API](https://the-odds-api.com)):

```bash
export ODDS_API_KEY=your_api_key_here
```

### Custom AGENT_PROMPT

Edit the `AGENT_PROMPT` variable in `v8_engine/v8_agent_orchestrator.py` to modify LLM report generation instructions.

---

## ⚠️ Disclaimer

- This project is **purely out of personal interest**, exploring quantitative algorithms in sports prediction
- It does **NOT constitute any betting advice** and does not encourage gambling
- I will continuously conduct **prediction vs actual result comparisons** to validate algorithm effectiveness
- Football is inherently uncertain — no model can guarantee outcomes
- If you're interested in the algorithms, feel free to study, discuss, and improve

---

## 📄 License

MIT License

---

## 👤 Author

**LuckyOneTwoThree** — [GitHub](https://github.com/LuckyOneTwoThree)

Built this prediction system out of my interest in quantitative analysis and football data. Feel free to discuss algorithm ideas.

---

## 🙏 Acknowledgments

- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system) — Arpad Elo
- [Dixon-Coles Model](https://doi.org/10.1111/1467-9574.00009) — Dixon & Coles (1997)
- [XGBoost](https://xgboost.readthedocs.io/) — Tianqi Chen
- [scikit-learn](https://scikit-learn.org/) — Pedregosa et al.
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — deedy5
