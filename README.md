# 🏆 FIFA World Cup 2026 Predictions — V8.0 Dynamic Engine

> **量化足球预测系统**：双引擎泊松积分 + 九维情报雷达 + LLM 智能研判

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/Engine-V8.0-orange.svg)](v8_engine_whitepaper.md)

---

> **⚠️ 免责声明**
>
> 本项目**纯粹出于个人兴趣**而编写，目的是研究量化算法在体育赛事预测中的应用能力。
> 作者将足球比赛视为一个有趣的**数据建模与算法验证问题**，而非投注工具。
>
> - 🚫 **本项目不构成任何投注建议**，也不鼓励任何形式的赌博行为
> - 🚫 **不建议任何人将本系统用于实际购买或投注决策**
> - ✅ 如果你对算法本身感兴趣，欢迎研究、讨论和改进
> - ✅ 后续会持续进行**预测结果 vs 实际赛果的数据比对**，验证算法是否真正具备预测能力
>
> 足球比赛充满不确定性，任何模型都无法保证结果。请理性对待。

---

## 📖 项目简介

这是一个**出于兴趣而构建的足球赛事量化预测系统**，专为 2026 FIFA 世界杯设计。作者希望通过**编程 + 数学建模 + 数据分析**的方式，探索算法能否对比赛结果做出有意义的预测。

系统采用"**量化底座 + 情报雷达 + LLM 融合研判**"的三层架构：

1. **V8 泊松数学引擎**：基于 49,000+ 场历史比赛训练的双引擎机器学习模型
2. **五维情报采集器**：实时爬取伤停、体能、天气、盘口、裁判等场外情报
3. **LLM 智能研判层**：将量化数据与定性情报融合，生成结构化的分析报告

> 📊 后续会持续跟踪预测准确率，用数据说话——算法到底靠不靠谱，让回测结果来回答。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    v8_agent_orchestrator.py                  │
│                      (编排中枢 / 调度器)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: 情报采集                Phase 2: 数学引擎          │
│  ┌─────────────────┐             ┌─────────────────┐        │
│  │ 🔍 News Harvester│             │  predict_v8.py  │        │
│  │ 🏃 Fatigue       │──impact──→  │  ┌───────────┐  │        │
│  │ 🎯 Game Theory   │   dict      │  │ Stacking   │  │        │
│  │ 💹 Market Odds   │             │  │ Classifier │  │        │
│  │ ⚖️ Micro Referee │             │  │ (30% 权重)  │  │        │
│  └─────────────────┘             │  ├───────────┤  │        │
│                                   │  │ Poisson   │  │        │
│  Phase 3: 上下文构建              │  │ Regressor │  │        │
│  ┌─────────────────┐             │  │ (70% 权重)  │  │        │
│  │ mega_context.md │◀────────────│  └───────────┘  │        │
│  │ (给 LLM 的输入)  │  results    └─────────────────┘        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  LLM 综合研判    │ → 最终决策报告 (pdata/)                 │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
worldcup/
├── results.csv                        # 49,000+ 场历史比赛数据 (1872-2026)
├── requirements.txt                   # Python 依赖
├── v8_engine_whitepaper.md            # 算法白皮书
├── .gitignore
│
├── v8_engine/                         # 🔧 核心引擎
│   ├── v8_shared.py                   # 共享模块 (缓存、翻译、模型加载)
│   ├── core_model.py                  # 模型训练 (Elo + Stacking + Poisson)
│   ├── predict_v8.py                  # 预测引擎主逻辑
│   ├── v8_agent_orchestrator.py       # LLM 编排器 (三阶段流水线)
│   ├── backtest_v8.py                 # 历史盲测
│   ├── evaluate_v8.py                 # WC 2022 全量评估
│   ├── robustness_test.py             # 参数网格搜索 (rho / 融合权重)
│   ├── train_model_v8.py              # 独立训练脚本
│   │
│   ├── orchestrator/                  # 📡 五维情报采集器
│   │   ├── harvester_news_weather.py  # 新伤停/天气 (DuckDuckGo)
│   │   ├── harvester_fatigue.py       # 体能/休息天数
│   │   ├── harvester_game_theory.py   # 战意/默契球风险
│   │   ├── harvester_market_odds.py   # 盘口/资金流向 (The-Odds-API)
│   │   └── harvester_micro_referee.py # 裁判-球队交互分析
│   │
│   └── data_scrapers/                 # 📊 静态数据
│       ├── squad_values.csv           # 球队身价 (31 队精确 + 200+ 队默认)
│       ├── tactical_styles.csv        # 战术数据 (控球率/PPDA/高空争顶)
│       ├── referee_stats.csv          # 裁判严厉指数 (12 人)
│       ├── xg_history.csv             # 历史 xG 数据 (49K 场)
│       ├── tactical_data_builder.py   # 战术数据生成器
│       ├── referee_data_builder.py    # 裁判数据生成器
│       ├── fbref_scraper.py           # FBRef 数据爬虫
│       ├── transfermarkt_scraper.py   # Transfermarkt 身价爬虫
│       └── update_pipeline.bat        # 数据更新流水线
│
├── pdata/                             # 📈 预测报告归档
│   ├── 2026_06_15_V8_Ultimate_Report.md
│   ├── 2026-06-16_V8_Hedge_Fund_Report.md
│   └── ...
│
└── Visualization/                     # 🎨 可视化前端
    ├── backend/main.py                # FastAPI 后端
    ├── frontend/                      # React + Vite 前端
    └── start_server.bat               # 一键启动脚本
```

---

## 🧮 核心算法

### 双引擎融合架构

| 引擎 | 模型 | 输出 | 权重 |
|------|------|------|------|
| **引擎 A (分类器)** | XGBoost + RandomForest + MLP → LogisticRegression → CalibratedClassifierCV | 胜/平/负概率 | 30% |
| **引擎 B (泊松流)** | XGBoost Poisson Regressor → Dixon-Coles 修正 (ρ=-0.05) → 15×15 比分矩阵 | xG → 比分概率 | 70% |

最终概率 = `P_engine_A × 0.3 + P_engine_B × 0.7`

### 五维输入特征

| 特征 | 说明 | 来源 |
|------|------|------|
| `elo_diff` | 双方 Elo 积分差 | 动态计算 (K=60/40/30/20) |
| `sv_diff` | 阵容身价差 (百万欧元) | Transfermarkt |
| `aerial_diff` | 高空争顶胜率差 | FBRef |
| `ppda_diff` | 前场逼抢强度差 | FBRef |
| `t_weight` | 赛事权重 (WC=60) | 赛程 |

### 动态 xG 修正 (V8 新增)

赛外情报通过乘数链直接修正 xG：

```
xG_final = xG_base × mult_home × mult_away

乘数因子：
├── 疲劳 (<5天休息 + 高逼抢): ×0.90
├── 雨天 + 传控型球队:        ×0.85
├── 低战意 / 默契球风险:      ×0.60
└── 伤停 (每单位):            ×(1 - 0.10)

硬限制：总乘数不低于 0.60 (最多降 40%)
```

### Dixon-Coles 低比分修正

标准泊松分布在 0-0、1-0、0-1、1-1 四个低比分上存在系统性偏差。Dixon-Coles 修正系数 ρ=-0.05 专门校正这一偏差：

```
τ(0,0) = 1 - λ₁λ₂ρ     # 修正 0-0
τ(0,1) = 1 + λ₁ρ        # 修正 0-1
τ(1,0) = 1 + λ₂ρ        # 修正 1-0
τ(1,1) = 1 - ρ           # 修正 1-1
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- pip

### 安装

```bash
git clone https://github.com/LuckyOneTwoThree/fifa-worldcup-predictions.git
cd fifa-worldcup-predictions
pip install -r requirements.txt
```

### 依赖

```
duckduckgo_search    # 新闻采集
xgboost              # 梯度提升模型
scikit-learn         # 机器学习管线
pandas               # 数据处理
numpy                # 数值计算
requests             # HTTP 请求 (盘口 API)
```

### 运行预测

```bash
# 方式 1：仅运行数学引擎 (无需 LLM)
cd v8_engine
python predict_v8.py 2026-06-15

# 方式 2：运行完整流水线 (数学引擎 + 情报采集 + LLM 研判)
python v8_agent_orchestrator.py 2026-06-15
```

### 运行回测

```bash
cd v8_engine

# WC 2022 全量盲测 (含 Brier Score、校准度检查)
python evaluate_v8.py

# 参数网格搜索 (rho 值 / 融合权重)
python robustness_test.py

# 自定义日期盲测
python backtest_v8.py
```

---

## 📡 五维情报采集器

| 采集器 | 数据源 | 采集内容 | 对 xG 的影响 |
|--------|--------|----------|-------------|
| **News Harvester** | DuckDuckGo News | 伤停、士气、突发新闻 | 伤停 ×0.90 per unit |
| **Fatigue Harvester** | results.csv 历史 | 上场比赛间隔天数 | <5天 ×0.90 (高逼抢队) |
| **Game Theory Harvester** | 规则引擎 | 赛事阶段、默契球风险 | LOW ×0.60 |
| **Market Odds Harvester** | The-Odds-API | 真实盘口 / 理论盘口 | 供 LLM 参考 |
| **Micro Referee Harvester** | 裁判数据 | 裁判-球队风格交互 | 供 LLM 参考 |

### 伤停检测关键词 (中英文)

```
injury, out, doubtful, sidelined, acl, knee, hamstring, ruled out,
缺阵, 受伤, 膝盖, 韧带, 报销, 伤退
```

---

## 📊 数据说明

### results.csv (核心数据)

- **来源**：国际足球比赛历史数据
- **时间跨度**：1872 年 - 2026 年
- **记录数**：~49,400 场
- **字段**：date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

### 静态特征数据

| 文件 | 条目数 | 说明 |
|------|--------|------|
| `squad_values.csv` | 244 队 | 球队身价 (Top 31 精确，其余默认 50M) |
| `tactical_styles.csv` | 243 队 | 控球率 / PPDA / 高空争顶胜率 |
| `referee_stats.csv` | 12 人 | 出牌数 / 点球数 / 严厉指数 |
| `xg_history.csv` | 49,400 场 | 历史预期进球数据 |

---

## 🧪 回测验证

### WC 2022 盲测方法

```python
# 对 WC 2022 每个比赛日：
# 1. 用该日之前的所有数据训练模型 (绝对盲测)
# 2. 预测该日所有比赛
# 3. 对比实际结果
```

### 评估指标

| 指标 | 说明 |
|------|------|
| **W/D/L Accuracy** | 胜平负预测准确率 |
| **Brier Score** | 概率校准度 (越低越好) |
| **Exact Score Hit** | 精确波胆命中率 |
| **Top 3 Coverage** | Top 3 波胆覆盖率 |
| **Calibration Check** | 高置信度场次的实际命中率 |

---

## 🤖 LLM 报告结构

编排器生成的 `mega_context.md` 包含完整的量化数据和场外情报，供 LLM 按以下结构生成报告：

### 五大版块

1. **📋 全局作战总览** — 一页速览所有比赛 (30 秒决策)
2. **🎯 单场独立研判**
   - 🧮 核心数据雷达 (7 项对位表格)
   - 🎲 泊松波胆矩阵 (Top 5)
   - 📡 场外风控情报 (伤停/体能/裁判/战意)
   - 💹 资金盘口与期望值
   - 📌 华尔街级操盘指令 (共振/背离 + Devil's Advocate)
3. **💰 全局资金风控** — 总敞口 + 防熔断机制
4. **🔍 模型自检与免责** — 回测表现 + 数据盲区

### 风控原则

- 所有仓位基于凯利公式，单场最高 5%
- 禁止"满仓"、"梭哈"等情绪化表述
- 量化与战术背离时强制推荐 No Bet
- 前序亏损时后续仓位自动减半

---

## 🔧 配置与扩展

### 添加新球队数据

1. 在 `v8_engine/data_scrapers/squad_values.csv` 添加身价
2. 在 `v8_engine/data_scrapers/tactical_styles.csv` 添加战术数据
3. 在 `v8_engine/v8_shared.py` 的 `zh_translation` 添加中文名

### 接入真实盘口 API

设置环境变量 `ODDS_API_KEY`（从 [The-Odds-API](https://the-odds-api.com) 获取）：

```bash
export ODDS_API_KEY=your_api_key_here
```

### 自定义 AGENT_PROMPT

编辑 `v8_engine/v8_agent_orchestrator.py` 中的 `AGENT_PROMPT` 变量，修改 LLM 的报告生成指令。

---

## ⚠️ 免责声明

- 本项目**纯粹出于个人兴趣**，研究量化算法在体育赛事预测中的应用
- **不构成任何投注建议**，不鼓励任何形式的赌博行为
- 后续将持续进行**预测 vs 实际赛果的数据比对**，验证算法有效性
- 足球比赛充满不确定性，任何模型都无法保证结果
- 如果你对算法感兴趣，欢迎研究、讨论和改进

---

## 📄 许可证

MIT License

---

## 👤 作者

**LuckyOneTwoThree** — [GitHub](https://github.com/LuckyOneTwoThree)

出于对量化分析和足球数据的兴趣，构建了这套预测系统。欢迎交流算法思路。

---

## 🙏 致谢

- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system) — Arpad Elo
- [Dixon-Coles Model](https://doi.org/10.1111/1467-9574.00009) — Dixon & Coles (1997)
- [XGBoost](https://xgboost.readthedocs.io/) — Tianqi Chen
- [scikit-learn](https://scikit-learn.org/) — Pedregosa et al.
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — deedy5
