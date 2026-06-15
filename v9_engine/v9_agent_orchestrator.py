import sys
import json
import os
import subprocess
from datetime import datetime
from orchestrator import harvester_news_weather
from orchestrator import harvester_fatigue
from orchestrator import harvester_game_theory
from orchestrator import harvester_market_odds
from orchestrator import harvester_micro_referee

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from predict_v9 import get_base_match_info, generate_v8_predictions
from v9_shared import get_zh_name

# SYSTEM PROMPT FOR LLM REPORT GENERATION
AGENT_PROMPT = """
你是一个顶尖的华尔街量化对冲基金操盘手和体育风控专家。
你将获得由 V9 泊松数学引擎计算出的纯粹量化数据（基准面），以及 9 维雷达采集到的场外情报（伤停、天气、疲劳度、资金异动、主裁风格等）。

请遵循以下极度严格的金融风控原则撰写《V9 终极实盘决策内参》，并且**必须**完全采用以下五大版块的结构排版：

【结构强制规范】
**📋 一、全局作战总览 (Global Overview)**
- 在报告最顶部提供一个 Markdown 表格，总结当日所有比赛（字段：场次、置信度、推荐方向、仓位、核心逻辑一句话）。让基金经理能在 30 秒内看懂全局。

**🎯 二、单场独立研判 (Match Analysis)** （针对每一场比赛循环输出）
- 使用 `### ⚽ 场次：XXX` 作为比赛级标题。
- 下属版块必须使用 `####` 级别标题强化视觉层级：
  - **#### 🧮 核心数据雷达**：严格按照 Markdown 表格展示 7 项对位：胜平负概率、Elo 底蕴、阵容身价、预期进球(xG)、对阵控球率、前场压迫(PPDA)、高空争顶(Aerial)。不可遗漏任何一项。
  - **#### 🎲 泊松波胆矩阵**：使用独立的 Markdown 表格横向展示 Top 5 波胆及其概率。
  - **#### 📡 场外风控与博弈情报**：必须完整列出：伤停舆情、体能储备、裁判尺度风格、**赛程博弈(动机与默契球风险 Biscotto Risk)**。
  - **#### 💹 资金盘口与期望值**：结合理论让球盘、大小球盘与市场数据分析 Value Edge。
  - **#### 📌 华尔街级操盘指令**：
    - 指出是**逻辑共振**还是**逻辑背离**。
    - 给出“主要剧本 (Base Case)”和“反面风险论证 (Devil's Advocate)”。
    - 基于凯利公式给出具体投资仓位上限（0-5%）和最终方向。

**💰 三、全局资金风控 (Portfolio Bankroll Management)**
- 汇总单日总资金敞口（如总计 6%）。
- 强调资金的时间叠加风险与防熔断机制。

**🔍 四、模型自检与免责声明 (Model Transparency & Limitations)**
- 透明公开 WC 2022 回测表现。
- 列出当日预测可能存在的数据盲区。
"""

def run_orchestrator(date_str):
    print(f"🚀 [Orchestrator] Initiating V9.0 Pure Predictive Sequence for {date_str}...")
    
    matches = get_base_match_info(date_str)
    
    if not matches:
        print(f"No match predictions returned for {date_str}.")
        return

    mega_context = f"# V9.0 Agent Ultimate Context: {date_str}\n\n"
    mega_context += f"## Agent Core Directives:\n{AGENT_PROMPT}\n\n---\n"
    
    impact_dict = {}
    
    # Harvester Phase FIRST
    for t1_en, t2_en, t1_zh, t2_zh in matches:
        print(f"\n⚡ [V9 Phase 1] Harvesting real-time intel for: {t1_zh} vs {t2_zh}")
        match_key = f"{t1_zh} vs {t2_zh}"
        impact_dict[match_key] = {}
        
        # 1. Weather & News (Pass English names for DDG query)
        news = harvester_news_weather.run(t1_en, t2_en, city="Stadium")
        weather = news.get("weather", "Clear")
        
        # Enhanced Injury Check
        def check_injury(news_list):
            if not isinstance(news_list, list): return 0
            keywords = ["injury", "缺阵", "out", "doubtful", "sidelined", "acl", "knee", "hamstring", "ruled out", "受伤", "膝盖", "韧带", "报销", "伤退"]
            for item in news_list:
                text = str(item.get("title", "")) + str(item.get("body", ""))
                text = text.lower()
                for k in keywords:
                    if k in text:
                        return 2
            return 0
            
        inj_home = check_injury(news.get("home_news", []))
        inj_away = check_injury(news.get("away_news", []))
        impact_dict[match_key]["weather"] = weather
        impact_dict[match_key]["injuries_home"] = inj_home
        impact_dict[match_key]["injuries_away"] = inj_away
        
        # 2. Fatigue (Pass English names to match CSV)
        fatigue = harvester_fatigue.run(t1_en, t2_en, date_str)
        impact_dict[match_key]["fatigue_home"] = fatigue.get("home_rest_days", 7)
        impact_dict[match_key]["fatigue_away"] = fatigue.get("away_rest_days", 7)
        
        # 3. Game Theory
        tour_stage = "FIFA World Cup"
        game_theory = harvester_game_theory.run(t1_zh, t2_zh, tour_stage, date_str)
        impact_dict[match_key]["motivation_index"] = game_theory.get("motivation_index", "HIGH")
        impact_dict[match_key]["biscotto_risk"] = game_theory.get("biscotto_risk", "NONE")
        
        impact_dict[match_key]["_raw_news"] = news
        impact_dict[match_key]["_raw_fatigue"] = fatigue
        impact_dict[match_key]["_raw_gt"] = game_theory
    
    print("\n🧠 [V9 Phase 2] Running V9 Math Engine with Dynamic Penalties...")
    results_dict, _ = generate_v8_predictions(date_str, impact_dict)
    
    # Phase 3: Construct Context
    for match_title, match_data in results_dict.items():
        t1_en, t2_en = match_title.split(" vs ")
        t1_zh = get_zh_name(t1_en)
        t2_zh = get_zh_name(t2_en)
        match_key = f"{t1_zh} vs {t2_zh}"
        
        mega_context += f"## ⚔️ {t1_zh} vs {t2_zh}\n"
        mega_context += "### V9 Dynamic Quant Base Data (Post-Penalty)\n```json\n"
        mega_context += json.dumps(match_data, indent=2) + "\n```\n"
        
        impacts = impact_dict.get(match_key, {})
        mega_context += f"### 1. Tournament Motivation\n```json\n{json.dumps(impacts.get('_raw_gt',{}), indent=2)}\n```\n"
        mega_context += f"### 2. Squad Fatigue\n```json\n{json.dumps(impacts.get('_raw_fatigue',{}), indent=2)}\n```\n"
        mega_context += f"### 3. News & Weather\n```json\n{json.dumps(impacts.get('_raw_news',{}), indent=2)}\n```\n"
        
        odds = harvester_market_odds.run(t1_en, t2_en, match_data["p_home"], match_data["p_draw"], match_data["p_away"])
        referee = harvester_micro_referee.run(t1_en, t2_en, match_data["referee"], match_data["ppda_home"], match_data["ppda_away"], match_data["strictness"])

        mega_context += f"### 4. Market Odds & Money Flow\n```json\n{json.dumps(odds, indent=2)}\n```\n"
        mega_context += f"### 5. Referee Micro-Analysis\n```json\n{json.dumps(referee, indent=2)}\n```\n\n"
        mega_context += "---\n"

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    out_file = f"{tmp_dir}/{date_str}_Ultimate_Context.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(mega_context)
    
    print(f"✅ [Orchestrator] Complete! V9 Dynamic Context saved to {out_file}")

if __name__ == '__main__':
    target_date_str = sys.argv[1] if len(sys.argv) > 1 else '2026-06-15'
    run_orchestrator(target_date_str)
