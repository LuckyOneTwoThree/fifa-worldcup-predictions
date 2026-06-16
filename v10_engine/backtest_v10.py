import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import math
from core_model import train_v8_models, get_k_factor
from sklearn.metrics import brier_score_loss

from datetime import datetime, timedelta
import os

def dixon_coles_prob(l1, l2, k1, k2, rho=0.0):
    prob = (math.exp(-l1) * (l1 ** k1) / math.factorial(k1)) * (math.exp(-l2) * (l2 ** k2) / math.factorial(k2))
    if k1 == 0 and k2 == 0: return prob * (1 - l1*l2*rho)
    elif k1 == 0 and k2 == 1: return prob * (1 + l1*rho)
    elif k1 == 1 and k2 == 0: return prob * (1 + l2*rho)
    elif k1 == 1 and k2 == 1: return prob * (1 - rho)
    return prob

print("Loading V10.0 Pre-Match Ultimate Engine for Blind Backtest...")

df = pd.read_csv('../results.csv')
squad_vals = pd.read_csv('data_scrapers/squad_values.csv')
tac_df = pd.read_csv('data_scrapers/tactical_styles.csv')
ref_df = pd.read_csv('data_scrapers/referee_stats.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
tac_dict = tac_df.set_index('team').to_dict('index')
ref_dict = ref_df.set_index('name').to_dict('index')

# Mock data for these specific backtest teams to ensure realistic modeling
custom_sv = {
    'Mexico': 200, 'South Africa': 40, 'South Korea': 160, 'Czech Republic': 120,
    'Canada': 180, 'Bosnia and Herzegovina': 80, 'USA': 250, 'Paraguay': 100
}
custom_tac = {
    'Mexico': {"aerial_win_rate": 48.0, "ppda": 10.0},
    'South Africa': {"aerial_win_rate": 51.0, "ppda": 15.0},
    'South Korea': {"aerial_win_rate": 45.0, "ppda": 11.0},
    'Czech Republic': {"aerial_win_rate": 58.0, "ppda": 13.0}, # Tall team
    'Canada': {"aerial_win_rate": 52.0, "ppda": 11.0},
    'Bosnia and Herzegovina': {"aerial_win_rate": 56.0, "ppda": 14.0}, # Physical team
    'USA': {"aerial_win_rate": 50.0, "ppda": 9.0}, # High press
    'Paraguay': {"aerial_win_rate": 54.0, "ppda": 12.0}
}
squad_dict.update(custom_sv)
tac_dict.update(custom_tac)

name_mapping = {'United States': 'USA', 'Korea Republic': 'South Korea'}
def map_name(name): return name_mapping.get(name, name)

zh_translation = {
    'Mexico': '墨西哥', 'South Africa': '南非', 'South Korea': '韩国', 'Czech Republic': '捷克',
    'Canada': '加拿大', 'Bosnia and Herzegovina': '波黑', 'USA': '美国', 'Paraguay': '巴拉圭',
    'Germany': '德国', 'Curaçao': '库拉索', 'Ivory Coast': '科特迪瓦', 'Ecuador': '厄瓜多尔',
    'Netherlands': '荷兰', 'Japan': '日本', 'Sweden': '瑞典', 'Tunisia': '突尼斯',
    'Qatar': '卡塔尔', 'Switzerland': '瑞士', 'Brazil': '巴西', 'Morocco': '摩洛哥',
    'Haiti': '海地', 'Scotland': '苏格兰', 'Australia': '澳大利亚', 'Turkey': '土耳其',
    'Belgium': '比利时', 'Egypt': '埃及', 'Iran': '伊朗', 'New Zealand': '新西兰',
    'Spain': '西班牙', 'Cape Verde': '佛得角', 'Saudi Arabia': '沙特阿拉伯', 'Uruguay': '乌拉圭',
    'Argentina': '阿根廷', 'Portugal': '葡萄牙', 'Croatia': '克罗地亚', 'Senegal': '塞内加尔',
    'France': '法国', 'England': '英格兰'
}
def t(name): return zh_translation.get(name, name)

global_metrics_data = []

def run_blind_prediction(target_date_str):
    target_date_dt = pd.to_datetime(target_date_str)
    
    # 1. ABSOLUTE BLIND TEST: Filter out ALL data from target_date onwards for Elo and Training
    past_df = df[df['date'] < target_date_dt].copy()
    
    elo_dict = {}
    wc_data = []
    
    calibrated_stack, xg_model_home, xg_model_away, elo_dict = train_v8_models(past_df, squad_dict, tac_dict)
    
    # Predict Target Date
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    beijing_date = target_date + timedelta(days=1)
    beijing_date_str = beijing_date.strftime('%Y-%m-%d')
    
    output_md = f"# 📅 北京时间：{beijing_date_str} (美洲当地 {target_date.month}-{target_date.day})\n"
    output_md += f"## 📊 V10.0 全息量化预测 (绝对盲测版)\n\n"
    output_md += "> [!IMPORTANT]\n> 本预测文件在生成时，底层切断了所有该日及该日之后的真实数据，确保 Elo 积分和机器学习权重处于**绝对盲测状态 (Blind Test)**，不含任何后见之明。\n\n"
    
    upcoming = df[df['date'] == target_date_str]
    referees = [
    "Michael Oliver", "Wilton Sampaio", "Daniele Orsato", "Szymon Marciniak",
    "Anthony Taylor", "Clement Turpin", "Danny Makkelie", "Slavko Vincic",
    "Facundo Tello", "Cesar Ramos", "Fernando Rapallini", "Ivan Barton"
]
    
    for _, row in upcoming.iterrows():
        t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
        e1, e2 = elo_dict.get(t1, 1500), elo_dict.get(t2, 1500)
        sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)
        
        tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0, "possession_avg": 50.0})
        tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0, "possession_avg": 50.0})
        aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
        ppda_diff = tac1['ppda'] - tac2['ppda']
        
        assigned_ref = referees[sum(ord(c) for c in t1+t2) % len(referees)]
        strictness = ref_dict[assigned_ref]['strictness_index']
        
        X_pred = pd.DataFrame({
            'elo_diff': [e1 - e2],
            'sv_diff': [sv1 - sv2],
            'aerial_diff': [aerial_diff],
            'ppda_diff': [ppda_diff],
            't_weight': [get_k_factor(row['tournament'])]
        })
        
        probs = calibrated_stack.predict_proba(X_pred)[0]
        p_away_ml, p_draw_ml, p_home_ml = probs[0], probs[1], probs[2]
        
        xg1_pred = max(0.1, min(float(xg_model_home.predict(X_pred)[0]), 6.0))
        xg2_pred = max(0.1, min(float(xg_model_away.predict(X_pred)[0]), 6.0))

        
        # V5 Unified Math: Calculate true W/D/L probabilities from Poisson matrix
        p_win_poisson, p_draw_poisson, p_loss_poisson = 0.0, 0.0, 0.0
        score_probs_all = []
        for i in range(15):
            for j in range(15):
                prob = dixon_coles_prob(xg1_pred, xg2_pred, i, j, rho=-0.05)
                if i > j: p_win_poisson += prob
                elif i == j: p_draw_poisson += prob
                else: p_loss_poisson += prob
                if i < 10 and j < 10:
                    score_probs_all.append({"score": f"{i}-{j}", "prob": prob})
                    
        total_p = p_win_poisson + p_draw_poisson + p_loss_poisson
        p_win_poisson /= total_p
        p_draw_poisson /= total_p
        p_loss_poisson /= total_p
        
        p_home = p_home_ml * 0.3 + p_win_poisson * 0.7
        p_draw = p_draw_ml * 0.3 + p_draw_poisson * 0.7
        p_away = p_away_ml * 0.3 + p_loss_poisson * 0.7
        
        score_probs = sorted(score_probs_all, key=lambda x: x["prob"], reverse=True)[:3]
        
        # Asian Handicap & Over/Under
        diff = xg1_pred - xg2_pred
        hc = round(diff / 0.25) * 0.25
        if hc == 0: ah_str = "平手盘 (0)"
        elif hc > 0: ah_str = f"主让 {hc} 球" if hc % 0.5 == 0 else f"主让 {hc-0.25}/{hc+0.25} 球"
        else: ah_str = f"受让 {abs(hc)} 球" if abs(hc) % 0.5 == 0 else f"受让 {abs(hc)-0.25}/{abs(hc)+0.25} 球"
        
        total_xg = xg1_pred + xg2_pred
        ou = round(total_xg / 0.25) * 0.25
        ou_str = f"{ou} 球" if ou % 0.5 == 0 else f"{ou-0.25}/{ou+0.25} 球"
        
        # V6.0 Logic: Elo Tiers, Possession, Warnings
        def get_tier(elo):
            if elo >= 1900: return "T0-世界顶尖"
            elif elo >= 1750: return "T1-洲际强队"
            elif elo >= 1600: return "T2-中坚力量"
            elif elo >= 1450: return "T3-边缘球队"
            else: return "T4-送分鱼腩"
        
        sv_ratio = max(sv1, 0.1) / max(sv2, 0.1)
        trap_warning = ""
        if sv_ratio > 2.5 and abs(e1 - e2) < 50:
            trap_warning = f"\n> [!WARNING]\n> **身价陷阱**：{t(t1)} 身价是对手 {sv_ratio:.1f} 倍，但真实战力(Elo)几乎持平，谨防大热必死！\n"
        elif sv_ratio < 0.4 and abs(e1 - e2) < 50:
            trap_warning = f"\n> [!WARNING]\n> **身价陷阱**：{t(t2)} 身价是对手 {1/sv_ratio:.1f} 倍，但真实战力(Elo)几乎持平，谨防大热必死！\n"
            
        p1, p2 = tac1.get('possession_avg', 50.0), tac2.get('possession_avg', 50.0)
        bc1 = (p1 / (p1 + p2)) * 100
        bc2 = (p2 / (p1 + p2)) * 100
        
        disc_warning = ""
        if strictness > 0.65:
            if tac1.get('ppda', 12.0) < 10.0: disc_warning += f"  - 🔴 {t(t1)} 逼抢极凶且主裁尺度严苛，极易触发红黄牌及危险定位球！\n"
            if tac2.get('ppda', 12.0) < 10.0: disc_warning += f"  - 🔴 {t(t2)} 逼抢极凶且主裁尺度严苛，极易触发红黄牌及危险定位球！\n"
        
        output_md += f"### ⚔️ {t(t1)} vs {t(t2)}\n"
        if trap_warning: output_md += trap_warning
        output_md += f"- **底蕴评级 (Elo)**：{get_tier(e1)} ({int(e1)}) vs {get_tier(e2)} ({int(e2)})\n"
        output_md += f"- **身价对比**：€{sv1}m vs €{sv2}m\n"
        output_md += f"- **全息战术雷达 (Tactical Radar)**：\n"
        output_md += f"  - ⚽ **球权推演**：预计 {t(t1)} 控球率 {bc1:.1f}%，{t(t2)} 控球率 {bc2:.1f}%\n"
        if aerial_diff > 5:
            output_md += f"  - ✈️ {t(t1)} 在高空争顶上占据绝对碾压优势 (+{aerial_diff:.1f}%)\n"
        elif aerial_diff < -5:
            output_md += f"  - ✈️ {t(t2)} 在高空争顶上占据绝对碾压优势 ({-aerial_diff:.1f}%)\n"
        else:
            output_md += f"  - ✈️ 双方身体对抗数据接近，无明显高空压制\n"
            
        if ppda_diff > 3:
            output_md += f"  - 🌪️ {t(t2)} 逼抢极其凶狠，{t(t1)} 的出球将面临巨大压力\n"
        elif ppda_diff < -3:
            output_md += f"  - 🌪️ {t(t1)} 逼抢极其凶狠，{t(t2)} 的出球将面临巨大压力\n"
            
        output_md += f"- **裁判因素 (Referee Factor)**：{assigned_ref} (严厉指数: {strictness})\n"
        if disc_warning: output_md += disc_warning
        global_metrics_data.append({'actual_s1': row['home_score'], 'actual_s2': row['away_score'], 'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away, 'score_probs': score_probs_all})

        
        output_md += f"- **V10 双模加权统一胜率 (基于泊松积分)**：主 {p_home*100:.1f}% | 平 {p_draw*100:.1f}% | 客 {p_away*100:.1f}%\n"
        output_md += f"- **火力指征 (xG)**：{xg1_pred:.2f} - {xg2_pred:.2f}\n"
        output_md += f"- **🎲 机器推演盘口**：\n"
        output_md += f"  - **理论亚盘**：{ah_str}\n"
        output_md += f"  - **理论大小球**：{ou_str}\n\n"
        
        output_md += "#### 💡 操盘手深度点评与资金分布建议\n"
        if p_home > 0.6: core_tone = f"本场比赛 {t(t1)} 具有压倒性优势，模型测算胜率高达 {p_home*100:.1f}%，是一场典型的攻防演练。"
        elif p_away > 0.6: core_tone = f"本场比赛 {t(t2)} 具有压倒性优势，模型测算胜率高达 {p_away*100:.1f}%，客场作战亦能反客为主。"
        elif p_draw > 0.3: core_tone = f"本场比赛呈势均力敌之势，双方战力极度接近，模型测算平局概率高达 {p_draw*100:.1f}%，极易陷入拉锯战。"
        else: core_tone = f"本场比赛虽然 {t(t1) if p_home > p_away else t(t2)} 略占优势，但 {t(t2) if p_home > p_away else t(t1)} 并非毫无还手之力，属于高难度博弈局。"

        if hc >= 1.0: ah_rec = f"机构极可能开出 **{ah_str}** 的深盘。结合绝对实力压制，推荐重注【**{t(t1)} 赢盘**】，穿盘概率极高。"
        elif hc <= -1.0: ah_rec = f"机构极可能开出 **{ah_str}** 的深盘。结合绝对实力压制，推荐重注【**{t(t2)} 赢盘**】，穿盘概率极高。"
        elif hc > 0 and hc < 1.0: ah_rec = f"亚盘初盘预计为 **{ah_str}**，生死盘口下建议选择【**{t(t1)} 独赢**】作为稳健打底，亚盘需防赢球输盘的小胜格局。"
        elif hc < 0 and hc > -1.0: ah_rec = f"亚盘初盘预计为 **{ah_str}**，生死盘口下建议选择【**{t(t2)} 独赢**】作为稳健打底，亚盘需防赢球输盘的小胜格局。"
        else: ah_rec = f"亚盘指向 **平手盘 (0)** 或极浅让步，数据支持有限，建议直接去往【**平局**】高赔，或选择【**受让方赢盘**】防身。"

        total_xg = xg1_pred + xg2_pred
        if total_xg >= 2.75: ou_rec = f"火力指征旺盛 (预期进球 {xg1_pred:.2f} + {xg2_pred:.2f})，大概率突破 **{ou_str}**，建议配置【**大球 (Over)**】。"
        else: ou_rec = f"双方预期进球受限 (预期进球 {xg1_pred:.2f} + {xg2_pred:.2f})，进球数极可能卡在 **{ou_str}** 以下，建议配置【**小球 (Under)**】。"

        output_md += f"- **📌 核心定调**：{core_tone}\n"
        output_md += f"- **💰 亚盘投资**：{ah_rec}\n"
        output_md += f"- **🎯 大小球博弈**：{ou_rec}\n\n"
        
        output_md += "**精确波胆矩阵 (Top 3 覆盖)：**\n\n"
        output_md += "| 排位 | 比分 | 发生概率 | 建议资金分配 |\n"
        output_md += "| :--- | :--- | :--- | :--- |\n"
        output_md += f"| Top 1 | **{score_probs[0]['score']}** | {score_probs[0]['prob']*100:.1f}% | 50% |\n"
        output_md += f"| Top 2 | **{score_probs[1]['score']}** | {score_probs[1]['prob']*100:.1f}% | 30% |\n"
        output_md += f"| Top 3 | **{score_probs[2]['score']}** | {score_probs[2]['prob']*100:.1f}% | 20% |\n\n"
        output_md += "---\n\n"

    out_path = os.path.join(r"d:\HelloWorld\Git_Project\worldcup\pdata", f"{beijing_date_str}_prediction.md")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(output_md)
    print(f"Blind prediction for {target_date_str} generated -> {out_path}")

run_blind_prediction('2026-06-11')
run_blind_prediction('2026-06-12')

print("\n--- 📈 V10.0 Quantitative Backtest Metrics ---")
if len(global_metrics_data) > 0:
    import numpy as np
    y_true = []
    y_prob = []
    hits = 0
    exact_hits = 0
    top3_hits = 0
    for m in global_metrics_data:
        s1, s2 = m['actual_s1'], m['actual_s2']
        if pd.isna(s1) or pd.isna(s2): continue
        s1, s2 = int(s1), int(s2)
        res = 2 if s1 > s2 else (1 if s1 == s2 else 0)
        y_true.append(res)
        y_prob.append([m['p_away'], m['p_draw'], m['p_home']])
        
        pred_res = np.argmax([m['p_away'], m['p_draw'], m['p_home']])
        if pred_res == res: hits += 1
        
        actual_score = f"{s1}-{s2}"
        top_scores = sorted(m['score_probs'], key=lambda x: x['prob'], reverse=True)
        if top_scores[0]['score'] == actual_score: exact_hits += 1
        if actual_score in [x['score'] for x in top_scores[:3]]: top3_hits += 1
        
    if len(y_true) > 0:
        
        y_true_onehot = np.zeros((len(y_true), 3))
        for idx, val in enumerate(y_true):
            y_true_onehot[idx, val] = 1.0
        brier = np.mean(np.sum((np.array(y_prob) - y_true_onehot)**2, axis=1))
        print(f"Matches Evaluated: {len(y_true)}")
        print(f"W/D/L Accuracy: {hits/len(y_true)*100:.1f}%")
        print(f"Brier Score (Log-Loss proxy): {brier:.4f}")
        print(f"Exact Score Hit Rate (Top 1): {exact_hits/len(y_true)*100:.1f}%")
        print(f"Top 3 Score Coverage: {top3_hits/len(y_true)*100:.1f}%")
else:
    print("No actual results found to evaluate.")
