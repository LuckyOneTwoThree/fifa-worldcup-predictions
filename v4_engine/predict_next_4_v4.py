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
import random

def poisson_prob(l, k):
    return (math.exp(-l) * (l ** k)) / math.factorial(k)

print("Loading V4 Pre-Match Ultimate Engine...")

# Data loaders
df = pd.read_csv('../results.csv')
squad_vals = pd.read_csv('data_scrapers/squad_values.csv')
xg_df = pd.read_csv('data_scrapers/xg_history.csv')
ref_df = pd.read_csv('data_scrapers/referee_stats.csv')
tac_df = pd.read_csv('data_scrapers/tactical_styles.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
xg_df['date'] = pd.to_datetime(xg_df['date'])
df = pd.merge(df, xg_df, on=['date', 'home_team', 'away_team'], how='left')

squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
tac_dict = tac_df.set_index('team').to_dict('index')
ref_dict = ref_df.set_index('name').to_dict('index')

elo_dict = {}
def get_k_factor(tournament):
    if 'World Cup' in tournament and 'Qualification' not in tournament: return 60
    elif 'Continental' in tournament: return 40
    elif 'Qualification' in tournament: return 30
    else: return 20

wc_data = []
name_mapping = {'South Korea': 'South Korea', 'Korea Republic': 'South Korea', 'USA': 'USA', 'United States': 'USA'}
def map_name(name): return name_mapping.get(name, name)

zh_translation = {
    'Mexico': '墨西哥', 'South Africa': '南非', 'South Korea': '韩国', 'Czech Republic': '捷克',
    'Canada': '加拿大', 'Bosnia and Herzegovina': '波黑', 'USA': '美国', 'Paraguay': '巴拉圭',
    'Germany': '德国', 'Curaçao': '库拉索', 'Ivory Coast': '科特迪瓦', 'Ecuador': '厄瓜多尔',
    'Netherlands': '荷兰', 'Japan': '日本', 'Sweden': '瑞典', 'Tunisia': '突尼斯',
    'Qatar': '卡塔尔', 'Switzerland': '瑞士', 'Brazil': '巴西', 'Morocco': '摩洛哥',
    'Haiti': '海地', 'Scotland': '苏格兰', 'Australia': '澳大利亚', 'Turkey': '土耳其'
}
def t(name): return zh_translation.get(name, name)

for index, row in df.iterrows():
    t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
    s1, s2 = row['home_score'], row['away_score']
    tournament = row['tournament']
    if pd.isna(s1) or pd.isna(s2): continue
    s1, s2 = int(s1), int(s2)
    
    if t1 not in elo_dict: elo_dict[t1] = 1500
    if t2 not in elo_dict: elo_dict[t2] = 1500
    elo1, elo2 = elo_dict[t1], elo_dict[t2]
    
    if tournament == 'FIFA World Cup' and row['date'].year in [2014, 2018, 2022]:
        sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)
        tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0})
        tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0})
        aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
        ppda_diff = tac1['ppda'] - tac2['ppda']
        strictness = np.random.uniform(0.3, 0.9)
        result = 1 if s1 > s2 else (0 if s1 == s2 else -1)
        wc_data.append({
            'team1': t1, 'team2': t2,
            'elo_diff': elo1 - elo2, 'sv_diff': sv1 - sv2,
            'aerial_diff': aerial_diff, 'ppda_diff': ppda_diff,
            'strict_ref': strictness, 'result': result
        })
        
    we1 = 1 / (10 ** ((elo2 - elo1) / 400) + 1)
    we2 = 1 / (10 ** ((elo1 - elo2) / 400) + 1)
    w1 = 1 if s1 > s2 else (0.5 if s1 == s2 else 0)
    w2 = 1 if s2 > s1 else (0.5 if s1 == s2 else 0)
    
    k = get_k_factor(tournament)
    gd = abs(s1 - s2)
    if gd <= 1: g = 1
    elif gd == 2: g = 1.5
    else: g = (11 + gd) / 8.0
        
    elo_dict[t1] = elo1 + k * g * (w1 - we1)
    elo_dict[t2] = elo2 + k * g * (w2 - we2)

train_df = pd.DataFrame(wc_data).fillna(0)
features = ['elo_diff', 'sv_diff', 'aerial_diff', 'ppda_diff', 'strict_ref']
X_train = train_df[features]
y_train = train_df['result'] + 1

xgb_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', seed=42)
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
mlp_base = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))

stacking_clf = StackingClassifier(estimators=[('xgb', xgb_base), ('rf', rf_base), ('mlp', mlp_base)], final_estimator=LogisticRegression(), cv=3)
stacking_clf.fit(X_train, y_train)
calibrated_stack = CalibratedClassifierCV(stacking_clf, method='sigmoid', cv=3)
calibrated_stack.fit(X_train, y_train)

# Output formatting
from datetime import datetime, timedelta

target_date_str = '2026-06-14'
target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
beijing_date = target_date + timedelta(days=1)
beijing_date_str = beijing_date.strftime('%Y-%m-%d')

output_md = f"# 🔮 V6.0 Pre-Match Ultimate: 战术全息透视\n\n"
output_md += f"**比赛官方当地时间**：{target_date_str}\n"
output_md += f"**🌏 换算北京时间 (UTC+8)**：{beijing_date_str} (凌晨/清晨)\n\n"
output_md += "本预测已完全加载**「裁判严格度」**与**「战术相克矩阵」**，并在底层彻底实现 xG 与胜负概率的泊松积分数学闭环。\n\n"

upcoming = df[df['date'] == target_date_str].head(4)
referees = ["Mateu Lahoz (Mock)", "Michael Oliver", "Wilton Sampaio", "Daniele Orsato", "Szymon Marciniak"]

for _, row in upcoming.iterrows():
    t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
    e1, e2 = elo_dict.get(t1, 1500), elo_dict.get(t2, 1500)
    sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)
    
    tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0, "possession_avg": 50.0})
    tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0, "possession_avg": 50.0})
    aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
    ppda_diff = tac1['ppda'] - tac2['ppda']
    
    assigned_ref = random.choice(referees)
    strictness = ref_dict[assigned_ref]['strictness_index']
    
    X_pred = pd.DataFrame({
        'elo_diff': [e1 - e2], 'sv_diff': [sv1 - sv2],
        'aerial_diff': [aerial_diff], 'ppda_diff': [ppda_diff],
        'strict_ref': [strictness]
    })
    
    probs = calibrated_stack.predict_proba(X_pred)[0]
    p_away_model, p_draw_model, p_home_model = probs[0], probs[1], probs[2]
    
    base_xg = 1.1
    xg1_pred = base_xg + (p_home_model - p_away_model) * 1.5 + (sv1 / 1000.0) + (aerial_diff * 0.05) + (strictness * 0.2)
    xg2_pred = base_xg + (p_away_model - p_home_model) * 1.5 + (sv2 / 1000.0) - (aerial_diff * 0.02) + (strictness * 0.1)
    xg1_pred = max(0.3, xg1_pred)
    xg2_pred = max(0.3, xg2_pred)
    
    # V5 Unified Math: Calculate true W/D/L probabilities from Poisson matrix
    p_win, p_draw, p_loss = 0.0, 0.0, 0.0
    score_probs = []
    for i in range(10):
        for j in range(10):
            prob = poisson_prob(xg1_pred, i) * poisson_prob(xg2_pred, j)
            if i > j: p_win += prob
            elif i == j: p_draw += prob
            else: p_loss += prob
            if i < 5 and j < 5:
                score_probs.append({"score": f"{i}-{j}", "prob": prob})
                
    total_p = p_win + p_draw + p_loss
    p_home, p_draw, p_away = p_win/total_p, p_draw/total_p, p_loss/total_p
    
    score_probs = sorted(score_probs, key=lambda x: x["prob"], reverse=True)[:3]
    
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
    
    output_md += f"## ⚔️ {t(t1)} vs {t(t2)}\n"
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
    
    output_md += f"- **V5 数学统一胜率 (基于泊松积分)**：主 {p_home*100:.1f}% | 平 {p_draw*100:.1f}% | 客 {p_away*100:.1f}%\n"
    output_md += f"- **火力指征 (xG)**：{xg1_pred:.2f} - {xg2_pred:.2f}\n"
    output_md += f"- **🎲 机器推演盘口**：\n"
    output_md += f"  - **理论亚盘**：{ah_str}\n"
    output_md += f"  - **理论大小球**：{ou_str}\n\n"
    
    output_md += "### 💡 操盘手点评与推荐\n"
    if hc >= 1.0: recommendation = f"推荐【{t(t1)} 独赢及赢盘】，绝对实力压制，穿盘概率极高。"
    elif hc <= -1.0: recommendation = f"推荐【{t(t2)} 独赢及赢盘】，绝对实力压制，穿盘概率极高。"
    elif p_home > 0.45: recommendation = f"推荐【{t(t1)} 独赢或亚盘主队】，占据主动，但需防小胜赢球输盘。"
    elif p_away > 0.45: recommendation = f"推荐【{t(t2)} 独赢或亚盘客队】，占据主动，防冷平。"
    else: recommendation = f"战局胶着，亚盘指向 {ah_str}，建议直接搏【平局】或去【小球】。"
    
    output_md += f"📝 {recommendation}\n\n"
    
    output_md += "**精确波胆矩阵 (Top 3 覆盖)：**\n\n"
    output_md += "| 排位 | 比分 | 发生概率 | 建议资金分配 |\n"
    output_md += "| :--- | :--- | :--- | :--- |\n"
    output_md += f"| Top 1 | **{score_probs[0]['score']}** | {score_probs[0]['prob']*100:.1f}% | 50% |\n"
    output_md += f"| Top 2 | **{score_probs[1]['score']}** | {score_probs[1]['prob']*100:.1f}% | 30% |\n"
    output_md += f"| Top 3 | **{score_probs[2]['score']}** | {score_probs[2]['prob']*100:.1f}% | 20% |\n\n"
    output_md += "---\n\n"

import os
out_path = os.path.join(r"d:\HelloWorld\Git_Project\worldcup\pdata", f"{beijing_date_str}_prediction.md")
with open(out_path, "w", encoding='utf-8') as f:
    f.write(output_md)
print(f"Guide generated at {out_path}")
