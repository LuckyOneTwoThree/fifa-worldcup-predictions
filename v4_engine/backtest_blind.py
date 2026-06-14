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
from datetime import datetime, timedelta
import os

def poisson_prob(l, k):
    return (math.exp(-l) * (l ** k)) / math.factorial(k)

print("Loading V4 Pre-Match Ultimate Engine for Blind Backtest...")

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
    'Haiti': '海地', 'Scotland': '苏格兰', 'Australia': '澳大利亚', 'Turkey': '土耳其'
}
def t(name): return zh_translation.get(name, name)

def get_k_factor(tournament):
    if 'World Cup' in tournament and 'Qualification' not in tournament: return 60
    elif 'Continental' in tournament: return 40
    elif 'Qualification' in tournament: return 30
    else: return 20

def run_blind_prediction(target_date_str):
    target_date_dt = pd.to_datetime(target_date_str)
    
    # 1. ABSOLUTE BLIND TEST: Filter out ALL data from target_date onwards for Elo and Training
    past_df = df[df['date'] < target_date_dt].copy()
    
    elo_dict = {}
    wc_data = []
    
    for index, row in past_df.iterrows():
        t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
        s1, s2 = row['home_score'], row['away_score']
        tournament = row['tournament']
        if pd.isna(s1) or pd.isna(s2): continue
        s1, s2 = int(s1), int(s2)
        
        if t1 not in elo_dict: elo_dict[t1] = 1500
        if t2 not in elo_dict: elo_dict[t2] = 1500
        elo1, elo2 = elo_dict[t1], elo_dict[t2]
        
        # Only use previous world cups for training
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

    # Train Model (100% blind to future)
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
    
    # Predict Target Date
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    beijing_date = target_date + timedelta(days=1)
    beijing_date_str = beijing_date.strftime('%Y-%m-%d')
    
    output_md = f"# 📅 北京时间：{beijing_date_str} (美洲当地 {target_date.month}-{target_date.day})\n"
    output_md += f"## 📊 赛前量化预测 (绝对盲测版)\n\n"
    output_md += "> [!IMPORTANT]\n> 本预测文件在生成时，底层切断了所有该日及该日之后的真实数据，确保 Elo 积分和机器学习权重处于**绝对盲测状态 (Blind Test)**，不含任何后见之明。\n\n"
    
    upcoming = df[df['date'] == target_date_str]
    referees = ["Mateu Lahoz (Mock)", "Michael Oliver", "Wilton Sampaio", "Daniele Orsato", "Szymon Marciniak"]
    
    for _, row in upcoming.iterrows():
        t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
        e1, e2 = elo_dict.get(t1, 1500), elo_dict.get(t2, 1500)
        sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)
        
        tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0})
        tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0})
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
        
        output_md += f"### ⚔️ {t(t1)} vs {t(t2)}\n"
        output_md += f"- **身价对比**：€{sv1}m vs €{sv2}m\n"
        output_md += f"- **模拟主裁**：{assigned_ref} (严厉指数: {strictness})\n"
        output_md += f"- **核心战术博弈 (Tactical Mismatch)**：\n"
        if aerial_diff > 5:
            output_md += f"  - ⚠️ {t(t1)} 在高空争顶上占据绝对碾压优势 (+{aerial_diff:.1f}%)\n"
        elif aerial_diff < -5:
            output_md += f"  - ⚠️ {t(t2)} 在高空争顶上占据绝对碾压优势 ({-aerial_diff:.1f}%)\n"
        else:
            output_md += f"  - 双方身体对抗数据接近，无明显高空压制\n"
            
        if ppda_diff > 3:
            output_md += f"  - ⚠️ {t(t2)} 逼抢极其凶狠，{t(t1)} 的出球将面临巨大压力\n"
        elif ppda_diff < -3:
            output_md += f"  - ⚠️ {t(t1)} 逼抢极其凶狠，{t(t2)} 的出球将面临巨大压力\n"
        
        output_md += f"- **V5 数学统一胜率 (基于泊松积分)**：主 {p_home*100:.1f}% | 平 {p_draw*100:.1f}% | 客 {p_away*100:.1f}%\n"
        output_md += f"- **火力指征 (xG)**：{xg1_pred:.2f} - {xg2_pred:.2f}\n"
        output_md += f"- **🎲 机器推演盘口**：\n"
        output_md += f"  - **理论亚盘**：{ah_str}\n"
        output_md += f"  - **理论大小球**：{ou_str}\n\n"
        
        output_md += "#### 💡 操盘手点评与推荐\n"
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

    out_path = os.path.join(r"d:\HelloWorld\Git_Project\worldcup\pdata", f"{beijing_date_str}_prediction.md")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(output_md)
    print(f"Blind prediction for {target_date_str} generated -> {out_path}")

run_blind_prediction('2026-06-11')
run_blind_prediction('2026-06-12')
