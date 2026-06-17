import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from core_model import train_v8_models
from v10_shared import dixon_coles_prob, load_v10_config, map_name, load_results_csv

config = load_v10_config()
xg_lb = config.get("poisson", {}).get("xg_lower_bound", 0.1)
xg_ub = config.get("poisson", {}).get("xg_upper_bound", 6.0)
rho_val = config.get("poisson", {}).get("rho", -0.05)
w_ml = config.get("fusion_weights", {}).get("classification", 0.3)
w_poisson = config.get("fusion_weights", {}).get("poisson", 0.7)
alpha_power = config.get("poisson", {}).get("alpha_sv_power", 0.15)

df = load_results_csv()
squad_vals = pd.read_csv('data_scrapers/squad_values.csv')
tac_df = pd.read_csv('data_scrapers/tactical_styles.csv')
ref_df = pd.read_csv('data_scrapers/referee_stats.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
tac_dict = tac_df.set_index('team').to_dict('index')

# Test on WC 2022
wc2022 = df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2022)]
dates = sorted(wc2022['date'].unique())
print(f'WC 2022: {len(dates)} match days, {len(wc2022)} total matches')

all_metrics = []
for target_date in dates:
    past_df = df[df['date'] < target_date].copy()
    calibrated_stack, xg_model_home, xg_model_away, elo_dict = train_v8_models(past_df, squad_dict, tac_dict)
    
    matches = wc2022[wc2022['date'] == target_date]
    for _, row in matches.iterrows():
        t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
        e1, e2 = elo_dict.get(t1, 1500), elo_dict.get(t2, 1500)
        sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)
        
        tac1 = tac_dict.get(t1, {'aerial_win_rate': 50.0, 'ppda': 12.0})
        tac2 = tac_dict.get(t2, {'aerial_win_rate': 50.0, 'ppda': 12.0})
        aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
        ppda_diff = tac1['ppda'] - tac2['ppda']
        
        is_neutral = str(row['neutral']).upper() == 'TRUE' if 'neutral' in row else True
        is_host = (t1 == row.get('country') or t2 == row.get('country'))
        home_adv = 1 if (is_host and not is_neutral) else 0
        
        X_pred = pd.DataFrame({
            'elo_diff': [e1 - e2],
            't_weight': [60],
            'home_adv': [home_adv]
        })
        
        probs = calibrated_stack.predict_proba(X_pred)[0]
        p_away_ml, p_draw_ml, p_home_ml = probs[0], probs[1], probs[2]
        
        xg1_base = float(xg_model_home.predict(X_pred)[0])
        xg2_base = float(xg_model_away.predict(X_pred)[0])
        
        sv_ratio_home = sv1 / (sv2 + 1e-5)
        sv_ratio_away = sv2 / (sv1 + 1e-5)
        
        xg1 = max(xg_lb, min(xg1_base * (sv_ratio_home ** alpha_power), xg_ub))
        xg2 = max(xg_lb, min(xg2_base * (sv_ratio_away ** alpha_power), xg_ub))
        
        p_win, p_draw, p_loss = 0.0, 0.0, 0.0
        score_probs = []
        for i in range(15):
            for j in range(15):
                prob = dixon_coles_prob(xg1, xg2, i, j, rho=rho_val)
                if i > j: p_win += prob
                elif i == j: p_draw += prob
                else: p_loss += prob
                if i < 5 and j < 5:
                    score_probs.append({'score': f'{i}-{j}', 'prob': prob})
        
        total_p = p_win + p_draw + p_loss
        p_home = p_home_ml * w_ml + (p_win/total_p) * w_poisson
        p_draw_final = p_draw_ml * w_ml + (p_draw/total_p) * w_poisson
        p_away = p_away_ml * w_ml + (p_loss/total_p) * w_poisson
        
        s1, s2 = int(row['home_score']), int(row['away_score'])
        res = 2 if s1 > s2 else (1 if s1 == s2 else 0)
        pred = np.argmax([p_away, p_draw_final, p_home])
        
        top_scores = sorted(score_probs, key=lambda x: x['prob'], reverse=True)
        actual_score = f'{s1}-{s2}'
        
        all_metrics.append({
            'match': f'{t1} vs {t2}',
            'actual': actual_score,
            'pred_WDL': ['L','D','W'][pred],
            'actual_WDL': ['L','D','W'][res],
            'correct': pred == res,
            'xg1': xg1, 'xg2': xg2,
            'p_home': p_home, 'p_draw': p_draw_final, 'p_away': p_away,
            'top1': top_scores[0]['score'],
            'top3': [s['score'] for s in top_scores[:3]],
            'exact_hit': top_scores[0]['score'] == actual_score,
            'top3_hit': actual_score in [s['score'] for s in top_scores[:3]]
        })

# Print results
print(f'\n=== WC 2022 Blind Backtest ({len(all_metrics)} matches) ===')
hits = sum(1 for m in all_metrics if m['correct'])
exact = sum(1 for m in all_metrics if m['exact_hit'])
top3 = sum(1 for m in all_metrics if m['top3_hit'])

for m in all_metrics:
    mark = 'Y' if m['correct'] else 'N'
    score_mark = 'EXACT' if m['exact_hit'] else ('TOP3' if m['top3_hit'] else '')
    ph = m['p_home']*100
    pd_ = m['p_draw']*100
    pa = m['p_away']*100
    print(f"  [{mark}] {m['match']:35s} Actual:{m['actual']:5s}({m['actual_WDL']}) Pred:{m['pred_WDL']} W:{ph:.0f}% D:{pd_:.0f}% L:{pa:.0f}% xG:{m['xg1']:.2f}-{m['xg2']:.2f} Top1:{m['top1']} {score_mark}")

print(f'\n--- Summary ---')
print(f'W/D/L Accuracy: {hits}/{len(all_metrics)} = {hits/len(all_metrics)*100:.1f}%')
print(f'Exact Score Hit: {exact}/{len(all_metrics)} = {exact/len(all_metrics)*100:.1f}%')
print(f'Top 3 Coverage: {top3}/{len(all_metrics)} = {top3/len(all_metrics)*100:.1f}%')

# Brier score
y_true = []
y_prob = []
for m in all_metrics:
    res = 2 if m['actual_WDL'] == 'W' else (1 if m['actual_WDL'] == 'D' else 0)
    y_true.append(res)
    y_prob.append([m['p_away'], m['p_draw'], m['p_home']])
y_true_onehot = np.zeros((len(y_true), 3))
for idx, val in enumerate(y_true):
    y_true_onehot[idx, val] = 1.0
brier = np.mean(np.sum((np.array(y_prob) - y_true_onehot)**2, axis=1))
print(f'Brier Score: {brier:.4f}')

# Calibration check
print(f'\n--- Calibration Check ---')
for threshold in [0.5, 0.6, 0.7]:
    high_conf = [m for m in all_metrics if max(m['p_home'], m['p_draw'], m['p_away']) >= threshold]
    if high_conf:
        conf_hits = sum(1 for m in high_conf if m['correct'])
        print(f'  Confidence >= {threshold*100:.0f}%: {conf_hits}/{len(high_conf)} = {conf_hits/len(high_conf)*100:.1f}% accuracy ({len(high_conf)} matches)')

# xG sanity check
xg_values = [m['xg1'] for m in all_metrics] + [m['xg2'] for m in all_metrics]
print(f'\n--- xG Sanity ---')
print(f'  xG range: {min(xg_values):.3f} to {max(xg_values):.3f}')
print(f'  xG mean: {np.mean(xg_values):.3f}')
print(f'  xG std: {np.std(xg_values):.3f}')
print(f'  xG < 0 count: {sum(1 for x in xg_values if x < 0)}')
