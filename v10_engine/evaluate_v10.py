import pandas as pd
import numpy as np
from core_model import train_v8_models, get_k_factor
from sklearn.metrics import accuracy_score
from v10_shared import dixon_coles_prob, load_v10_config, map_name, load_results_csv

print("Loading data for V10 Backtest Grid Search...")
df = load_results_csv()
squad_vals = pd.read_csv('data_scrapers/squad_values.csv')
tac_df = pd.read_csv('data_scrapers/tactical_styles.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
tac_dict = tac_df.set_index('team').to_dict('index')

config = load_v10_config()
xg_lb = config.get("poisson", {}).get("xg_lower_bound", 0.1)
xg_ub = config.get("poisson", {}).get("xg_upper_bound", 6.0)
alpha_power = config.get("poisson", {}).get("alpha_sv_power", 0.15)
c_rho = config.get("poisson", {}).get("rho", -0.05)
c_w_ml = config.get("fusion_weights", {}).get("classification", 0.3)
c_w_pois = config.get("fusion_weights", {}).get("poisson", 0.7)

# We will test on 2022 World Cup matches
target_tournament = 'FIFA World Cup'
test_df = df[(df['tournament'] == target_tournament) & (df['date'].dt.year == 2022)].copy()

# Train model up to 2022-11-20
cutoff_date = pd.to_datetime('2022-11-20')
past_df = df[df['date'] < cutoff_date].copy()

print(f"Training V10 on {len(past_df)} matches up to {cutoff_date.date()}...")
calibrated_stack, xg_model_home, xg_model_away, elo_dict = train_v8_models(past_df, squad_dict, tac_dict)

print(f"Evaluating on {len(test_df)} World Cup 2022 matches...")

def evaluate(rho, w_ml, w_poisson):
    y_true = []
    y_prob = []
    
    for index, row in test_df.iterrows():
        t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
        s1, s2 = row['home_score'], row['away_score']
        if pd.isna(s1) or pd.isna(s2): continue
        s1, s2 = int(s1), int(s2)
        
        actual_result = 2 if s1 > s2 else (1 if s1 == s2 else 0)
        y_true.append(actual_result)
        
        e1 = elo_dict.get(t1, 1500)
        e2 = elo_dict.get(t2, 1500)
        sv1 = squad_dict.get(t1, 50)
        sv2 = squad_dict.get(t2, 50)
        tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0})
        tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0})
        aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
        ppda_diff = tac1['ppda'] - tac2['ppda']
        
        is_neutral = str(row['neutral']).upper() == 'TRUE' if 'neutral' in row else True
        is_host = (t1 == row.get('country') or t2 == row.get('country'))
        home_adv = 1 if (is_host and not is_neutral) else 0
        
        X_pred = pd.DataFrame({
            'elo_diff': [e1 - e2],
            't_weight': [get_k_factor(row['tournament'])],
            'home_adv': [home_adv]
        })
        
        probs = calibrated_stack.predict_proba(X_pred)[0]
        p_away_ml, p_draw_ml, p_home_ml = probs[0], probs[1], probs[2]
        
        xg1_base = float(xg_model_home.predict(X_pred)[0])
        xg2_base = float(xg_model_away.predict(X_pred)[0])
        
        sv_ratio_home = sv1 / (sv2 + 1e-5)
        sv_ratio_away = sv2 / (sv1 + 1e-5)
        
        xg1_pred = max(xg_lb, min(xg1_base * (sv_ratio_home ** alpha_power), xg_ub))
        xg2_pred = max(xg_lb, min(xg2_base * (sv_ratio_away ** alpha_power), xg_ub))
        
        p_win_poisson, p_draw_poisson, p_loss_poisson = 0.0, 0.0, 0.0
        for i in range(15):
            for j in range(15):
                prob = dixon_coles_prob(xg1_pred, xg2_pred, i, j, rho=rho)
                if i > j: p_win_poisson += prob
                elif i == j: p_draw_poisson += prob
                else: p_loss_poisson += prob
                
        total_p = p_win_poisson + p_draw_poisson + p_loss_poisson
        p_win_poisson /= total_p
        p_draw_poisson /= total_p
        p_loss_poisson /= total_p
        
        p_home = p_home_ml * w_ml + p_win_poisson * w_poisson
        p_draw = p_draw_ml * w_ml + p_draw_poisson * w_poisson
        p_away = p_away_ml * w_ml + p_loss_poisson * w_poisson
        
        y_prob.append([p_away, p_draw, p_home])
        
    y_true_onehot = np.zeros((len(y_true), 3))
    for idx, val in enumerate(y_true):
        y_true_onehot[idx, val] = 1.0
    brier = np.mean(np.sum((np.array(y_prob) - y_true_onehot)**2, axis=1))
    
    y_pred_classes = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred_classes)
    return brier, acc

print(f"\n--- Current Config Evaluation (rho={c_rho}, w_ml={c_w_ml}, w_poisson={c_w_pois}) ---")
b, a = evaluate(c_rho, c_w_ml, c_w_pois)
print(f"Brier: {b:.4f} | Accuracy: {a:.2%}")

print("\n--- Rho Optimization (w_ml=0.3, w_poisson=0.7) ---")
for r in [-0.05, 0.0, 0.05]:
    b, a = evaluate(r, 0.3, 0.7)
    print(f"Rho = {r:5.2f} | Brier: {b:.4f} | Accuracy: {a:.2%}")

print("\n--- Fusion Weight Optimization (rho=0.0) ---")
for w_ml, w_poisson in [(0.0, 1.0), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.7, 0.3), (1.0, 0.0)]:
    b, a = evaluate(0.0, w_ml, w_poisson)
    print(f"ML={w_ml}, Poisson={w_poisson} | Brier: {b:.4f} | Accuracy: {a:.2%}")
