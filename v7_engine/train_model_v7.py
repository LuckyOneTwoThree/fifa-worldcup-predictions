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
import warnings
warnings.filterwarnings('ignore')

print("=== V4.0 PRE-MATCH ULTIMATE ENGINE INITIALIZATION ===")

# --- 1. Load Data ---
try:
    df = pd.read_csv('../results.csv')
    squad_vals = pd.read_csv('data_scrapers/squad_values.csv')
    xg_df = pd.read_csv('data_scrapers/xg_history.csv')
    ref_df = pd.read_csv('data_scrapers/referee_stats.csv')
    tac_df = pd.read_csv('data_scrapers/tactical_styles.csv')
except FileNotFoundError:
    df = pd.read_csv('../../results.csv')
    squad_vals = pd.read_csv('../squad_values.csv')
    xg_df = pd.read_csv('../xg_history.csv')
    ref_df = pd.read_csv('referee_stats.csv')
    tac_df = pd.read_csv('tactical_styles.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
xg_df['date'] = pd.to_datetime(xg_df['date'])
df = pd.merge(df, xg_df, on=['date', 'home_team', 'away_team'], how='left')

# Mappings
squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
tac_dict = tac_df.set_index('team').to_dict('index')
ref_dict = ref_df.set_index('name').to_dict('index')

# --- 2. Feature Engineering ---
elo_dict = {}
def get_k_factor(tournament):
    if 'World Cup' in tournament and 'Qualification' not in tournament: return 60
    elif 'Continental' in tournament: return 40
    elif 'Qualification' in tournament: return 30
    else: return 20

wc_data = []

name_mapping = {
    'South Korea': 'South Korea', 'Korea Republic': 'South Korea',
    'USA': 'USA', 'United States': 'USA',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Czech Republic': 'Czech Republic'
}
def map_name(name): return name_mapping.get(name, name)

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
        
        # Tactical mismatch
        tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0})
        tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0})
        aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
        ppda_diff = tac1['ppda'] - tac2['ppda']
        
        # Random referee strictness for historical data
        strictness = np.random.uniform(0.3, 0.9)
        
        result = 1 if s1 > s2 else (0 if s1 == s2 else -1)
        wc_data.append({
            'team1': t1, 'team2': t2,
            'elo_diff': elo1 - elo2,
            'sv_diff': sv1 - sv2,
            'aerial_diff': aerial_diff,
            'ppda_diff': ppda_diff,
            'strict_ref': strictness,
            'result': result
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

# --- 3. Train V4.0 Stacking Model ---
train_df = pd.DataFrame(wc_data).fillna(0)
features = ['elo_diff', 'sv_diff', 'aerial_diff', 'ppda_diff', 'strict_ref']
X_train = train_df[features]
y_train = train_df['result'] + 1

xgb_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', seed=42)
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
mlp_base = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))

stacking_clf = StackingClassifier(
    estimators=[('xgb', xgb_base), ('rf', rf_base), ('mlp', mlp_base)],
    final_estimator=LogisticRegression(), cv=3
)
print("Training V4 Stacking Ensemble with Tactical and Referee Features...")
stacking_clf.fit(X_train, y_train)

calibrated_stack = CalibratedClassifierCV(stacking_clf, method='sigmoid', cv=3)
calibrated_stack.fit(X_train, y_train)

# --- 4. Validation: Australia vs Turkey ---
print("\n--- 🛡️ V4.0 VALIDATION TARGET: AUSTRALIA VS TURKEY ---")
t1, t2 = "Australia", "Turkey"
e1, e2 = elo_dict.get(t1, 1500), elo_dict.get(t2, 1500)
sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)

tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0})
tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0})
aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
ppda_diff = tac1['ppda'] - tac2['ppda']

# Assign strict referee "Mateu Lahoz" to increase chaos
strictness = 0.95 

X_pred = pd.DataFrame({
    'elo_diff': [e1 - e2],
    'sv_diff': [sv1 - sv2],
    'aerial_diff': [aerial_diff],
    'ppda_diff': [ppda_diff],
    'strict_ref': [strictness]
})

probs = calibrated_stack.predict_proba(X_pred)[0]
p_away, p_draw, p_home = probs[0], probs[1], probs[2]

# Dynamic xG using tactical mismatch + referee chaos
base_xg = 1.1
xg1_pred = base_xg + (p_home - p_away) * 1.5 + (sv1 / 1000.0) + (aerial_diff * 0.05) + (strictness * 0.2)
xg2_pred = base_xg + (p_away - p_home) * 1.5 + (sv2 / 1000.0) - (aerial_diff * 0.02) + (strictness * 0.1)

xg1_pred = max(0.3, xg1_pred)
xg2_pred = max(0.3, xg2_pred)

def poisson_prob(l, k):
    return (math.exp(-l) * (l ** k)) / math.factorial(k)

score_probs = []
for i in range(5):
    for j in range(5):
        prob = poisson_prob(xg1_pred, i) * poisson_prob(xg2_pred, j)
        score_probs.append({"score": f"{i}-{j}", "prob": prob})
score_probs = sorted(score_probs, key=lambda x: x["prob"], reverse=True)[:3]

print(f"Match: {t1} vs {t2}")
print(f"Tactical Clash -> Aerial Diff: +{aerial_diff:.1f}% (Massive Advantage for AUS)")
print(f"V4 Probabilities -> Home(AUS): {p_home*100:.1f}% | Draw: {p_draw*100:.1f}% | Away(TUR): {p_away*100:.1f}%")
print(f"V4 xG Output -> AUS {xg1_pred:.2f} - {xg2_pred:.2f} TUR")
print(f"Top Score Cluster: {score_probs[0]['score']} ({score_probs[0]['prob']*100:.1f}%), {score_probs[1]['score']}, {score_probs[2]['score']}")
print("V4.0 ULTIMATE ENGINE READY.")
