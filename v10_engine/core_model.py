import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_k_factor(tournament):
    if 'World Cup' in tournament and 'Qualification' not in tournament: return 60
    elif tournament in ['UEFA Euro', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup', 'CONCACAF Championship']: return 40
    elif 'Qualification' in tournament: return 30
    else: return 20

def train_v8_models(df, squad_dict, tac_dict):
    """
    Trains the V10 Dual-Engine ML Pipeline.
    Returns:
    - calibrated_stack: classification model for Win/Draw/Loss
    - xg_model_home: Poisson regressor for home xG
    - xg_model_away: Poisson regressor for away xG
    - elo_dict: final elo ratings at cutoff
    """
    

    elo_dict = {}
    train_data = []
    
    name_mapping = {
        'South Korea': 'South Korea', 'Korea Republic': 'South Korea',
        'USA': 'USA', 'United States': 'USA',
        'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
        'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
        'Czech Republic': 'Czech Republic'
    }
    def map_name(name): return name_mapping.get(name, name)
    
    major_tournaments = [
        'FIFA World Cup', 'UEFA Euro', 'Copa América', 
        'African Cup of Nations', 'AFC Asian Cup', 'CONCACAF Championship', 
        'FIFA World Cup qualification', 'UEFA Euro qualification'
    ]
    
    for index, row in df.iterrows():
        t1, t2 = map_name(row['home_team']), map_name(row['away_team'])
        s1, s2 = row['home_score'], row['away_score']
        tournament = row['tournament']
        if pd.isna(s1) or pd.isna(s2): continue
        s1, s2 = int(s1), int(s2)
        
        if t1 not in elo_dict: elo_dict[t1] = 1500
        if t2 not in elo_dict: elo_dict[t2] = 1500
        elo1, elo2 = elo_dict[t1], elo_dict[t2]
        
        # Build training set using V10 rules
        if tournament in major_tournaments and row['date'].year >= 2010:
            # Prevent Data Leakage: only use static 2026 tactical/squad data if match is in 2026.
            # Prevent Data Leakage
            if row['date'].year >= 2025:
                sv1, sv2 = squad_dict.get(t1, 50), squad_dict.get(t2, 50)
                tac1 = tac_dict.get(t1, {"aerial_win_rate": 50.0, "ppda": 12.0})
                tac2 = tac_dict.get(t2, {"aerial_win_rate": 50.0, "ppda": 12.0})
                aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
                ppda_diff = tac1['ppda'] - tac2['ppda']
            else:
                sv1, sv2 = 50, 50
                aerial_diff = 0.0
                ppda_diff = 0.0
            
            t_weight = get_k_factor(tournament)
            
            result = 1 if s1 > s2 else (0 if s1 == s2 else -1)
            train_data.append({
                'team1': t1, 'team2': t2,
                'elo_diff': elo1 - elo2,
                'sv_diff': sv1 - sv2,
                'aerial_diff': aerial_diff,
                'ppda_diff': ppda_diff,
                't_weight': t_weight,
                'result': result,
                'home_score': s1,
                'away_score': s2
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
        
    train_df = pd.DataFrame(train_data).fillna(0)
    features = ['elo_diff', 'sv_diff', 'aerial_diff', 'ppda_diff', 't_weight']
    X_train = train_df[features]
    y_train = train_df['result'] + 1 # 0: Loss, 1: Draw, 2: Win
    
    # Train Model A: Classification (Stacking)
    xgb_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', seed=42)
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp_base = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    
    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb_base), ('rf', rf_base), ('mlp', mlp_base)],
        final_estimator=LogisticRegression(), cv=3
    )
    calibrated_stack = CalibratedClassifierCV(stacking_clf, method='sigmoid', cv=3)
    calibrated_stack.fit(X_train, y_train)
    
    # Train Model B: xG Predictors (Poisson Regression)
    # Target variables are actual home and away scores
    xg_model_home = xgb.XGBRegressor(objective='count:poisson', n_estimators=100, learning_rate=0.05, random_state=42)
    xg_model_away = xgb.XGBRegressor(objective='count:poisson', n_estimators=100, learning_rate=0.05, random_state=42)
    
    xg_model_home.fit(X_train, train_df['home_score'])
    xg_model_away.fit(X_train, train_df['away_score'])
    
    return calibrated_stack, xg_model_home, xg_model_away, elo_dict
