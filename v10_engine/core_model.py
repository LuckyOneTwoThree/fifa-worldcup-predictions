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
            # Use current 2026 data as a static proxy for historical tactical/squad data
            # Dead SV/Tactical code removed to prevent data leakage and speed up training
            t_weight = get_k_factor(tournament)
            
            is_neutral = str(row['neutral']).upper() == 'TRUE'
            is_host = (t1 == row.get('country') or t2 == row.get('country'))
            home_adv = 1 if (is_host and not is_neutral) else 0
            
            result = 1 if s1 > s2 else (0 if s1 == s2 else -1)
            train_data.append({
                'team1': t1, 'team2': t2,
                'elo_diff': elo1 - elo2,
                't_weight': t_weight,
                'home_adv': home_adv,
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
    features = ['elo_diff', 't_weight', 'home_adv']
    X_train = train_df[features]
    y_train = train_df['result'] + 1 # 0: Loss, 1: Draw, 2: Win
    
    # Train Model A: Classification
    # Since features are just elo_diff, t_weight, home_adv, a Stacking Ensemble is overkill.
    # LogisticRegression is ideal for mapping continuous distance (elo_diff) to probability.
    base_clf = LogisticRegression(max_iter=1000, random_state=42)
    calibrated_stack = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    calibrated_stack.fit(X_train, y_train)
    
    # Train Model B: xG Predictors (Poisson Regression)
    # Target variables are actual home and away scores
    xg_model_home = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, n_estimators=100, learning_rate=0.05, random_state=42)
    xg_model_away = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, n_estimators=100, learning_rate=0.05, random_state=42)
    
    xg_model_home.fit(X_train, train_df['home_score'])
    xg_model_away.fit(X_train, train_df['away_score'])
    
    return calibrated_stack, xg_model_home, xg_model_away, elo_dict
