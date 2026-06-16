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
from core_model import get_k_factor
from v10_shared import load_results_csv, get_zh_name, get_cached_models

def dixon_coles_prob(l1, l2, k1, k2, rho=0.0, elo_diff=0.0):
    prob = (math.exp(-l1) * (l1 ** k1) / math.factorial(k1)) * (math.exp(-l2) * (l2 ** k2) / math.factorial(k2))
    
    # Core Dixon Coles adjustment
    if k1 == 0 and k2 == 0: prob = prob * (1 - l1*l2*rho)
    elif k1 == 0 and k2 == 1: prob = prob * (1 + l1*rho)
    elif k1 == 1 and k2 == 0: prob = prob * (1 + l2*rho)
    elif k1 == 1 and k2 == 1: prob = prob * (1 - rho)
    
    # Phase 2: Avalanche Effect removed due to V10 redesign (Park-the-bus resistance replaces this)
        
    return prob

def get_base_match_info(target_date_str="2026-06-15"):
    df = load_results_csv()
    df = df.sort_values('date')

    name_mapping = {'South Korea': 'South Korea', 'Korea Republic': 'South Korea', 'USA': 'USA', 'United States': 'USA'}
    def map_name(name): return name_mapping.get(name, name)

    upcoming = df[df['date'] == target_date_str].head(4)
    matches = []
    for _, row in upcoming.iterrows():
        t1_en, t2_en = map_name(row['home_team']), map_name(row['away_team'])
        matches.append((t1_en, t2_en, get_zh_name(t1_en), get_zh_name(t2_en)))
    return matches

def generate_v10_predictions(target_date_str, impact_dict=None):
    print("Loading V10.0 Pre-Match Ultimate Engine...")

    df = load_results_csv()
    models = get_cached_models()
    calibrated_stack, xg_model_home, xg_model_away, elo_dict = models
    
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    squad_vals = pd.read_csv(os.path.join(base_dir, 'data_scrapers/squad_values.csv'))
    ref_df = pd.read_csv(os.path.join(base_dir, 'data_scrapers/referee_stats.csv'))
    tac_df = pd.read_csv(os.path.join(base_dir, 'data_scrapers/tactical_styles.csv'))

    squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
    tac_dict = tac_df.set_index('team').to_dict('index')
    ref_dict = ref_df.set_index('name').to_dict('index')

    name_mapping = {'South Korea': 'South Korea', 'Korea Republic': 'South Korea', 'USA': 'USA', 'United States': 'USA'}
    def map_name(name): return name_mapping.get(name, name)

    upcoming = df[df['date'] == target_date_str].head(4)
    referees = [
        "Michael Oliver", "Wilton Sampaio", "Daniele Orsato", "Szymon Marciniak",
        "Anthony Taylor", "Clement Turpin", "Danny Makkelie", "Slavko Vincic",
        "Facundo Tello", "Cesar Ramos", "Fernando Rapallini", "Ivan Barton"
    ]
    
    results_dict = {}
    output_md = ""

    if len(upcoming) == 0:
        print(f"No matches found for {target_date_str}")
        return results_dict, output_md

    for _, row in upcoming.iterrows():
        t1_en, t2_en = map_name(row['home_team']), map_name(row['away_team'])
        t1, t2 = get_zh_name(t1_en), get_zh_name(t2_en)
        e1, e2 = elo_dict.get(t1_en, 1500), elo_dict.get(t2_en, 1500)
        sv1, sv2 = squad_dict.get(t1_en, 50), squad_dict.get(t2_en, 50)
        
        tac1 = tac_dict.get(t1_en, {"aerial_win_rate": 50.0, "ppda": 12.0, "possession_avg": 50.0})
        tac2 = tac_dict.get(t2_en, {"aerial_win_rate": 50.0, "ppda": 12.0, "possession_avg": 50.0})
        aerial_diff = tac1['aerial_win_rate'] - tac2['aerial_win_rate']
        ppda_diff = tac1['ppda'] - tac2['ppda']
        
        assigned_ref = referees[sum(ord(c) for c in t1_en+t2_en) % len(referees)]
        strictness = ref_dict.get(assigned_ref, {}).get('strictness_index', 0.5)
        
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

        if impact_dict is not None:
            match_key = f"{t1} vs {t2}"
            impacts = impact_dict.get(match_key, {})
            
            try: fatigue_home = float(impacts.get("fatigue_home", 7))
            except: fatigue_home = 7
            try: fatigue_away = float(impacts.get("fatigue_away", 7))
            except: fatigue_away = 7
            weather = impacts.get("weather", "Clear")
            motivation = impacts.get("motivation_index", "HIGH")
            inj_home = impacts.get("injuries_home", 0)
            inj_away = impacts.get("injuries_away", 0)

            # Base multiplier starts at 1.0
            mult1, mult2 = 1.0, 1.0
            
            if fatigue_home < 5 and tac1.get('ppda', 12.0) < 11.0: mult1 *= 0.90
            if fatigue_away < 5 and tac2.get('ppda', 12.0) < 11.0: mult2 *= 0.90
            
            if 'Rain' in weather or 'Storm' in weather:
                if tac1.get('possession_avg', 50) > 55: mult1 *= 0.85
                if tac2.get('possession_avg', 50) > 55: mult2 *= 0.85
                
            # Phase 4: Geo-Climatic Debuff for Extreme Heat
            if 'Heat' in weather or 'Hot' in weather or '35C' in weather:
                nordic_alpine_teams = ["Norway", "Sweden", "Denmark", "Finland", "Iceland", "Switzerland", "Scotland", "Wales", "Republic of Ireland", "Northern Ireland"]
                if t1 in nordic_alpine_teams and fatigue_home >= 5: mult1 *= 0.85  # Severe stamina penalty
                if t2 in nordic_alpine_teams and fatigue_away >= 5: mult2 *= 0.85
                
            biscotto_risk = impacts.get("biscotto_risk", "NONE")
            
            # Phase 5: Park-the-Bus Resistance (V10)
            elo_diff_raw = e1 - e2
            if elo_diff_raw > 200:
                # Home is heavy favorite. If away team plays conservative, penalize Home xG.
                if tac2.get('ppda', 12.0) >= 10.5 or tac2.get('possession_avg', 50) <= 45:
                    mult1 *= 0.70
            elif elo_diff_raw < -200:
                if tac1.get('ppda', 12.0) >= 10.5 or tac1.get('possession_avg', 50) <= 45:
                    mult2 *= 0.70
            
            # 1. Extreme low motivation or confirmed biscotto (MD3 High Risk)
            if motivation == "LOW" or biscotto_risk == "HIGH":
                mult1 *= 0.60
                mult2 *= 0.60
            # 2. Potential conservative play (MD3 permutation risk)
            elif biscotto_risk == "MEDIUM":
                mult1 *= 0.85
                mult2 *= 0.85
            # 3. Extreme tension (Knockouts - defensive posturing early on)
            elif motivation == "EXTREME":
                mult1 *= 0.90
                mult2 *= 0.90
            # 4. Standard Group Stage (MD1/MD2)
            # mult1 *= 1.0 (no penalty)
                
            if inj_home > 0: mult1 *= (1.0 - (inj_home * 0.1))
            if inj_away > 0: mult2 *= (1.0 - (inj_away * 0.1))

            # Apply hard cap to prevent extreme distortion
            mult1 = max(0.60, mult1)
            mult2 = max(0.60, mult2)
            
            xg1_pred *= mult1
            xg2_pred *= mult2

        score_probs = []
        p_home_pois, p_draw_pois, p_away_pois = 0.0, 0.0, 0.0
        for i in range(15):
            for j in range(15):
                # Pass elo_diff for Avalanche Effect
                elo_diff = elo_dict.get(t1, 1500) - elo_dict.get(t2, 1500)
                p = dixon_coles_prob(xg1_pred, xg2_pred, i, j, rho=-0.05, elo_diff=elo_diff)
                score_probs.append({'score': f'{i}-{j}', 'prob': p})
                if i > j: p_home_pois += p
                elif i == j: p_draw_pois += p
                else: p_away_pois += p

        # Normalize score probs
        total_prob = sum(x['prob'] for x in score_probs)
        for x in score_probs:
            x['prob'] /= total_prob
        score_probs = sorted(score_probs, key=lambda x: x['prob'], reverse=True)

        w_ml, w_pois = 0.3, 0.7
        p_home = p_home_ml * w_ml + p_home_pois * w_pois
        p_draw = p_draw_ml * w_ml + p_draw_pois * w_pois
        p_away = p_away_ml * w_ml + p_away_pois * w_pois
        
        norm = p_home + p_draw + p_away
        p_home /= norm; p_draw /= norm; p_away /= norm

        # True AH based on original xG difference
        diff = xg1_pred - xg2_pred
        hc = round(diff / 0.25) * 0.25
        if hc == 0: ah_str = "平手盘(0)"
        elif hc > 0: ah_str = f"主让 {hc} 球" if hc % 0.5 == 0 else f"主让 {hc-0.25}/{hc+0.25} 球"
        else: ah_str = f"受让 {abs(hc)} 球" if abs(hc) % 0.5 == 0 else f"受让 {abs(hc)-0.25}/{abs(hc)+0.25} 球"

        # True OU based on original total xG
        total_xg = xg1_pred + xg2_pred
        ou = round(total_xg / 0.25) * 0.25
        ou_str = f"{ou} 球" if ou % 0.5 == 0 else f"{ou-0.25}/{ou+0.25} 球"
        
        p1 = tac1.get('possession_avg', 50.0)
        p2 = tac2.get('possession_avg', 50.0)
        norm_pos1 = (p1 / (p1 + p2)) * 100
        norm_pos2 = (p2 / (p1 + p2)) * 100
        
        base_elo1, base_elo2 = e1, e2
        def get_tier(elo):
            if elo >= 1900: return "T0-世界顶尖"
            elif elo >= 1750: return "T1-洲际强队"
            elif elo >= 1600: return "T2-中坚力量"
            elif elo >= 1450: return "T3-边缘球队"
            else: return "T4-送分鱼腩"
            
        tier1 = get_tier(base_elo1)
        tier2 = get_tier(base_elo2)
        val_home, val_away = sv1, sv2

        results_dict[f"{t1_en} vs {t2_en}"] = {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "xg_home": xg1_pred, "xg_away": xg2_pred,
            "ah_num": round(hc * 2) / 2, "ah_str": ah_str,
            "ou_num": round(total_xg * 2) / 2, "ou_str": ou_str,
            "elo_home": base_elo1, "elo_away": base_elo2,
            "tier_home": tier1, "tier_away": tier2,
            "sv_home": val_home, "sv_away": val_away,
            "score_probs": score_probs[:5],
            "referee": assigned_ref, "strictness": strictness,
            "ppda_home": tac1.get('ppda', 12.0), "ppda_away": tac2.get('ppda', 12.0),
            "possession_home": tac1.get('possession_avg', 50.0),
            "possession_away": tac2.get('possession_avg', 50.0),
            "possession_home_norm": norm_pos1, "possession_away_norm": norm_pos2,
            "aerial_diff": aerial_diff,
            "confidence_index": "HIGH" if abs(p_home - p_away) > 0.4 else "MEDIUM"
        }

    return results_dict, output_md

if __name__ == '__main__':
    import sys
    target_date_str = sys.argv[1] if len(sys.argv) > 1 else '2026-06-15'
    generate_v10_predictions(target_date_str)
