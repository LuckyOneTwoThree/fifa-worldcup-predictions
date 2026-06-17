import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from v10_shared import get_zh_name, get_cached_models, load_results_csv, load_v10_config, get_tier, dixon_coles_prob, get_assigned_referee, map_name
from core_model import get_k_factor

def get_base_match_info(target_date_str="2026-06-15"):
    df = load_results_csv()
    df = df.sort_values('date')

    upcoming = df[df['date'] == target_date_str].head(4)
    matches = []
    for _, row in upcoming.iterrows():
        t1_en, t2_en = map_name(row['home_team']), map_name(row['away_team'])
        city = row.get('city', 'Stadium')
        matches.append((t1_en, t2_en, get_zh_name(t1_en), get_zh_name(t2_en), city))
    return matches

def generate_v10_predictions(target_date_str, impact_dict=None):
    print("Loading V10.0 Pre-Match Ultimate Engine...")
    config = load_v10_config()
    poisson_cfg = config.get("poisson", {})
    fusion_cfg = config.get("fusion_weights", {})
    penalty_cfg = config.get("penalties", {})
    
    rho_val = poisson_cfg.get("rho", -0.05)
    xg_lb = poisson_cfg.get("xg_lower_bound", 0.1)
    xg_ub = poisson_cfg.get("xg_upper_bound", 6.0)
    alpha_power = poisson_cfg.get("alpha_sv_power", 0.15)
    w_ml = fusion_cfg.get("classification", 0.3)
    w_pois = fusion_cfg.get("poisson", 0.7)

    df = load_results_csv()
    calibrated_stack, xg_model_home, xg_model_away, elo_dict = get_cached_models()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    squad_vals = pd.read_csv(os.path.join(base_dir, 'data_scrapers/squad_values.csv'))
    tac_df = pd.read_csv(os.path.join(base_dir, 'data_scrapers/tactical_styles.csv'))
    ref_df = pd.read_csv(os.path.join(base_dir, 'data_scrapers/referee_stats.csv'))

    squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
    tac_dict = tac_df.set_index('team').to_dict('index')
    ref_dict = ref_df.set_index('name').to_dict('index')
    referees = list(ref_dict.keys())
    
    upcoming = df[df['date'] == target_date_str]
    
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
        
        date_str = str(row['date'])[:10]
        assigned_ref = get_assigned_referee(t1_en, t2_en, date_str, referees)
        strictness = ref_dict.get(assigned_ref, {}).get('strictness_index', 0.5)
        
        is_neutral = str(row['neutral']).upper() == 'TRUE' if 'neutral' in row else True
        is_host = (t1_en == row.get('country') or t2_en == row.get('country'))
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
        
        # V10.4 Alpha Injection: Re-apply 2026 Squad Value scaling safely outside the tree models
        sv_ratio_home = sv1 / (sv2 + 1e-5)
        sv_ratio_away = sv2 / (sv1 + 1e-5)
        
        xg1_pred = xg1_base * (sv_ratio_home ** alpha_power)
        xg2_pred = xg2_base * (sv_ratio_away ** alpha_power)

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
                nordic_alpine_teams = penalty_cfg.get("nordic_alpine_teams", [])
                if t1_en in nordic_alpine_teams and fatigue_home >= 5: mult1 *= penalty_cfg.get("heat_exhaustion", 0.85)
                if t2_en in nordic_alpine_teams and fatigue_away >= 5: mult2 *= penalty_cfg.get("heat_exhaustion", 0.85)
                
            biscotto_risk = impacts.get("biscotto_risk", "NONE")
            
            # Phase 5: Park-the-Bus Resistance (V10)
            superstar_teams = penalty_cfg.get("superstar_teams", [])
            is_home_superstar = (t1_en in superstar_teams) and (fatigue_home >= 5) and (inj_home == 0)
            is_away_superstar = (t2_en in superstar_teams) and (fatigue_away >= 5) and (inj_away == 0)

            elo_diff_raw = e1 - e2
            if elo_diff_raw > 200:
                # Home is heavy favorite. If away team plays conservative, penalize Home xG.
                if tac2.get('ppda', 12.0) >= 10.5 or tac2.get('possession_avg', 50) <= 45:
                    if not is_home_superstar:
                        mult1 *= penalty_cfg.get("park_the_bus", 0.70)
            elif elo_diff_raw < -200:
                if tac1.get('ppda', 12.0) >= 10.5 or tac1.get('possession_avg', 50) <= 45:
                    if not is_away_superstar:
                        mult2 *= penalty_cfg.get("park_the_bus", 0.70)
            
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

            # OSINT Probability Shift (V10.3)
            # Ensure the ML classifier also respects the dynamic OSINT penalties
            p_home_ml_adj = p_home_ml * mult1
            p_away_ml_adj = p_away_ml * mult2
            
            home_drop = p_home_ml - p_home_ml_adj
            away_drop = p_away_ml - p_away_ml_adj
            
            p_home_ml = p_home_ml_adj + (away_drop * 0.3)
            p_away_ml = p_away_ml_adj + (home_drop * 0.3)
            p_draw_ml = p_draw_ml + (home_drop * 0.7) + (away_drop * 0.7)

        # Apply final clipping to ensure xG remains within logical physical bounds
        xg1_pred = max(xg_lb, min(xg1_pred, xg_ub))
        xg2_pred = max(xg_lb, min(xg2_pred, xg_ub))

        score_probs = []
        p_home_pois, p_draw_pois, p_away_pois = 0.0, 0.0, 0.0
        for i in range(15):
            for j in range(15):
                p = dixon_coles_prob(xg1_pred, xg2_pred, i, j, rho=rho_val)
                score_probs.append({'score': f'{i}-{j}', 'prob': p})
                if i > j: p_home_pois += p
                elif i == j: p_draw_pois += p
                else: p_away_pois += p

        # Normalize score probs
        total_prob = sum(x['prob'] for x in score_probs)
        for x in score_probs:
            x['prob'] /= total_prob
        score_probs = sorted(score_probs, key=lambda x: x['prob'], reverse=True)

        p_home_pois /= total_prob
        p_draw_pois /= total_prob
        p_away_pois /= total_prob

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
