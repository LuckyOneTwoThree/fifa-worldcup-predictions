import pandas as pd
import numpy as np
import os
import hashlib

def generate_tactical_data():
    print("Generating comprehensive tactical style profiles based on squad values...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load squad values
    sv_path = os.path.join(base_dir, 'squad_values.csv')
    if not os.path.exists(sv_path):
        print(f"Error: {sv_path} not found.")
        return
        
    squad_df = pd.read_csv(sv_path)
    
    # Derive tactical stats based on squad value as a proxy
    # High squad value = dominant possession, high press (low PPDA)
    # Low squad value = low possession, low press (high PPDA)
    max_sv = squad_df['squad_value_m'].max()
    
    df_rows = []
    for _, row in squad_df.iterrows():
        team = row['team']
        sv = row['squad_value_m']
        
        # Logarithmic scaling for base stats
        sv_ratio = np.log1p(sv) / np.log1p(max_sv)
        
        # Deterministic hashing for noise
        hash_int = int(hashlib.md5(team.encode()).hexdigest(), 16)
        
        # Symmetrical noise: % 201 -> [0, 200] -> / 20.0 -> [0.0, 10.0] -> - 5.0 -> [-5.0, 5.0]
        noise_aerial = (hash_int % 201) / 20.0 - 5.0
        noise_poss = ((hash_int // 100) % 201) / 20.0 - 5.0
        noise_ppda = ((hash_int // 10000) % 201) / 40.0 - 2.5 # [-2.5, 2.5]
        
        # Possession ranges from 35% to 65% with noise
        possession = 35.0 + (30.0 * sv_ratio) + noise_poss
        
        # PPDA ranges from 18.0 (passive) down to 8.0 (aggressive high press) with noise
        ppda = 18.0 - (10.0 * sv_ratio) + noise_ppda
        
        # Taller/stronger teams (proxied slightly by SV) might be better, base 48-52
        base_aerial = 48.0 + (4.0 * sv_ratio)
        aerial_win_rate = base_aerial + noise_aerial
        
        # Original overrides for known specific styles
        overrides = {
            "Brazil": {"possession_avg": 58.5, "ppda": 10.2, "aerial_win_rate": 52.1},
            "Spain": {"possession_avg": 65.0, "ppda": 8.5, "aerial_win_rate": 45.0},
            "England": {"possession_avg": 55.2, "ppda": 11.0, "aerial_win_rate": 55.4},
            "France": {"possession_avg": 52.0, "ppda": 12.5, "aerial_win_rate": 54.0},
            "Morocco": {"possession_avg": 42.0, "ppda": 15.2, "aerial_win_rate": 50.5},
            "Australia": {"possession_avg": 44.5, "ppda": 14.1, "aerial_win_rate": 62.8}, 
            "Turkey": {"possession_avg": 51.0, "ppda": 11.5, "aerial_win_rate": 43.2},   
            "Switzerland": {"possession_avg": 53.0, "ppda": 10.8, "aerial_win_rate": 49.0},
            "Qatar": {"possession_avg": 48.0, "ppda": 16.0, "aerial_win_rate": 45.5},
            "Haiti": {"possession_avg": 38.0, "ppda": 18.5, "aerial_win_rate": 47.0},
            "Scotland": {"possession_avg": 46.0, "ppda": 13.0, "aerial_win_rate": 56.5},
            "Germany": {"possession_avg": 62.0, "ppda": 9.5, "aerial_win_rate": 54.5},
            "Curaçao": {"possession_avg": 35.0, "ppda": 19.0, "aerial_win_rate": 42.0},
            "Ivory Coast": {"possession_avg": 45.0, "ppda": 14.0, "aerial_win_rate": 60.5},
            "Ecuador": {"possession_avg": 48.0, "ppda": 11.0, "aerial_win_rate": 51.0},
            "Netherlands": {"possession_avg": 57.0, "ppda": 10.5, "aerial_win_rate": 58.0},
            "Japan": {"possession_avg": 52.0, "ppda": 8.5, "aerial_win_rate": 38.0}, 
            "Sweden": {"possession_avg": 42.0, "ppda": 15.5, "aerial_win_rate": 61.0},
            "Tunisia": {"possession_avg": 40.0, "ppda": 17.0, "aerial_win_rate": 48.0}
        }
        
        tactics = {
            "team": team,
            "possession_avg": round(possession, 1),
            "ppda": round(ppda, 1),
            "aerial_win_rate": round(aerial_win_rate, 1)
        }
        
        if team in overrides:
            tactics.update(overrides[team])
            
        df_rows.append(tactics)
        
    df = pd.DataFrame(df_rows)
    df.to_csv(os.path.join(base_dir, "tactical_styles.csv"), index=False)
    print(f"Generated extended tactical_styles.csv for {len(df)} teams.")

if __name__ == "__main__":
    generate_tactical_data()
