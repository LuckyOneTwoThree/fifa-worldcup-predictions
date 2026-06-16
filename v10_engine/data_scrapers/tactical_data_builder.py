import pandas as pd
import numpy as np

def generate_tactical_data():
    print("Generating extended tactical style profiles...")
    
    tactics = {
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
        
        # New Teams
        "Germany": {"possession_avg": 62.0, "ppda": 9.5, "aerial_win_rate": 54.5},
        "Curaçao": {"possession_avg": 35.0, "ppda": 19.0, "aerial_win_rate": 42.0},
        "Ivory Coast": {"possession_avg": 45.0, "ppda": 14.0, "aerial_win_rate": 60.5},
        "Ecuador": {"possession_avg": 48.0, "ppda": 11.0, "aerial_win_rate": 51.0},
        "Netherlands": {"possession_avg": 57.0, "ppda": 10.5, "aerial_win_rate": 58.0},
        "Japan": {"possession_avg": 52.0, "ppda": 8.5, "aerial_win_rate": 38.0}, # Japan weak in air, high press
        "Sweden": {"possession_avg": 42.0, "ppda": 15.5, "aerial_win_rate": 61.0},
        "Tunisia": {"possession_avg": 40.0, "ppda": 17.0, "aerial_win_rate": 48.0}
    }
    
    df_rows = []
    for team, stats in tactics.items():
        row = {"team": team}
        row.update(stats)
        df_rows.append(row)
        
    df = pd.DataFrame(df_rows)
    df.to_csv("tactical_styles.csv", index=False)
    print("Generated extended tactical_styles.csv")

if __name__ == "__main__":
    generate_tactical_data()
