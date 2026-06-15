import pandas as pd
from datetime import datetime

def run(home_team, away_team, match_date_str):
    print(f"[Fatigue Radar] Calculating bio-rhythm for {home_team} vs {away_team}...")
    try:
        import sys
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shared_dir = os.path.abspath(os.path.join(base_dir, '..'))
        if shared_dir not in sys.path:
            sys.path.append(shared_dir)
        from v9_shared import load_results_csv
        
        df = load_results_csv()
        match_date = datetime.strptime(match_date_str.split(' ')[0], '%Y-%m-%d')
        
        # Filter past matches
        past_df = df[df['date'] < match_date]
        
        def get_last_match_date(team):
            team_matches = past_df[(past_df['home_team'] == team) | (past_df['away_team'] == team)]
            if not team_matches.empty:
                return team_matches['date'].max()
            return None

        h_last = get_last_match_date(home_team)
        a_last = get_last_match_date(away_team)
        
        h_rest = (match_date - h_last).days if h_last else 999
        a_rest = (match_date - a_last).days if a_last else 999
        
        def format_rest(rest_days):
            if rest_days > 30:
                return "30+ days (Off-season / Systematic Prep)"
            return rest_days

        return {
            "home_rest_days": format_rest(h_rest),
            "away_rest_days": format_rest(a_rest),
            "fatigue_warning": "HIGH" if (h_rest < 4 or a_rest < 4) else "NORMAL"
        }
    except Exception as e:
        print(f"[Fatigue Radar] Error: {e}")
        return {"home_rest_days": "Unknown", "away_rest_days": "Unknown", "fatigue_warning": "UNKNOWN"}

if __name__ == "__main__":
    print(run("Germany", "Curacao", "2026-06-15"))
