from datetime import datetime

def run(home_team, away_team, tournament, date_str):
    print(f"[Game Theory Radar] Analyzing tournament motivation for {tournament} on {date_str}...")
    
    try:
        match_date = datetime.strptime(date_str, '%Y-%m-%d')
        group_stage_end = datetime.strptime('2026-06-26', '%Y-%m-%d')
    except ValueError:
        return {"error": "Invalid date format."}
        
    if "World Cup" in tournament:
        if match_date <= group_stage_end:
            return {
                "motivation_index": "HIGH",
                "biscotto_risk": "MEDIUM",
                "analysis": f"World Cup Group Stage match. Both {home_team} and {away_team} need points. Watch out for 'Biscotto' (mutually beneficial draw) in the 3rd matchday."
            }
        else:
            return {
                "motivation_index": "EXTREME",
                "biscotto_risk": "NONE",
                "analysis": "World Cup Knockout phase. Teams will fight to the death. Tactical conservatism is expected until a goal is scored."
            }
    elif "Friendly" in tournament:
        return {
            "motivation_index": "LOW",
            "biscotto_risk": "N/A",
            "analysis": "International friendly. High risk of rotation and low competitive motivation."
        }
    else:
        return {
            "motivation_index": "NORMAL",
            "biscotto_risk": "N/A",
            "analysis": "Standard competitive match."
        }

if __name__ == "__main__":
    print(run("Netherlands", "Japan", "FIFA World Cup", "2026-06-15"))
