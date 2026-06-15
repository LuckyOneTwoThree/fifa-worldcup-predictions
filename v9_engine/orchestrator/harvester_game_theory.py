from datetime import datetime

def run(home_team, away_team, tournament, date_str):
    print(f"[Game Theory Radar] Analyzing tournament motivation for {tournament} on {date_str}...")
    
    try:
        match_date = datetime.strptime(date_str, '%Y-%m-%d')
        # MD1 and MD2 approx dates
        md2_end = datetime.strptime('2026-06-22', '%Y-%m-%d')
        group_stage_end = datetime.strptime('2026-06-26', '%Y-%m-%d')
    except ValueError:
        return {"error": "Invalid date format."}
        
    if "World Cup" in tournament:
        if match_date <= md2_end:
            return {
                "motivation_index": "HIGH",
                "biscotto_risk": "NONE",
                "analysis": f"World Cup Group Stage (MD1/MD2). Both {home_team} and {away_team} are fighting for points. No biscotto risk."
            }
        elif match_date <= group_stage_end:
            return {
                "motivation_index": "HIGH",
                "biscotto_risk": "MEDIUM",
                "analysis": f"World Cup Group Stage (MD3). Watch out for 'Biscotto' (mutually beneficial draw) or conservative play."
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
    print(run("Netherlands", "Japan", "FIFA World Cup", "2026-06-25"))
