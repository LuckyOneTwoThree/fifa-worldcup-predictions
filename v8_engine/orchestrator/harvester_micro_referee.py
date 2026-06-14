def run(home_team, away_team, referee_name, home_ppda, away_ppda, strictness_index):
    print(f"[Micro-Referee Radar] Analyzing {referee_name} against team styles...")
    
    # PPDA lower means higher pressing intensity -> more fouls -> more cards if ref is strict
    home_risk = "LOW"
    away_risk = "LOW"
    
    if strictness_index > 0.6:
        if home_ppda < 10.5: home_risk = "HIGH (Red Card / Penalty Risk)"
        if away_ppda < 10.5: away_risk = "HIGH (Red Card / Penalty Risk)"
        
    analysis = "Referee is relatively lenient or teams don't press aggressively."
    if strictness_index > 0.6:
        analysis = f"{referee_name} is highly strict (Index: {strictness_index}). Teams that rely on aggressive tactical fouling will be heavily penalized."
        
    return {
        "referee": referee_name,
        "strictness": strictness_index,
        "home_foul_risk": home_risk,
        "away_foul_risk": away_risk,
        "micro_analysis": analysis
    }

if __name__ == "__main__":
    print(run("Germany", "Curacao", "Cesar Ramos", 9.5, 14.0, 0.65))
