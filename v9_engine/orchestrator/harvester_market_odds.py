import os
import requests

def run(home_team, away_team, v8_prob_home, v8_prob_draw, v8_prob_away):
    print(f"[Smart Money Radar] Scanning market odds for {home_team} vs {away_team}...")
    
    odds_api_key = os.environ.get("ODDS_API_KEY")
    market_home, market_draw, market_away = 0, 0, 0
    source = ""

    if odds_api_key:
        try:
            # Example call to the odds api
            url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"
            params = {
                "apiKey": odds_api_key,
                "regions": "eu",
                "markets": "h2h",
                "bookmakers": "pinnacle"
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # In a real scenario we'd match the team names
                # For this MVP, if we got data, we extract it. 
                # But since the World Cup might not be active on API yet, we fallback if no matches found.
                if len(data) > 0:
                    # Parse the actual odds from The-Odds-API response
                    # Find the match containing the home_team
                    match = next((m for m in data if home_team.lower() in m['home_team'].lower() or away_team.lower() in m['away_team'].lower()), None)
                    if match and match.get('bookmakers'):
                        pinnacle = next((b for b in match['bookmakers'] if b['key'] == 'pinnacle'), match['bookmakers'][0])
                        h2h_market = next((m for m in pinnacle['markets'] if m['key'] == 'h2h'), None)
                        if h2h_market:
                            outcomes = h2h_market['outcomes']
                            # Decimal odds to implied probability
                            for outcome in outcomes:
                                name = outcome['name'].lower()
                                if name == match['home_team'].lower() or name == home_team.lower():
                                    market_home = 1.0 / outcome['price']
                                elif name == 'draw':
                                    market_draw = 1.0 / outcome['price']
                                else:
                                    market_away = 1.0 / outcome['price']
                            source = f"The-Odds-API ({pinnacle['title']})"
                        else:
                            source = "Market 'h2h' not found. Fallback to Theoretical."
                    else:
                        source = "Match found but no bookmakers. Fallback to Theoretical."
                else:
                    source = "No active API matches. Fallback to Theoretical."
            else:
                source = f"API Error {response.status_code}. Fallback to Theoretical."
        except Exception as e:
            source = f"API Request Failed ({str(e)}). Fallback to Theoretical."
    else:
        source = "ODDS_API_KEY not set. Graceful Degradation to V9 Theoretical Odds."

    # Graceful Fallback: Generate theoretical odds with a 5% bookmaker overround
    if "Fallback" in source or not source.startswith("The-Odds"):
        overround = 1.05
        market_home = min(0.99, v8_prob_home * overround)
        market_draw = min(0.99, v8_prob_draw * overround)
        market_away = min(0.99, v8_prob_away * overround)
        
        # Normalize just in case
        total = market_home + market_draw + market_away
        market_home /= total
        market_draw /= total
        market_away /= total

    # Find the Alpha (Value bet)
    alpha_home = v8_prob_home - market_home
    alpha_draw = v8_prob_draw - market_draw
    alpha_away = v8_prob_away - market_away
    
    best_alpha = max(alpha_home, alpha_draw, alpha_away)
    value_pick = ""
    if best_alpha == alpha_home: value_pick = f"{home_team} (Home)"
    elif best_alpha == alpha_draw: value_pick = "Draw"
    else: value_pick = f"{away_team} (Away)"
    
    return {
        "data_source": source,
        "market_implied_probs": {
            "home": market_home,
            "draw": market_draw,
            "away": market_away
        },
        "v8_probs": {
            "home": v8_prob_home,
            "draw": v8_prob_draw,
            "away": v8_prob_away
        },
        "value_analysis": {
            "best_value_bet": value_pick,
            "edge": f"+{best_alpha*100:.1f}%" if best_alpha > 0 else "No Edge"
        }
    }

if __name__ == "__main__":
    print(run("Germany", "Curacao", 0.877, 0.092, 0.032))
