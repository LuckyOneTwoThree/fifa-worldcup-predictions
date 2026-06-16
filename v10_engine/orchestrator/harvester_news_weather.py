import json
from duckduckgo_search import DDGS

def harvest_news(team_name):
    print(f"[News Radar] Scanning latest news for {team_name}...")
    results = []
    try:
        with DDGS() as ddgs:
            # We search for injuries, morale, and general news
            query = f"{team_name} national football team injury news morale"
            news = list(ddgs.news(query, max_results=3))
            results = news
    except Exception as e:
        print(f"[News Radar] Warning: Failed to fetch for {team_name}. Error: {e}")
    return results

def harvest_weather(match_city="Unknown"):
    import hashlib
    hash_val = int(hashlib.md5(match_city.encode('utf-8')).hexdigest(), 16)
    
    # Base states
    weather_states = ["Clear sky", "Cloudy", "Rain", "Storm", "Clear sky"]
    w = weather_states[hash_val % len(weather_states)]
    
    # Override for known hot cities during summer
    hot_cities = ["Miami", "Houston", "Dallas", "Atlanta", "Monterrey", "Guadalajara", "Los Angeles"]
    if any(city in match_city for city in hot_cities):
        if hash_val % 3 == 0:
            w = "Extreme Heat (35C+)"
        elif hash_val % 3 == 1:
            w = "Hot and Humid"
            
    return f"{w}, optimal conditions for {match_city} (Simulated)"

def run(home_team, away_team, city="Stadium"):
    return {
        "home_news": harvest_news(home_team),
        "away_news": harvest_news(away_team),
        "weather": harvest_weather(city)
    }

if __name__ == "__main__":
    print(run("Germany", "Curacao"))
