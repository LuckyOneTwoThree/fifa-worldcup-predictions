from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def parse_fbref_table(page, url, table_id):
    print(f"Navigating to {url}...")
    page.goto(url, timeout=60000)
    time.sleep(6) # FBref requires slow scraping
    
    html = page.content()
    # FBref hides some tables in comments, but playwright might render them if visible
    # We can also regex search for the table id
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'id': table_id})
    
    # If not found, look inside comments
    if not table:
        import bs4
        comments = soup.find_all(string=lambda text: isinstance(text, bs4.Comment))
        for c in comments:
            if table_id in c:
                c_soup = BeautifulSoup(c, 'html.parser')
                table = c_soup.find('table', {'id': table_id})
                if table: break

    data = {}
    if table:
        rows = table.find('tbody').find_all('tr')
        for row in rows:
            th = row.find('th', {'data-stat': 'team'})
            if not th: continue
            team_name = th.text.strip()
            
            # Extract data
            tds = row.find_all('td')
            row_data = {td.get('data-stat'): td.text.strip() for td in tds}
            data[team_name] = row_data
            
    return data

def scrape_fbref():
    print("Launching Playwright to scrape FBref...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        page = context.new_page()
        
        # 1. Possession
        pos_data = parse_fbref_table(page, 'https://fbref.com/en/comps/1/possession/World-Cup-Stats', 'stats_squads_possession_for')
        
        # 2. Defense (for PPDA proxy: tackles in att 3rd)
        def_data = parse_fbref_table(page, 'https://fbref.com/en/comps/1/defense/World-Cup-Stats', 'stats_squads_defense_for')
        
        # 3. Misc (Aerials)
        misc_data = parse_fbref_table(page, 'https://fbref.com/en/comps/1/misc/World-Cup-Stats', 'stats_squads_misc_for')
        
        browser.close()
        
    # Merge Data
    all_teams = set(pos_data.keys()) | set(def_data.keys()) | set(misc_data.keys())
    final_data = []
    
    for team in all_teams:
        # FBref appends country codes like 'ar Argentina', we split it
        clean_team = ' '.join(team.split(' ')[1:]) if ' ' in team else team
        if not clean_team: clean_team = team
        
        # Default fallbacks
        pos = 50.0
        ppda = 12.0
        aerial = 50.0
        
        if team in pos_data and 'possession' in pos_data[team] and pos_data[team]['possession']:
            pos = float(pos_data[team]['possession'])
            
        if team in def_data and 'tackles_att_3rd' in def_data[team] and def_data[team]['tackles_att_3rd']:
            # Mock PPDA inverse logic: more tackles in att 3rd = lower PPDA
            tackles = float(def_data[team]['tackles_att_3rd'])
            ppda = max(5.0, 20.0 - tackles) 
            
        if team in misc_data and 'aerials_won_pct' in misc_data[team] and misc_data[team]['aerials_won_pct']:
            aerial = float(misc_data[team]['aerials_won_pct'])
            
        final_data.append({
            "team": clean_team,
            "possession_avg": pos,
            "ppda": ppda,
            "aerial_win_rate": aerial
        })
        
    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv("tactical_styles.csv", index=False)
        print(f"Successfully saved {len(df)} tactical profiles to tactical_styles.csv")
    else:
        print("Failed to scrape FBref data.")

if __name__ == "__main__":
    scrape_fbref()
