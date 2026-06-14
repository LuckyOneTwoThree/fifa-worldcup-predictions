from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time

def extract_value(val_str):
    val_str = val_str.lower().replace('€', '').strip()
    if 'm' in val_str:
        return float(val_str.replace('m', ''))
    elif 'k' in val_str:
        return float(val_str.replace('k', '')) / 1000
    elif 'bn' in val_str:
        return float(val_str.replace('bn', '')) * 1000
    return 50.0 # Default if parsing fails

def scrape_transfermarkt():
    print("Launching Playwright to scrape Transfermarkt...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        page = context.new_page()
        
        print("Navigating to National Teams page...")
        # Page 1
        page.goto('https://www.transfermarkt.com/nationalmannschaften/marktwertetop/statistik', timeout=60000)
        time.sleep(5)
        
        squad_data = []
        
        for p_num in range(1, 10): # Scrape first 9 pages
            print(f"Scraping page {p_num}...")
            soup = BeautifulSoup(page.content(), 'html.parser')
            
            table = soup.find('table', {'class': 'items'})
            if not table:
                print("Could not find table, maybe Cloudflare blocked us?")
                break
                
            rows = table.find('tbody').find_all('tr', recursive=False)
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 5:
                    team_a = cols[1].find('a')
                    if team_a:
                        team_name = team_a.get('title')
                        value_str = cols[5].text.strip()
                        squad_value_m = extract_value(value_str)
                        squad_data.append({"team": team_name, "squad_value_m": squad_value_m})
                        
            # Go to next page
            next_btn = page.locator("li.tm-pagination__list-item--icon-next-page a")
            if next_btn.count() > 0 and next_btn.is_visible():
                next_btn.click()
                time.sleep(3)
            else:
                break
                
        browser.close()
        
    if squad_data:
        df = pd.DataFrame(squad_data)
        df.to_csv("squad_values.csv", index=False)
        print(f"Successfully saved {len(df)} squad values to squad_values.csv")
    else:
        print("Failed to scrape any data.")

if __name__ == "__main__":
    scrape_transfermarkt()
