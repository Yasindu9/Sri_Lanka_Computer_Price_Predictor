import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

def scrape_ikman_computers():
    base_url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    all_data = []
    page = 1
    target_records = 5600 # Slightly more to ensure we have 5500+
    
    print(f"Starting scraping... Target records: {target_records}", flush=True)
    
    while len(all_data) < target_records:
        url = f"{base_url}?page={page}"
        print(f"Scraping page {page}... (Current records: {len(all_data)})", flush=True)
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch page {page}. Status code: {response.status_code}")
                if response.status_code == 429:
                    print("Rate limited. Sleeping for 30s...")
                    time.sleep(30)
                    continue
                else:
                    break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            cards = soup.find_all('li')
            
            ads_on_page = 0
            for card in cards:
                a_tag = card.find('a')
                if not a_tag:
                    continue
                href = a_tag.get('href', '')
                if '/en/ad/' in href:
                    title_elem = card.find('h2')
                    title = title_elem.text.strip() if title_elem else None
                    
                    texts = list(card.stripped_strings)
                    price = None
                    location_category = None
                    member = False
                    
                    for text in texts:
                        if 'Rs' in text:
                            price = text
                        elif text == 'MEMBER':
                            member = True
                        elif ',' in text and not 'Rs' in text and len(text.split(',')) >= 2:
                            location_category = text
                    
                    # We can also capture the link
                    link = "https://ikman.lk" + href if href.startswith('/') else href
                    
                    all_data.append({
                        'Title': title,
                        'Price': price,
                        'Location_Category': location_category,
                        'Link': link,
                        'Member': member
                    })
                    ads_on_page += 1
            
            if ads_on_page == 0:
                print(f"No ads found on page {page}. Either end of results or blocked.")
                break
                
            page += 1
            time.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
            time.sleep(5)
            
    print(f"Total records scraped: {len(all_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"Original shape: {df.shape}")
    # Requirement: "remove empty rows and column"
    # Drop rows where all elements are NaN
    df.dropna(how='all', inplace=True)
    # Drop columns where all elements are NaN
    df.dropna(axis=1, how='all', inplace=True)
    print(f"Cleaned shape: {df.shape}")
    
    output_file = 'computers_data_withoutprocessed.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    scrape_ikman_computers()
