import requests
from bs4 import BeautifulSoup

def test_scrape():
    url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    print(f"Fetching {url}")
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Let's just find all list items that look like ads
        # Typically ikman uses something like 'ul.list--3NxGO > li' or similar
        # Let's dump some HTML to see
        cards = soup.find_all('li')
        count = 0
        for card in cards:
            a_tag = card.find('a')
            if a_tag and '/en/ad/' in a_tag.get('href', ''):
                count += 1
                title = card.find('h2')
                print("Ad:", title.text.strip() if title else "No title", "| Link:", a_tag['href'])
                print(card.prettify())
                break
        print(f"Total list items tested: {len(cards)}. Ads found: {count}")
    else:
        print("Failed to fetch.")

if __name__ == "__main__":
    test_scrape()
