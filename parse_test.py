import requests
from bs4 import BeautifulSoup

def test_parse():
    url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print("Failed to fetch")
        return
    soup = BeautifulSoup(resp.content, 'html.parser')
    cards = soup.find_all('li')
    for card in cards:
        a_tag = card.find('a')
        if a_tag and '/en/ad/' in a_tag.get('href', ''):
            title = card.find('h2')
            print("Title:", title.text.strip() if title else "None")
            for text in card.stripped_strings:
                print(text)
            break

if __name__ == "__main__":
    test_parse()
