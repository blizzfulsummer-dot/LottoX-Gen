# utils/result_history.py

import json
from playwright.sync_api import sync_playwright

def scrape_pcso_results():
    url = "https://www.pcso.gov.ph/SearchLottoResult.aspx"
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # set False for debugging
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.pcso.gov.ph/",
            }
        )

        page = context.new_page()

        print(f"Navigating to {url} ...")
        page.goto(url, wait_until="networkidle", timeout=60000)

        # Debug: dump first 500 chars of HTML
        html = page.content()
        print("=== First 500 chars of page ===")
        print(html[:500])

        # Try to find result table
        tables = page.query_selector_all("table")
        if not tables:
            print("❌ No <table> found. Probably still blocked.")
            browser.close()
            return results

        rows = tables[0].query_selector_all("tr")
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) >= 3:
                game_name = cells[0].inner_text().strip()
                draw_date = cells[1].inner_text().strip()
                result_numbers = cells[2].inner_text().strip()

                results.append({
                    "game": game_name,
                    "date": draw_date,
                    "results": result_numbers,
                })

        browser.close()

    return results


if __name__ == "__main__":
    data = scrape_pcso_results()

    if data:
        with open("pcso_results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(data)} results to pcso_results.json")
    else:
        print("❌ No results scraped. Check if site is blocking us.")
