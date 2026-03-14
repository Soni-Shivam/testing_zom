from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('http://127.0.0.1:8000/', wait_until='networkidle')
    print("BODY INNER HTML:")
    print(page.evaluate("document.body.innerHTML")[:1000])
    browser.close()
