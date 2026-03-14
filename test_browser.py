from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Browser Error: {err}"))
    response = page.goto("http://127.0.0.1:8000/")
    page.wait_for_timeout(2000)
    print("Page Title:", page.title())
    print("DOM Snippet:", page.content()[:500])
    browser.close()
