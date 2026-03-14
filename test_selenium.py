import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)
driver.get("http://127.0.0.1:8000/")
time.sleep(2)
driver.save_screenshot("test_gui.png")

logs = driver.get_log("browser")
for log in logs:
    print("Browser Log:", log)

print("Title:", driver.title)
source = driver.page_source
print("Source length:", len(source))
driver.quit()
