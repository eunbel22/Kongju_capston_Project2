#realtime_crawling_EHH.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import time

app = FastAPI()

# CORS 설정 (필요 시 제한 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Selenium 드라이버 설정 함수
def init_driver():
    chromedriver_autoinstaller.install()
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    return driver

@app.get("/crawl")
def crawl_kongju():
    driver = init_driver()
    result = {
        "meal": [],
    }

    try:
        driver.get("https://dormi.kongju.ac.kr/HOME/sub.php?code=041301")
        time.sleep(2)

        # 명시적 대기 (최대 10초)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-board.food tbody tr"))
        )

        rows = driver.find_elements("css selector", "table.table-board.food tbody tr")
        print(f"행 개수: {len(rows)}")
        for row in rows:
            tds = row.find_elements("tag name", "td")
            print([td.get_attribute("textContent").strip() for td in tds])
            if len(tds) >= 5:
                day_info = {
                    "day": tds[0].get_attribute("textContent").strip(),
                    "date": tds[1].get_attribute("textContent").strip(),
                    "breakfast": tds[2].get_attribute("textContent").strip(),
                    "lunch": tds[3].get_attribute("textContent").strip(),
                    "dinner": tds[4].get_attribute("textContent").strip(),
                }
                result["meal"].append(day_info)

    except Exception as e:
        return {"error": f"❗ 크롤링 중 오류 발생: {e}"}
    finally:
        driver.quit()

    return result
