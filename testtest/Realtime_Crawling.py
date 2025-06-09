#Realtime_Crawling.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import time

app = FastAPI(title="공주대학교 기숙사 식단 크롤러")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 캠퍼스별 코드 매핑
CAMPUS_CODES = {
    "cheonan": "041304",
    "dream":   "041303",
    "ehh":     "041301",
    "vb":      "041302",
    "yesan":   "041305",
}

# Selenium 드라이버 초기화

def init_driver():
    chromedriver_autoinstaller.install()
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    return webdriver.Chrome(options=options)

@app.get("/crawl/{campus}")
def crawl_meal(campus: str):
    code = CAMPUS_CODES.get(campus.lower())
    if not code:
        raise HTTPException(status_code=404, detail=f"Unknown campus: {campus}")

    url = f"https://dormi.kongju.ac.kr/HOME/sub.php?code={code}"
    driver = init_driver()
    result = {"campus": campus, "meal": []}

    try:
        driver.get(url)
        time.sleep(2)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-board.food tbody tr"))
        )

        rows = driver.find_elements(By.CSS_SELECTOR, "table.table-board.food tbody tr")
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 5:
                day_info = {
                    "day":      cols[0].text.strip(),
                    "date":     cols[1].text.strip(),
                    "breakfast":cols[2].text.strip(),
                    "lunch":    cols[3].text.strip(),
                    "dinner":   cols[4].text.strip(),
                }
                result["meal"].append(day_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"크롤링 오류: {e}")
    finally:
        driver.quit()

    return result
