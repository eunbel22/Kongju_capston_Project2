# try_crawling_updated.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime

app = FastAPI()

# CORS 설정(필요 시 수정 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_uc_driver(proxy_url=None):
    options = uc.ChromeOptions()

    # = headless 해제: 반드시 주석 처리해서 Chrome 창이 뜨도록 합니다 =
    # options.add_argument("--headless")

    # 사람처럼 보이는 User-Agent
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.7151.56 Safari/537.36"
    )
    # 자동화 탐지 플래그 제거
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    # navigator.webdriver 숨기기
    options.add_argument("--disable-infobars")
    if proxy_url:
        options.add_argument(f"--proxy-server={proxy_url}")

    driver = uc.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false
                });
            """
        },
    )
    return driver

@app.get("/crawl")
def crawl_kongju():
    proxy_url = None  # 내부망 연결이므로 보통 None
    driver = init_uc_driver(proxy_url)

    result = {
        "meal": {},
        "sns": [],
        "hot_news": []
    }

    try:
        # 1) 공주대 메인 → SNS 크롤링
        driver.get("https://www.kongju.ac.kr/kor/main.do")
        time.sleep(random.uniform(2.0, 3.0))
        soup_main = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup_main.select(".sns > ul > li > a"):
            href = a.get("href")
            title = a.get("title")
            if href:
                result["sns"].append({"title": title, "href": href})

        # 2) 공지사항 페이지 → HOT 뉴스 크롤링
        driver.get("https://www.kongju.ac.kr/kor/395/subview.do")
        time.sleep(random.uniform(2.5, 4.0))
        soup_news = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup_news.select(".board-list ul li a")[:5]:
            href = a.get("href")
            txt = a.get_text(strip=True)
            if href:
                result["hot_news"].append({
                    "title": txt,
                    "href": "https://www.kongju.ac.kr" + href
                })

        # 3) 기숙사 식단 페이지 열기 (내부망만 가능)
        dormi_url = "https://dormi.kongju.ac.kr/HOME/sub.php?code=041304"
        driver.get(dormi_url)
        # 팝업은 닫지 않음 → 내부망이라면 곧장 테이블이 뜰 것
        time.sleep(random.uniform(2.0, 3.0))

        # 테이블 로딩 대기 (최대 10초)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-board.food"))
        )

        soup_dormi = BeautifulSoup(driver.page_source, "html.parser")
        table = soup_dormi.select_one("table.table-board.food")

        # 오늘 날짜(“MM월 DD일”) 생성
        today = datetime.now()
        today_str = f"{today.month:02d}월 {today.day:02d}일"

        if not table:
            result["meal"] = "❗ 식단 테이블을 찾을 수 없습니다. (내부망에서만 확인 가능)"
        else:
            found = False
            for row in table.select("tbody tr"):
                date_td = row.select_one("td[data-mqtitle='date']")
                if date_td and date_td.get_text(strip=True) == today_str:
                    bf = row.select_one("td[data-mqtitle='breakfast']").get_text(strip=True)
                    ln = row.select_one("td[data-mqtitle='lunch']").get_text(strip=True)
                    dn = row.select_one("td[data-mqtitle='dinner']").get_text(strip=True)
                    result["meal"] = {"아침": bf, "점심": ln, "저녁": dn}
                    found = True
                    break
            if not found:
                result["meal"] = f"❗ 오늘({today_str}) 식단이 테이블에 없습니다."

    except Exception as e:
        return {"error": f"❗ 크롤링 중 오류 발생: {e}"}
    finally:
        driver.quit()

    return result
