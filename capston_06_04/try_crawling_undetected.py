# try_crawling_final2_internal_fixed.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime

app = FastAPI()

# CORS 전체 허용 (필요 시 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_uc_driver(proxy_url=None):
    """
    undetected_chromedriver로 Chrome을 실행하면서 탐지 회피용 옵션을 단순히 추가합니다.
    • 내부망(학교 와이파이 또는 VPN)에서만 기숙사 식단 페이지가 열립니다.
    • proxy_url이 필요하다면 "http://IP:PORT" 형식으로 넣어 주세요.
    """
    options = uc.ChromeOptions()

    # ────────────────────────────────────────────────────────────────────────
    # (1) Headless 모드 해제: 반드시 주석 처리 상태여야 Chrome 창이 뜹니다.
    #     개발/디버깅 시 최우선으로 확인하세요!
    # options.add_argument("--headless")
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # (2) 실제 사용자처럼 보이는 User-Agent
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.7151.56 Safari/537.36"
    )
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # (3) 자동화 탐지 우회를 위한 Blink Features 비활성화
    options.add_argument("--disable-blink-features=AutomationControlled")
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # (4) Proxy 설정 (내부망이 아닐 경우에만 필요)
    if proxy_url:
        options.add_argument(f"--proxy-server={proxy_url}")
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # (5) undetected-chromedriver 인스턴스 생성
    driver = uc.Chrome(options=options)
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # (6) navigator.webdriver 속성을 false로 정의해 자동화 흔적을 최대한 감춥니다.
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
    # ────────────────────────────────────────────────────────────────────────

    return driver


@app.get("/crawl")
def crawl_kongju():
    # ————————————————————————
    # proxy_url 사용 예시:
    #    - 외부망에서 학교 내부망으로 우회 접속해야 할 경우 여기에 입력
    #    - 예: proxy_url = "http://123.45.67.89:8080"
    #    - 그러나 일반적으로 학교 와이파이 사용 시 None 으로 둡니다.
    proxy_url = None

    driver = init_uc_driver(proxy_url)

    result = {
        "meal": {},       # 오늘 식단 (아침·점심·저녁)
        "sns": [],        # SNS 링크
        "hot_news": []    # HOT 뉴스(공지사항) 상위 5개
    }

    try:
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ┃ 1) 공주대 메인에서 SNS 링크 크롤
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        driver.get("https://www.kongju.ac.kr/kor/main.do")
        time.sleep(random.uniform(2.0, 3.0))
        soup_main = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup_main.select(".sns > ul > li > a"):
            href = a.get("href")
            title = a.get("title")
            if href:
                result["sns"].append({"title": title, "href": href})

        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ┃ 2) 공지사항 페이지에서 HOT 뉴스 크롤
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ┃ 3) 기숙사 식단 페이지 열기 (내부망 전용)
        # ┃    • 외부망 접속 시 “Inform…사이트가 존재하지 않습니다” 팝업만 뜹니다.
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        dormi_url = "https://dormi.kongju.ac.kr/HOME/sub.php?code=041304"
        driver.get(dormi_url)
        time.sleep(random.uniform(2.0, 3.0))

        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ┃ 3-A) 팝업(“Confirm” 또는 “확인”) 자동 클릭
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            confirm_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    # 영어 Confirm 버튼 또는 한글 확인 버튼
                    (By.XPATH, "//button[contains(text(),'Confirm') or contains(text(),'확인')]")
                )
            )
            confirm_btn.click()
            time.sleep(random.uniform(0.8, 1.5))
        except (TimeoutException, NoSuchElementException):
            # 팝업이 뜨지 않거나, 버튼에 못 맞춰도 그냥 진행
            pass

        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ┃ 3-B) 식단 테이블 로딩 대기 (최대 10초)
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-board.food"))
        )

        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ┃ 3-C) 실제 식단 테이블 파싱 (“MM월 DD일”과 비교해 오늘 메뉴 추출)
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        soup_dormi = BeautifulSoup(driver.page_source, "html.parser")
        table = soup_dormi.select_one("table.table-board.food")

        today = datetime.now()
        today_str = f"{today.month:02d}월 {today.day:02d}일"

        if not table:
            result["meal"] = "❗ 식단 테이블을 찾을 수 없습니다. (내부망 확인 필요)"
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

    except Exception as exc:
        return {"error": f"❗ 크롤링 중 오류 발생: {exc}"}

    finally:
        driver.quit()

    return result
