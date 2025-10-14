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
import logging
import traceback

# 로깅 설정 추가
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("Chrome 드라이버 초기화 시작")
    try:
        chromedriver_autoinstaller.install()
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options)
        logger.info("Chrome 드라이버 초기화 성공")
        return driver
    except Exception as e:
        logger.error(f"Chrome 드라이버 초기화 실패: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise

@app.get("/crawl/{campus}")
def crawl_meal(campus: str):
    logger.info(f"크롤링 요청 - 캠퍼스: {campus}")
    
    code = CAMPUS_CODES.get(campus.lower())
    if not code:
        logger.error(f"알 수 없는 캠퍼스: {campus}")
        raise HTTPException(status_code=404, detail=f"Unknown campus: {campus}")

    url = f"https://dormi.kongju.ac.kr/HOME/sub.php?code={code}"
    logger.info(f"접속할 URL: {url}")
    
    driver = init_driver()
    result = {"campus": campus, "meal": []}

    try:
        logger.info("웹페이지 접속 중...")
        driver.get(url)
        time.sleep(2)
        logger.info("페이지 로드 완료, 테이블 요소 대기 중...")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-board.food tbody tr"))
        )
        logger.info("테이블 요소 로드 완료")

        rows = driver.find_elements(By.CSS_SELECTOR, "table.table-board.food tbody tr")
        logger.info(f"찾은 테이블 행 개수: {len(rows)}")
        
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
        
        logger.info(f"크롤링 완료 - 수집된 식단 데이터: {len(result['meal'])}개")

    except Exception as e:
        logger.error(f"크롤링 중 오류 발생: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"크롤링 오류: {e}")
    finally:
        logger.info("드라이버 종료 중...")
        driver.quit()
        logger.info("드라이버 종료 완료")

    return result
