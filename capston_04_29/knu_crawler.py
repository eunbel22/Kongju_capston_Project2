# crawler.py

import os
import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import time
import json

# Selenium WebDriver 설정
options = Options()
options.add_argument('--headless')  # 브라우저 창을 띄우지 않음
options.add_argument('--disable-gpu')  # GPU 비활성화 (Linux 환경에서 필요)
options.add_argument('--no-sandbox')  # 권한 문제 방지 (Linux 환경에서 필요)

# ChromeDriver 자동 설치 및 설정
chromedriver_autoinstaller.install()
driver = webdriver.Chrome(options=options)

# 공주대학교 메인 페이지 URL
base_url = "https://www.kongju.ac.kr"


# --- 공주대학교 크롤링 함수 ---
def crawl_kongju_website():
    driver.get(base_url)
    time.sleep(3)  # 페이지 로딩 대기

    # 메뉴 탐색: BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(driver.page_source, "lxml")

    # 모든 메뉴 링크 추출 (예: <a> 태그에서 href 속성을 가져옴)
    menu_links = soup.select("nav a")  # 네비게이션 바에서 <a> 태그 선택

    menu_data = []

    # 메뉴별 데이터 크롤링 및 저장
    for link in menu_links:
        menu_name = link.get_text(strip=True)  # 메뉴 이름 추출
        href = link.get("href")  # 링크 URL 추출

        if not href or href.startswith("#"):  # 유효하지 않은 링크는 제외
            continue

        # 절대 URL 생성 (상대 경로 처리)
        if not href.startswith("http"):
            href = base_url + href

        try:
            # 해당 메뉴 페이지 접속
            driver.get(href)
            time.sleep(3)  # 페이지 로딩 대기

            # 페이지 소스 가져오기 및 텍스트 정리
            page_soup = BeautifulSoup(driver.page_source, "lxml")
            text_content = page_soup.get_text()
            cleaned_text = "\n".join(line.strip() for line in text_content.splitlines() if line.strip())

            menu_data.append({"menu": menu_name, "content": cleaned_text})

        except Exception as e:
            print(f"오류 발생: {menu_name} ({href}) - {e}")

    return menu_data


# 공주대학교 데이터 크롤링 결과 저장
if __name__ == "__main__":
    kongju_data = crawl_kongju_website()
    with open("kongju_data.json", "w", encoding="utf-8") as f:
        json.dump(kongju_data, f, ensure_ascii=False, indent=4)

    print("공주대학교 크롤링 완료!")

    # WebDriver 종료
    driver.quit()
