import uvicorn
import os
import sys
import time
import subprocess
import requests
from multiprocessing import Process

def start_crawling_server():
    """크롤링 서버 시작 (백그라운드)"""
    print("🕷️  크롤링 서버를 시작합니다... (포트: 8001)")
    try:
        uvicorn.run(
            "Realtime_Crawling:app",
            host="127.0.0.1",
            port=8001,
            reload=False,  # 백그라운드에서는 reload 비활성화
            log_level="warning"  # 로그 레벨을 warning으로 설정하여 출력 최소화
        )
    except Exception as e:
        print(f"❌ 크롤링 서버 실행 중 오류 발생: {e}")

def wait_for_server(url, max_attempts=30, delay=1):
    """서버가 준비될 때까지 대기"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            time.sleep(delay)
            print(f"   서버 대기 중... ({attempt + 1}/{max_attempts})")
    
    return False

def start_ai_server():
    """AI 서버 시작 (메인 프로세스)"""
    print("🤖 AI 서버를 시작합니다... (포트: 8000)")
    print("📍 서버 주소: http://127.0.0.1:8000")
    print("⏹️  서버 종료: Ctrl+C")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "ai_server:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✅ AI 서버가 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ AI 서버 실행 중 오류 발생: {e}")
        sys.exit(1)

def main():
    """서버 시작 메인 함수"""
    print("🚀 공주대학교 포티 서버를 시작합니다...")
    print("=" * 50)
    
    # 1. 크롤링 서버를 별도 프로세스로 시작
    crawling_process = Process(target=start_crawling_server)
    crawling_process.daemon = True  # 메인 프로세스 종료 시 함께 종료
    crawling_process.start()
    
    # 2. 크롤링 서버가 준비될 때까지 대기
    print("⏳ 크롤링 서버 준비 대기 중...")
    crawling_ready = wait_for_server("http://127.0.0.1:8001/docs")
    
    if crawling_ready:
        print("✅ 크롤링 서버 준비 완료!")
    else:
        print("⚠️  크롤링 서버 준비 시간 초과 (계속 진행)")
    
    print("-" * 50)
    
    # 3. AI 서버 시작 (메인 프로세스)
    try:
        start_ai_server()
    finally:
        # 프로그램 종료 시 크롤링 서버도 종료
        if crawling_process.is_alive():
            print("🛑 크롤링 서버를 종료합니다...")
            crawling_process.terminate()
            crawling_process.join(timeout=5)
            if crawling_process.is_alive():
                crawling_process.kill()

if __name__ == "__main__":
    main()
