import uvicorn
import os
import sys

def main():
    """FastAPI 서버 시작"""
    print("🚀 공주대학교 AI 서버를 시작합니다...")
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
        print("\n✅ 서버가 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

