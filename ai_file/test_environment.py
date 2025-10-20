# test_environment.py

from config import IS_SERVER, USE_AI_SAFEGUARD, SAFEGUARD_DEVICE, ENVIRONMENT
import torch

print("=" * 60)
print("환경 감지 테스트")
print("=" * 60)

print(f"\n🌍 환경: {ENVIRONMENT}")
print(f"🖥️  서버 여부: {IS_SERVER}")
print(f"🛡️  AI Safeguard: {USE_AI_SAFEGUARD}")
print(f"📍 Device: {SAFEGUARD_DEVICE}")

if torch.cuda.is_available():
    print(f"\n✅ CUDA 사용 가능")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
else:
    print(f"\n❌ CUDA 사용 불가 (CPU 모드)")

print("\n" + "=" * 60)