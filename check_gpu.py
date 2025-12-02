"""
GPU 및 CUDA 확인 스크립트
RTX 4070 GPU 사용 가능 여부 확인
"""
import sys

print("=" * 60)
print("GPU 및 CUDA 환경 확인")
print("=" * 60)

# 1. Python 버전 확인
print(f"\n1. Python 버전: {sys.version}")

# 2. torch 설치 확인
try:
    import torch
    print(f"\n2. PyTorch 버전: {torch.__version__}")
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA 사용 가능: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   cuDNN 버전: {torch.backends.cudnn.version()}")
        print(f"   GPU 개수: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     이름: {torch.cuda.get_device_name(i)}")
            print(f"     메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"     Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # 간단한 GPU 테스트
        print("\n3. GPU 테스트:")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   ✅ GPU 연산 테스트 성공!")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ GPU 연산 테스트 실패: {e}")
    else:
        print("\n   ⚠️ CUDA를 사용할 수 없습니다.")
        print("   💡 GPU 버전 torch 설치 방법:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
except ImportError:
    print("\n2. ❌ PyTorch가 설치되지 않았습니다.")
    print("   💡 설치 방법:")
    print("      CPU 버전: pip install torch torchvision torchaudio")
    print("      GPU 버전: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# 3. transformers 확인
try:
    import transformers
    print(f"\n4. Transformers 버전: {transformers.__version__}")
except ImportError:
    print("\n4. ❌ Transformers가 설치되지 않았습니다.")
    print("   💡 설치: pip install transformers")

print("\n" + "=" * 60)

