# 유방암 이미지 분석 웹 애플리케이션

XAI (Explainable AI) 기술을 활용한 유방암 세포 이미지 분류 및 설명 시스템

## 📋 프로젝트 개요

이 프로젝트는 유방암 세포 이미지를 분석하여 양성(Benign)과 악성(Malignant)을 분류하고, SHAP, LIME, VLM 등의 XAI 기술을 사용하여 예측 결과를 설명하는 웹 애플리케이션입니다.

### 주요 기능

- **이미지 분류**: CNN 기반 딥러닝 모델과 전통적인 특징 추출 기반 모델을 사용한 이중 분류 시스템
- **XAI 설명**: 
  - SHAP (SHapley Additive exPlanations) - 특징 기여도 분석
  - LIME (Local Interpretable Model-agnostic Explanations) - 지역적 설명
  - VLM (Vision Language Model) - 자연어 기반 이미지 설명
- **세포 감지**: 이미지에서 세포를 자동으로 감지하고 바운딩 박스 표시
- **실시간 분석**: 웹 인터페이스를 통한 실시간 이미지 업로드 및 분석

## 🛠️ 기술 스택

- **Backend**: Flask
- **Machine Learning**: 
  - PyTorch (CNN 모델)
  - scikit-learn (Random Forest)
  - SHAP, LIME (XAI)
- **Computer Vision**: OpenCV, scikit-image
- **Frontend**: HTML, CSS, JavaScript
- **VLM**: LLaVA (Vision Language Model)

## 📦 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd test_breast_pj
```

### 2. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. PyTorch 설치 (GPU 지원)

GPU를 사용하는 경우 (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU만 사용하는 경우:

```bash
pip install torch torchvision torchaudio
```

### 5. GPU 확인 (선택사항)

```bash
python check_gpu.py
```

## 🚀 사용 방법

### 1. 애플리케이션 실행

```bash
python app.py
```

### 2. 웹 브라우저 접속

```
http://localhost:5000
```

### 3. 이미지 업로드 및 분석

1. 웹 인터페이스에서 이미지 파일(.tif, .png, .jpg 등)을 업로드
2. 자동으로 세포 감지 및 분류 수행
3. 예측 결과 및 XAI 설명 확인

## 📁 프로젝트 구조

```
test_breast_pj/
├── app.py                 # Flask 메인 애플리케이션
├── model_utils.py         # 모델 학습 및 예측 유틸리티
├── image_classifier.py    # CNN 기반 이미지 분류기
├── image_utils.py         # 이미지 처리 및 특징 추출
├── vlm_utils.py           # VLM 설명 생성 유틸리티
├── analyze_data.py        # 데이터 분석 스크립트
├── check_gpu.py           # GPU 환경 확인 스크립트
├── requirements.txt       # Python 패키지 의존성
├── templates/
│   └── index.html        # 웹 인터페이스
├── static/
│   ├── css/
│   │   └── style.css    # 스타일시트
│   └── images/          # 정적 이미지 파일
└── uploads/             # 업로드된 이미지 저장 폴더
```

## 🔧 모델 학습

### 수치 기반 모델 (Random Forest)

모델은 `kr_data.csv` 파일을 사용하여 자동으로 학습됩니다. 모델 파일이 없으면 첫 실행 시 자동으로 학습됩니다.

### CNN 이미지 분류 모델

이미지 분류 모델은 `image/Images` 폴더의 이미지를 사용하여 학습됩니다:

```python
from image_classifier import ImageClassifier

classifier = ImageClassifier()
classifier.train(image_dir="image/Images", epochs=15, batch_size=8)
```

## 📊 데이터 형식

- **입력 이미지**: TIF, PNG, JPG 형식 지원
- **학습 데이터**: `kr_data.csv` - 30개 특징과 진단 결과(B/M)

## ⚙️ 설정

### 환경 변수

필요한 경우 `.env` 파일을 생성하여 설정할 수 있습니다.

### 모델 경로

- 수치 기반 모델: `breast_cancer_model.joblib`, `scaler.joblib`
- CNN 모델: `image_classifier_model.pth`

## 🐛 문제 해결

### GPU를 사용할 수 없는 경우

1. CUDA가 설치되어 있는지 확인: `python check_gpu.py`
2. PyTorch GPU 버전 재설치:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### VLM 모델 로드 실패

VLM 모델은 첫 사용 시 자동으로 다운로드됩니다. 인터넷 연결이 필요하며, 시간이 걸릴 수 있습니다.

### 이미지에서 세포를 감지하지 못하는 경우

- 이미지 품질 확인
- 전처리 파라미터 조정 (`image_utils.py`의 `detect_cells` 함수)

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 👥 기여자

프로젝트에 기여해주신 모든 분들께 감사드립니다.

## 📧 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.

