"""
VLM (Vision Language Model) 유틸리티
- LLaVA / BLIP / GIT 기반 이미지 설명
- AI 예측 + SHAP/LIME 특징 요약
- 이미지 기반 세포 형태 정량 분석(analyze_morphology)까지 붙여서
  병리학적으로 읽기 쉬운 리포트 형태로 출력
"""

from PIL import Image
import os

# 이미지 처리 모듈에서 형태 정량 분석 함수만 가져옴
from image_utils import analyze_morphology


class VLMExplainer:
    """
    LLaVA / BLIP / GIT 기반 이미지 설명 + 모델 특징 요약 + 형태 정량 분석 리포트
    """

    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device=None):
        self.model_name = model_name
        self.device = None
        self.model = None
        self.processor = None

        # 1) torch / device 설정
        try:
            import torch
            # torch 버전 체크
            torch_version = torch.__version__
            major, minor = map(int, torch_version.split('.')[:2])
            if major < 2 or (major == 2 and minor < 6):
                print(f"⚠️ torch 버전이 낮습니다 (현재: {torch_version}, 필요: >= 2.6)")
                print("💡 pip install --upgrade torch torchvision torchaudio")
            
            if device is not None:
                self.device = device
            else:
                if torch.cuda.is_available():
                    self.device = "cuda"
                    print("✅ GPU 감지:", torch.cuda.get_device_name(0))
                    print("✅ CUDA 버전:", torch.version.cuda)
                else:
                    self.device = "cpu"
                    print("⚠️ GPU 미사용, CPU 사용")
        except Exception as e:
            print(f"⚠️ torch 로드 실패: {e}")
            print("💡 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return

        # 2) 모델 로드
        self._load_model()

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            print(f"VLM 모델 로드 시도 (device={self.device})")

            # 작은 모델부터 시도 (메모리 문제 방지)
            # torch 버전 문제로 blip-image-captioning-base는 제외
            candidates = [
                "microsoft/git-base",
                "llava-hf/llava-1.5-7b-hf",
            ]

            last_error = None
            for name in candidates:
                try:
                    print(f"🔍 모델 후보: {name}")
                    # use_fast=False 명시적으로 설정하여 경고 방지
                    processor = AutoProcessor.from_pretrained(
                        name, trust_remote_code=True, use_fast=False
                    )
                    kwargs = {
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                    }
                    if self.device == "cuda":
                        kwargs["torch_dtype"] = torch.float16
                        kwargs["device_map"] = "auto"
                    else:
                        kwargs["torch_dtype"] = torch.float32

                    model = AutoModelForVision2Seq.from_pretrained(name, **kwargs)
                    if self.device == "cpu":
                        model = model.to(self.device)

                    self.model_name = name
                    self.processor = processor
                    self.model = model.eval()

                    print(f"✅ VLM 로드 성공: {name}")
                    return
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    # torch 버전 관련 오류는 무시하고 다음 모델 시도
                    if "torch.load" in error_msg or "torch 2.6" in error_msg.lower():
                        print(f"⚠️ {name} 로드 실패 (torch 버전 문제): 다음 모델 시도...")
                    else:
                        print(f"⚠️ {name} 로드 실패: {error_msg[:200]}")

            print("❌ 사용 가능한 VLM 모델을 찾지 못했습니다.")
            if last_error:
                print("  마지막 오류:", last_error)

        except Exception as e:
            print(f"⚠️ transformers/모델 로드 실패: {e}")
            self.model = None
            self.processor = None

    # ------------------------------------------------------------------
    # 유효성 체크
    # ------------------------------------------------------------------
    def is_available(self):
        return self.model is not None and self.processor is not None

    # ------------------------------------------------------------------
    # 내부: 형태 정량 분석 (VLM 프롬프트 & 리포트용)
    # ------------------------------------------------------------------
    def _compute_morphology_summary(self, image_path):
        try:
            return analyze_morphology(image_path)
        except Exception as e:
            print(f"⚠️ 형태 정량 분석 중 오류: {e}")
            return None

    # ------------------------------------------------------------------
    # 메인: 이미지 설명 생성
    # ------------------------------------------------------------------
    def explain_image(self, image_path, prediction_result=None, features_info=None):
        """
        이미지에 대한 설명 생성:
        - VLM 캡션 (의학적 사전 지식 + 정량 분석 포함 프롬프트)
        - 예측/특징 요약
        - 세포 형태 정량 분석 리포트
        ==> 하나의 markdown 문자열로 반환
        """
        # 0) 형태 정량 분석
        morph_summary = self._compute_morphology_summary(image_path)

        vlm_text = None

        if self.is_available():
            try:
                vlm_text = self._run_vlm(
                    image_path, prediction_result, features_info, morph_summary
                )
            except Exception as e:
                print(f"⚠️ VLM 추론 중 오류: {e}")
                vlm_text = None

        if not vlm_text:
            vlm_text = self._generate_fallback_visual_description()

        summary_text = self._build_prediction_feature_summary(
            prediction_result, features_info
        )
        morph_md = self._build_morphology_summary_md(morph_summary)

        final_md = "## 이미지 기반 설명 (VLM)\n\n"
        final_md += vlm_text.strip() + "\n\n"
        final_md += "---\n\n"
        final_md += summary_text.strip()
        if morph_md:
            final_md += "\n\n---\n\n" + morph_md.strip()

        print("\n================= [VLM 최종 설명] =================")
        print(final_md[:1500])
        print("==================================================\n")

        return final_md

    # ------------------------------------------------------------------
    # 내부: 실제 VLM 호출
    # ------------------------------------------------------------------
    def _run_vlm(
        self,
        image_path,
        prediction_result=None,
        features_info=None,
        morph_summary=None
    ):
        import torch

        # 이미지 로드
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                print(f"⚠️ 이미지 경로 없음: {image_path}")
                return None
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path  # 이미 PIL.Image 인스턴스로 넘어오는 경우

        # 해상도 제한(메모리 보호)
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            ratio = min(max_size / image.size[0], max_size / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        prompt = self._build_prompt(prediction_result, features_info, morph_summary)

        print(f"📸 VLM 이미지 분석 시작 (size={image.size}, model={self.model_name})")

        # BLIP / GIT 계열: 캡셔닝 방식
        if "blip" in self.model_name.lower() or "git" in self.model_name.lower():
            # 프롬프트를 더 짧게 만들기
            short_prompt = self._build_short_prompt(prediction_result, features_info, morph_summary)
            text_prompt = (
                "이것은 유방암 세포 현미경 사진입니다. "
                "세포의 형태, 크기, 배열, 핵의 특징을 한국어로 설명해주세요.\n\n"
                + short_prompt
            )
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                # 입력 길이 확인 및 max_new_tokens 설정
                input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                # 모델의 최대 길이를 고려하여 더 보수적으로 설정
                max_model_length = 512  # GIT/BLIP 모델의 일반적인 최대 길이
                if input_length > max_model_length - 100:
                    # 입력이 너무 길면 잘라내기
                    print(f"⚠️ 입력 토큰 길이({input_length})가 너무 깁니다. 프롬프트를 단축합니다.")
                    # 더 짧은 프롬프트로 재시도
                    text_prompt = "이것은 유방암 세포 현미경 사진입니다. 세포의 형태와 핵의 특징을 한국어로 간단히 설명해주세요."
                    inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                
                max_new_tokens = min(256, max(50, max_model_length - input_length - 50))
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                )

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            if text.startswith(text_prompt):
                text = text[len(text_prompt):].strip()

            print(f"📝 VLM 원본 캡션: {text[:200]}...")
            return text if text else None

        # LLaVA 계열: 대화형 프롬프트
        else:
            # 프롬프트를 더 짧게 만들기
            short_prompt = self._build_short_prompt(prediction_result, features_info, morph_summary)
            inputs = self.processor(
                images=image,
                text=short_prompt,
                return_tensors="pt"
            )
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                # 입력 길이 확인 및 max_new_tokens 설정
                input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                # 모델의 최대 길이를 고려하여 더 보수적으로 설정
                max_model_length = 2048  # LLaVA 모델의 일반적인 최대 길이
                if input_length > max_model_length - 200:
                    # 입력이 너무 길면 더 짧은 프롬프트로 재시도
                    print(f"⚠️ 입력 토큰 길이({input_length})가 너무 깁니다. 프롬프트를 단축합니다.")
                    short_prompt = "이것은 유방암 세포 현미경 사진입니다. 세포의 형태와 핵의 특징을 한국어로 간단히 설명해주세요."
                    inputs = self.processor(images=image, text=short_prompt, return_tensors="pt")
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                
                max_new_tokens = min(1024, max(50, max_model_length - input_length - 100))
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=(
                        self.processor.tokenizer.eos_token_id
                        if hasattr(self.processor, "tokenizer")
                        else None
                    ),
                )

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            print(f"📝 VLM 원본 응답: {text[:200]}...")
            text = text.replace("</s>", "").replace("<pad>", "").strip()
            return text if text else None

    # ------------------------------------------------------------------
    # 프롬프트 구성 (유방암 세포 특징 + 예측결과 + 형태 정량 분석)
    # ------------------------------------------------------------------
    def _build_prompt(self, prediction_result=None, features_info=None,
                      morph_summary=None):
        # 병리학적 설명을 위한 base 텍스트
        base = """이것은 유방암 세포 현미경 사진입니다. 반드시 한국어로 답변해주세요.

【양성(Benign) 세포의 전형적 특징】
- 비교적 균일한 세포 크기와 모양 (저도/경도 세포 이형성)
- 세포가 군집을 이루더라도 경계가 비교적 매끄럽고, 핵/세포질 비율(N/C ratio)이 낮음
- 핵 크기가 서로 비슷하고, 염색 강도(chromatin)가 고르게 분포하며 과도하게 어둡지 않음
- 분열상(mitotic figure)이 거의 보이지 않거나 드묾
- 세포 사이 간격이 어느 정도 유지되고, 조직 구조가 비교적 보존됨

【악성(Malignant) 세포의 전형적 특징】
- 크기가 서로 다른 세포들이 섞여 있는 다형성(nuclear pleomorphism)
- 핵/세포질 비율(N/C ratio)이 증가하고, 핵이 비정상적으로 크거나 찌그러져 보임
- 핵막(nuclear membrane)이 불규칙하고, 염색 강도가 진하거나 거칠게 분포(coarse chromatin)
- 뚜렷한 핵소체(prominent nucleoli)가 관찰되는 경우가 많음
- 세포들이 조밀하게 군집하거나, 시트(sheet) 형태 또는 무질서한 배열을 보이며 주변 조직과의 경계가 불명확해짐
- 분열상(mitosis)이 증가하며, 비정형 분열상(atypical mitosis)이 관찰될 수 있음
"""

        # 이미지 기반 정량 분석 결과를 프롬프트에 추가
        if morph_summary and morph_summary.get("total_cells", 0) > 0:
            total = morph_summary["total_cells"]
            ir_cnt = morph_summary["irregular_boundary_cells"]
            ir_ratio = morph_summary["irregular_boundary_ratio"] * 100
            lg_cnt = morph_summary["large_cells"]
            lg_ratio = morph_summary["large_cell_ratio"] * 100
            hc_cnt = morph_summary["high_contrast_cells"]
            hc_ratio = morph_summary["high_contrast_ratio"] * 100

            base += "\n【이미지 전처리 기반 세포 형태 정량 분석 요약】\n"
            base += f"- 감지된 세포 수: {total}개\n"
            base += f"- 경계가 들쑥날쑥한(불규칙한) 세포: 약 {ir_ratio:.1f}% ({ir_cnt}개)\n"
            base += f"- 상대적으로 큰 세포(면적 상위 25%): 약 {lg_ratio:.1f}% ({lg_cnt}개)\n"
            base += f"- 텍스처 대비(명암 변화)가 높은 세포: 약 {hc_ratio:.1f}% ({hc_cnt}개)\n"
            base += "- 일반적으로 악성으로 갈수록 경계가 불규칙한 세포와 큰 세포, 핵 염색이 진한 세포의 비율이 증가합니다.\n"

        # 예측 결과
        if prediction_result:
            pred = prediction_result.get("prediction", "")
            malignant_prob = prediction_result.get("malignant_prob", 0)
            benign_prob = prediction_result.get("benign_prob", 0)
            base += f"\n【AI 예측 결과(참고용)】\nAI 예측: {pred} (악성 {malignant_prob:.1%}, 양성 {benign_prob:.1%})\n"

        # XAI 상위 특징 이름 힌트
        if features_info and features_info.get("top_features"):
            base += "\n【모델이 중요하게 본 정량 특징(상위 일부)】\n"
            names = [f["feature"] for f in features_info["top_features"][:3]]
            base += "- 예: " + ", ".join(names) + "\n"
            base += "이들 특징은 종양의 크기(radius/area/perimeter), 경계 불규칙성(concavity/convexity), 핵 주변 질감(texture contrast/homogeneity) 등을 반영합니다.\n"

        # 의학적 추론을 요구하는 구체 질문
        base += """
이미지를 직접 관찰하고, 위의 병리학적 특징과 정량 분석 결과를 참고하여 다음을 한국어로 **차분하게, 병리과 의사가 구두 소견을 기술하듯** 자세히 설명해주세요. 가능하면 추상적인 표현 대신, 실제로 눈에 보이는 양상을 근거로 서술해주세요.

1. **세포 크기와 모양**
   - 세포 크기가 전반적으로 균일한지, 크기가 매우 다양한지(경도/중등도/고도 세포 이형성 중 어디에 가까운지) 서술해주세요.
   - 원형에 가까운 세포가 많은지, 찌그러진/길게 늘어난 세포가 많은지 설명해주세요.

2. **핵의 형태와 염색 양상**
   - 핵의 크기와 모양이 서로 비슷한지, 크기 차이가 큰지(핵 다형성) 기술해주세요.
   - 핵 염색 강도(chromatin)가 균일한지, 일부 세포에서 유난히 진하거나 거친 패턴이 보이는지,
     뚜렷한 핵소체가 보이는 세포가 많은지 관찰한 대로 설명해주세요.

3. **핵/세포질 비율(N/C ratio)과 세포 배열**
   - 세포질에 비해 핵이 차지하는 비율이 전반적으로 낮은지/높은지, 악성에 가까운 패턴인지 판단해 보세요.
   - 세포들이 느슨하게 퍼져 있는지, 군집·시트(sheet)·중첩된 배열로 밀집되어 있는지,
     단일 세포(single cell)들이 많이 떨어져 보이는 양상인지 서술해주세요.

4. **경계 불규칙성과 주변 조직과의 관계**
   - 세포 또는 군집의 외곽 경계가 매끄러운지, 톱니 모양/불규칙한지 서술해주세요.
   - 정량 분석에서 제시된 '경계가 들쑥날쑥한 세포 비율'이 실제 눈으로 보이는 경계 불규칙성과 잘 맞는지,
     혹은 특정 영역에 집중되어 있는지 설명해주세요.

5. **세포 크기 분포와 다형성**
   - '큰 세포 비율'이 높은 경우, 실제로 큰 세포들이 어느 영역에 집중되는지,
     작은 세포와 섞여 있는 다형성 패턴으로 보이는지 관찰한 대로 설명해주세요.

6. **염색 농도와 텍스처**
   - 고대비 텍스처 세포 비율이 높은 경우, 핵 또는 세포질에서 명암 차이가 두드러지는 부위가 있는지,
     괴사(central necrosis)나 염증성 세포 침윤이 의심되는 영역이 있는지 간단히 언급해주세요.

7. **종합적 인상**
   - 위의 소견을 종합했을 때, 전형적인 양성 변화에 가까운지, '비정형이 동반된 양성 변화'인지,
     혹은 악성 병변에 더 가까운 인상인지 **진단명은 언급하지 말고**, 
     "악성에 가까운 소견", "전형적 양성에 가까운 소견" 등의 표현으로 정리해주세요.
"""
        return base

    # ------------------------------------------------------------------
    # 짧은 프롬프트 생성 (토큰 길이 제한용)
    # ------------------------------------------------------------------
    def _build_short_prompt(self, prediction_result=None, features_info=None,
                           morph_summary=None):
        """토큰 길이 제한을 위한 짧은 프롬프트"""
        prompt = "이것은 유방암 세포 현미경 사진입니다. 한국어로 설명해주세요.\n\n"
        
        # 예측 결과만 간단히 추가
        if prediction_result:
            pred = prediction_result.get("prediction", "")
            malignant_prob = prediction_result.get("malignant_prob", 0)
            benign_prob = prediction_result.get("benign_prob", 0)
            prompt += f"예측: {pred} (악성 {malignant_prob:.1%}, 양성 {benign_prob:.1%})\n"
        
        # 정량 분석 요약만 간단히 추가
        if morph_summary and morph_summary.get("total_cells", 0) > 0:
            total = morph_summary["total_cells"]
            ir_ratio = morph_summary["irregular_boundary_ratio"] * 100
            lg_ratio = morph_summary["large_cell_ratio"] * 100
            prompt += f"세포 {total}개, 불규칙 경계 {ir_ratio:.1f}%, 큰 세포 {lg_ratio:.1f}%\n"
        
        prompt += "\n세포의 형태, 크기, 핵의 특징을 간단히 설명해주세요."
        return prompt

    # ------------------------------------------------------------------
    # 예측 + 특징 요약 섹션
    # ------------------------------------------------------------------
    def _build_prediction_feature_summary(self, prediction_result, features_info):
        md = "## 모델 예측 및 특징 요약\n\n"

        # 예측 결과
        if prediction_result:
            pred = prediction_result.get("prediction", "알 수 없음")
            prob = prediction_result.get("probability", 0)
            malignant_prob = prediction_result.get("malignant_prob", 0)
            benign_prob = prediction_result.get("benign_prob", 0)

            md += "### 예측 결과\n"
            md += f"- **진단 방향(모델 출력)**: {pred}\n"
            md += f"- **전체 확률**: {prob:.2%}\n"
            md += f"- **악성 쪽으로 기운 확률**: {malignant_prob:.2%}\n"
            md += f"- **양성 쪽으로 기운 확률**: {benign_prob:.2%}\n\n"

        # 병리학적 의미를 붙이기 위한 간단한 맵
        patho_notes = {
            "radius_mean": "종양 덩어리의 평균 크기(반경)에 해당하며, 악성 병변일수록 전반적인 크기가 커지는 경향과 관련됩니다.",
            "radius_worst": "가장 큰 세포/덩어리의 크기를 반영하며, 고악성 병변에서 대형 세포 또는 종괴가 동반되는 소견과 연결될 수 있습니다.",
            "area_mean": "세포 또는 군집의 평균 면적을 나타내며, 덩어리가 클수록 악성 가능성이 높아지는 경향과 연관됩니다.",
            "area_worst": "가장 큰 군집/세포의 면적을 반영하며, 국소적으로 매우 큰 병변이 있는지와 관련됩니다.",
            "perimeter_mean": "세포/군집의 둘레 길이를 나타내며, 불규칙한 경계가 많을수록 길어지는 경향이 있습니다.",
            "concavity_mean": "세포/군집 경계의 '파고든 정도'를 나타내며, 악성에서 더 불규칙한 경계와 연관됩니다.",
            "concave_points_mean": "경계가 안쪽으로 꺾이는 포인트 개수로, 톱니 모양의 불규칙한 경계를 반영합니다.",
            "texture_contrast_mean": "핵/세포질 내 명암 대비를 반영하며, 핵 염색이 불균일하거나 진한 세포가 많을수록 증가할 수 있습니다.",
            "texture_homogeneity_mean": "텍스처의 균질성을 의미하며, 값이 낮을수록 조직이 더 불균질하고 복잡한 패턴을 보일 수 있습니다.",
            "mean_intensity_mean": "평균 밝기로, 염색 강도 및 세포 밀도와 관련된 간접 지표로 사용할 수 있습니다.",
        }

        if features_info and features_info.get("top_features"):
            md += "### 중요 특징 기여도 (SHAP/LIME 기반)\n\n"
            md += "모델이 예측을 내릴 때 특히 크게 참고한 특징들과 그 병리학적 의미를 함께 정리했습니다:\n\n"

            for i, feat in enumerate(features_info["top_features"][:5], 1):
                name = feat["feature"]
                contrib = feat["contribution"]
                direction = "값이 클수록 **악성 쪽**으로 기여" if contrib > 0 else "값이 클수록 **양성 쪽**으로 기여"

                # 병리 코멘트
                note = None
                # 기본 키 그대로가 없는 경우를 위해 접두사/접미사 제거 후 매칭 시도
                if name in patho_notes:
                    note = patho_notes[name]
                else:
                    base_key = (
                        name.replace("_mean", "")
                            .replace("_se", "")
                            .replace("_worst", "")
                    )
                    for k in patho_notes.keys():
                        if base_key in k:
                            note = patho_notes[k]
                            break

                md += f"{i}. **{name}**: 기여도 {abs(contrib):.4f} ({direction})\n"
                if note:
                    md += f"   - 병리학적 해석: {note}\n"

            md += "\n"

        if not prediction_result and not (features_info and features_info.get("top_features")):
            md += "- 예측 결과 및 특징 정보가 전달되지 않아, 모델 관점의 해석은 제공할 수 없습니다.\n"

        return md

    # ------------------------------------------------------------------
    # 형태 정량 분석 리포트 섹션
    # ------------------------------------------------------------------
    def _build_morphology_summary_md(self, morph_summary):
        if not morph_summary or morph_summary.get("total_cells", 0) == 0:
            return ""

        total = morph_summary["total_cells"]
        ir_cnt = morph_summary["irregular_boundary_cells"]
        ir_ratio = morph_summary["irregular_boundary_ratio"] * 100
        lg_cnt = morph_summary["large_cells"]
        lg_ratio = morph_summary["large_cell_ratio"] * 100
        hc_cnt = morph_summary["high_contrast_cells"]
        hc_ratio = morph_summary["high_contrast_ratio"] * 100

        md = "## 세포 형태 정량 분석 (이미지 기반)\n\n"
        md += f"- 감지된 세포는 총 **{total}개**입니다.\n"
        md += f"- 이 중 **경계가 들쭉날쑥한 세포**(원형도 감소, convexity/solidity 감소)는 약 **{ir_ratio:.1f}%**인 **{ir_cnt}개**입니다.\n"
        md += f"- **상대적으로 큰 세포(면적 상위 25%)**는 약 **{lg_ratio:.1f}%**인 **{lg_cnt}개**입니다.\n"
        md += f"- **텍스처 대비(명암 변화)가 높은 세포**는 약 **{hc_ratio:.1f}%**인 **{hc_cnt}개**입니다.\n\n"

        # 정량값을 이용해 병리학적 인상 코멘트(완전 진단은 아님)
        md += "### 병리학적 관점에서의 해석적 코멘트\n\n"

        # 경계 불규칙 비율 코멘트
        if ir_ratio < 15:
            md += "- 경계가 불규칙한 세포 비율이 비교적 낮아, 전체적으로는 **경도 이형성 또는 양성 변화에 가까운 경계 패턴**으로 볼 수 있습니다.\n"
        elif ir_ratio < 40:
            md += "- 경계가 불규칙한 세포가 일정 비율 존재하여, **부분적으로 비정형 세포가 섞여 있는 중간 정도의 이형성 패턴**으로 해석될 수 있습니다.\n"
        else:
            md += "- 경계가 들쑥날쑥한 세포 비율이 상당히 높아, **고도 이형성 또는 악성에 가까운 경계 불규칙 패턴**이 관찰될 가능성이 있습니다.\n"

        # 큰 세포 비율 코멘트
        if lg_ratio < 15:
            md += "- 큰 세포 비율이 낮아, 세포 크기 분포는 비교적 균일한 편으로 보입니다.\n"
        elif lg_ratio < 40:
            md += "- 큰 세포가 눈에 띄게 존재하지만 극단적으로 많지는 않아, **중등도 정도의 세포 크기 변화(pleomorphism)**로 볼 수 있습니다.\n"
        else:
            md += "- 큰 세포 비율이 높아, **세포 크기 다형성이 뚜렷한 패턴**으로 악성 변화와 더 잘 맞는 양상일 수 있습니다.\n"

        # 고대비 텍스처 비율 코멘트
        if hc_ratio < 15:
            md += "- 텍스처 대비가 높은 세포 비율이 낮아, 핵 염색 강도나 명암 차이는 비교적 균일한 편으로 추정됩니다.\n"
        elif hc_ratio < 40:
            md += "- 텍스처 대비가 높은 세포가 부분적으로 존재하여, 일부 영역에서 **핵 염색 강도가 더 진하거나 불균일한 소견**이 있을 수 있습니다.\n"
        else:
            md += "- 텍스처 대비가 높은 세포 비율이 높아, **핵 염색이 불균일하거나 거친 크로마틴 패턴**이 광범위하게 존재할 가능성이 있습니다.\n"

        md += (
            "\n일반적으로 악성 병변에서는\n"
            "- 경계가 불규칙한 세포 비율이 증가하고,\n"
            "- 크기가 큰 세포와 작은 세포가 섞여 나타나며(세포/핵 다형성),\n"
            "- 핵 염색이 더 진하거나 불균일한 경우가 많습니다.\n"
            "위 정량 분석은 이러한 병리학적 패턴이 어느 정도 존재하는지 **정량적 단서**를 제공하며, "
            "실제 슬라이드 판독 시 육안 소견과 함께 참고되는 보조 지표로 활용될 수 있습니다.\n"
            "※ 이 분석은 연구·교육 목적의 정량 요약이며, 단독으로 임상 진단에 사용되어서는 안 됩니다.\n"
        )

        return md

    # ------------------------------------------------------------------
    # VLM이 아예 안될 때 사용할 기본 설명
    # ------------------------------------------------------------------
    def _generate_fallback_visual_description(self):
        return (
            "VLM 모델을 로드하지 못해, 이미지를 직접 분석한 설명을 제공하지 못했습니다. "
            "다만 아래 예측 결과, 특징 중요도, 세포 형태 정량 분석을 함께 참고하여 "
            "세포의 크기·형태·밀집도·경계 불규칙성, 핵 염색 패턴 등을 병리학적으로 해석할 수 있습니다."
        )


# 전역 싱글톤 인스턴스
_vlm_instance = None


def get_vlm_explainer():
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = VLMExplainer()
    return _vlm_instance
