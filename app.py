"""
유방암 이미지 분석 웹 애플리케이션
XAI (SHAP)를 사용한 설명 가능한 AI 시스템
"""
from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
import io
from PIL import Image
import numpy as np
import cv2
from model_utils import BreastCancerModel
from image_utils import extract_image_features, draw_bbox_with_labels, load_image, preprocess_image, detect_cells
from image_classifier import ImageClassifier
# vlm_utils는 지연 로딩 (torch DLL 문제 방지)
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안 함
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 한글 인코딩 설정
app.config['JSON_AS_ASCII'] = False

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 모델 로드
print("모델 로드 중...")
model = BreastCancerModel()
try:
    model.load_model()
except:
    print("모델 학습 중...")
    model.train_model()
    model.save_model()

# 이미지 분류 모델 로드 (CNN 기반)
print("이미지 분류 모델 로드 중...")
image_classifier = ImageClassifier()
image_model_loaded = image_classifier.load_model()

if not image_model_loaded:
    print("⚠️ 이미지 분류 모델이 없습니다. 학습을 시작합니다...")
    try:
        # GPU 사용 가능 여부 확인
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CPU 사용 (학습이 느릴 수 있습니다)")
        
        # 이미지에서 직접 학습
        success = image_classifier.train(image_dir="image/Images", epochs=15, batch_size=8)
        if success:
            print("✅ 이미지 분류 모델 학습 완료!")
        else:
            print("⚠️ 이미지 분류 모델 학습 실패")
            image_classifier = None
    except Exception as e:
        print(f"⚠️ 이미지 분류 모델 학습 중 오류: {e}")
        image_classifier = None
else:
    print("✅ 이미지 분류 모델 로드 완료")

# VLM 모델은 필요할 때만 로드 (지연 로딩)
vlm_explainer = None
print("⚠️ VLM 모델은 첫 사용 시 로드됩니다 (시간이 걸릴 수 있습니다)")

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """이미지 예측 및 설명 생성"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 없습니다.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        # 파일 저장
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 파일명에서 실제 레이블 추출 (검증용)
        actual_label = None
        if 'benign' in filename.lower():
            actual_label = '양성(B)'
        elif 'malignant' in filename.lower():
            actual_label = '악성(M)'
        
        # 이미지 특징 추출 (세포 감지용)
        print(f"이미지 처리 중: {filename}")
        if actual_label:
            print(f"파일명 기반 실제 레이블: {actual_label}")
        
        features, cells, processed_img = extract_image_features(filepath)
        
        # 예측 수행 - 이미지 분류 모델 우선 사용
        if image_classifier is not None and image_classifier.model is not None:
            try:
                print("이미지 분류 모델(CNN) 사용 중...")
                prediction = image_classifier.predict(filepath)
                prediction_method = "CNN (이미지 직접 학습)"
                use_cnn = True
            except Exception as e:
                print(f"⚠️ CNN 예측 실패: {e}, 수치 기반 모델 사용")
                if features is None:
                    return jsonify({'error': '이미지에서 세포를 감지할 수 없습니다.'}), 400
                prediction = model.predict(features)
                prediction_method = "수치 기반 (특징 추출)"
                use_cnn = False
        else:
            print("수치 기반 모델 사용 중...")
            if features is None:
                return jsonify({'error': '이미지에서 세포를 감지할 수 없습니다.'}), 400
            prediction = model.predict(features)
            prediction_method = "수치 기반 (특징 추출)"
            use_cnn = False
        
        # 실제 레이블과 비교하여 정확도 확인
        if actual_label:
            predicted_label = prediction['prediction']
            is_correct = actual_label == predicted_label
            print(f"예측: {predicted_label}, 실제: {actual_label}, 정확도: {'✓' if is_correct else '✗'}")
            if not is_correct:
                print(f"⚠️ 분류 오류 감지! 예측 확률: {prediction['probability']:.2%}")
        
        # SHAP 설명 생성
        if use_cnn:
            # CNN 사용 시 수치 기반 설명도 함께 제공 (세포가 감지된 경우)
            if features is not None and len(cells) > 0:
                explanation = model.explain(features)
            else:
                explanation = {
                    'top_features': [],
                    'all_features': [],
                    'method': 'CNN'
                }
        else:
            explanation = model.explain(features)
        
        # 각 세포별 예측 (간단한 버전 - 전체 이미지 예측을 각 세포에 적용)
        cell_predictions = []
        cell_explanations = []
        for i, cell in enumerate(cells):
            cell_predictions.append(prediction)
            cell_explanations.append(explanation)
        
        # 박스 정보 추출 (동적 렌더링을 위해)
        boxes_data = []
        for i, cell in enumerate(cells):
            x, y, w, h = cell['bbox']
            pred = cell_predictions[i] if i < len(cell_predictions) else None
            expl = cell_explanations[i] if i < len(cell_explanations) else None
            
            # 예측 정보
            is_malignant = '악성' in pred['prediction'] if pred else False
            prob = pred['probability'] if pred else 0
            label = pred['prediction'] if pred else '세포'
            color = '#ff6b6b' if is_malignant else '#51cf66' if pred else '#888888'
            
            # 특징 정보
            top_features = []
            if expl and 'top_features' in expl:
                top_features = expl['top_features'][:3]
            
            # Contour 정보 추출
            contours_data = []
            try:
                if 'full_mask' in cell:
                    mask = cell['full_mask']
                else:
                    mask = cell['mask']
                    full_mask = np.zeros((processed_img.shape[0], processed_img.shape[1]), dtype=np.uint8)
                    full_mask[y:y+h, x:x+w] = mask
                    mask = full_mask
                
                mask_uint8 = np.uint8(mask * 255)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if len(contour) >= 3:
                        # Contour 좌표를 리스트로 변환
                        contour_points = contour.reshape(-1, 2).tolist()
                        contours_data.append(contour_points)
            except:
                pass
            
            boxes_data.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'label': label,
                'probability': float(prob),
                'is_malignant': is_malignant,
                'color': color,
                'top_features': top_features,
                'contours': contours_data
            })
        
        # 결과 이미지 생성 (바운딩 박스만, segmentation 제외)
        from image_utils import draw_bbox_with_labels, draw_instance_segmentation, draw_mask_based_segmentation, load_mask_image
        result_img = draw_bbox_with_labels(processed_img, cells, cell_predictions, cell_explanations, show_segmentation=False)
        
        # Instance segmentation 시각화 이미지 생성
        segmentation_img = draw_instance_segmentation(processed_img, cells, cell_predictions)
        
        # 마스크 파일 기반 시각화 이미지 생성
        mask_based_img = None
        mask_base64 = None
        mask_img = load_mask_image(filepath)
        if mask_img is not None:
            try:
                # 원본 이미지 로드 (컬러 가능)
                original_img = load_image(filepath)
                if len(original_img.shape) == 3:
                    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                else:
                    original_gray = original_img
                
                # 마스크 기반 시각화 생성
                mask_based_img = draw_mask_based_segmentation(original_gray, mask_img, cell_predictions)
                
                # 마스크 기반 이미지 인코딩
                mask_buffer = io.BytesIO()
                mask_based_img.savefig(mask_buffer, format='png', bbox_inches='tight', dpi=150)
                mask_buffer.seek(0)
                mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
                plt.close(mask_based_img)
                print("✅ 마스크 기반 시각화 이미지 생성 완료")
            except Exception as e:
                print(f"⚠️ 마스크 기반 시각화 생성 실패: {e}")
                import traceback
                traceback.print_exc()
                mask_base64 = None
        
        # 이미지를 base64로 인코딩
        img_buffer = io.BytesIO()
        result_img.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(result_img)
        
        # Segmentation 이미지 인코딩
        seg_buffer = io.BytesIO()
        segmentation_img.savefig(seg_buffer, format='png', bbox_inches='tight', dpi=150)
        seg_buffer.seek(0)
        seg_base64 = base64.b64encode(seg_buffer.getvalue()).decode('utf-8')
        plt.close(segmentation_img)
        
        # 특징 값 딕셔너리 생성
        feature_dict = {}
        for i, feat_name in enumerate(model.feature_names):
            feature_dict[feat_name] = float(features[i])
        
        # 설명에 method 추가
        if 'method' not in explanation:
            explanation['method'] = 'Feature Importance'
        
        # VLM 설명 생성 (지연 로딩)
        vlm_explanation = None
        try:
            # VLM이 아직 로드되지 않았으면 로드 시도
            global vlm_explainer
            if vlm_explainer is None:
                print("VLM 모델 로드 중... (처음 사용 시 시간이 걸릴 수 있습니다)")
                from vlm_utils import VLMExplainer
                try:
                    vlm_explainer = VLMExplainer()
                    if vlm_explainer.model is None:
                        print("⚠️ VLM 모델을 사용할 수 없습니다.")
                        vlm_explainer = None
                except Exception as e:
                    print(f"⚠️ VLM 모델 로드 실패: {e}")
                    vlm_explainer = None
            
            if vlm_explainer is not None and vlm_explainer.model is not None:
                try:
                    print("VLM 설명 생성 중...")
                    vlm_explanation = vlm_explainer.explain_image(filepath, prediction, explanation)
                    print(f"✅ VLM 설명 생성 완료: {vlm_explanation[:100] if vlm_explanation else 'None'}...")
                    if not vlm_explanation:
                        vlm_explanation = "VLM 설명이 생성되지 않았습니다."
                except Exception as e:
                    print(f"⚠️ VLM 설명 생성 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    vlm_explanation = "VLM 설명 생성 중 오류가 발생했습니다."
            else:
                vlm_explanation = "VLM 모델을 사용할 수 없습니다. 이미지 분류 결과를 기반으로 분석해주세요."
        except Exception as e:
            print(f"⚠️ VLM 처리 중 오류: {e}")
            vlm_explanation = None
        
        # 응답 데이터 구성
        response_data = {
            'prediction': prediction['prediction'],
            'probability': prediction['probability'],
            'malignant_prob': prediction['malignant_prob'],
            'benign_prob': prediction['benign_prob'],
            'explanation': {
                'top_features': explanation.get('top_features', []),
                'all_features': explanation.get('all_features', []),
                'method': explanation.get('method', prediction_method)
            },
            'features': feature_dict if image_classifier is None else {},
            'image': f"data:image/png;base64,{img_base64}",
            'base_image': f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', processed_img)[1]).decode('utf-8')}",  # 원본 이미지 (박스 없음)
            'boxes_data': boxes_data,  # 박스 정보 (동적 렌더링용)
            'segmentation_image': f"data:image/png;base64,{seg_base64}",
            'mask_based_image': f"data:image/png;base64,{mask_base64}" if mask_base64 else None,
            'num_cells': len(cells),
            'vlm_explanation': vlm_explanation if vlm_explanation else None,
            'actual_label': actual_label,  # 실제 레이블 (파일명 기반)
            'prediction_method': prediction_method  # 사용된 모델 방법
        }
        
        # VLM 설명 로그
        if vlm_explanation:
            print(f"📤 VLM 설명 생성 완료 (길이: {len(vlm_explanation)})")
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

@app.route('/sample_images')
def sample_images():
    """샘플 이미지 목록 반환"""
    image_dir = 'image/Images'
    sample_images = []
    
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith('.tif') and not filename.endswith('.xml'):
                # 파일명에서 benign/malignant 판단
                is_malignant = 'malignant' in filename.lower()
                sample_images.append({
                    'filename': filename,
                    'path': os.path.join(image_dir, filename),
                    'label': '악성' if is_malignant else '양성'
                })
    
    return jsonify({'images': sample_images[:10]})  # 최대 10개만 반환

@app.route('/comparison_visualization')
def comparison_visualization():
    """비교 시각화 이미지 생성 (의학적 스타일)"""
    try:
        from image_utils import load_image, preprocess_image, detect_cells, load_mask_image, detect_cells_from_mask
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        
        # 샘플 이미지 선택 (양성/악성 혼합)
        image_dir = 'image/Images'
        masks_dir = 'image/Masks'
        
        if not os.path.exists(image_dir):
            return jsonify({'error': '이미지 폴더를 찾을 수 없습니다.'}), 404
        
        # 샘플 이미지 파일 목록
        sample_files = [
            'ytma49_111003_benign1_ccd.tif',
            'ytma49_111003_malignant1_ccd.tif',
            'ytma49_111003_benign2_ccd.tif',
            'ytma49_111003_malignant2_ccd.tif',
            'ytma49_111003_benign3_ccd.tif',
        ]
        
        # 실제 존재하는 파일만 필터링
        available_samples = []
        for filename in sample_files:
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                available_samples.append(filename)
            if len(available_samples) >= 5:
                break
        
        if len(available_samples) == 0:
            return jsonify({'error': '샘플 이미지를 찾을 수 없습니다.'}), 404
        
        # 비교 시각화 생성
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(len(available_samples) + 1, 5, figure=fig, 
                              hspace=0.4, wspace=0.3, 
                              left=0.05, right=0.95, top=0.95, bottom=0.05,
                              height_ratios=[0.5] + [1] * len(available_samples))
        
        # 컬럼 헤더
        column_headers = ['Image', 'Red Channel', 'Ground Truth', 'Ours', 'Farsight']
        for col_idx, header in enumerate(column_headers):
            ax_header = fig.add_subplot(gs[0, col_idx])
            ax_header.text(0.5, 0.5, header, ha='center', va='center', 
                          fontsize=16, fontweight='bold', transform=ax_header.transAxes,
                          bbox=dict(boxstyle='round', facecolor='#667eea', alpha=0.2, edgecolor='#667eea', linewidth=2))
            ax_header.axis('off')
        
        for row_idx, filename in enumerate(available_samples):
            img_path = os.path.join(image_dir, filename)
            
            # 각 컬럼을 개별적으로 처리하여 오류 격리
            axes = []
            for col_idx in range(5):
                axes.append(fig.add_subplot(gs[row_idx + 1, col_idx]))
            
            try:
                # 원본 이미지 로드
                img = load_image(img_path)
                if img is None or img.size == 0:
                    raise ValueError(f"이미지 로드 실패: {img_path}")
                
                # 이미지 형태 확인 및 변환
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    gray = img.copy()
                    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    raise ValueError(f"지원하지 않는 이미지 형태: {img.shape}")
                
                # 이미지 크기 확인
                if gray.shape[0] == 0 or gray.shape[1] == 0:
                    raise ValueError(f"이미지 크기가 0입니다: {gray.shape}")
                
                # 이미지 크기 조정 (일관된 크기로)
                target_size = (400, 400)
                gray_resized = cv2.resize(gray, target_size)
                img_rgb_resized = cv2.resize(img_rgb, target_size)
                
                # 1. Image (원본)
                try:
                    axes[0].imshow(img_rgb_resized)
                    axes[0].axis('off')
                    label = 'Benign' if 'benign' in filename.lower() else 'Malignant'
                    axes[0].set_title(f'Sample {row_idx + 1} ({label})', fontsize=11, fontweight='bold', pad=8)
                except Exception as e:
                    print(f"⚠️ Image 컬럼 오류: {e}")
                    axes[0].text(0.5, 0.5, 'Error', ha='center', va='center', 
                                transform=axes[0].transAxes, fontsize=12, color='red', fontweight='bold')
                    axes[0].axis('off')
                
                # 2. Red Channel (그레이스케일)
                try:
                    axes[1].imshow(gray_resized, cmap='gray')
                    axes[1].axis('off')
                except Exception as e:
                    print(f"⚠️ Red Channel 컬럼 오류: {e}")
                    axes[1].text(0.5, 0.5, 'Error', ha='center', va='center', 
                                transform=axes[1].transAxes, fontsize=12, color='red', fontweight='bold')
                    axes[1].axis('off')
                
                # 3. Ground Truth (마스크 기반)
                try:
                    mask_img = load_mask_image(img_path)
                    if mask_img is not None and mask_img.size > 0:
                        # 마스크 이미지 전처리
                        if len(mask_img.shape) == 3:
                            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                        elif len(mask_img.shape) == 2:
                            mask_gray = mask_img.copy()
                        else:
                            mask_gray = None
                        
                        if mask_gray is not None and mask_gray.shape[0] > 0 and mask_gray.shape[1] > 0:
                            # 마스크 크기 조정
                            mask_gray_resized = cv2.resize(mask_gray, target_size)
                            
                            # 마스크에서 고유한 세포 ID 찾기
                            unique_values = np.unique(mask_gray_resized)
                            unique_values = unique_values[unique_values > 0]
                            
                            # 각 세포를 흰색으로 표시 (Ground Truth 스타일)
                            mask_display = np.zeros_like(gray_resized)
                            for cell_id in unique_values:
                                mask_display[mask_gray_resized == cell_id] = 255
                            
                            axes[2].imshow(gray_resized, cmap='gray', alpha=0.5)
                            axes[2].imshow(mask_display, cmap='gray', alpha=0.8)
                        else:
                            axes[2].imshow(gray_resized, cmap='gray')
                            axes[2].text(0.5, 0.5, 'Invalid Mask', ha='center', va='center', 
                                        transform=axes[2].transAxes, fontsize=10, color='orange', fontweight='bold')
                    else:
                        axes[2].imshow(gray_resized, cmap='gray')
                        axes[2].text(0.5, 0.5, 'No Mask', ha='center', va='center', 
                                    transform=axes[2].transAxes, fontsize=12, color='red', fontweight='bold')
                    axes[2].axis('off')
                except Exception as e:
                    print(f"⚠️ Ground Truth 컬럼 오류: {e}")
                    axes[2].imshow(gray_resized, cmap='gray')
                    axes[2].text(0.5, 0.5, 'Error', ha='center', va='center', 
                                transform=axes[2].transAxes, fontsize=12, color='red', fontweight='bold')
                    axes[2].axis('off')
                
                # 4. Ours (현재 detect_cells 방법)
                try:
                    cells_ours, processed, _ = detect_cells(img)
                    axes[3].imshow(gray_resized, cmap='gray')
                    
                    # 각 세포를 다른 색으로 표시
                    if len(cells_ours) > 0:
                        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(cells_ours))))
                        overlay = np.zeros((gray_resized.shape[0], gray_resized.shape[1], 3), dtype=np.float32)
                        
                        for idx, cell in enumerate(cells_ours[:50]):  # 최대 50개만 표시
                            try:
                                x, y, w, h = cell['bbox']
                                # 원본 이미지 크기에 맞춰 좌표 스케일링
                                if gray.shape[0] > 0 and gray.shape[1] > 0:
                                    scale_x = target_size[0] / gray.shape[1]
                                    scale_y = target_size[1] / gray.shape[0]
                                else:
                                    continue
                                
                                if 'full_mask' in cell:
                                    mask = cell['full_mask']
                                    if mask is not None and mask.size > 0:
                                        mask_resized = cv2.resize(mask.astype(np.uint8), target_size)
                                    else:
                                        continue
                                else:
                                    mask = cell.get('mask')
                                    if mask is not None and mask.size > 0:
                                        full_mask = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
                                        if y + h <= gray.shape[0] and x + w <= gray.shape[1]:
                                            full_mask[y:y+h, x:x+w] = mask
                                        mask_resized = cv2.resize(full_mask, target_size)
                                    else:
                                        continue
                                
                                color = colors[idx % len(colors)]
                                mask_bool = mask_resized > 0
                                for c in range(3):
                                    overlay[:, :, c][mask_bool] = color[c]
                            except Exception as e:
                                continue  # 개별 셀 처리 오류는 무시
                        
                        if np.any(overlay > 0):
                            axes[3].imshow(overlay, alpha=0.6)
                except Exception as e:
                    print(f"⚠️ Ours 컬럼 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    axes[3].imshow(gray_resized, cmap='gray')
                    axes[3].text(0.5, 0.5, 'Error', ha='center', va='center', 
                                transform=axes[3].transAxes, fontsize=12, color='red', fontweight='bold')
                axes[3].axis('off')
                
                # 5. Farsight (간단한 connectedComponents 방법)
                try:
                    processed_simple, _ = preprocess_image(img)
                    if processed_simple is not None and processed_simple.size > 0:
                        _, binary = cv2.threshold(processed_simple, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kernel = np.ones((3, 3), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                        
                        binary_resized = cv2.resize(binary, target_size)
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_resized, connectivity=8)
                        
                        axes[4].imshow(gray_resized, cmap='gray')
                        
                        # 각 컴포넌트를 다른 색으로 표시
                        if num_labels > 1:
                            colors_farsight = plt.cm.Set3(np.linspace(0, 1, min(12, num_labels)))
                            overlay_farsight = np.zeros((gray_resized.shape[0], gray_resized.shape[1], 3), dtype=np.float32)
                            
                            for i in range(1, min(num_labels, 50)):  # 최대 50개만 표시
                                try:
                                    area = stats[i, cv2.CC_STAT_AREA]
                                    if 20 <= area <= 20000:
                                        mask_farsight = (labels == i).astype(np.uint8)
                                        color = colors_farsight[i % len(colors_farsight)]
                                        mask_bool = mask_farsight > 0
                                        for c in range(3):
                                            overlay_farsight[:, :, c][mask_bool] = color[c]
                                except Exception:
                                    continue
                            
                            if np.any(overlay_farsight > 0):
                                axes[4].imshow(overlay_farsight, alpha=0.6)
                    else:
                        axes[4].imshow(gray_resized, cmap='gray')
                        axes[4].text(0.5, 0.5, 'Process Error', ha='center', va='center', 
                                    transform=axes[4].transAxes, fontsize=10, color='orange', fontweight='bold')
                except Exception as e:
                    print(f"⚠️ Farsight 컬럼 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    axes[4].imshow(gray_resized, cmap='gray')
                    axes[4].text(0.5, 0.5, 'Error', ha='center', va='center', 
                                transform=axes[4].transAxes, fontsize=12, color='red', fontweight='bold')
                axes[4].axis('off')
                
            except Exception as e:
                print(f"⚠️ 샘플 {filename} 처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 발생 시 빈 이미지 표시
                for col_idx in range(5):
                    axes[col_idx].text(0.5, 0.5, f'Error\n{str(e)[:30]}', ha='center', va='center', 
                                       transform=axes[col_idx].transAxes, fontsize=10, color='red', fontweight='bold')
                    axes[col_idx].axis('off')
        
        # 이미지를 base64로 인코딩
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'비교 시각화 생성 중 오류: {str(e)}'}), 500

@app.route('/vlm_explain', methods=['POST'])
def vlm_explain():
    """VLM을 사용한 이미지 설명 생성 (별도 엔드포인트)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 없습니다.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        # 예측 결과 정보 받기
        prediction_result = request.form.get('prediction_result')
        features_info = request.form.get('features_info')
        
        # 파일 저장
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"vlm_{filename}")
        file.save(filepath)
        
        # VLM 설명 생성 (지연 로딩)
        global vlm_explainer
        if vlm_explainer is None:
            try:
                from vlm_utils import get_vlm_explainer
                vlm_explainer = get_vlm_explainer()
            except Exception as e:
                print(f"⚠️ VLM 모델 로드 실패: {e}")
                vlm_explainer = None
        
        if vlm_explainer and vlm_explainer.is_available():
            import json
            pred_data = json.loads(prediction_result) if prediction_result else None
            feat_data = json.loads(features_info) if features_info else None
            
            explanation = vlm_explainer.explain_image(
                filepath,
                prediction_result=pred_data,
                features_info=feat_data
            )
            
            return jsonify({'explanation': explanation})
        else:
            return jsonify({'error': 'VLM 모델을 사용할 수 없습니다.'}), 503
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("유방암 이미지 분석 웹 애플리케이션 시작")
    print("=" * 50)
    print("브라우저에서 http://localhost:5000 접속")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

