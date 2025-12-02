"""
이미지 처리 및 특징 추출 유틸리티
- 세포 검출
- 세포별 형태/텍스처/강도 특징 추출
- UCI 30 feature 스케일로 매핑 (kr_data.csv 기준)
- 경계 불규칙 세포 비율/큰 세포 비율/고대비 세포 비율 등 형태 정량 분석
- 바운딩 박스 + XAI 특징 레이블 시각화
"""

import cv2
import numpy as np
from PIL import Image
import os
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection

# =============================================================================
# 한글 폰트 설정
# =============================================================================
_korean_font_prop = None


def setup_korean_font():
    """한글 폰트 설정 및 FontProperties 반환"""
    global _korean_font_prop
    import platform
    system = platform.system()

    if system == 'Windows':
        font_list = ['Malgun Gothic', 'NanumGothic', 'Nanum Gothic', 'Gulim',
                     'Batang', 'Gungsuh', 'Dotum']
        for font_name in font_list:
            try:
                font_prop = fm.FontProperties(family=font_name)
                font_path = fm.findfont(font_prop)
                if font_path and os.path.exists(font_path):
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False
                    _korean_font_prop = font_prop
                    print(f"✅ 한글 폰트 설정 완료: {font_name} ({font_path})")
                    return font_prop
            except Exception:
                continue

        # 직접 폰트 경로 찾기 시도
        font_dirs = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
            'C:\\Windows\\Fonts',
            os.path.expanduser('~\\AppData\\Local\\Microsoft\\Windows\\Fonts')
        ]
        font_files = ['malgun.ttf', 'NanumGothic.ttf', 'NanumGothicRegular.ttf',
                      'gulim.ttc', 'batang.ttc', 'gungsuh.ttc']

        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_file in font_files:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        try:
                            font_prop = fm.FontProperties(fname=font_path)
                            plt.rcParams['font.family'] = font_prop.get_name()
                            plt.rcParams['axes.unicode_minus'] = False
                            _korean_font_prop = font_prop
                            print(f"✅ 한글 폰트 직접 로드 완료: {font_path}")
                            return font_prop
                        except Exception:
                            continue

    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        _korean_font_prop = fm.FontProperties(family='AppleGothic')
        return _korean_font_prop

    else:  # Linux
        font_list = ['NanumGothic', 'Nanum Gothic', 'Noto Sans CJK KR', 'DejaVu Sans']
        for font_name in font_list:
            try:
                font_prop = fm.FontProperties(family=font_name)
                font_path = fm.findfont(font_prop)
                if font_path:
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False
                    _korean_font_prop = font_prop
                    return font_prop
            except Exception:
                continue

    # 폰트를 찾지 못한 경우 기본 설정
    plt.rcParams['axes.unicode_minus'] = False
    try:
        _korean_font_prop = fm.FontProperties()
        print("⚠️ 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
    except Exception:
        _korean_font_prop = None
    return _korean_font_prop


_korean_font_prop = setup_korean_font()

# =============================================================================
# 이미지 로드 / 전처리 / 세포 검출
# =============================================================================


def load_image(image_path):
    """이미지 로드 (TIFF 포함)"""
    if image_path.lower().endswith('.tif'):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(image_path))
    else:
        img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    return img


def preprocess_image(img):
    """이미지 전처리 (그레이스케일 + 노이즈 제거 + 대비 향상)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced, gray


def detect_cells(img, min_area=15, max_area=30000):
    """
    세포 감지 및 바운딩 박스 생성 (개선된 방법)
    - Watershed 알고리즘으로 더 정확한 세포 분리
    - connectedComponents 기반 blob → 세포 후보
    - min_area와 max_area를 조정하여 더 많은 세포 검출
    - min_area: 15 (더 작은 세포도 검출)
    - max_area: 30000 (더 큰 세포도 검출)
    """
    processed, gray = preprocess_image(img)

    # 이진화
    _, binary = cv2.threshold(
        processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 모폴로지 연산으로 노이즈 제거 및 세포 분리 개선
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 거리 변환을 사용한 Watershed 알고리즘으로 세포 분리 개선
    try:
        from skimage.segmentation import watershed
        from scipy import ndimage as ndi
        
        # 거리 변환
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 로컬 최대값 찾기 (세포 중심) - scipy.ndimage 사용
        # peak_local_maxima 대신 maximum_filter 사용
        from scipy.ndimage import maximum_filter
        local_maxima_mask = (dist_transform == maximum_filter(dist_transform, size=10))
        local_maxima_mask = local_maxima_mask & (dist_transform > 0.3 * dist_transform.max())
        local_maxima = np.where(local_maxima_mask)
        
        if len(local_maxima[0]) > 0 and len(local_maxima[1]) > 0:
            # 마커 생성
            markers = np.zeros_like(binary, dtype=np.int32)
            for i, (y, x) in enumerate(zip(local_maxima[0], local_maxima[1])):
                if 0 <= y < binary.shape[0] and 0 <= x < binary.shape[1]:
                    markers[y, x] = i + 1
            
            # Watershed 분할
            labels_ws = watershed(-dist_transform, markers, mask=binary)
            
            # Watershed 결과를 사용하여 세포 추출
            num_labels = np.max(labels_ws) + 1
            labels = labels_ws.astype(np.uint8)
        else:
            # Watershed 실패 시 기본 방법 사용
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
    except Exception as e:
        print(f"⚠️ Watershed 알고리즘 실패, 기본 방법 사용: {e}")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

    # 통계 계산 (Watershed 사용 시)
    use_watershed = 'labels_ws' in locals()
    if use_watershed:
        # Watershed 결과에서 통계 계산
        stats = np.zeros((num_labels, 5), dtype=np.int32)
        centroids = np.zeros((num_labels, 2), dtype=np.float64)
        for i in range(1, num_labels):
            mask = (labels == i).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            area = cv2.countNonZero(mask)
            stats[i] = [x, y, w, h, area]
            M = cv2.moments(mask)
            if M['m00'] > 0:
                centroids[i] = (M['m10'] / M['m00'], M['m01'] / M['m00'])
            else:
                centroids[i] = (x + w/2, y + h/2)
    else:
        # 기본 방법 사용 시 stats는 이미 계산됨
        pass

    cells = []
    for i in range(1, num_labels):  # 0은 배경
        area = stats[i, cv2.CC_STAT_AREA] if len(stats.shape) > 1 else stats[i]
        if min_area <= area <= max_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            cells.append({
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': (int(cx), int(cy)),
                'mask': (labels == i).astype(np.uint8)
            })

    return cells, processed, gray


def detect_cells_from_mask(img, mask_img):
    """
    마스크 이미지를 사용하여 정확한 세포 감지
    - 마스크 이미지에서 각 세포 영역 추출
    """
    processed, gray = preprocess_image(img)
    
    # 마스크 이미지 전처리
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img.copy()
    
    # 마스크 이진화
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 연결된 컴포넌트 찾기
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )
    
    cells = []
    for i in range(1, num_labels):  # 0은 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 50:  # 최소 크기 필터링
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 마스크 영역 추출
            cell_mask = (labels == i).astype(np.uint8)
            
            cells.append({
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': (int(centroids[i][0]), int(centroids[i][1])),
                'mask': cell_mask,
                'full_mask': cell_mask  # 전체 이미지 크기의 마스크
            })
    
    print(f"✅ 마스크 기반 세포 감지: {len(cells)}개 세포 발견")
    return cells, processed, gray


def load_mask_image(img_path):
    """
    Masks 폴더에서 해당 이미지의 마스크 파일 찾기 및 로드
    """
    try:
        # 파일명 추출
        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # _ccd 같은 접미사 제거
        if '_ccd' in name_without_ext:
            name_without_ext = name_without_ext.replace('_ccd', '')
        
        # Masks 폴더 경로 찾기
        masks_dirs = [
            'image/Masks',
            os.path.join('image', 'Masks'),
            os.path.join(os.path.dirname(img_path), '..', 'Masks'),
            os.path.join(os.path.dirname(os.path.dirname(img_path)), 'Masks'),
        ]
        
        masks_dir = None
        for dir_path in masks_dirs:
            abs_path = os.path.abspath(dir_path)
            if os.path.exists(abs_path):
                masks_dir = abs_path
                break
        
        if not masks_dir:
            print(f"⚠️ Masks 폴더를 찾을 수 없습니다")
            return None
        
        # 마스크 파일 찾기 (여러 패턴 시도)
        import glob
        
        # 패턴 1: 정확한 파일명 매칭
        patterns = [
            os.path.join(masks_dir, f"{name_without_ext}.TIF"),
            os.path.join(masks_dir, f"{name_without_ext}.tif"),
            os.path.join(masks_dir, f"{base_name}"),
        ]
        
        # 패턴 2: 부분 매칭 (예: ytma49_111003_malignant3 -> ytma49_111003_malignant*)
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            if len(parts) >= 3:
                # 마지막 숫자 제거하고 매칭
                base_pattern = '_'.join(parts[:-1])
                patterns.extend([
                    os.path.join(masks_dir, f"{base_pattern}_*.TIF"),
                    os.path.join(masks_dir, f"{base_pattern}_*.tif"),
                ])
        
        # 파일 찾기
        for pattern in patterns:
            if '*' in pattern:
                matches = glob.glob(pattern)
                if matches:
                    mask_path = matches[0]
                    try:
                        mask_img = load_image(mask_path)
                        print(f"✅ 마스크 로드 성공: {mask_path}")
                        return mask_img
                    except Exception as e:
                        print(f"⚠️ 마스크 로드 실패: {e}")
                        continue
            else:
                if os.path.exists(pattern):
                    try:
                        mask_img = load_image(pattern)
                        print(f"✅ 마스크 로드 성공: {pattern}")
                        return mask_img
                    except Exception as e:
                        print(f"⚠️ 마스크 로드 실패: {e}")
                        continue
        
        print(f"⚠️ 마스크 파일을 찾을 수 없습니다: {base_name} (검색 경로: {masks_dir})")
        return None
    except Exception as e:
        print(f"⚠️ load_mask_image 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_cells_from_mask(img, mask_img):
    """
    마스크 이미지를 사용하여 정확한 세포 감지
    - 마스크 이미지에서 각 세포 영역 추출
    """
    processed, gray = preprocess_image(img)
    
    # 마스크 이미지 전처리
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img.copy()
    
    # 마스크 이진화
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 연결된 컴포넌트 찾기
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )
    
    cells = []
    for i in range(1, num_labels):  # 0은 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 50:  # 최소 크기 필터링
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 마스크 영역 추출
            cell_mask = (labels == i).astype(np.uint8)
            
            cells.append({
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': (int(centroids[i][0]), int(centroids[i][1])),
                'mask': cell_mask,
                'full_mask': cell_mask  # 전체 이미지 크기의 마스크
            })
    
    print(f"✅ 마스크 기반 세포 감지: {len(cells)}개 세포 발견")
    return cells, processed, gray

# =============================================================================
# 세포별 특징 추출
# =============================================================================


def extract_cell_features(cell_region, gray_img):
    """
    개별 세포에서 형태/텍스처/강도 특징 추출
    - area, perimeter, circularity, solidity, convexity
    - GLCM 기반 텍스처 (contrast, homogeneity, energy)
    - mean / std intensity
    """
    x, y, w, h = cell_region['bbox']
    cell_img = gray_img[y:y + h, x:x + w]

    if cell_img.size == 0:
        return None

    mask = cell_region['mask'][y:y + h, x:x + w]
    masked_cell = cell_img * mask

    features = {}
    area = cell_region['area']

    mask_uint8 = np.uint8(mask * 255)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 둘레 길이
    if len(contours) > 0 and len(contours[0]) >= 3:
        try:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter <= 0 or not np.isfinite(perimeter):
                perimeter = 1
        except Exception:
            perimeter = 1
    else:
        perimeter = 1

    features['area'] = area
    features['perimeter'] = perimeter
    features['circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # convex hull 기반 solidity / convexity
    if len(contours) > 0 and len(contours[0]) >= 3:
        try:
            hull = cv2.convexHull(contours[0])
            if len(hull) >= 3:
                hull_area = cv2.contourArea(hull)
                features['solidity'] = area / hull_area if hull_area > 0 else 0

                hull_perimeter = cv2.arcLength(hull, True)
                if hull_perimeter > 0 and np.isfinite(hull_perimeter):
                    features['convexity'] = hull_perimeter / perimeter if perimeter > 0 else 0
                else:
                    features['convexity'] = 0
            else:
                features['solidity'] = 0
                features['convexity'] = 0
        except Exception:
            features['solidity'] = 0
            features['convexity'] = 0
    else:
        features['solidity'] = 0
        features['convexity'] = 0

    # 텍스처 특징 (GLCM)
    if cell_img.size > 0:
        cell_normalized = (masked_cell / 255.0 * 15).astype(np.uint8)
        if np.max(cell_normalized) > 0:
            try:
                glcm = graycomatrix(
                    cell_normalized, [1], [0], 16,
                    symmetric=True, normed=True
                )
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]

                features['texture_contrast'] = contrast
                features['texture_homogeneity'] = homogeneity
                features['texture_energy'] = energy
            except Exception:
                features['texture_contrast'] = 0
                features['texture_homogeneity'] = 0
                features['texture_energy'] = 0
        else:
            features['texture_contrast'] = 0
            features['texture_homogeneity'] = 0
            features['texture_energy'] = 0

    # 강도 특징
    if np.sum(mask) > 0:
        features['mean_intensity'] = float(np.mean(masked_cell[mask > 0]))
        features['std_intensity'] = float(np.std(masked_cell[mask > 0]))
    else:
        features['mean_intensity'] = 0.0
        features['std_intensity'] = 0.0

    return features

# =============================================================================
# 이미지 전체 특징 (UCI 30 feature 스케일에 맞추기)
# =============================================================================


def extract_image_features(img_path):
    """
    이미지에서 전체 특징 추출 (kr_data.csv 형식에 맞춤)
    - kr_data.csv의 30개 feature 스케일과 평균/표준편차를 이용해 정규화
    - Masks 폴더의 마스크 이미지 활용
    """
    import pandas as pd

    img = load_image(img_path)
    
    # 마스크 이미지 로드 시도
    mask_img = load_mask_image(img_path)
    
    # 마스크가 있으면 마스크 기반으로 세포 감지, 없으면 기존 방법 사용
    if mask_img is not None:
        cells, processed, gray = detect_cells_from_mask(img, mask_img)
    else:
        cells, processed, gray = detect_cells(img)

    # 기존 UCI 기반 테이블 통계 로드
    try:
        df_ref = pd.read_csv("kr_data.csv", encoding="utf-8")
        feature_cols = list(df_ref.columns[2:])
        df_features = df_ref[feature_cols]
        feature_means = df_features.mean().values
        feature_stds = df_features.std().values
    except Exception:
        feature_means = np.zeros(30)
        feature_stds = np.ones(30)

    if len(cells) == 0:
        return feature_means, [], processed

    all_features = []
    for cell in cells:
        feat = extract_cell_features(cell, gray)
        if feat is not None:
            all_features.append(feat)

    if len(all_features) == 0:
        return feature_means, [], processed

    feature_names = [
        'area', 'perimeter', 'circularity', 'solidity', 'convexity',
        'texture_contrast', 'texture_homogeneity', 'texture_energy',
        'mean_intensity', 'std_intensity'
    ]

    stats = {}
    for feat_name in feature_names:
        values = [f[feat_name] for f in all_features if feat_name in f]
        if len(values) > 0:
            stats[f'{feat_name}_mean'] = np.mean(values)
            stats[f'{feat_name}_se'] = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            stats[f'{feat_name}_worst'] = np.max(values)
        else:
            stats[f'{feat_name}_mean'] = 0
            stats[f'{feat_name}_se'] = 0
            stats[f'{feat_name}_worst'] = 0

    feature_vector = []

    # 반경: area → radius 추정
    area_scale = 0.1
    radius_mean = np.sqrt((stats.get('area_mean', 0) * area_scale) / np.pi) if stats.get('area_mean', 0) > 0 else 0
    radius_se = np.sqrt((stats.get('area_se', 0) * area_scale) / np.pi) if stats.get('area_se', 0) > 0 else 0
    radius_worst = np.sqrt((stats.get('area_worst', 0) * area_scale) / np.pi) if stats.get('area_worst', 0) > 0 else 0

    intensity_scale = 0.1
    tissue_mean = stats.get('mean_intensity_mean', 0) * intensity_scale

    smoothness_mean = 0.05 + (1 - stats.get('texture_homogeneity_mean', 0)) * 0.11
    compactness_mean = 0.08 + stats.get('texture_energy_mean', 0) * 0.20
    concavity_mean = (1 - stats.get('solidity_mean', 0)) * 0.27
    concave_points_mean = stats.get('texture_contrast_mean', 0) * 0.20
    symmetry_mean = 0.11 + stats.get('circularity_mean', 0) * 0.19
    fractal_dimension_mean = 0.02 + (1 - stats.get('convexity_mean', 0)) * 0.13

    # 평균 특징 10개
    feature_vector.extend([
        max(6, min(30, radius_mean)),        # radius_mean
        max(9, min(30, tissue_mean)),        # texture_mean
        stats.get('perimeter_mean', 0) * 0.1,
        stats.get('area_mean', 0) * area_scale,
        max(0.05, min(0.16, smoothness_mean)),
        max(0.08, min(0.28, compactness_mean)),
        max(0.03, min(0.30, concavity_mean)),
        max(0.00, min(0.20, concave_points_mean)),
        max(0.11, min(0.30, symmetry_mean)),
        max(0.02, min(0.15, fractal_dimension_mean)),
    ])

    # 표준오차 10개
    feature_vector.extend([
        radius_se,
        stats.get('mean_intensity_se', 0),
        stats.get('perimeter_se', 0),
        stats.get('area_se', 0),
        stats.get('texture_homogeneity_se', 0),
        stats.get('texture_energy_se', 0),
        stats.get('solidity_se', 0),
        stats.get('texture_contrast_se', 0),
        stats.get('circularity_se', 0),
        stats.get('convexity_se', 0),
    ])

    # 최악값 10개
    feature_vector.extend([
        radius_worst,
        stats.get('mean_intensity_worst', 0),
        stats.get('perimeter_worst', 0),
        stats.get('area_worst', 0),
        stats.get('texture_homogeneity_worst', 0),
        stats.get('texture_energy_worst', 0),
        1 - stats.get('solidity_worst', 0),
        stats.get('texture_contrast_worst', 0),
        stats.get('circularity_worst', 0),
        1 - stats.get('convexity_worst', 0),
    ])

    feature_vector = np.array(feature_vector)

    # Z-score 기반 정규화 후 다시 원래 스케일 복원 (clip)
    normalized_features = (feature_vector - feature_means) / (feature_stds + 1e-8)
    final_features = normalized_features * feature_stds + feature_means

    final_features = np.clip(
        final_features,
        feature_means - 3 * feature_stds,
        feature_means + 3 * feature_stds
    )

    return final_features, cells, processed

# =============================================================================
# 형태 정량 분석 (경계 불규칙 세포 비율 등)
# =============================================================================


def analyze_morphology(img_path):
    """
    세포 수준 형태 분석:
    - 총 세포 수
    - 경계가 들쑥날쑥한 세포 비율 (circularity/solidity/convexity 기준)
    - 큰 세포 비율 (area 상위 25%)
    - 고대비 텍스처 세포 비율 (contrast 상위 25%)
    """
    img = load_image(img_path)
    cells, processed, gray = detect_cells(img)

    if len(cells) == 0:
        return {
            "total_cells": 0,
            "irregular_boundary_cells": 0,
            "irregular_boundary_ratio": 0.0,
            "large_cells": 0,
            "large_cell_ratio": 0.0,
            "high_contrast_cells": 0,
            "high_contrast_ratio": 0.0,
            "mean_circularity": 0.0,
            "mean_solidity": 0.0,
            "mean_convexity": 0.0,
        }

    all_features = []
    for cell in cells:
        feat = extract_cell_features(cell, gray)
        if feat is not None:
            all_features.append(feat)

    if len(all_features) == 0:
        return {
            "total_cells": 0,
            "irregular_boundary_cells": 0,
            "irregular_boundary_ratio": 0.0,
            "large_cells": 0,
            "large_cell_ratio": 0.0,
            "high_contrast_cells": 0,
            "high_contrast_ratio": 0.0,
            "mean_circularity": 0.0,
            "mean_solidity": 0.0,
            "mean_convexity": 0.0,
        }

    total = len(all_features)
    areas = np.array([f["area"] for f in all_features])
    circularities = np.array([f["circularity"] for f in all_features])
    solidities = np.array([f["solidity"] for f in all_features])
    convexities = np.array([f["convexity"] for f in all_features])
    contrasts = np.array([f["texture_contrast"] for f in all_features])

    # 경계 불규칙 세포: circularity 낮거나, solidity 낮거나, convexity 낮은 경우
    irregular_mask = (
        (circularities < 0.75) |
        (solidities < 0.9) |
        (convexities < 0.95)
    )
    irregular_count = int(np.sum(irregular_mask))

    # 큰 세포: 면적 상위 25%
    area_threshold = np.percentile(areas, 75)
    large_mask = areas >= area_threshold
    large_count = int(np.sum(large_mask))

    # 고대비 텍스처 세포: contrast 상위 25%
    contrast_threshold = np.percentile(contrasts, 75)
    high_contrast_mask = contrasts >= contrast_threshold
    high_contrast_count = int(np.sum(high_contrast_mask))

    summary = {
        "total_cells": total,
        "irregular_boundary_cells": irregular_count,
        "irregular_boundary_ratio": irregular_count / total if total > 0 else 0.0,
        "large_cells": large_count,
        "large_cell_ratio": large_count / total if total > 0 else 0.0,
        "high_contrast_cells": high_contrast_count,
        "high_contrast_ratio": high_contrast_count / total if total > 0 else 0.0,
        "mean_circularity": float(np.mean(circularities)),
        "mean_solidity": float(np.mean(solidities)),
        "mean_convexity": float(np.mean(convexities)),
    }

    return summary

# =============================================================================
# 바운딩 박스 시각화
# =============================================================================


def draw_bbox_with_labels(img, cells, predictions=None, explanations=None, show_segmentation=False):
    """
    바운딩 박스와 레이블 그리기 - XAI 특징 반영
    - predictions: [{ 'prediction': '악성', 'probability': 0.9 }, ...]
    - explanations: [{ 'top_features': [{'feature': 'radius_worst', 'contribution': 0.3}, ...]}, ...]
    - show_segmentation: 세포 경계(contour) 표시 여부 (기본값: False)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img, cmap='gray')

    # 더 많은 세포 표시 (최대 100개까지)
    max_cells_to_show = min(100, len(cells))

    cell_data = []
    for i, cell in enumerate(cells[:max_cells_to_show]):
        if predictions and i < len(predictions):
            pred = predictions[i]
            prob = pred['probability']
            cell_data.append((i, cell, pred, prob))
        else:
            cell_data.append((i, cell, None, 0))

    cell_data.sort(key=lambda x: x[3], reverse=True)

    bbox_styles = [
        {'linewidth': 3, 'alpha': 0.9, 'linestyle': '-', 'edgecolor': 'red'},
        {'linewidth': 2.5, 'alpha': 0.8, 'linestyle': '-', 'edgecolor': '#ff4444'},
        {'linewidth': 2, 'alpha': 0.7, 'linestyle': '--', 'edgecolor': '#ff8888'},
        {'linewidth': 3, 'alpha': 0.9, 'linestyle': '-', 'edgecolor': 'green'},
        {'linewidth': 2.5, 'alpha': 0.8, 'linestyle': '-', 'edgecolor': '#44ff44'},
        {'linewidth': 2, 'alpha': 0.7, 'linestyle': '--', 'edgecolor': '#88ff88'},
    ]

    for idx, (i, cell, pred, prob) in enumerate(cell_data):
        x, y, w, h = cell['bbox']

        if pred:
            is_malignant = '악성' in pred['prediction']
            label = pred['prediction']

            if is_malignant:
                if prob >= 0.8:
                    style_idx = 0
                elif prob >= 0.6:
                    style_idx = 1
                else:
                    style_idx = 2
                color = '#ff6b6b'
            else:
                if prob >= 0.8:
                    style_idx = 3
                elif prob >= 0.6:
                    style_idx = 4
                else:
                    style_idx = 5
                color = '#51cf66'
        else:
            style_idx = 2
            color = '#888888'
            label = '세포'
            prob = 0

        style = bbox_styles[style_idx]

        rect = Rectangle(
            (x, y), w, h,
            linewidth=style['linewidth'],
            edgecolor=style['edgecolor'] if pred else color,
            facecolor=style['edgecolor'] if pred else color,
            alpha=style['alpha'] * 0.3,
            linestyle=style['linestyle']
        )
        ax.add_patch(rect)

        rect_border = Rectangle(
            (x, y), w, h,
            linewidth=style['linewidth'],
            edgecolor=style['edgecolor'] if pred else color,
            facecolor='none',
            alpha=style['alpha'],
            linestyle=style['linestyle']
        )
        ax.add_patch(rect_border)
        
        # 세포 경계(contour) 시각화 - 전문적인 스타일
        if show_segmentation:
            try:
                # full_mask가 있으면 사용 (마스크 기반), 없으면 bbox 내 mask 사용
                if 'full_mask' in cell:
                    mask = cell['full_mask']
                else:
                    mask = cell['mask']
                    # bbox 영역만 있는 경우 전체 이미지 크기로 확장
                    full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    full_mask[y:y+h, x:x+w] = mask
                    mask = full_mask
                
                mask_uint8 = np.uint8(mask * 255)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if len(contours) > 0:
                    # 모든 contour 그리기 (더 정확한 경계)
                    for contour in contours:
                        if len(contour) >= 3:
                            contour_coords = contour.reshape(-1, 2)
                            
                            # 전문적인 경계선 스타일
                            contour_color = color if pred else '#888888'
                            
                            # 두꺼운 외곽선 + 얇은 내부선 (레이어 효과)
                            ax.plot(
                                contour_coords[:, 0], contour_coords[:, 1],
                                color='white',
                                linewidth=3.5,
                                alpha=0.8,
                                linestyle='-',
                                zorder=1
                            )
                            ax.plot(
                                contour_coords[:, 0], contour_coords[:, 1],
                                color=contour_color,
                                linewidth=2.0,
                                alpha=0.9,
                                linestyle='-',
                                zorder=2
                            )
                            
                            # 세포 내부 반투명 채우기 (선택적)
                            if pred and prob >= 0.7:
                                from matplotlib.patches import Polygon as MPLPolygon
                                polygon = MPLPolygon(
                                    contour_coords,
                                    facecolor=contour_color,
                                    alpha=0.15,
                                    edgecolor='none',
                                    zorder=0
                                )
                                ax.add_patch(polygon)
            except Exception as e:
                # contour 그리기 실패 시 무시
                pass

        # 레이블 표시 - XAI 특징 반영
        if pred and explanations and i < len(explanations):
            expl = explanations[i]
            top_features = expl.get('top_features', [])[:3]  # 상위 3개 특징 표시

            if top_features:
                feature_texts = []
                for feat in top_features:
                    feat_name = feat['feature']
                    contrib = feat.get('contribution', 0)
                    # 특징 이름을 간단하게 표시
                    short_name = feat_name.replace('_mean', '').replace('_worst', '').replace('_se', '')
                    if len(short_name) > 15:
                        short_name = short_name[:12] + '...'
                    feature_texts.append(f"{short_name}({contrib:+.2f})")

                label_text = f"{label} ({prob:.1%})\n" + "\n".join(feature_texts)
            else:
                label_text = f"{label} ({prob:.1%})"
        else:
            label_text = f"{label} ({prob:.1%})" if pred else f"세포 {idx+1}"

        # 텍스트 색: 악성/양성 구분
        if pred and '악성' in pred['prediction']:
            text_color = '#cc0000'
        elif pred:
            text_color = '#006600'
        else:
            text_color = '#333333'

        # 레이블 위치 최적화 (겹침 방지)
        label_x = x
        label_y = y - 5
        
        # 다른 레이블과 겹치지 않도록 위치 조정
        if idx > 0:
            # 이전 레이블들과의 거리 확인 및 조정
            for prev_idx in range(max(0, idx - 10), idx):
                if prev_idx < len(cell_data):
                    prev_i, prev_cell, prev_pred, prev_prob = cell_data[prev_idx]
                    prev_x, prev_y, prev_w, prev_h = prev_cell['bbox']
                    prev_label_y = prev_y - 5
                    
                    # 수직 거리가 너무 가까우면 오프셋 추가
                    if abs(label_y - prev_label_y) < 30 and abs(label_x - prev_x) < 100:
                        label_y = prev_label_y - 35
        
        # 레이블이 이미지 밖으로 나가지 않도록 조정
        label_y = max(10, min(label_y, img.shape[0] - 5))

        font_prop = _korean_font_prop if _korean_font_prop else None

        if font_prop:
            ax.text(
                label_x, label_y, label_text, color=text_color, fontsize=8,
                fontweight='bold',
                fontproperties=font_prop,
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.95,
                    edgecolor=color,
                    linewidth=2.5,
                    pad=3
                ),
                verticalalignment='bottom'
            )
        else:
            ax.text(
                label_x, label_y, label_text, color=text_color, fontsize=8,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.95,
                    edgecolor=color,
                    linewidth=2.5,
                    pad=3
                ),
                verticalalignment='bottom'
            )

    ax.axis('off')
    plt.tight_layout()
    return fig


def extract_cells_from_mask_file(mask_img):
    """
    마스크 파일에서 직접 세포 경계 추출
    - 마스크 이미지의 각 고유 값이 하나의 세포를 나타냄
    - 정확한 경계 추출
    """
    # 마스크 이미지 전처리
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img.copy()
    
    # 마스크에서 고유한 세포 ID 찾기 (0은 배경)
    unique_values = np.unique(mask_gray)
    unique_values = unique_values[unique_values > 0]  # 배경 제외
    
    cells = []
    h, w = mask_gray.shape
    
    for cell_id in unique_values:
        # 각 세포 영역 추출
        cell_mask = (mask_gray == cell_id).astype(np.uint8)
        area = np.sum(cell_mask)
        
        if area >= 50:  # 최소 크기 필터링
            # Bounding box 계산
            coords = np.where(cell_mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                x, y = x_min, y_min
                w_cell, h_cell = x_max - x_min + 1, y_max - y_min + 1
                
                # 중심점 계산
                M = cv2.moments(cell_mask)
                if M['m00'] > 0:
                    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                else:
                    cx, cy = x + w_cell/2, y + h_cell/2
                
                cells.append({
                    'id': int(cell_id),
                    'bbox': (x, y, w_cell, h_cell),
                    'area': int(area),
                    'centroid': (int(cx), int(cy)),
                    'mask': cell_mask,
                    'full_mask': cell_mask
                })
    
    print(f"✅ 마스크 파일에서 {len(cells)}개 세포 추출")
    return cells


def draw_mask_based_segmentation(img, mask_img, predictions=None):
    """
    Masks 폴더의 마스크 파일을 직접 사용한 전문적인 segmentation 시각화
    - 각 세포를 고유한 색상으로 구분
    - 정확한 경계선 표시
    - 사진처럼 전문적인 시각화
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 원본 이미지 표시
    if len(img.shape) == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.6)
    else:
        ax.imshow(img, cmap='gray', alpha=0.6)
    
    # 마스크 이미지 전처리
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img.copy()
    
    # 마스크에서 고유한 세포 ID 찾기
    unique_values = np.unique(mask_gray)
    unique_values = unique_values[unique_values > 0]  # 배경 제외
    
    # 색상 팔레트 생성 (더 많은 색상)
    num_cells = len(unique_values)
    if num_cells > 0:
        # tab20, Set3, Pastel1 등을 조합하여 더 많은 색상
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
        all_colors = np.vstack([colors1, colors2, colors3])
        
        # 전체 segmentation 맵 생성
        h, w = mask_gray.shape
        segmentation_map = np.zeros((h, w, 3), dtype=np.float32)
        
        # 각 세포를 다른 색으로 채우기
        for idx, cell_id in enumerate(unique_values):
            color = all_colors[idx % len(all_colors)]
            color_rgb = color[:3]
            
            # 해당 세포 영역에 색상 적용
            cell_mask = (mask_gray == cell_id)
            for c in range(3):
                segmentation_map[:, :, c][cell_mask] = color_rgb[c]
        
        # Segmentation 맵 오버레이
        ax.imshow(segmentation_map, alpha=0.7)
        
        # 각 세포의 경계선 그리기
        for idx, cell_id in enumerate(unique_values):
            cell_mask = (mask_gray == cell_id).astype(np.uint8)
            
            # Contour 추출
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 예측 결과에 따른 경계선 색상 결정
            if predictions and idx < len(predictions):
                pred = predictions[idx]
                is_malignant = '악성' in pred['prediction']
                border_color = '#ff0000' if is_malignant else '#00ff00'
                linewidth = 2.0
            else:
                border_color = '#ffffff'  # 기본 흰색 경계선
                linewidth = 1.5
            
            # 모든 contour 그리기
            for contour in contours:
                if len(contour) >= 3:
                    contour_coords = contour.reshape(-1, 2)
                    
                    # 두꺼운 외곽선 + 얇은 내부선 (레이어 효과)
                    ax.plot(
                        contour_coords[:, 0], contour_coords[:, 1],
                        color='black',
                        linewidth=linewidth + 1.5,
                        alpha=0.8,
                        zorder=1
                    )
                    ax.plot(
                        contour_coords[:, 0], contour_coords[:, 1],
                        color=border_color,
                        linewidth=linewidth,
                        alpha=0.95,
                        zorder=2
                    )
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_instance_segmentation(img, cells, predictions=None):
    """
    Instance segmentation 시각화 (각 세포를 다른 색으로 표시)
    - 전문적인 segmentation 시각화 (사진처럼)
    - 각 세포를 고유한 색상으로 채우기
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 원본 이미지 표시 (그레이스케일)
    ax.imshow(img, cmap='gray', alpha=0.7)
    
    # 각 세포를 다른 색으로 표시
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(cells))))
    
    # 전체 마스크 이미지 생성
    h, w = img.shape[:2]
    segmentation_map = np.zeros((h, w, 3), dtype=np.float32)
    
    for idx, cell in enumerate(cells):
        # 색상 선택 (20개 색상 순환)
        color = colors[idx % len(colors)]
        color_rgb = color[:3]
        
        # 마스크 추출
        if 'full_mask' in cell:
            mask = cell['full_mask']
        else:
            x, y, w_cell, h_cell = cell['bbox']
            mask = cell['mask']
            # 전체 이미지 크기로 확장
            full_mask = np.zeros((h, w), dtype=np.uint8)
            if y + h_cell <= h and x + w_cell <= w:
                full_mask[y:y+h_cell, x:x+w_cell] = mask
            mask = full_mask
        
        # 마스크를 색상으로 채우기
        mask_bool = mask > 0
        for c in range(3):
            segmentation_map[:, :, c][mask_bool] = color_rgb[c]
    
    # Segmentation 맵 오버레이
    ax.imshow(segmentation_map, alpha=0.6)
    
    # 예측 결과에 따른 경계선 그리기
    for idx, cell in enumerate(cells):
        x, y, w_cell, h_cell = cell['bbox']
        
        # 예측 결과에 따라 경계선 색상 결정
        if predictions and idx < len(predictions):
            pred = predictions[idx]
            is_malignant = '악성' in pred['prediction']
            border_color = '#ff0000' if is_malignant else '#00ff00'
            linewidth = 2.5
        else:
            border_color = '#888888'
            linewidth = 1.5
        
        # 마스크에서 contour 추출
        if 'full_mask' in cell:
            mask = cell['full_mask']
        else:
            mask = cell['mask']
            full_mask = np.zeros((h, w), dtype=np.uint8)
            if y + h_cell <= h and x + w_cell <= w:
                full_mask[y:y+h_cell, x:x+w_cell] = mask
            mask = full_mask
        
        mask_uint8 = np.uint8(mask * 255)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            for contour in contours:
                if len(contour) >= 3:
                    contour_coords = contour.reshape(-1, 2)
                    ax.plot(
                        contour_coords[:, 0], contour_coords[:, 1],
                        color=border_color,
                        linewidth=linewidth,
                        alpha=0.9
                    )
    
    ax.axis('off')
    plt.tight_layout()
    return fig
