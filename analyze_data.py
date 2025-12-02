"""데이터 분석 스크립트 - 양성/악성 특징 분포 확인"""
import pandas as pd
import numpy as np

df = pd.read_csv('kr_data.csv', encoding='utf-8')
print("컬럼명:", df.columns.tolist()[:5])

target_col = df.columns[1]  # 진단 컬럼
print(f"\n진단 컬럼: {target_col}")

benign = df[df[target_col] == 'B']
malignant = df[df[target_col] == 'M']

print(f"\n양성(B) 개수: {len(benign)}")
print(f"악성(M) 개수: {len(malignant)}")

# 주요 특징 비교
features_to_check = ['반경 평균', '면적 평균', '매끄러움 평균', '치밀도 평균', '오목함 평균']

print("\n" + "="*60)
print("주요 특징 비교 (양성 vs 악성)")
print("="*60)

for feat in features_to_check:
    if feat in df.columns:
        b_mean = benign[feat].mean()
        b_std = benign[feat].std()
        m_mean = malignant[feat].mean()
        m_std = malignant[feat].std()
        
        print(f"\n{feat}:")
        print(f"  양성(B): 평균 {b_mean:.4f} ± {b_std:.4f}, 범위 [{benign[feat].min():.4f}, {benign[feat].max():.4f}]")
        print(f"  악성(M): 평균 {m_mean:.4f} ± {m_std:.4f}, 범위 [{malignant[feat].min():.4f}, {malignant[feat].max():.4f}]")
        print(f"  차이: {m_mean - b_mean:.4f} ({((m_mean - b_mean) / b_mean * 100):.1f}%)")

