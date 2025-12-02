"""
모델 학습 및 예측 유틸리티
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# shap은 지연 로딩 (torch DLL 문제 방지)

class BreastCancerModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        self.lime_explainer = None
        self.training_data = None  # LIME을 위한 학습 데이터 저장
        
    def train_model(self, data_path="kr_data.csv"):
        """모델 학습"""
        print("데이터 로드 중...")
        df = pd.read_csv(data_path, encoding="utf-8")
        
        # 진단 컬럼 및 특징 컬럼 설정
        target_col = df.columns[1]  # 진단
        feature_cols = list(df.columns[2:])  # 수치형 특징들
        
        # 진단 인코딩 (M=1, B=0)
        y = (df[target_col] == "M").astype(int)
        X = df[feature_cols].values
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 학습
        print("모델 학습 중...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # 정확도 출력
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"학습 정확도: {train_score:.4f}")
        print(f"테스트 정확도: {test_score:.4f}")
        
        # 특징 이름 저장
        self.feature_names = feature_cols
        
        # 학습 데이터 저장 (LIME용)
        self.training_data = X_train_scaled
        
        # SHAP explainer 생성 (TreeExplainer 사용) - 지연 로딩
        print("SHAP explainer 생성 중...")
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP explainer 생성 완료")
        except ImportError:
            print("⚠️ shap 라이브러리를 사용할 수 없습니다. LIME을 사용합니다.")
            self.explainer = None
            self._setup_lime(X_train_scaled, feature_cols)
        except Exception as e:
            print(f"⚠️ SHAP explainer 생성 실패: {e}")
            print("⚠️ LIME을 사용합니다.")
            self.explainer = None
            self._setup_lime(X_train_scaled, feature_cols)
        
        return self.model, self.scaler
    
    def _setup_lime(self, training_data, feature_names):
        """LIME explainer 설정"""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            self.lime_explainer = LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=['양성(B)', '악성(M)'],
                mode='classification'
            )
            print("✅ LIME explainer 생성 완료")
        except ImportError:
            print("⚠️ lime 라이브러리가 설치되지 않았습니다.")
            print("⚠️ pip install lime 를 실행하세요.")
            self.lime_explainer = None
        except Exception as e:
            print(f"⚠️ LIME explainer 생성 실패: {e}")
            self.lime_explainer = None
    
    def predict(self, features):
        """예측 수행"""
        if self.model is None or self.scaler is None:
            raise ValueError("모델이 학습되지 않았습니다. train_model()을 먼저 호출하세요.")
        
        # 스케일링
        features_scaled = self.scaler.transform([features])
        
        # 예측
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': '악성(M)' if prediction == 1 else '양성(B)',
            'probability': float(max(probability)),
            'malignant_prob': float(probability[1]),
            'benign_prob': float(probability[0])
        }
    
    def explain(self, features):
        """SHAP 또는 LIME을 사용한 설명 생성"""
        # 스케일링
        features_scaled = self.scaler.transform([features])
        
        # 1. SHAP 시도
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(features_scaled)
                
                # 양성/악성에 대한 SHAP 값 (악성 클래스 사용)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]  # 악성 클래스
                else:
                    shap_vals = shap_values
                
                # 특징별 기여도 계산
                feature_contributions = []
                for i, feature_name in enumerate(self.feature_names):
                    contribution = float(shap_vals[0][i])
                    feature_contributions.append({
                        'feature': feature_name,
                        'contribution': contribution,
                        'abs_contribution': abs(contribution)
                    })
                
                # 절댓값 기준으로 정렬 (상위 10개)
                feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
                top_features = feature_contributions[:10]
                
                return {
                    'top_features': top_features,
                    'all_features': feature_contributions,
                    'method': 'SHAP'
                }
            except Exception as e:
                print(f"⚠️ SHAP 계산 중 오류: {e}, LIME으로 전환")
        
        # 2. LIME 시도
        if self.lime_explainer is not None and self.training_data is not None:
            try:
                explanation = self.lime_explainer.explain_instance(
                    features_scaled[0],
                    self.model.predict_proba,
                    num_features=len(self.feature_names),
                    top_labels=1
                )
                
                # LIME 설명에서 특징 기여도 추출
                exp_list = explanation.as_list()
                feature_contributions = []
                
                for feature_name, contribution in exp_list:
                    feature_contributions.append({
                        'feature': feature_name,
                        'contribution': float(contribution),
                        'abs_contribution': abs(float(contribution))
                    })
                
                # 절댓값 기준으로 정렬 (상위 10개)
                feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
                top_features = feature_contributions[:10]
                
                return {
                    'top_features': top_features,
                    'all_features': feature_contributions,
                    'method': 'LIME'
                }
            except Exception as e:
                print(f"⚠️ LIME 계산 중 오류: {e}")
        
        # 3. 기본 설명 (feature_importances_ 사용)
        return self._generate_basic_explanation(features)
    
    def _generate_basic_explanation(self, features):
        """SHAP/LIME을 사용할 수 없을 때 기본 설명 생성 (feature_importances_ 사용)"""
        if self.model is None or self.scaler is None:
            return {
                'top_features': [],
                'all_features': [],
                'method': 'None'
            }
        
        # 모델의 feature_importances_ 사용
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_contributions = []
            
            # 예측 확률에 따라 기여도 조정
            features_scaled = self.scaler.transform([features])
            proba = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            
            # 예측 결과에 따라 기여도 부호 조정
            for i, feature_name in enumerate(self.feature_names):
                base_importance = float(importances[i])
                # 예측이 악성인 경우 양수, 양성인 경우 음수로 조정
                if prediction == 1:  # 악성
                    contribution = base_importance * proba[1]
                else:  # 양성
                    contribution = -base_importance * proba[0]
                
                feature_contributions.append({
                    'feature': feature_name,
                    'contribution': contribution,
                    'abs_contribution': abs(contribution)
                })
            
            # 중요도 기준으로 정렬
            feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            top_features = feature_contributions[:10]
            
            return {
                'top_features': top_features,
                'all_features': feature_contributions,
                'method': 'Feature Importance'
            }
        else:
            # feature_importances_도 없으면 빈 리스트 반환
            return {
                'top_features': [],
                'all_features': [],
                'method': 'None'
            }
    
    def save_model(self, model_path="breast_cancer_model.joblib", scaler_path="scaler.joblib"):
        """모델 저장"""
        if self.model is None or self.scaler is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"모델 저장 완료: {model_path}, {scaler_path}")
    
    def load_model(self, model_path="breast_cancer_model.joblib", scaler_path="scaler.joblib"):
        """모델 로드"""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("저장된 모델이 없습니다. 새로 학습합니다...")
            self.train_model()
            self.save_model(model_path, scaler_path)
            return
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # 특징 이름 설정 (데이터에서 로드)
        df = pd.read_csv("kr_data.csv", encoding="utf-8")
        self.feature_names = list(df.columns[2:])
        
        # 학습 데이터 로드 (LIME용)
        try:
            df = pd.read_csv("kr_data.csv", encoding="utf-8")
            target_col = df.columns[1]
            feature_cols = list(df.columns[2:])
            y = (df[target_col] == "M").astype(int)
            X = df[feature_cols].values
            X_scaled = self.scaler.transform(X)
            self.training_data = X_scaled
        except:
            self.training_data = None
        
        # SHAP explainer 재생성 (지연 로딩)
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP explainer 생성 완료")
        except ImportError:
            print("⚠️ shap 라이브러리를 사용할 수 없습니다. LIME을 사용합니다.")
            self.explainer = None
            if self.training_data is not None:
                self._setup_lime(self.training_data, self.feature_names)
        except Exception as e:
            print(f"⚠️ SHAP explainer 생성 실패: {e}")
            print("⚠️ LIME을 사용합니다.")
            self.explainer = None
            if self.training_data is not None:
                self._setup_lime(self.training_data, self.feature_names)
        
        print(f"모델 로드 완료: {model_path}, {scaler_path}")

