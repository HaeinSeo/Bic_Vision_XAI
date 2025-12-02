"""
이미지 기반 CNN 분류 모델
benign/malignant 이미지에서 직접 학습
"""
import os
import numpy as np
import cv2
from PIL import Image

# PyTorch 지연 로딩
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch가 설치되지 않았습니다. 이미지 분류 기능을 사용할 수 없습니다.")

try:
    from tqdm import tqdm
except ImportError:
    # tqdm이 없으면 간단한 진행 표시
    def tqdm(iterable, desc=""):
        return iterable

class BreastCancerImageDataset(Dataset):
    """유방암 이미지 데이터셋"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 이미지 파일 수집
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.endswith('.tif') and not filename.endswith('.xml'):
                    filepath = os.path.join(image_dir, filename)
                    
                    # 파일명에서 레이블 추출
                    if 'benign' in filename.lower():
                        label = 0  # 양성
                    elif 'malignant' in filename.lower():
                        label = 1  # 악성
                    else:
                        continue
                    
                    self.images.append(filepath)
                    self.labels.append(label)
        
        print(f"로드된 이미지: 양성 {self.labels.count(0)}개, 악성 {self.labels.count(1)}개")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        try:
            img = Image.open(img_path).convert('RGB')
            # 이미지 크기 조정
            img = img.resize((224, 224))
        except:
            # 실패 시 빈 이미지
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class BreastCancerCNN(nn.Module):
    """유방암 분류 CNN 모델 (ResNet 기반)"""
    def __init__(self, num_classes=2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
        
        super(BreastCancerCNN, self).__init__()
        # ResNet18 전이학습
        self.model = models.resnet18(pretrained=True)
        # 마지막 레이어 수정
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ImageClassifier:
    """이미지 분류기"""
    def __init__(self, device=None):
        if not TORCH_AVAILABLE:
            self.device = None
            self.model = None
            return
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
        ])
    
    def train(self, image_dir="image/Images", epochs=10, batch_size=8):
        """모델 학습"""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch가 필요합니다. pip install torch torchvision")
            return False
        
        print(f"디바이스: {self.device}")
        
        # 데이터셋 생성
        dataset = BreastCancerImageDataset(image_dir, transform=self.transform)
        
        if len(dataset) == 0:
            print("⚠️ 학습할 이미지가 없습니다.")
            return False
        
        # 데이터 분할
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 모델 생성
        self.model = BreastCancerCNN(num_classes=2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        # 학습
        best_val_acc = 0
        print(f"\n학습 시작: {epochs} epochs, {len(train_dataset)} train, {len(val_dataset)} val")
        
        for epoch in range(epochs):
            # 학습 단계
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 검증 단계
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, "
                  f"Val Acc={val_acc:.2f}%")
            
            scheduler.step()
            
            # 최고 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("image_classifier_model.pth")
                print(f"✅ 최고 모델 저장 (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n✅ 학습 완료! 최고 검증 정확도: {best_val_acc:.2f}%")
        return True
    
    def predict(self, image_path):
        """이미지 예측"""
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch를 사용할 수 없습니다.")
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        self.model.eval()
        
        # 이미지 로드 및 전처리
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {e}")
            raise ValueError(f"이미지를 로드할 수 없습니다: {e}")
        
        # 예측
        try:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            prob = probabilities[0].cpu().numpy()
            
            # 예측 결과 반환 (0=양성, 1=악성)
            prediction_label = '악성(M)' if predicted.item() == 1 else '양성(B)'
            
            return {
                'prediction': prediction_label,
                'probability': float(max(prob)),
                'malignant_prob': float(prob[1]),
                'benign_prob': float(prob[0])
            }
        except Exception as e:
            print(f"⚠️ 예측 실패: {e}")
            raise ValueError(f"예측 중 오류 발생: {e}")
    
    def save_model(self, path="image_classifier_model.pth"):
        """모델 저장"""
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        import torch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': self.device
        }, path)
        print(f"모델 저장: {path}")
    
    def load_model(self, path="image_classifier_model.pth"):
        """모델 로드"""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch가 필요합니다.")
            return False
        
        if not os.path.exists(path):
            print(f"⚠️ 모델 파일이 없습니다: {path}")
            return False
        
        try:
            import torch
            checkpoint = torch.load(path, map_location=self.device)
            self.model = BreastCancerCNN(num_classes=2).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ 모델 로드 완료: {path}")
            return True
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            return False

