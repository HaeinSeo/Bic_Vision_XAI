# 💡 Bic_Vision_XAI 💡  

<p align="center">
  <img src="https://github.com/HaeinSeo/Bic_Vision_XAI/blob/main/hae.png" width="210" alt="Bic_Vision_XAI Logo">
</p>

<h1 align="center"> ✨ Bic_Vision_XAI ✨  
<p align="center" style="font-size:18px;">
Vision-driven Explainable AI System for Breast Cancer Diagnosis
</p>
</h1>

<p align="center">
This project leverages <b>Explainable AI (XAI)</b> to provide transparent and interpretable predictions for breast cancer cell image classification.  
It integrates <b>CNN-based models, feature-based ML models, and Vision-Language Models (VLM)</b> to enhance trust and usability in clinical decision support. 🩺
</p>

---

### 🔎 Project Overview

The **Bic_Vision_XAI** web application analyzes breast cancer cell images through **two complementary ML systems**:

- **CNN-based Deep Learning Model** — Extracts visual patterns directly from microscopic images  
- **Random Forest based on 30 numerical features** — Leverages classical ML to ensure stability and transparency  

The system integrates **SHAP, LIME, and LLaVA (VLM)** to visually and linguistically explain predictions.

---

### ✨ Key Features

| Feature | Description |
|--------|------------|
| 🔬 **Dual Classification** | CNN + Random Forest hybrid inference |
| 💡 **XAI Interpretation** | SHAP (global/feature), LIME (local), VLM (text explanation) |
| 🖼 **Cell Detection** | Automatic bounding box localization |
| ⚡ **Real-time Web UI** | Upload → Predict → Explain on browser |

---

### 🛠 Tech Stack

| Category | Technology |
|--------|------------|
| Backend | Flask |
| ML Core | PyTorch |
| Feature ML & XAI | scikit-learn, SHAP, LIME |
| Vision | OpenCV, scikit-image |
| VLM | LLaVA |
| Frontend | HTML, CSS, JavaScript |

---

### 📊 Datasets Used

This project uses **two independent datasets** to support both numerical-feature modeling and image-based modeling:

#### **1️⃣ Breast Cancer Wisconsin (Diagnostic) Dataset**
🔗 <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>  
- **569 samples / 30 continuous features**
- Labels: **Malignant / Benign**

#### **2️⃣ Breast Cancer Cell Segmentation Dataset (Andrewmvd)**
🔗 <https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation>  
- TIFF & PNG breast cancer cell images
- Used for **CNN image classification** & **cell detection**

---

### 📦 Installation & Execution

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/HaeinSeo/Bic_Vision_XAI.git
cd Bic_Vision_XAI
2️⃣ Create environment
bash
코드 복사
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
bash
코드 복사
pip install -r requirements.txt
4️⃣ Run the app
bash
코드 복사
python app.py
Then open:

➡ http://localhost:5000

📁 Project Structure
cpp
코드 복사
Bic_Vision_XAI/
├── app.py                   
├── model_utils.py           
├── image_classifier.py      
├── image_utils.py           
├── vlm_utils.py             
├── requirements.txt         
├── templates/
│   └── index.html           
├── static/                 
├── uploads/               
└── models/
    ├── breast_cancer_model.joblib
    └── image_classifier_model.pth
🔧 Model Training
Random Forest — Uses 30-feature CSV

CNN Image Model — Trained directly with cell images

If model files missing → auto-trigger training at launch

🐛 Troubleshooting
Issue	Solution
GPU not recognized	reinstall PyTorch with CUDA
VLM fails	ensure stable internet
Weak cell detection	tune parameters in image_utils.py

👤 Developer / Research Lead
Name	Role
Seo Haein	Creator & Lead Developer (ML/XAI Backend)

📧 Contact & Issues
➡ https://github.com/HaeinSeo/Bic_Vision_XAI/issues

📚 Citation (Datasets)
Breast Cancer Wisconsin (Diagnostic) Data Set — UCI ML Repository
Breast Cancer Cell Segmentation Dataset — Kaggle (Andrewmvd)

<p align="center"> ⭐ If this project inspires you, please consider giving it a star! ⭐ </p>
