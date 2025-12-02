<!-- ========================== --> <!-- Custom Font 적용 --> <!-- ========================== --> <style> @font-face { font-family: 'SchoolSafetyWing'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2511-1@1.0/HakgyoansimNalgaeR.woff2') format('woff2'); font-weight: normal; font-display: swap; } h1, h2, h3, h4, .custom-title { font-family: 'SchoolSafetyWing', 'Noto Sans KR', sans-serif; letter-spacing: 0.5px; } .readme-sub { font-family: 'Noto Sans KR', sans-serif; font-size: 16px; color: #444; } .section-title { font-family: 'SchoolSafetyWing', sans-serif; font-size: 22px; color: #993A6B; margin-top: 30px; } </style> <p align="center"> <img src="https://github.com/HaeinSeo/Bic_Vision_XAI/blob/main/hae.png" width="210" alt="Bic_Vision_XAI Logo"> </p> <h1 align="center" class="custom-title">✨ Bic_Vision_XAI ✨ <p align="center" style="font-size:18px; margin-top:8px;"> Vision-driven Explainable AI System for Breast Cancer Diagnosis </p> </h1> <p align="center" class="readme-sub"> This project leverages <b>Explainable AI (XAI)</b> to provide transparent and interpretable predictions for breast cancer cell image classification.<br> It integrates <b>CNN-based models, feature-based ML models, and Vision-Language Models (VLM)</b> to enhance trust and usability in clinical decision support. 🩺 </p>
<span class="section-title">🔎 Project Overview</span>

The Bic_Vision_XAI web application analyzes breast cancer cell images through two complementary ML systems:

CNN-based Deep Learning Model — Extracts visual patterns directly from microscopic images

Random Forest based on 30 numerical features — Leverages classical ML to ensure stability and transparency

The system integrates SHAP, LIME, and LLaVA (VLM) to visually and linguistically explain predictions.

<span class="section-title">✨ Key Features</span>
Feature	Description
🔬 Dual Classification	CNN + Random Forest hybrid inference
💡 XAI Interpretation	SHAP (global/feature), LIME (local), VLM (text explanation)
🖼 Cell Detection	Automatic bounding box localization
⚡ Real-time Web UI	Upload → Predict → Explain on browser
<span class="section-title">🛠 Tech Stack</span>
Category	Technology
Backend	Flask
ML Core	PyTorch
Feature ML & XAI	scikit-learn, SHAP, LIME
Vision	OpenCV, scikit-image
VLM	LLaVA
Frontend	HTML, CSS, JavaScript
<span class="section-title">📊 Datasets Used</span>

This project uses two independent datasets to support both numerical-feature modeling and image-based modeling:

1️⃣ Breast Cancer Wisconsin (Diagnostic) Dataset

🔗 https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

569 samples / 30 continuous features

Labels: Malignant / Benign

2️⃣ Breast Cancer Cell Segmentation Dataset (Andrewmvd)

🔗 https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation

TIFF & PNG breast cancer cell images

Used for CNN image classification & cell detection

<span class="section-title">🌺 Live Demo Video & Captured Images</span>

###🔗 Demo Video

https://github.com/user-attachments/assets/09491f00-293a-4764-b665-9f8fb0a628c1

<span class="section-title">🌿 Benign Samples</span>
![Image](https://github.com/user-attachments/assets/547c2952-3707-46c6-b5b0-2a5c0f6ab8bc)
![Image](https://github.com/user-attachments/assets/1368d6da-6137-4c1a-91e0-9dacb8c6b686)
![Image](https://github.com/user-attachments/assets/750d309f-c68c-4100-af5d-55cd205843f2)
![Image](https://github.com/user-attachments/assets/8e60211c-d840-4302-ad22-d2d01ef866f9)
![Image](https://github.com/user-attachments/assets/7d9af188-2d00-4dcc-bc23-f3931a03e9bd)
![Image](https://github.com/user-attachments/assets/6fe5fbf7-c5dc-44cc-a70d-5b24b1cd6fb6)
![Image](https://github.com/user-attachments/assets/cbd4576e-bf6f-4e9f-b55a-fa63dca21580)
![Image](https://github.com/user-attachments/assets/b7489ade-2b2b-4be4-9633-96c3ec424546)



<span class="section-title">🚨 Malignant Samples</span>
![Image](https://github.com/user-attachments/assets/a5040e23-43fb-4719-83f2-27db9b515c88)
![Image](https://github.com/user-attachments/assets/7b7c52e8-d8e1-4384-9a5d-2af02a2e787a)




<span class="section-title">📦 Installation & Execution</span>
1️⃣ Clone the repository
git clone https://github.com/HaeinSeo/Bic_Vision_XAI.git
cd Bic_Vision_XAI

2️⃣ Create environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
python app.py


➡ Open in browser:
http://localhost:5000

<span class="section-title">📁 Project Structure</span>
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

<span class="section-title">🔧 Model Training</span>

Random Forest — Uses 30-feature CSV

CNN Image Model — Trained using Breast Cancer Cell images

If model files are missing, the system auto-triggers training at launch.

<span class="section-title">🐛 Troubleshooting</span>
Issue	Solution
GPU not recognized	reinstall PyTorch with CUDA
VLM fails	ensure stable internet
Weak cell detection	tune parameters in image_utils.py
<span class="section-title">👤 Developer / Research Lead</span>
Name	Role
Seo Haein	Creator & Lead Developer (ML/XAI Backend)
<span class="section-title">📧 Contact & Issues</span>

➡ https://github.com/HaeinSeo/Bic_Vision_XAI/issues

<p align="center" style="font-family:'SchoolSafetyWing'; font-size:20px; color:#993A6B;"> ⭐ If this project inspires you, please consider giving it a star! ⭐ </p>
