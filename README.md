# 💡 Bic_Vision_XAI 💡  
<p align="center">
  <img src="https://github.com/HaeinSeo/Bic_Vision_XAI/blob/main/hae.png" width="200" alt="Bic_Vision_XAI Logo">
</p>

<p align="center" style="font-family: 'SchoolSafetyWing', 'Garamond', cursive; color:#993A6B;">
  <h1> Bic_Vision_XAI: A Vision-driven Explainable AI System for Breast Cancer Diagnosis 💥</h1>
</p>

<p align="center" style="font-size:16px; line-height:1.6;">
  This project leverages <b>Explainable AI (XAI)</b> technologies to provide transparent and interpretable predictions for breast cancer cell image classification. The system integrates advanced machine learning models and XAI techniques to enhance medical professionals' trust in automated diagnostic systems. 🩺
</p>

<style>
  @font-face {
    font-family: 'SchoolSafetyWing';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2511-1@1.0/HakgyoansimNalgaeR.woff2') format('woff2');
    font-weight: normal;
    font-display: swap;
}
</style>

---

### 🔎 **Project Overview**

The **Bic_Vision_XAI** is a web application designed to analyze breast cancer cell images using deep learning models. This system uses **CNN-based deep learning models** and **traditional feature-based Random Forest models** for classifying images as either **Benign** or **Malignant**. Additionally, the application uses cutting-edge **XAI techniques** such as **SHAP**, **LIME**, and **VLM** (Vision Language Models) to explain predictions and provide detailed visual and textual explanations of the model's decisions.

---

### ✨ **Key Features**

- **🔬 Dual System Image Classification**:  
   - **CNN-based Deep Learning Model**: Learns complex patterns in images.
   - **Traditional Feature-based Random Forest**: Classifies based on 30 numerical features extracted from the images.
   
- **💡 XAI (Explainable AI) Explanation**:  
   - **SHAP (SHapley Additive exPlanations)**: Quantifies the contribution of each feature to the prediction.
   - **LIME (Local Interpretable Model-agnostic Explanations)**: Explains specific predictions with localized insights.
   - **VLM (Vision Language Model - LLaVA)**: Generates detailed natural language explanations based on image and classification results.
   
- **🖼️ Cell Detection**:  
   - Automatically detects breast cancer cell regions within images and highlights them using **bounding boxes**.

- **🚀 Real-time Analysis**:  
   - Upload images directly to the web interface and receive immediate classification results and XAI explanations.

---

### 🛠️ **Tech Stack**

| Category        | Technology           | Description                                        |
| --------------- | -------------------- | -------------------------------------------------- |
| **Backend**     | Flask                | Web application server framework                   |
| **ML Core**     | PyTorch              | Deep learning framework for CNN models             |
| **ML/XAI**      | scikit-learn, SHAP, LIME | Classical ML models & XAI libraries                |
| **Vision**      | OpenCV, scikit-image | Image processing, feature extraction, cell detection |
| **VLM**         | LLaVA                | Vision Language Model for generating natural language explanations |
| **Frontend**    | HTML, CSS, JavaScript | Web interface and user experience design           |

---

### 📊 **Datasets Used**

This project utilizes two major datasets for training and evaluating the models:

1. **[Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** (UCI ML Repository)  
   - **Content**: 569 samples with 30 numerical features (e.g., radius, texture, perimeter, area) to classify tumors as **Malignant** or **Benign**.
   - **Source**: University of Wisconsin Diagnostic Center

2. **[Breast Cancer Cell Segmentation Dataset (Andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation)** (Kaggle)  
   - **Content**: Includes **TIFF** and **PNG** images of breast cancer cells, along with the diagnosis results (Benign/Malignant).
   - **Data Use**: Utilized for image-based deep learning classification and segmentation tasks, including **cell detection** and **image classification**.
   - **Link**: [Kaggle Breast Cancer Cell Segmentation Dataset](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation)

---

### 📦 **Installation and Setup**

1. **Clone the Repository**:

```bash
git clone https://github.com/HaeinSeo/bic_vision_xai.git
cd bic_vision_xai
