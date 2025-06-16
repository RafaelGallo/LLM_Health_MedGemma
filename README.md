# 🏥 LLM Health MedGemma

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-blueviolet)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Loguru](https://img.shields.io/badge/Loguru-Logging-lightgrey)
![TQDM](https://img.shields.io/badge/TQDM-Progress%20Bars-blue)
![JupyterLab](https://img.shields.io/badge/JupyterLab-Notebook-yellowgreen)
![Google Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-9cf)

![image](https://github.com/user-attachments/assets/36588b73-5196-47e1-90d3-e4386d3674ca)


## 📌 Business Problem: Development of a Multimodal AI System for Medical Image Diagnosis Support

### Business Context

Hospitals, clinics, and diagnostic labs face challenges analyzing large volumes of medical images like **MRI**, **Mammograms**, **Histopathology slides**, and **Dermoscopic images**.
Delays in critical diagnoses can directly impact patient outcomes.

### Project Objective

Develop an AI-based system that:

* **Automatically analyzes multiple types of medical images**
* Detects and classifies:

  * Brain Tumors
  * Breast Cancer
  * Skin Cancer
  * Histopathological Abnormalities
* **Generates AI-based preliminary diagnostic reports**

## 📂 Datasets Used

| Dataset Path                                                                                    | Image Type                   | Diagnosis / Condition                         |
| ----------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------- |
| `/kaggle/input/brain-tumor-dataset-segmentation-and-classification`                             | MRI (Brain Imaging)          | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| `/kaggle/input/breast-cancer-detection`                                                         | Mammography (X-ray)          | Breast Cancer, BI-RADS abnormalities          |
| `/kaggle/input/processedoriginalmonuseg`                                                        | Histopathology (Cell Slides) | Nuclei analysis, cell segmentation            |
| `/kaggle/input/skin-cancer-detection` + `/kaggle/input/ham1000-segmentation-and-classification` | Dermoscopy (Skin Lesions)    | Melanoma, Nevi, Basal Cell Carcinoma, etc.    |

## ✅ System Inputs & Outputs

**Inputs:**

* Upload of medical image
* Image type selection
* AI LLM Inference (MedGemma with prompt)

**Outputs:**

* Lesion/Tumor classification
* Radiology/Dermatology/Pathology Report (Markdown style)
* Optional CSV or PDF export

## 🎯 Business Impact

| Metric                       | Target                 |
| ---------------------------- | ---------------------- |
| Diagnostic time per case     | -50% vs manual review  |
| Classification accuracy      | ≥ 85% across all tasks |
| Critical case identification | ≥ 90% sensitivity      |

## 🛠️ Technologies Used

* **LLM:** MedGemma 4b-it
* **Frameworks:** Hugging Face Transformers, PyTorch
* **Environment:** Kaggle, Google Colab
* **Language:** Python
* **Visualization:** Jupyter Notebooks + Markdown

## 📁 Project Directory Structure

```
├── data/
├── docs/
├── env/
├── img/
├── input/
├── models/
├── notebooks/
├── output/
├── references/
├── reports/
├── src/
└── tests/
```

## 📓 Main Notebook

**Notebook:**
`notebooks/breast-segmentation-llm-medgemma.ipynb`

**Features:**

* Kaggle authentication
* Dataset extraction
* Image reading and preview
* Prompt Engineering
* LLM inference with MedGemma
* Structured markdown-based medical reports

## ✅ Example LLM Diagnostic Results (MedGemma Outputs)

### Histopathology - Cellular Analysis

![image](https://github.com/user-attachments/assets/b2ceb40e-517c-464c-9180-0c111d3ede2c)

**Prompt:**

> Classify tissue abnormality and identify cancer indicators.

**MedGemma Response:**

> Pleomorphism, hyperchromasia, and nucleoli prominence suggest malignancy.

---

### Breast Cancer - Mammogram BI-RADS Assessment

![image](https://github.com/user-attachments/assets/03b180c6-3a88-439f-9853-8d3930e88d75)

![image](https://github.com/user-attachments/assets/365a3741-6bf1-40d9-9766-a46b1f76bfd7)

**Response Summary:**

* **Density:** C (Heterogeneously Dense)
* **Mass:** Well-defined, irregular shape
* **BI-RADS:** Category 4 (Probably Benign)

### Brain Tumor Detection - Multiple MRI Cases

**Example 1: Sphenoid Wing Meningioma**

![image](https://github.com/user-attachments/assets/be9bcf7d-19c7-47b0-bc7b-a2906f4befa0)

**Example 2: Temporal Meningioma**

![image](https://github.com/user-attachments/assets/53e56e63-6d5c-47d8-8c19-772a03f34666)

**Example 3: Suprasellar Meningioma**

![image](https://github.com/user-attachments/assets/3aa92b36-2c0f-48e7-b762-e783166db9f3)

**Example 4: No Tumor Case**

![image](https://github.com/user-attachments/assets/cd2aea53-de92-464d-833f-81a7f5bf435d)

### Skin Lesion - Dermoscopy Analysis (Melanocytic Nevus)

![image](https://github.com/user-attachments/assets/87ae3289-4170-46e7-a101-0921a667048d)

**AI Report Summary:**

* **Asymmetry:** Present
* **Border:** Irregular
* **Color:** Multiple pigmentation areas
* **Size:** \~6mm
* **Classification:** Melanocytic Nevus (likely benign)

## 🧑‍💻 Environment Setup

### Python Version:

Python 3.8+

### Requirements:

```
black
flake8
ipython
isort
jupyterlab
loguru
matplotlib
mkdocs
notebook
numpy
pandas
pip
pytest
python-dotenv
scikit-learn
tqdm
typer
-e .
transformers
torch
Pillow
```

If using **Colab**:

```bash
pip install kaggle huggingface_hub
```

Also configure Hugging Face API Token (`HF_TOKEN`).

## ✅ How to Run (Colab / Local)

1. Upload your `kaggle.json` file.
2. Download datasets via Kaggle API.
3. Extract datasets to `/content/dataset/llm/` (Colab) or `/data/` (local).
4. Open and run:

```
notebooks/breast-segmentation-llm-medgemma.ipynb
```

## 📚 References

* **Google Health - MedGemma Official Repository:**
  Google Health. *MedGemma: Multimodal Language Model for Medical Image Understanding.*
  Available at: [https://github.com/Google-Health/medgemma](https://github.com/Google-Health/medgemma)
  Accessed: June 15, 2025.

* **Fine-Tuning MedGemma with Hugging Face Transformers:**
  Google Health. *Fine-tune MedGemma with Hugging Face Transformers (Example Notebook).*
  Available at: [https://github.com/Google-Health/medgemma/blob/main/notebooks/fine\_tune\_with\_hugging\_face.ipynb](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
  Accessed: June 15, 2025.
  
## ⚠️ Disclaimer

This project is for **research and educational purposes only**. **Not for clinical use or patient treatment.**
