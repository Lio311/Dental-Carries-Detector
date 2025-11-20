# 🦷 Dental Caries (Cavity) Detector - YOLOv8 Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dentalcarriesdetector.streamlit.app)

This is an end-to-end Computer Vision project designed to detect dental caries (cavities) in X-ray images. The system utilizes a custom **YOLOv8-OBB** (Oriented Bounding Box) model trained via **Transfer Learning** and features an interactive deployment on **Streamlit Cloud**.

---

## ⚙️ Technologies and Environment

* **Language:** **Python 3.10+**
* **Model:** **Ultralytics YOLOv8n** (Pre-trained on COCO)
* **Core Libraries:** **PyTorch** and **OpenCV (cv2)**
* **Application:** **Streamlit** (for interactive UI)
* **Training Environment:** Google Colab (Tesla T4 GPU)
* **Deployment Environment:** Streamlit Cloud (Linux container)
* **Data Source:** Roboflow Universe

---

## 🚀 Project Pipeline: Methodology & Results

The project was defined by the transition from a contaminated multi-class dataset to a highly focused, accurate single-class detection model.

### 1. Data Processing and Cleanup (The Critical Step)

* **Initial Problem:** The raw dataset was **contaminated**, containing 12 classes (e.g., `Caries`, `RA1`, `class_10` texts). This lack of focus skewed initial training results.
* **Solution (Label Fusion):** A custom preprocessing script was executed to clean and unify the labels:
    1.  It scanned all **2,706 annotations** across all label files (`.txt`).
    2.  It **merged all 12 original classes** (including `Caries`, `RAx`, and metadata classes) into a single, unified target class: **ID 0** (`Caries`).
    3.  The final `data.yaml` was rewritten to define only one class (`nc: 1`), ensuring the model was focused exclusively on the caries detection task.
* **Result:** The model was trained on a high volume of unified, relevant data.

### 2. Advanced Model Training (Fine-Tuning)

Training utilized **Transfer Learning** and **Manual Hyperparameter Tuning** to optimize the detection process:

| Parameter | Value | Academic Rationale |
| :--- | :--- | :--- |
| **Model Base** | `yolov8n.pt` | Used Transfer Learning (COCO weights) for high initial visual understanding. |
| **Optimizer** | `AdamW` | Manually selected highly efficient optimizer for stable convergence. |
| **Learning Rate (`lr0`)** | `0.002` | Custom setting to control step size during gradient descent. |
| **Epochs/Patience** | `120 / 30` | Defined 120 cycles with Early Stopping (`patience=30`) to prevent overfitting. |
| **Augmentations** | `degrees=15`, `flipud=0.5`, `scale=0.6` | Manually specified augmentations to artificially expand the dataset and improve model robustness. |

### 3. Final Performance Metrics

* **The final model (`best.pt`) achieved a very strong score on the validation data:**

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **mAP50 (Main Score)** | **82.6%** | **Excellent:** The overall detection accuracy is very strong for a final project. |
| **Precision** | **86.2%** | **Reliability:** When the model marks a cavity, it is correct in 86.2% of cases. |
| **Recall** | **75.7%** | **Effectiveness:** The model successfully finds 75.7% of all actual cavities in the test images. |

---

## 🚀 Local Setup & Run

### Required Files in Root Directory:

* `app.py`
* `requirements.txt`
* `packages.txt` (for Linux system dependencies)
* `best.pt` (The trained model file)

### How to Run:

1.  Clone this repository and navigate to the root directory.
    ```bash
    git clone [https://github.com/your-username/dental-caries-detector.git](https://github.com/your-username/dental-caries-detector.git)
    cd dental-caries-detector
    ```
2.  Install the required Python packages (ensure your virtual environment is active):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
