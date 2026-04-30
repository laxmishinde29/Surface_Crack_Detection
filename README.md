#  Surface Crack Detection using CNN & MLOps Pipeline
---

## 📌 Project Overview

This project automatically detects **cracks in concrete surfaces** using a **Convolutional Neural Network (CNN)**.  
It follows an end-to-end **Industrial MLOps Pipeline** — from raw image data collection all the way to deployment-ready model saving.

Manual inspection of buildings and infrastructure is slow, error-prone, and risky.  
This system replaces that process with an AI model that can analyze surface images and instantly classify them as **Crack** or **No Crack**.

## 📦 Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

Or click here → [Concrete Crack Images Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

---

## 🎯 Problem Statement

| Problem | Solution |
|---|---|
| Manual inspection is slow | Automated image classification |
| Human errors possible | CNN-based objective detection |
| Large buildings need fast scanning | Scalable ML model |
| Safety risks if cracks missed | High-recall model with threshold tuning |

---

## 🏗️ MLOps Pipeline

```
Business Problem ──► Data Collection ──► Data Validation ──► Data Splitting
                                                                    │
                                                                    ▼
                         CNN Model Architecture ◄── Data Augmentation ◄── Image Preprocessing
                                    │
                                    ▼
                             Model Training
                                    │
                                    ▼
                         Validation Monitoring + Industrial Callbacks
                                    │
                                    ▼
               Final Testing ──► Confusion Matrix ──► Model Saving ──► Deployment
                                                                            │
                                                                            ▼
                                                               Monitoring & Retraining
```

### Pipeline Stages

| Step | Stage | Description |
|---|---|---|
| 1 | Business Problem | Detect surface cracks automatically |
| 2 | Data Collection | Positive (crack) & Negative (no crack) images |
| 3 | Data Validation | Check corrupt files, duplicates, class balance |
| 4 | Data Splitting | 70% Train / 15% Validation / 15% Test |
| 5 | Image Preprocessing | Resize 128x128, normalize pixels 0 to 1 |
| 6 | Data Augmentation | Rotation, zoom, flip to prevent overfitting |
| 7 | CNN Architecture | 3 Conv blocks + Dense + Dropout + Sigmoid |
| 8 | Model Training | Adam optimizer, binary cross-entropy loss |
| 9 | Validation Monitoring | val_accuracy and val_loss tracked per epoch |
| 10 | Industrial Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| 11 | Testing & Evaluation | Confusion matrix, classification report |
| 12 | Model Saving | Saved as .keras for reuse without retraining |
| 13 | Deployment | Mobile, CCTV, drone, web, factory |
| 14 | Monitoring | Continuous improvement with new data |

---

## 🧠 CNN Model Architecture

```
Input Image (128 x 128 x 3)
        │
        ▼
Conv2D - 32 filters - 3x3 - ReLU     ← Detects edges, lines, basic patterns
        │
        ▼
MaxPooling2D (2x2)                    ← Reduces size: 128x128 → 64x64
        │
        ▼
Conv2D - 64 filters - 3x3 - ReLU     ← Detects crack-like shapes
        │
        ▼
MaxPooling2D (2x2)                    ← Reduces size: 64x64 → 32x32
        │
        ▼
Conv2D - 128 filters - 3x3 - ReLU    ← Detects full crack structures
        │
        ▼
MaxPooling2D (2x2)                    ← Reduces size: 32x32 → 16x16
        │
        ▼
Flatten                               ← 3D feature maps → 1D vector
        │
        ▼
Dense (128 neurons, ReLU)             ← Fully connected decision layer
        │
        ▼
Dropout (0.5)                         ← Drops 50% neurons to reduce overfitting
        │
        ▼
Dense (1 neuron, Sigmoid)             ← Output probability: 0.0 to 1.0
        │
        ▼
Output: Crack / No Crack
```

---

## 📁 Dataset Structure

```
Concrete Crack Dataset/         ← Original Dataset
├── Positive/                   → Crack Images     (~20,000 images)
└── Negative/                   → No Crack Images  (~20,000 images)


CrackDataset/                   ← After Splitting
├── train/
│   ├── Crack/                  → 80% of Positive images
│   └── NoCrack/                → 80% of Negative images
└── test/
    ├── Crack/                  → 20% of Positive images
    └── NoCrack/                → 20% of Negative images
```

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| Deep Learning Framework | TensorFlow 2.x, Keras |
| Data Handling | NumPy, OS, Shutil, Random |
| Visualization | Matplotlib |
| Model Evaluation | Scikit-learn |
| Model Save Format | .h5 / .keras |

---

## ⚙️ How to Run

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/surface-crack-detection.git
cd surface-crack-detection
```

### Step 2 — Install Dependencies
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Step 3 — Prepare Dataset
Place your dataset in the project folder with this structure:
```
Concrete Crack Dataset/
├── Positive/
└── Negative/
```

### Step 4 — Run the Script
```bash
python crack_detection.py
```

### Step 5 — Predict a Single Image
```python
Predict_Crack("path/to/your/image.jpg")
```

---

## 📊 Model Evaluation Metrics

| Metric | What It Means |
|---|---|
| Accuracy | Overall correct predictions out of total |
| Precision | Out of predicted cracks, how many were actually cracks |
| Recall | Out of actual cracks, how many were successfully detected |
| F1-Score | Balance between Precision and Recall |
| Confusion Matrix | Shows TP / FP / FN / TN in a table |

### Confusion Matrix

```
                  Predicted
                Crack    No Crack
Actual  Crack    TP         FN       ← FN = Crack Missed (dangerous!)
       NoCrack   FP         TN       ← FP = False Alarm (acceptable)
```

> ⚠️ For safety-critical systems, Recall is the most important metric.
> Missing a real crack (FN) is far more dangerous than a false alarm (FP).

---

## 🔁 Industrial Callbacks Used

| Callback | Purpose |
|---|---|
| EarlyStopping | Stops training if validation loss does not improve |
| ModelCheckpoint | Saves only the best model during training |
| ReduceLROnPlateau | Reduces learning rate when model gets stuck |

---

## 🚀 Real-World Deployment Use Cases

| Use Case | Description |
|---|---|
| 📱 Mobile App | On-site inspection by workers |
| 📷 CCTV Camera | Real-time monitoring of construction sites |
| 🚁 Drone Inspection | Large infrastructure like bridges, dams |
| 🌐 Web Upload System | Remote image analysis via browser |
| 🏭 Factory Quality Check | Surface defect detection in manufacturing |

---

## 📈 Key Concepts Demonstrated

- ✅ Binary Image Classification using CNN
- ✅ Data Augmentation (Rotation, Zoom, Horizontal Flip)
- ✅ End-to-End Industrial MLOps Pipeline (14 stages)
- ✅ Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- ✅ Confusion Matrix and Classification Report Analysis
- ✅ Model Saving and Reloading without Retraining
- ✅ Single Image Prediction Function
- ✅ Proper Train / Validation / Test Split Strategy

---

## 👤 Author : Laxmi Shinde
