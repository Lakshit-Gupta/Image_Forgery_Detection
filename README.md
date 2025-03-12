#  🖼️Image Forgery Detection using CNN
This project leverages deep learning techniques to detect splicing forgeries in images. By utilizing **Error Level Analysis (ELA)** for feature extraction and a CNN-based classifier optimized with focal loss, the model effectively identifies manipulated images. It is designed for forensic applications, ensuring reliable detection of image tampering through robust evaluation metrics.

---

##  💻Run Locally

Clone the project

```bash
git clone https://github.com/Lakshit-Gupta/Image_Forgery_Detection.git
```

Go to the project directory

```bash
cd Image_Forgery_Detection
```


## 📥Dataset And Model Download
The dataset and model for this project is available in the **[Releases](https://github.com/Lakshit-Gupta/Image_Forgery_Detection/releases)** section. Click the link to download the files.  

## ⚙️Resolving Dependencies
If using anaconda use the following command in base(root) terminal of anaconda to resolve all the depedencies
```bash
conda env create -f tf.yaml ```
```
```bash
conda activate tf ```
```
## 🚀How to run the Application
Open the main_app.py and run the followind command in the Terminal
```bash
streamlit run main_app.py  ```
``` 
---
 
##  🐳 **Docker Setup**

### CPU Version

```bash
docker pull lakshitgupta/forgery_detection_image:v14.0
```
```bash
docker run -p 8501:8501 lakshitgupta/forgery_detection_image:v14.0
```
Then go to:
```bash
localhost:8501
```
### GPU Version

```bash
docker pull lakshitgupta/forgery_detection_prediction-image:v15.0
```
```bash
docker run --gpus all -p 8501:8501 forgery_detection_prediction-image:v15.0
```
Then go to:
```bash
localhost:8501
```

---
## 📂 Dataset

The dataset is split into three categories:

- **Train:** 4498 Authentic (Class 0), 965 Tampered (Class 1)
- **Validation:** 964 Authentic (Class 0), 206 Tampered (Class 1)
- **Test:** 965 Authentic (Class 0), 208 Tampered (Class 1)

The dataset is structured as follows:

```
train_val_test_split_384x384/
│── train/
│   ├── au/   # Authentic Images
│   ├── tp/   # Tampered Images
│
│── val/
│   ├── au/
│   ├── tp/
│
│── test/
│   ├── au/
│   ├── tp/
```




---

## 🏗️ Model Architecture

This model uses transfer learning with a **VGG19** backbone, pretrained on ImageNet, to extract features from images. The final classification layers are customized for splicing forgery detection.

- **Feature Extraction**: Uses Error Level Analysis (ELA) to highlight manipulated regions.
- **Pretrained Backbone**: VGG19 (transfer learning from ImageNet).
- **Fully Connected Layers**: Batch normalization and dropout (0.5) for regularization.
- **Optimizer**: Adam for stable training.
- **Loss Function**: Focal Loss to handle class imbalance.
### 🔹 Model Summary

```
Input: (384, 384, 3)
VGG19 Backbone (Pretrained on ImageNet, Transfer Learning)
Fully Connected Layers (With BatchNorm & Dropout)
Dropout (0.5)
Output: Sigmoid Activation (Binary Classification)
```

---

## 🎯 Training & Optimization

- **Optimizer Choices:** AdamW(Recommended), RMSprop, SGD with Momentum
- **Loss Functions Explored:**
  - Binary Cross-Entropy (BCE)
  - Focal Loss (for imbalanced classes)
  - Dice Loss (for better foreground-background separation)
- **Callbacks:** Early Stopping, ReduceLROnPlateau, Model Checkpointing

---

## 📊 Results & Evaluation  

The model was evaluated on the **test set**, achieving the following performance:  

| Metric         | Score  |
|---------------|--------|
| **Accuracy**  | **96.20%** |
| **Precision** | **89.00%** |
| **Recall**    | **89.42%** |
| **F1-Score**  | **89.21%** |

### 📌 Confusion Matrix  
The confusion matrix helps in understanding model performance in terms of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):
| Actual \ Predicted | ✅0 (Authentic) | ❌1 (Tampered) |
|--------------------|--------------|--------------|
| **0 (Authentic)** |🟢 952           | 🔴24           |
| **1 (Tampered)**  |🔴 16           | 🟢192           |

- **🟢True Negative (TN)**: Authentic image correctly classified as authentic.  
- **🔴False Positive (FP)**: Authentic image misclassified as tampered.  
- **🔴False Negative (FN)**: Tampered image misclassified as authentic.  
- **🟢True Positive (TP)**: Tampered image correctly classified as tampered. 
---
## 📌 Future Improvements  

- **🖼️ Enhance noise analysis using PRNU-based techniques.**  
- **🛠️Integrate attention mechanisms for improved localization of forged regions.**  
- **🔍Explore contrastive learning for improved feature representation.**  
- **📡Leverage Fourier Transform Analysis for frequency-based anomaly detection.**  
- **🖌️Incorporate Residual Pixel Analysis (RPA) with Error Level Analysis (ELA).**  
- **🧩 Refine splicing detection using edge and boundary irregularities.**  
- **💡Improve model robustness by analyzing lighting and shadow inconsistencies.**  

---

## 🤝 Contributors

- **[Lakshit Gupta]**
- **[Manan Goel]**

For any questions, feel free to reach out!

---

## 🏆 Acknowledgments

- CASIA 2.0 Dataset
- Research papers on image forensics

