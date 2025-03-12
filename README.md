# Splicing Forgery Detection using Multi-Feature CNN

## 📌 Project Overview

This project focuses on detecting splicing forgery in images using a convolutional neural network (CNN). The model utilizes forensic features like:

- **Error Level Analysis (ELA)**
- **Local Noise Variance**
- **Fourier Transform Magnitude**

These features are extracted and stacked into a three-channel image representation, which is then passed to the CNN for classification.

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

## ⚙️ Preprocessing Pipeline

1. **Error Level Analysis (ELA):** Highlights inconsistencies in image compression.
2. **Local Noise Variance Estimation:** Captures noise inconsistencies between different regions.
3. **Fourier Transform Analysis:** Detects unnatural frequency patterns in an image.
4. **Feature Stacking:** Combines the three extracted features into a three-channel image.

---

## 🚀 Model Architecture

The model is based on **VGG19**, with custom modifications:

- Feature extraction from ELA, noise variance, and Fourier transform.
- Fully connected layers with dropout to prevent overfitting.
- Optimizer: **AdamW / RMSprop / SGD with Momentum**
- Loss Function: **Binary Cross-Entropy / Focal Loss / Dice Loss**

### 🔹 Model Summary

```
Input: (384, 384, 3)
VGG19 Backbone (Pretrained)
Fully Connected Layers
Dropout (0.5)
Output: Sigmoid Activation
```

---

## 🔧 Training & Optimization

- **Optimizer Choices:** AdamW, RMSprop, SGD with Momentum
- **Loss Functions Explored:**
  - Binary Cross-Entropy (BCE)
  - Focal Loss (for imbalanced classes)
  - Dice Loss (for better foreground-background separation)
- **Callbacks:** Early Stopping, ReduceLROnPlateau, Model Checkpointing

---

## 📊 Results & Evaluation

- **Metrics Used:** Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix Analysis**
- **Precision-Recall Curve for class imbalance handling**

---

## 🛠 How to Run the Project

### 🔹 Step 1: Install Dependencies

```bash
pip install tensorflow tensorflow-addons numpy matplotlib seaborn scikit-learn pillow pywavelets opencv-python
```

### 🔹 Step 2: Prepare Dataset

Ensure your dataset is stored in `train_val_test_split_384x384/` as per the structure mentioned earlier.

### 🔹 Step 3: Run Training Script

```bash
python train.py
```

### 🔹 Step 4: Evaluate the Model

```bash
python evaluate.py
```

---

## 📌 Future Improvements

- **Incorporate PRNU-based noise analysis**
- **Use attention mechanisms for better localization**
- **Try contrastive learning for better representation learning**

---

## 🤝 Contributors

- **[Your Name]**

For any questions, feel free to reach out!

---

## 🏆 Acknowledgments

- CASIA 2.0 Dataset
- Research papers on image forensics

