# **Art_Classification**
# Abstract
This project aims to classify Style in the WikiArt dataset using a convolutional-recurrent architecture. The model follows a multi-phase training strategy with fine-tuning and semi-supervised learning. Evaluation metrics include accuracy, precision, recall, F1-score, and Top-5 accuracy. Outlier detection methods like Isolation Forest and SHAP will be used to identify misclassified artworks. The goal is to achieve >85% accuracy while handling ambiguous or mislabeled paintings.
# Approach
# Dataset
Dataset: https://www.google.com/url?q=https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%2520Dataset/README.md&sa=D&source=editors&ust=1743430855089395&usg=AOvVaw2yJFpGiN9h1XH20jOs9qhY

https://github.com/nithika987/Art_Classification/blob/main/utils/dataset_utils.py

1. Filtering & Cleaning: The dataset is preprocessed to remove duplicate images and entries.

2. Merging & Sampling: Classes can be merged or dropped based on predefined criteria and a fixed number of images per class can be sampled using different strategies (Drop, Replace, Max).

3. Train-Validation-Test Split: The dataset is split into train, validation, and test sets based on the selected classification target (Style, Genre, or Artist).

4. Choosing 8 Styles out of 27: 8 styles with clear visual distinctions are selected for classification.

# Model
https://github.com/nithika987/Art_Classification/blob/main/models/crn.py

1. Data Augmentation & Preprocessing:
   
Extensive augmentation (flip, rotation, zoom, brightness, contrast, translation, crop, Gaussian noise).

Uses EfficientNetV2S preprocessing for optimized feature extraction.

2. EfficientNetV2-S as Feature Extractor:
   
Pretrained EfficientNetV2-S backbone with first 50% of layers frozen.

Extra Conv2D and BatchNormalization layers to refine features.

3. ConvLSTM for Temporal Feature Learning:
   
Reshapes CNN output into a time-series format.

Two ConvLSTM2D layers capture sequential spatial dependencies.

4. Self-Attention Mechanism:

Applies Attention() to learn feature importance dynamically.

Skip connection with Add() enhances feature representation.

5. Feature Pyramid Network (FPN) for Multi-Scale Learning:
   
Generates multi-resolution features via Conv2D + Resizing.

Aggregates with Concatenate() for better classification robustness.

6. Regularization & Final Classification:
   
Dropout (0.4) to prevent overfitting.

Fully connected Dense layer with softmax activation for classification.
# Evaluation Metrics
Accuracy & Loss – Measure overall model performance during training and validation. Accuracy shows correct predictions, while loss indicates model confidence.

Precision, Recall, and F1-score – Evaluate class-wise performance, especially for imbalanced datasets, ensuring fair assessment beyond just accuracy.

Macro & Weighted Averages – Macro averages treat all classes equally, while weighted averages account for class imbalance, providing a balanced evaluation.
# Result Analysis

1. **Overfitting Risk** – Train and validation accuracy are similar (~75%), but test accuracy drops to 71.3%, indicating moderate generalization but possible overfitting.
   
2. **Class Imbalance Issue** – Some classes (e.g., Class_0, Class_1) have very low precision and recall, while dominant classes (Class_4, Class_5) perform better.
   
3. **Poor Macro F1-score** – The model struggles with rare classes, as indicated by the low macro F1-score (0.12), suggesting the need for better data balancing or loss adjustments.  

---

### **Evaluation Summary**  

| Metric         | Value  |  
|---------------|--------|  
| **Train Accuracy**   | 75.06% |  
| **Train Loss**       | 1.0000 |  
| **Validation Accuracy** | 74.44% |  
| **Validation Loss**  | 1.0389 |  
| **Test Accuracy**    | 71.30% |  
| **Macro Precision**  | 0.12   |  
| **Macro Recall**     | 0.12   |  
| **Macro F1-score**   | 0.12   |  
| **Weighted Precision** | 0.17  |  
| **Weighted Recall**   | 0.17  |  
| **Weighted F1-score** | 0.17  |  

Test 1: Correct prediction
![image](https://github.com/user-attachments/assets/ad452e05-87eb-4d5a-88f4-1600495f2952)
Test 2: Correct Prediction
![image](https://github.com/user-attachments/assets/2b3dbfe3-411f-4e1a-9b5a-015eb0e4eebb)

Test 3: Wrong Prediction
![image](https://github.com/user-attachments/assets/a538ab17-52f2-4b41-a287-42c8ecff8bd2)



# Future Scope
Deal with class imbalance

Expandin to Art, Style and Genre classification

Improving accuracy


# 🎨 Art_Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract
This project classifies **art styles** in the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%2520Dataset/README.md) using a **convolutional-recurrent network**.  
The model employs **multi-phase training** with fine-tuning and semi-supervised learning.  

**Goals:**
- Achieve **>85% accuracy**
- Handle ambiguous or mislabeled artworks
- Identify misclassified artworks using **Outlier Detection** (Isolation Forest, SHAP)

---

## Dataset
- **Source:** [WikiArt Dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%2520Dataset/README.md)  
- **Utilities:** [dataset_utils.py](https://github.com/nithika987/Art_Classification/blob/main/utils/dataset_utils.py)  

**Preprocessing Steps:**
1. **Filtering & Cleaning:** Remove duplicate images and irrelevant entries
2. **Merging & Sampling:** Merge/drop classes; fixed samples per class
3. **Train/Validation/Test Split:** Based on Style, Genre, or Artist
4. **Select 8 Styles:** Choose styles with clear visual distinction

---

## Model Architecture
- **Implementation:** [crn.py](https://github.com/nithika987/Art_Classification/blob/main/models/crn.py)

**Components:**
1. **Data Augmentation & Preprocessing:** Flip, rotation, zoom, brightness/contrast, Gaussian noise; EfficientNetV2S preprocessing
2. **EfficientNetV2-S Feature Extractor:** Pretrained backbone, first 50% layers frozen; additional Conv2D + BatchNorm layers
3. **ConvLSTM Layers:** Reshapes CNN output into a time-series format; Two ConvLSTM2D layers capture sequential spatial dependencies.
4. **Self-Attention:** Learn feature importance dynamically; Attention mechanism with skip connections enhances feature representation
5. **Feature Pyramid Network (FPN) for Multi-Scale Learning :** Generates multi-resolution features via Conv2D + Resizing; Aggregates with Concatenate() for better classification robustness.
6. **Regularization & Classification:** Dropout (0.4), Dense softmax output

---

## Evaluation Metrics
| Metric | Description |
|--------|-------------|
| Accuracy & Loss | Overall performance during training & validation |
| Precision, Recall, F1 | Class-wise performance for imbalanced datasets |
| Macro & Weighted | Macro: all classes equally; Weighted: account for class imbalance |

---

## Results & Analysis
| Metric         | Value  |  
|---------------|--------|  
| **Train Accuracy**   | 75.06% |  
| **Train Loss**       | 1.0000 |  
| **Validation Accuracy** | 74.44% |  
| **Validation Loss**  | 1.0389 |  
| **Test Accuracy**    | 71.30% |  
| **Macro Precision**  | 0.12   |  
| **Macro Recall**     | 0.12   |  
| **Macro F1-score**   | 0.12   |  
| **Weighted Precision** | 0.17  |  
| **Weighted Recall**   | 0.17  |  
| **Weighted F1-score** | 0.17  |  

**Observations:**
- 1. **Overfitting Risk** – Train and validation accuracy are similar (~75%), but test accuracy drops to 71.3%, indicating moderate generalization but possible overfitting.
- 2. **Class Imbalance Issue** – Some classes (e.g., Class_0, Class_1) have very low precision and recall, while dominant classes (Class_4, Class_5) perform better.
- 3. **Poor Macro F1-score** – The model struggles with rare classes, as indicated by the low macro F1-score (0.12), suggesting the need for better data balancing or loss adjustments.  


**Test Predictions:**
- ✅ Test 1: Correct Prediction

 ![image](https://github.com/user-attachments/assets/ad452e05-87eb-4d5a-88f4-1600495f2952)
 
- ✅ Test 2: Correct Prediction

![image](https://github.com/user-attachments/assets/2b3dbfe3-411f-4e1a-9b5a-015eb0e4eebb)

- ❌ Test 3: Wrong Prediction

 ![image](https://github.com/user-attachments/assets/a538ab17-52f2-4b41-a287-42c8ecff8bd2)

---

## Future Scope
- Address **class imbalance**
- Expand to **Art, Style & Genre classification**
- Improve **overall accuracy** with advanced techniques

---

## Tags
`#DeepLearning` `#ComputerVision` `#ConvLSTM` `#Attention` `#FeaturePyramidNetwork` `#ArtClassification` `#EfficientNetV2` `#Python` `#PyTorch` `#TensorFlow`

---

## License
This project is licensed under the **MIT License** – see the [LICENSE](https://github.com/nithika987/Art_Classification/blob/main/LICENSE) file for details.



