# Indian Sign Language Recognition (IndiSignLang)

A CNN-based image classifier that recognizes Indian Sign Language alphabet signs from photos of hands, built with transfer learning on MobileNetV2.

## Overview

- **Task:** Multi-class image classification
- **Classes:** 24 letters (A–Y, excluding J and Z)
- **Approach:** MobileNetV2 (ImageNet weights, frozen) + custom classification head
- **Framework:** TensorFlow / Keras
- **Test accuracy:** 99.46% (742 / 746 correct)

## Dataset

- 4,972 images across 24 classes, one folder per class
- Class sizes range from 116 to 259 images (mild imbalance)
- No corrupted files
- Images resized to 224×224 for training
- Split: 70% train / 15% validation / 15% test (stratified, seed=42)

## Model Architecture

| Layer | Purpose |
|---|---|
| Input (224, 224, 3) | RGB image |
| Data augmentation | Random flip / rotation / zoom (training only) |
| Preprocessing | MobileNetV2-specific pixel scaling |
| MobileNetV2 (frozen) | Pretrained feature extractor |
| GlobalAveragePooling2D | 1,280 features |
| Dense(256, relu) | Learned combinations of features |
| Dropout(0.3) | Regularization |
| Dense(24, softmax) | Class probabilities |

- Total parameters: 2,592,088
- Trainable parameters: 334,104 (12.9%)
- Optimizer: Adam (lr=0.001)
- Loss: Sparse categorical cross-entropy
- Early stopping on validation loss (patience=10, best weights restored)

## Results

| Metric | Value |
|---|---|
| Best epoch | 16 |
| Validation accuracy (best epoch) | 99.60% |
| Validation loss (best epoch) | 0.0109 |
| Test accuracy | 99.46% |
| Test loss | 0.0148 |
| Training stopped at | Epoch 26 (of 40 max) |
| Saved model size | 13.01 MB |

20 of 24 classes scored a perfect 1.00 on precision, recall and F1. The only notable confusion was **N predicted as T** (3 images) and **G predicted as A** (1 image) — 4 total errors out of 746 test images.

## Repository Contents

- `indisignlang_documented.ipynb` — full notebook: EDA, data pipeline, model training, evaluation, error analysis, and single-image inference, with documentation of each step

## Limitations

- Results come from a single train/test split and a single training run (no cross-validation)
- Dataset images were taken against plain, controlled backgrounds; performance on cluttered backgrounds or new signers is untested
- Only static letters are covered (J and Z, which involve motion, are excluded)
- Class sizes are imbalanced (116–259 images per class) with no resampling or class weighting applied

## Possible Next Steps

- Evaluate on new, real-world images (different lighting, backgrounds, signers)
- Fine-tune the MobileNetV2 backbone instead of keeping it frozen
- Address the N/T confusion with targeted data or augmentation
- Export to TensorFlow Lite for a live webcam or mobile demo

## Tech Stack

TensorFlow, Keras, MobileNetV2, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

## Author

**Abhay Kadam**
[GitHub](https://github.com/ENGABHAY) · [LinkedIn](https://linkedin.com/in/kadamabhay)
