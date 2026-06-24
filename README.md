# Flower Recognition Using ResNet-18 & Custom CNN

> A deep learning image classification project comparing Transfer Learning (ResNet-18) with custom CNN architectures for multi-class flower recognition, achieving 93.76% test accuracy.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Best_Accuracy-93.76%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This project tackles multi-class flower image classification using deep learning. The goal is to compare the effectiveness of **Transfer Learning** (fine-tuning a pre-trained ResNet-18) against **custom CNNs trained from scratch** on a dataset of 4,317 flower images across 5 classes.

The project systematically experiments with multiple architectures and improvement strategies:

| Experiment | Architecture | Strategy | Test Accuracy |
|-----------|-------------|----------|---------------|
| 1 | ResNet-18 (frozen) | Transfer Learning - Feature Extraction | 89.61% |
| 2 | ResNet-18 (layer4 unfrozen) | Transfer Learning - Fine-Tuning | **93.76%** |
| 3 | CNN-2 (2 conv blocks) | From Scratch | ~62.80% |
| 4 | CNN-4 (4 conv blocks) | From Scratch | ~69.78% |
| 5 | CNN-4 + Early Stopping | From Scratch + Regularization | ~70.89% |
| 6 | CNN-4 + Data Augmentation + ES | From Scratch + Aug + Regularization | ~75.21% |

---

## Business Problem

Automated plant identification has applications across:

- **Agriculture** - Crop and weed identification for precision farming
- **Biodiversity monitoring** - Automated species cataloging for environmental surveys
- **Education** - Interactive botanical learning tools
- **E-commerce** - Visual search for flower delivery services

This project demonstrates that transfer learning dramatically outperforms training from scratch on small-to-medium image datasets, providing a practical blueprint for similar classification tasks.

---

## Dataset Description

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) |
| **Total Images** | 4,317 |
| **Classes** | 5 (Daisy, Dandelion, Rose, Sunflower, Tulip) |
| **Distribution** | Balanced (~800 images per class) |
| **Resolution** | Variable (approx. 320x240, ranging widely) |
| **Sources** | Scraped from Flickr, Google Images, Yandex |
| **Split** | 80% Train / 10% Validation / 10% Test |

### Data Characteristics
- **High variability**: Images differ significantly in lighting, angles, backgrounds, and aspect ratios
- **Balanced classes**: No oversampling needed
- **Variable resolutions**: Requires standardized preprocessing pipeline

---

## Methodology

### Data Preprocessing Pipeline

```
Raw Image --> Resize(255) --> CenterCrop(224x224) --> ToTensor() --> Normalize(ImageNet stats)
```

1. **Resize & CenterCrop** - Standardize to 224x224 (required by ResNet-18 architecture)
2. **Normalization** - ImageNet channel-wise stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
3. **Tensor Conversion** - Convert pixel values from [0-255] integers to [0.0-1.0] floats

### Data Augmentation (Improvement Experiment)
```
RandomResizedCrop --> RandomHorizontalFlip --> RandomRotation(20) --> ColorJitter --> Normalize
```

### Model Architectures

#### ResNet-18 (Transfer Learning)
- Pre-trained on ImageNet (1.28M images, 1,000 classes)
- **Feature Extraction**: All convolutional layers frozen; only new FC head trained
- **Fine-Tuning**: Layer4 + FC head unfrozen with differential learning rates (layer4: 1e-4, FC: 1e-3)
- Optimizer: Adam | Loss: CrossEntropyLoss | Batch Size: 32

#### Custom CNN-2 (From Scratch)
- 2 convolutional blocks (Conv2d > ReLU > MaxPool2d)
- Channels: 3 > 32 > 64
- FC classifier: Flatten > Linear(128) > ReLU > Linear(5)

#### Custom CNN-4 (From Scratch)
- 4 convolutional blocks
- Channels: 3 > 32 > 64 > 128 > 256
- FC classifier: Flatten > Linear(256) > ReLU > Linear(5)
- Variants tested with Early Stopping (patience=4) and Data Augmentation

---

## Exploratory Data Analysis

### Class Distribution
The dataset is well-balanced with approximately 800 images per class, eliminating the need for oversampling techniques.

### Image Resolution Analysis
Raw images exhibit high spatial heterogeneity, ranging from small thumbnails to high-resolution photographs. A scatter plot analysis confirmed the need for a standardized preprocessing pipeline.

### Sample Inspection
Qualitative review of random samples confirmed variations in:
- Aspect ratios (landscape vs. portrait)
- Lighting conditions (indoor, outdoor, overcast)
- Backgrounds (gardens, close-ups, mixed flora)
- Viewing angles (top-down, side, macro)

---

## Data Modeling

### Training Results

#### ResNet-18 - Feature Extraction (10 epochs)
```
Epoch  1: Train Acc: 0.7301 | Val Acc: 0.8260
Epoch  5: Train Acc: 0.8894 | Val Acc: 0.8747
Epoch 10: Train Acc: 0.9067 | Val Acc: 0.8910
--> Test Accuracy: 89.61%
```

#### ResNet-18 - Fine-Tuning (5 additional epochs)
```
Epoch 1: Train Acc: 0.8955 | Val Acc: 0.9072
Epoch 5: Train Acc: 0.9977 | Val Acc: 0.9095
--> Test Accuracy: 93.76%  (Best Model)
```

#### CNN-2 From Scratch (20 epochs)
```
Severe overfitting: Train Acc reached 0.999 but Val Acc plateaued at ~0.63
--> Evidence of insufficient model generalization
```

#### CNN-4 From Scratch (20 epochs)
```
Similar overfitting pattern: Train 0.999 vs Val ~0.70
--> Deeper architecture helped but still overfit
```

#### CNN-4 + Early Stopping
```
Early stopping triggered at epoch 9 (patience=4, best epoch=5)
--> Prevented further overfitting, val_loss improved to 0.8865
```

---

## Key Findings

1. **Transfer Learning dominates** - ResNet-18 fine-tuned achieved 93.76% accuracy vs. best CNN from scratch at ~75%, demonstrating a **+18.76% improvement** with less training time.

2. **Fine-tuning > Feature Extraction** - Unfreezing layer4 with differential learning rates boosted accuracy from 89.61% to 93.76% (+4.15%).

3. **Custom CNNs overfit severely** - Both CNN-2 and CNN-4 reached ~99.9% training accuracy but only 62-70% validation accuracy, showing the models memorized rather than learned.

4. **Early Stopping is essential** - Reduced CNN-4 training from 20 to 9 epochs and improved generalization by restoring best-epoch weights.

5. **Data Augmentation helps but isn't enough** - Augmentation improved CNN-4 generalization but still couldn't match pre-trained features from ImageNet.

6. **Pre-trained features are powerful** - ImageNet features (edges, textures, shapes) transfer remarkably well to botanical classification, even though ImageNet contains no dedicated flower categories.

---

## Recommendations

1. **Use Transfer Learning for small datasets** - With < 10K images, always start with a pre-trained model rather than training from scratch.

2. **Progressive unfreezing** - Start with frozen feature extraction, then fine-tune deeper layers with lower learning rates.

3. **Monitor validation loss** - Use Early Stopping to prevent overfitting, especially with custom architectures.

4. **Data Augmentation as standard practice** - Apply augmentation to training data to improve generalization, but don't rely on it as a substitute for transfer learning.

5. **Consider larger architectures for production** - ResNet-50 or EfficientNet could further improve accuracy with proper regularization.

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core programming language |
| PyTorch | Deep learning framework |
| torchvision | Pre-trained models & transforms |
| NumPy | Numerical computations |
| Pandas | Data organization |
| Matplotlib | Training curves & visualizations |
| Seaborn | Statistical plots & confusion matrices |
| scikit-learn | Classification reports & metrics |
| PIL/Pillow | Image loading & processing |
| Google Colab (T4 GPU) | Training environment |

---

## Project Structure

```
flower-recognition-cnn-resnet/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── notebooks/
│   └── Flowers_Recognition_Project.ipynb   # Main notebook (all experiments)
│
├── data/
│   └── README.md                      # Dataset download instructions
│
├── reports/
│   └── Flower_Recognition_Report.pdf  # Final project report
│
└── images/
    ├── class_distribution.png
    ├── resnet_training_curves.png
    ├── cnn_training_curves.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

---

## How to Run

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hVxHGPWd14wQFdytwwaFfW5zKM1XVPBS?usp=sharing)

1. Open the notebook in Google Colab
2. Ensure GPU runtime is enabled: `Runtime > Change runtime type > T4 GPU`
3. Upload the `flowers.zip` dataset to your Google Drive
4. Run all cells sequentially

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/Rolw4-53872/flower-recognition-cnn-resnet.git
cd flower-recognition-cnn-resnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/Flowers_Recognition_Project.ipynb
```

> **Note**: A CUDA-compatible GPU is recommended for training. CPU training will be significantly slower.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) on Kaggle
- Pre-trained weights: [ResNet-18 (IMAGENET1K_V1)](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- Built as part of a Data Modeling course project
