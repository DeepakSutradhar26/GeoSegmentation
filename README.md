# GeoVision: Turf/Grass Segmentation from Aerial Imagery

## Overview

GeoVision is an end-to-end semantic segmentation pipeline developed for the **Ottermap Open Vision / ML Engineer Intern Technical Challenge**.

The project trains a deep learning model to identify **Turf/Grass** regions from high-resolution aerial imagery and exports predictions as GIS-compatible outputs.

The solution includes:

- Semantic segmentation using SegFormer
- GeoJSON annotation processing
- Binary mask generation
- Training and inference pipelines
- GIS-compatible prediction outputs

# Project Structure

```
project/
│
├── config.py
├── train.py
├── inference.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── prepare_data.py
│   └── dataset.py
│
├── model/
│   └── segformer.py
│
├── trainer/
│   └── train.py
│
├── weights/
│   └── best_model.pt
│
├── outputs/
│   ├── predictions/
│   ├── overlays/
│   └── geojson/
│
└── notebooks/
```

# Dataset

The provided dataset contains:

- High-resolution aerial imagery (.tiff)
- GeoJSON annotations
- Shapefile annotations

Each GeoJSON file contains polygon annotations representing Turf/Grass regions.

# Approach

## Model

The project uses **SegFormer-B2** pretrained on ImageNet.

The classification head is replaced with a single-channel output for binary segmentation.

```
Input Image
        │
        ▼
SegFormer Encoder
        │
        ▼
Decoder Head
        │
        ▼
1-channel Binary Mask
```

# Dataset Preparation

1. Load TIFF aerial image
2. Read GeoJSON annotations
3. Convert polygons into binary masks
4. Resize image and mask to 512×512
5. Convert to PyTorch tensors

# Training

Training uses:

- Transfer Learning
- Adam Optimizer
- Mixed Precision Training (AMP)
- Best model checkpoint saving

Current loss:

```
MSE Loss
```

# Installation

Clone the repository

```bash
git clone https://github.com/DeepakSutradhar26/GeoSegmentation.git
cd project
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

# Training

Run

```bash
python main.py
```

The best model will be saved to

```
weights/best_model.pt
```

# Inference

Run inference on a single image

```bash
python inference.py --image path/to/image.tiff
```

or

```bash
python inference.py --input ./images/
```

# GIS Output

Predicted segmentation masks are converted into GIS-compatible vector polygons.

Supported output formats

- GeoJSON
- Polygon Masks

These outputs can be directly imported into GIS software.

# External Generalization Test

To evaluate model robustness, inference was performed on an external aerial imagery dataset obtained from Kaggle.

The objective is to evaluate model performance on imagery captured from a different geographic location than the training dataset.

# Results

Example outputs include

- Training predictions
- Validation predictions
- External dataset predictions
- Binary masks
- Overlay visualizations
- GeoJSON vector outputs

# Future Improvements

With additional time, the following improvements would be implemented:

- Dice + BCE Loss
- Data augmentation using Albumentations
- Sliding-window inference for large images
- Polygon post-processing
- Test-Time Augmentation (TTA)
- Larger SegFormer backbone
- Cross-validation
- Hyperparameter optimization

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- SegFormer
- OpenCV
- GeoPandas
- NumPy
- Matplotlib
- Scikit-learn

# Reproducibility

All dependencies are listed in

```
requirements.txt
```

Training can be reproduced using

```bash
python train.py
```

Inference can be reproduced using

```bash
python inference.py --image input_image.tiff
```

# Deliverables

- Source Code
- Training Pipeline
- Inference Pipeline
- Trained Model
- Sample Predictions
- GeoJSON Outputs
- Technical Summary