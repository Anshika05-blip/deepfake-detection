# Deepfake Image Detection

A binary image classifier that detects whether a face image is real or AI-generated (deepfake). Built using a hybrid CNN + Transformer architecture in PyTorch, designed to run on Google Colab.

## How it works

The model combines a custom CNN backbone for extracting local features with a Transformer encoder for capturing global context. Face regions are first cropped using MTCNN before being passed to the model.

**Architecture:**

* CNN backbone — 4 convolutional blocks with BatchNorm and MaxPooling
* Transformer head — 4-layer encoder with a learnable class token
* Output — sigmoid activation for binary classification (Real / Fake)

## Dataset

Uses the [Real and Fake Face Detection dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) from Kaggle.

```
data/
  real/    # real face images
  fake/    # deepfake face images
```

Alternatively supports FaceForensics++ and Celeb-DF v2.

## Setup

```bash
pip install torch torchvision torchaudio timm facenet-pytorch albumentations opencv-python scikit-learn matplotlib pandas numpy tqdm Pillow kaggle
```

Recommended: Run on Google Colab with a T4 GPU (`Runtime > Change runtime type > T4 GPU`).

## Usage

1. Add your Kaggle API key (`kaggle.json`) to `/content/`
2. The notebook will download and split the dataset automatically
3. Run all cells in order

To run inference on a single image after training:

```python
label, prob = predict\_image(model, 'your\_image.jpg')
print(f"Prediction: {label} | Fake probability: {prob:.4f}")
```

## Results

Trained for 2 epochs on a small subset as a proof of concept.

|Epoch|Train Loss|Val Loss|Val Accuracy|Val F1|Val AUC|
|-|-|-|-|-|-|
|1|0.7356|0.7062|50.0%|0.634|0.493|
|2|0.6926|0.7163|53.3%|0.611|0.498|

Performance is expected to improve significantly with more epochs and a larger dataset.

## Features

* Face extraction using MTCNN
* Data augmentation (flips, brightness, rotation)
* Test-Time Augmentation (TTA) for better inference
* Grad-CAM visualisation to highlight regions the model focuses on
* Metrics: Accuracy, F1, AUC, ROC curve, Confusion Matrix

## Tech Stack

Python · PyTorch · timm · facenet-pytorch · albumentations · OpenCV · scikit-learn

