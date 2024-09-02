#test

# Solar Panel Fault Detection System

This project is focused on building a Convolutional Neural Network (CNN) to detect various types of faults in solar panels using image data. The model is trained using a ResNet-50 architecture and fine-tuned on a dataset of solar panel images categorized into six different classes.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Results](#results)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Overview

The goal of this project is to develop a machine learning model that can accurately classify faults in solar panels. The faults are categorized into six classes:

- Bird-drop
- Clean
- Dusty
- Electrical-damage
- Physical-Damage
- Snow-Covered

The model leverages transfer learning with a pre-trained ResNet-50 network, fine-tuned on the specific dataset to achieve optimal performance.

## Dataset

The dataset used for this project can be found on Kaggle: [Solar Panel Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images).

The dataset consists of images of solar panels categorized into six classes representing different types of faults. The images are preprocessed by resizing, center cropping, and normalization using the calculated mean and standard deviation of the training dataset.

### Directory Structure

```plaintext
├── train/
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
├── test/
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
```

## Model Architecture

The model uses a ResNet-50 architecture pre-trained on ImageNet. The final fully connected layer is replaced with a new one that outputs six classes.

```python
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)
```

## Training Procedure

The model is trained using the following steps:

1. **Data Preprocessing**: Images are resized, center-cropped, and normalized using the mean and standard deviation of the training dataset.
2. **Training**: The model was trained for 200 epochs with early stopping triggered at epoch 116 based on validation accuracy. After some hyperparameter tuning, the model achieved optimal performance.
3. **Validation**: After each epoch, the model is evaluated on the validation set. The best model (based on validation accuracy) is saved.

```python
# Optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Training loop with early stopping
```

## Results

The best model achieved a validation accuracy of **90.96%**. The model was trained for 200 epochs, with early stopping at epoch 116.

## How to Use

### Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Numpy

### Installation

1. Clone the repository:

```bash
git clone https://github.com/n01syboii/Solar-Panel-Fault-Detection.git
cd Solar
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. **Training the Model**: Run the training script to train the model on your dataset.

```bash
python train.py
```

2. **Evaluating the Model**: After training, evaluate the model on the test dataset.

```bash
python evaluate.py --model best_solar_panel_model_0.9096.pth --data_dir test/
```

3. **Inference**: Use the trained model to predict the class of a new solar panel image.

```python
# Load the trained model and use it for prediction
python predict.py --model best_solar_panel_model_0.9096.pth --image path_to_image
```

### Example

```bash
python train.py
python evaluate.py --model best_solar_panel_model_0.9096.pth --data_dir test/
python predict.py --model best_solar_panel_model_0.9096.pth --image test/Clean/image.jpg
```
