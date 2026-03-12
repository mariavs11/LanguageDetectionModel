# LanguageDetectionModel

## Overview

This project implements a multi-class speech classification system capable of distinguishing between **English**, **Russian**, and **Portuguese** audio recordings.

The system leverages machine learning and deep learning techniques applied to extracted audio features, using two approaches:

- **Support Vector Machines (SVM)**
- **Convolutional Neural Networks (CNN)**

---

## Approaches

### CNN
Extracts Mel-spectrograms from speech signals, treating them as 2D time-frequency images that serve as input to a CNN model.

### SVM
Extracts compact features from speech signals that feed into an SVM model. The features consist primarily of MFCCs along with additional speech-related coefficients.

---
## Details about CNN model:

The CNN architecture consists of three convolutional blocks with increasing filter counts (32, 64, 128), allowing the network to learn progressively complex features as it passes through each layer. The choice of 32, 64, and 128 filters was made after analyzing well stablished CNN implementations. In the initial layers, the model captures low-level features such as edges and frequency patterns, while deeper layers learn higher-level and abstract features. Each block employs 3×3 kernels with padding=1 to preserve image dimensions (image size stays the same), followed by ReLU activation to introduce non-linearity. 

MaxPooling layers are added after the first two Convolutional blocks; it applies a 2x2 window to select the maximum values in each region of the image, preserving the most important features and reducing the number of features (prevents overfitting). The third convolution block (128 filters) is followed by an Adaptive Average Pooling Layer, that takes average of features and enforces a fixed-sized 4x4 feature map as output, regardless of input size. The 4x4 feature map is obtained from each of the 128 filters (total of 128 feature maps) that feed into a fully connected layer. 

The last stage of the network is a fully connected layer that flattens all feature maps into a single vector. This single vector is fed to a neural network with a hidden layer comprised of 128 neurons. A dropout layer (0.3) is applied to reduce overfitting, followed by a final linear layer that outputs class scores for each target language.

To train our model, we monitored the CrossEntropy Loss metric to determine if convergence was reached. 


## Installation

Install the required dependencies:

```bash
pip install torch librosa numpy scikit-learn matplotlib soundfile
```

---

## Usage

### Train the CNN Model

The `train.py` script trains the CNN model and evaluates it on the test data. Before running, manually edit the number of epochs and the model name inside the script.

By default, it trains for **15 epochs** and saves the model as `cnn_model_15epochs.pt`.

```bash
python3 scripts/train.py
```

---

### Evaluate the CNN Model

The `predict.py` script handles evaluation only — it loads a saved model and predicts each class. It runs the test set through `cnn_model_15epochs_redo.pth`.

```bash
python3 scripts/predict.py
```

---

### Train the SVM Model

The `svm_voice_detector_train.py` script trains and evaluates the SVM model.

```bash
python3 scripts/svm_voice_detector_train.py
```
## Dataset

The training and testing data are not included in this repository.  
Audio recordings are available from the LibriSpeech dataset:

LibriSpeech ASR Corpus  
Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur  
https://www.openslr.org/12  

LibriSpeech is distributed under the Creative Commons Attribution 4.0 (CC BY 4.0) license.
