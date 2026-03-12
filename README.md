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
