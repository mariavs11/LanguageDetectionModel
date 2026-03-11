# LanguageDetectionModel

## Overview

This project implements a multi-class speech classification system capable of distinguishing between English, Russian, and Portuguese audio recordings.

The system leverages machine learning and deep learning techniques applied to extracted audio features. Specifically, it uses:

  - Support Vector Machines (SVM)
  - Convolutional Neural Networks (CNN)

CNN approach: 

  - Extracts Mel-spectrogram from speech signals, treating them as 2D time-frequency images that serve as input to a CNN model.

SVM approach:

  - Extracts compact features from speech signals that feed into SVM model. The features consist primarily of MFCCs along with additional speech-related coefficients. 

