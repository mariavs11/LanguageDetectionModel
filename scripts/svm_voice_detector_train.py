# svm_language_detector.py
import os
import glob
import random
import pickle
import numpy as np
import librosa
from librosa import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def extract_features(audio_path):
    """Extract audio features for SVM classification."""
    try:
        audio, sr = librosa.load(audio_path, duration=13.0, sr=16000)

        if len(audio) < sr * 0.3:
            print(f"Skipping {audio_path}: too short")
            return None

        features = []

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(rolloff))

        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(bandwidth))

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend(np.mean(contrast, axis=1))

        return np.array(features)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def load_dataset(language_dirs, max_samples=2300):
    X, y = [], []
    all_files = {}

    for lang, dir_path in language_dirs.items():
        files = glob.glob(os.path.join(dir_path, "**/*.*"), recursive=True)
        files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]

        # Truncate per language
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]

        all_files[lang] = files
        print(f"Using {len(files)} files from {lang}")

    # Feature extraction
    for label, (lang, files) in enumerate(all_files.items()):
        for f in files:
            feat = extract_features(f)
            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y), list(language_dirs.keys())


def train_model(X, y, label_names):
    """Train SVM model with train/test split."""

    scaler = StandardScaler() # for normalization

    # Normalize and train training set
    X_train_scaled = scaler.fit_transform(X)

    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight='balanced'
    )

    #Training
    svm.fit(X_train_scaled, y)

    return svm, scaler


def save_model(svm, scaler, label_names, path='svm_model.pkl'):
    """Save trained model and scaler."""
    with open(path, 'wb') as f:
        pickle.dump({'model': svm, 'scaler': scaler, 'labels': label_names}, f)
    print(f"\nModel saved to {path}")



def load_model(path='svm_model.pkl'):
    """Load trained model and scaler."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['labels']

def predict_folder(folder_path, model_path='svm_model.pkl', recursive=True):
    svm, scaler, labels = load_model(model_path)

    # Collect audio files
    pattern = "**/*.*" if recursive else "*.*"
    files = glob.glob(os.path.join(folder_path, pattern), recursive=recursive)
    files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]

    print(f"\nPredicting {len(files)} files in: {folder_path}\n")

    results = []

    for f in files:
        features = extract_features(f)
        if features is None:
            continue
        #Normalize features
        features_scaled = scaler.transform([features])
        probs = svm.predict_proba(features_scaled)[0]
        idx = np.argmax(probs)

        language = labels[idx]
        confidence = probs[idx]

        results.append((f, language, confidence))

        print(f"{os.path.basename(f):40s} → {language:12s} ({confidence:.2%})")

    return results


def predict(audio_path, model_path='svm_model.pkl'):
    # Predict language of given audio
    #Load model
    svm, scaler, labels = load_model(model_path)

    # Extract features of input
    features = extract_features(audio_path)
    if features is None:
        return None, None

    features_scaled = scaler.transform([features])

    pred = svm.predict(features_scaled)[0]  # prediction result
    proba = svm.predict_proba(features_scaled)[0] # highest prob prediction

    return labels[pred], proba[pred]
if __name__ == "__main__":
    TRAIN_LANGS = {
        'english': './data/train/en',
        'russian': './data/train/ru',
        'portuguese': './data/train/portuguese',
    }

    TEST_LANGS = {
        'english': './data/test/en',
        'russian': './data/test/ru',
        'portuguese': './data/test/portuguese',
    }

    # =======================
    # Load TRAIN data
    # =======================
    X_train, y_train, label_names = load_dataset(TRAIN_LANGS, max_samples=2300)
    print(f"\nTraining samples: {len(X_train)}")

    # =======================
    # Train model
    # =======================
    svm, scaler = train_model(X_train, y_train, label_names)

    # =======================
    # Load TEST data
    # =======================
    X_test, y_test, _ = load_dataset(TEST_LANGS, max_samples=530)
    print(f"Test samples: {len(X_test)}")

    X_test_scaled = scaler.transform(X_test)

    # =======================
    # Predict on test set
    # =======================
    y_pred = svm.predict(X_test_scaled)

    # =======================
    # Evaluation
    # =======================
    print("\n" + "=" * 50)
    print("FINAL TEST SET EVALUATION")
    print("=" * 50)
    print(f"Test accuracy: {np.mean(y_pred == y_test):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =======================
    # Save model
    # =======================
    save_model(svm, scaler, label_names)



