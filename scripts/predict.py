import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import librosa
from librosa import feature
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

 # MODEL - Simple CNN

class LanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(LanguageCNN, self).__init__()

        #Build Network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



# ============================================
# 1. DATA LOADING - Convert audio to spectrograms
# ============================================
def normalize(signal):
    rms = np.sqrt(np.mean(signal ** 2) + 1e-9)
    return signal / rms


def predict(audio_path, model_path='cnn_model_15epochs.pth'):
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]

    train_mean = checkpoint.get("train_mean", 0.0)  # Get normalization stats
    train_std = checkpoint.get("train_std", 1.0)

    model = LanguageCNN(num_classes=len(labels))  # create empty network
    model.load_state_dict(checkpoint["model_state_dict"])  # loads weights from model
    model.eval()

    print(f" Loaded CNN model with classes: {labels}")
    print(f" Using normalization: mean={train_mean:.4f}, std={train_std:.4f}")


    # Extract spectrogram
    spec = audio_to_spectrogram(audio_path)

    if spec is None:
        return None, None
    spec = (spec - train_mean) / (train_std + 1e-9)

    # Convert to tensor: (64, 430) → (1, 1, 64, 430)
    x = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    idx = int(np.argmax(probs))
    language = labels[idx]
    confidence = float(probs[idx]) * 100

    return language, confidence

def predict_all_cnn( language_dirs, model_path='cnn_model_15epochs.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data spectograms and store in arrays
    X_test, y_test= load_data(language_dirs)

    # Get trained model info
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Access labels
    labels = checkpoint["labels"]

    # Get mean and std of model to normalize test data
    train_mean = checkpoint.get("train_mean", 0.0)
    train_std = checkpoint.get("train_std", 1.0)

    # Instantiate empty CNN
    model = LanguageCNN(num_classes=len(labels))

    # loads weights from trained model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Normalize test data
    X_test= (X_test -train_mean)/(train_std + 1e-9)

    print(f" Loaded CNN model with classes: {labels}")
    print(f" Using normalization: mean={train_mean:.4f}, std={train_std:.4f}")

    # Convert X_test to PyTorch tensor
    X_test = torch.FloatTensor(X_test).unsqueeze(1) # add dimension

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)  # Fit on the model's labels
    y_test_encoded = le.transform(y_test)

    # Create dataloader
    y_test = torch.LongTensor(y_test_encoded) # 0, 1, 2 instead of names
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16)  # 16 samples

    test_correct = 0
    test_total = 0

    y_true = []
    y_pred = []
    all_probs =[]
    # Predict
    with torch.no_grad(): # turn off gradient calc
        for X_batch, y_batch in test_loader: # process test data in batches
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs*100)
            _, predicted = torch.max(outputs, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item() # increments if prediction is correct
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    all_probs = np.vstack(all_probs)  # Combine all batch probabilities
    print(all_probs)

    indices = np.argmax(all_probs, axis=1) # indices of max prob along the rows
    language = [labels[idx] for idx in indices] # saves array of language names

    test_acc = test_correct / test_total


    print(f"\n Test Results:")
    print(f"   Test Accuracy: {test_acc:.2%}")
    print(f"   Correct:       {test_correct}/{test_total}")
    print("=" * 70)

    return language



def audio_to_spectrogram(audio_path, n_mels=96, max_len=430):
    try:
        #Process 13s of audio from file
        audio, sr = librosa.load(audio_path, duration=13.0, sr=16000)

        # Skip if too short
        if len(audio) < sr * 0.3:
            return None
        #RMS Normalization of audio
        audio = normalize(audio)

        # Create mel spectrogram

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Force exact size
        result = np.zeros((n_mels, max_len))  # Create fixed-size array

        if mel_spec_db.shape[1] >= max_len:
            result = mel_spec_db[:, :max_len]  # Truncate
        else:
            result[:, :mel_spec_db.shape[1]] = mel_spec_db  # Pad with zeros


        return result

    except Exception as e:
        print(f"Error: {audio_path}: {e}")
        return None


def load_data(language_dirs, max_samples=528):
    """Load audio files and convert to spectrograms."""
    X, y = [], []

    for lang, dir_path in language_dirs.items():
        files = glob.glob(os.path.join(dir_path, "**/*.*"), recursive=True)
        files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]

        # Truncate if too many
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]

        print(f"Using {len(files)} files from {lang}")

        for f in files:
            spec = audio_to_spectrogram(f)
            if spec is not None:
                X.append(spec)
                y.append(lang)

    return np.array(X), np.array(y)


#
#
# ============================================
# 3. TRAINING
# ============================================


if __name__ == "__main__":
    # Define languages
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

    lang = predict_all_cnn(TEST_LANGS, model_path='cnn_model_15epochs_redo.pth')












