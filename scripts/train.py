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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# ============================================
# 1. DATA LOADING - Convert audio to spectrograms
# ============================================
def normalize(signal):
    rms = np.sqrt(np.mean(signal ** 2) + 1e-9)
    return signal / rms


def audio_to_spectrogram(audio_path, n_mels=96, max_len=430):
    try:
        # Process 13s of audio from file
        audio, sr = librosa.load(audio_path, duration=13.0, sr=16000)

        # Skip if too short
        if len(audio) < sr * 0.3:
            return None

        # RMS Normalization of audio
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


def load_data(language_dirs, max_samples=9200):
    """Load audio files, convert to spectrogram, store as numpy array."""
    X, y = [], []

    for lang, dir_paths in language_dirs.items():
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]

        all_files = []
        for dir_path in dir_paths:
            files = glob.glob(os.path.join(dir_path, "**/*.*"), recursive=True)
            files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]
            all_files.extend(files)

        # Truncate if too many
        if max_samples and len(all_files) > max_samples:
            all_files = all_files[:max_samples]

        print(f"Using {len(all_files)} files from {lang}")

        for f in all_files:
            spec = audio_to_spectrogram(f)
            if spec is not None:
                X.append(spec)
                y.append(lang)

    return np.array(X), np.array(y)

# ============================================
# 2. MODEL
# ============================================
class LanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(LanguageCNN, self).__init__()

        # Build Network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32 filters, 3x3 kernel,
            nn.ReLU(), # introduce non-linearity
            nn.MaxPool2d(2), #pooling max in 2x2 matrix

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
            nn.Dropout(0.3),  # turns off neurons to prevent overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============================================
# 3. TRAINING WITH OVERFITTING DETECTION
# ============================================
def train_model(model, train_loader, val_loader, epochs=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Gap':<8}")
    print("=" * 70)
    patience =4 # patience to trigger stop
    for epoch in range(epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad() # reset gradient
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ========== VALIDATION PHASE ==========
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Calculate accuracy gap (indicator of overfitting)
        acc_gap = train_acc - val_acc

        # Store metrics
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Print with overfitting warning
        warning = ""
        if acc_gap > 0.15:  # 15% gap
            warning = " OVERFITTING!"
        elif acc_gap > 0.10:  # 10% gap
            warning = "Warning"

        print(f"{epoch + 1:<8} {avg_train_loss:<12.4f} {train_acc:<12.2%} "
              f"{avg_val_loss:<12.4f} {val_acc:<12.2%} {acc_gap:<8.2%}{warning}")

    print("=" * 70 + "\n")

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)

    return model


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics to detect overfitting."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-o',
             label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in val_accs], 'r-s',
             label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Overfitting
    final_gap = (train_accs[-1] - val_accs[-1]) * 100
    if final_gap > 10:
        ax2.text(0.5, 0.05, f'Final Gap: {final_gap:.1f}% - Possible Overfitting',
                 transform=ax2.transAxes, ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    print(" Learning curves saved to 'learning_curves.png'")
    plt.show()



if __name__ == "__main__":
    # Define languages
    TRAIN_LANGS = {
        'english': ['./data/train/en'],
        'russian': ['./data/train/ru'],
        'portuguese': ['./data/train/portuguese'],
    }

    TEST_LANGS = {
        'english': './data/test/en',
        'russian': './data/test/ru',
        'portuguese': './data/test/portuguese',
    }


    print("Loading TRAIN data...")
    X_train, y_train = load_data(TRAIN_LANGS, max_samples=2300)

    # Calculate normalization statistics from training data
    train_mean = X_train.mean()
    train_std = X_train.std()

    # Normalize training data
    X_train = (X_train - train_mean) / (train_std + 1e-9)

    print("Loading TEST data...")
    X_test, y_test = load_data(TEST_LANGS, max_samples=528)

    # Normalize test data using training mean and std
    X_test = (X_test - train_mean) / (train_std + 1e-9)
    print(f"Total test samples: {len(X_test)}")

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    print(f"\nNumber of classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")

    # Split training data into train and validation (80/20 split)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
    )

    print(f"\nDataset sizes:")
    print(f"  Training:   {len(X_train_split)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    # Convert to PyTorch tensors
    X_train_split = torch.FloatTensor(X_train_split).unsqueeze(1)
    X_val = torch.FloatTensor(X_val).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)

    y_train_split = torch.LongTensor(y_train_split)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test_encoded)

    # Create data loaders
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Build and train model
    num_classes = len(le.classes_)
    model = LanguageCNN(num_classes=num_classes)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model with validation set
    print("\n Starting training...")
    model = train_model(model, train_loader, val_loader, epochs=15)

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluation mode, activate all neurons
    model.eval()

    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss() # metric
    test_loss = 0
    y_true = []
    y_pred =[]
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch) # raw predictions

            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion matrix:\n{cm}")

    test_acc = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    print(f"\n Test Results:")
    print(f"   Test Loss:     {avg_test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2%}")
    print(f"   Correct:       {test_correct}/{test_total}")
    print("=" * 70)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'labels': le.classes_.tolist(),
        'train_mean': train_mean,
        'train_std': train_std,
        'num_classes': num_classes,
        'test_accuracy': test_acc
    }, 'cnn_model_15epochs_redo.pth')

    print(f"\n Model saved to 'cnn_model_15epochs_redo.pth'")
    print(f" Learning curves saved to 'learning_curves.png'")