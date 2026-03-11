import librosa
import numpy as np
import glob
import os
from librosa import effects
import soundfile as sf


def augment_audio(input_path, output_dir, sr=16000):
    """Augment a single audio file with speed variations."""
    try:
        y, sr = librosa.load(input_path, sr=sr, duration=13.0)

        # Get original filename without extension
        filename = os.path.splitext(os.path.basename(input_path))[0]

        # Time stretch variations
        variations = {
            'faster': 1.2,
        }

        for name, rate in variations.items():
            y_aug = librosa.effects.time_stretch(y, rate=rate)
            y_aug = librosa.util.normalize(y_aug)

            output_path = os.path.join(output_dir, f"out_{filename}_{name}.wav")
            sf.write(output_path, y_aug, sr)
            print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def augment_folder(input_dir, output_dir, sr=16000):
    """Augment all audio files in a folder."""
    files = glob.glob(os.path.join(input_dir, "**/*.*"), recursive=True)
    files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]

    print(f"Found {len(files)} files in {input_dir}")

    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] Processing: {os.path.basename(f)}")
        augment_audio(f, output_dir, sr)

    print(f"\n Augmented files saved to {output_dir}")


if __name__ == "__main__":
    LANGUAGES = {
        'english': ('./data/train/en', './data/train/en_augmented'),
        'russian': ('./data/train/ru', './data/train/ru_augmented'),
        'portuguese': ('./data/train/portuguese', './data/train/por_augmented'),
    }

    LANGUAGEST = {
        'english': ('./data/test/en', './data/test/en_augmented'),
        'russian': ('./data/test/ru', './data/test/ru_augmented'),
        'portuguese': ('./data/test/portuguese', './data/test/por_augmented'),
    }

    for lang, (input_dir, output_dir) in LANGUAGEST.items():
        print(f"\n{'=' * 50}")
        print(f"Augmenting {lang}...")
        print(f"{'=' * 50}")
        augment_folder(input_dir, output_dir)

