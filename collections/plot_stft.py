#!/usr/bin/env python3
import os
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob

def save_stft(wav_path, fft_size, win_size, hop_size):
    y, sr = librosa.load(wav_path, sr=None, mono=False)
    if y.ndim > 1:
        y = y[0]
    S = librosa.stft(
        y,
        n_fft=fft_size,
        win_length=win_size,
        hop_length=hop_size,
        window='hann'
    )
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')
    plt.tight_layout()
    out_path = os.path.splitext(wav_path)[0] + '_stft.png'
    plt.savefig(out_path)
    plt.close()

def process_directory(base_dir, fft_size, win_size, hop_size):
    # match exactly x_channels/model_name/audio_name.wav
    pattern = os.path.join(base_dir, '8_channels', 'nbc','*.wav')
    for wav_path in glob.glob(pattern):
        save_stft(wav_path, fft_size, win_size, hop_size)

def main():
    p = argparse.ArgumentParser(
        description='Compute & save STFT plots (first channel only) for every .wav'
    )
    p.add_argument('base_dir', nargs='?', default='.',
                   help='root directory to scan (default: current directory)')
    p.add_argument('--fft_size', type=int, default=512,
                   help='FFT size for STFT (default: 512)')
    p.add_argument('--win_size', type=int, default=512,
                   help='Window size for STFT (default: 512)')
    p.add_argument('--hop_size', type=int, default=128,
                   help='Hop size for STFT (default: 128)')
    args = p.parse_args()

    process_directory(
        args.base_dir,
        args.fft_size,
        args.win_size,
        args.hop_size
    )

if __name__ == '__main__':
    main()
