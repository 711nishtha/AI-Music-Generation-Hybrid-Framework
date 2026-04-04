"""Audio utility functions."""

from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa


def mel_to_wav(
    mel_db: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    n_iter: int = 64,
) -> np.ndarray:
    """
    Convert normalised mel spectrogram (values in [-1,1]) back to waveform
    using Griffin-Lim phase reconstruction.
    """
    mel_db_raw = mel_db * 80.0
    mel_power  = librosa.db_to_power(mel_db_raw)

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                     fmin=20.0, fmax=8000.0)
    inv_mel = np.linalg.pinv(mel_basis)
    stft_mag = np.dot(inv_mel, mel_power)
    stft_mag = np.maximum(stft_mag, 0.0)

    audio = librosa.griffinlim(
        stft_mag, n_iter=n_iter,
        hop_length=hop_length, win_length=n_fft
    )
    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalise audio array."""
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio * (target_peak / peak)
    return audio


def save_wav(audio: np.ndarray, path: str, sr: int = 22050) -> None:
    """Save audio array as WAV file."""
    audio = normalize_audio(audio)
    sf.write(path, audio, sr)


def concatenate_wavs(wav_paths: list, output_path: str,
                     gap_seconds: float = 0.5, sr: int = 22050) -> None:
    """Concatenate multiple WAV files with a short silence gap."""
    import os
    segments = []
    gap = np.zeros(int(gap_seconds * sr), dtype=np.float32)
    for p in wav_paths:
        if os.path.exists(p):
            y, _ = librosa.load(p, sr=sr, mono=True)
            segments.append(y.astype(np.float32))
            segments.append(gap)
    if segments:
        out = np.concatenate(segments)
        save_wav(out, output_path, sr)
