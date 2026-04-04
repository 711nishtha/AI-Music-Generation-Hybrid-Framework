"""
PyTorch Dataset classes for the three training stages:
  1. SymbolicMusicDataset     — for symbolic planning model
  2. AudioRendererDataset     — for audio rendering model
  3. PreferenceDataset        — for DPO alignment
"""

from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import pretty_midi
import librosa
import soundfile as sf

from .tokenizer import MusicTokenizer


class SymbolicMusicDataset(Dataset):
    """
    Returns token sequences for causal language-model training of the
    hierarchical symbolic planner.
    """

    def __init__(
        self,
        token_files: List[str],
        max_seq_len: int = 2048,
        tokenizer: Optional[MusicTokenizer] = None,
    ):
        self.token_files = token_files
        self.max_seq_len = max_seq_len
        self.tokenizer   = tokenizer or MusicTokenizer()

        self.sequences: List[List[int]] = []
        for path in token_files:
            path = Path(path)
            if path.suffix == ".npy":
                seq = np.load(str(path)).tolist()
            elif path.suffix == ".json":
                with open(path) as f:
                    seq = json.load(f)
            else:
                continue
            if len(seq) > 8:
                self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        if len(seq) > self.max_seq_len:
            start = random.randint(0, len(seq) - self.max_seq_len)
            seq = seq[start: start + self.max_seq_len]

        ids, mask = self.tokenizer.pad(seq, self.max_seq_len)

        ids_t  = torch.tensor(ids,  dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.long)

        return {
            "input_ids":      ids_t[:-1],
            "labels":         ids_t[1:],
            "attention_mask": mask_t[:-1],
        }


class AudioRendererDataset(Dataset):
    """
    Paired (symbolic_token_sequence, mel_spectrogram) dataset for training
    the conditional diffusion audio renderer.
    """

    N_FFT    = 1024
    HOP_LEN  = 256
    N_MELS   = 80
    SR       = 22050
    MAX_MELS = 512

    def __init__(
        self,
        pair_manifest: str,
        max_seq_len:   int = 1024,
        tokenizer:     Optional[MusicTokenizer] = None,
        augment:       bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer   = tokenizer or MusicTokenizer()
        self.augment     = augment

        with open(pair_manifest) as f:
            self.pairs = json.load(f)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_mel(self, audio_path: str) -> np.ndarray:
        mel_path = audio_path.replace(".wav", "_mel.npy")
        if os.path.exists(mel_path):
            return np.load(mel_path)

        y, _ = librosa.load(audio_path, sr=self.SR, mono=True)
        y = y / (np.abs(y).max() + 1e-8)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.SR, n_fft=self.N_FFT,
            hop_length=self.HOP_LEN, n_mels=self.N_MELS,
            fmin=20.0, fmax=8000.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.clip(mel_db / 80.0, -1.0, 1.0)
        np.save(mel_path, mel_db.astype(np.float32))
        return mel_db.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair       = self.pairs[idx]
        token_file = pair["token_file"]
        audio_file = pair["audio_file"]

        tok_path = Path(token_file)
        if tok_path.suffix == ".npy":
            seq = np.load(str(tok_path)).tolist()
        else:
            with open(tok_path) as f:
                seq = json.load(f)

        ids, mask = self.tokenizer.pad(seq, self.max_seq_len)

        mel = self._load_mel(audio_file)

        T = mel.shape[1]
        if T > self.MAX_MELS:
            if self.augment:
                start = random.randint(0, T - self.MAX_MELS)
            else:
                start = 0
            mel = mel[:, start: start + self.MAX_MELS]
        else:
            pad_w = self.MAX_MELS - T
            mel = np.pad(mel, ((0, 0), (0, pad_w)), mode="constant", constant_values=-1.0)

        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "mel":            torch.tensor(mel,  dtype=torch.float32).unsqueeze(0),
        }


class PreferenceDataset(Dataset):
    """
    Dataset of (prompt_tokens, chosen_continuation, rejected_continuation)
    for Direct Preference Optimisation.
    """

    def __init__(
        self,
        preference_file: str,
        max_seq_len: int = 1024,
        tokenizer: Optional[MusicTokenizer] = None,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer   = tokenizer or MusicTokenizer()

        with open(preference_file) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item    = self.data[idx]
        prompt  = item["prompt"]
        chosen  = item["chosen"]
        rejected = item["rejected"]

        def _encode(seq):
            ids, mask = self.tokenizer.pad(seq, self.max_seq_len)
            return torch.tensor(ids, dtype=torch.long), \
                   torch.tensor(mask, dtype=torch.long)

        p_ids,  p_mask  = _encode(prompt)
        c_ids,  c_mask  = _encode(chosen)
        r_ids,  r_mask  = _encode(rejected)

        return {
            "prompt_ids":        p_ids,
            "prompt_mask":       p_mask,
            "chosen_ids":        c_ids,
            "chosen_mask":       c_mask,
            "rejected_ids":      r_ids,
            "rejected_mask":     r_mask,
        }
