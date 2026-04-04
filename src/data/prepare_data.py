"""
Full data-preparation pipeline:
  1. Discover MIDI files in provided directories
  2. Tokenise each MIDI using MusicTokenizer
  3. Render MIDI -> WAV using FluidSynth
  4. Save token sequences as .npy files
  5. Build pair manifest (token <-> audio)
  6. Generate synthetic preference data
  7. Write small train/val subset manifests
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.tokenizer import MusicTokenizer
from src.utils.midi_utils import heuristic_score


def find_midi_files(dirs: List[str], max_files: int = 2000) -> List[str]:
    files = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
            files.extend([str(p) for p in d.rglob(ext)])
    random.shuffle(files)
    return files[:max_files]


def render_midi_to_wav(midi_path: str, wav_path: str, soundfont: str,
                       sample_rate: int = 22050) -> bool:
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = [
        "fluidsynth", "-ni",
        "-F", wav_path,
        "-r", str(sample_rate),
        soundfont, midi_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000
    except Exception:
        return False


def generate_synthetic_preference(
    sequences: List[List[int]],
    tokenizer: MusicTokenizer,
    n_pairs: int = 500
) -> List[dict]:
    scored = []
    for seq in sequences:
        score = heuristic_score(seq, tokenizer)
        scored.append((seq, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    pairs = []
    n = len(scored)
    if n < 4:
        return pairs

    for _ in range(min(n_pairs, n // 2)):
        avg_len = int(np.mean([len(s) for s, _ in scored]))
        prefix_len = max(10, int(avg_len * 0.3))

        c_idx = random.randint(0, max(1, int(n * 0.3) - 1))
        chosen_full = scored[c_idx][0]
        if len(chosen_full) <= prefix_len:
            continue

        r_idx = random.randint(max(0, int(n * 0.7)), n - 1)
        rejected_full = scored[r_idx][0]
        if len(rejected_full) <= prefix_len:
            continue

        prompt   = chosen_full[:prefix_len]
        chosen   = chosen_full[prefix_len: prefix_len + 256]
        rejected = rejected_full[prefix_len: prefix_len + 256]

        pairs.append({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
        })

    return pairs


def main(args):
    random.seed(42)
    np.random.seed(42)

    tokenizer = MusicTokenizer()

    print(f"[Data] Finding MIDI files in {args.midi_dirs}...")
    midi_files = find_midi_files(args.midi_dirs, args.max_files)
    print(f"[Data] Found {len(midi_files)} MIDI files.")

    os.makedirs(args.output_dir, exist_ok=True)
    token_dir = os.path.join(args.output_dir, "tokens")
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    token_paths: List[str] = []
    pair_records: List[dict] = []
    all_sequences: List[List[int]] = []

    print("[Data] Tokenising MIDI files and rendering audio...")
    for midi_path in tqdm(midi_files):
        stem = Path(midi_path).stem.replace(" ", "_")[:60]
        token_path = os.path.join(token_dir, f"{stem}.npy")
        wav_path   = os.path.join(audio_dir, f"{stem}.wav")

        if not os.path.exists(token_path):
            try:
                pm = pretty_midi.PrettyMIDI(midi_path)
                ids = tokenizer.encode(pm, style="POP", emotion="HAPPY")
                np.save(token_path, np.array(ids, dtype=np.int32))
            except Exception as e:
                continue

        if not os.path.exists(wav_path) and os.path.exists(args.soundfont):
            render_midi_to_wav(midi_path, wav_path, args.soundfont)

        token_paths.append(token_path)
        if os.path.exists(wav_path):
            pair_records.append({
                "token_file": token_path,
                "audio_file": wav_path,
            })

        seq = np.load(token_path).tolist()
        all_sequences.append(seq)

    print(f"[Data] Tokenised {len(token_paths)} files, "
          f"{len(pair_records)} with audio.")

    os.makedirs(args.subset_dir, exist_ok=True)

    random.shuffle(token_paths)
    n_sub  = min(args.subset_size, len(token_paths))
    n_val  = max(1, int(n_sub * 0.1))
    n_train = n_sub - n_val

    with open(os.path.join(args.subset_dir, "train_tokens.json"), "w") as f:
        json.dump(token_paths[:n_train], f)
    with open(os.path.join(args.subset_dir, "val_tokens.json"), "w") as f:
        json.dump(token_paths[n_train: n_sub], f)

    random.shuffle(pair_records)
    n_pair_sub = min(args.subset_size, len(pair_records))
    n_pair_val = max(1, int(n_pair_sub * 0.1))
    with open(os.path.join(args.subset_dir, "train_pairs.json"), "w") as f:
        json.dump(pair_records[:n_pair_sub - n_pair_val], f)
    with open(os.path.join(args.subset_dir, "val_pairs.json"), "w") as f:
        json.dump(pair_records[n_pair_sub - n_pair_val: n_pair_sub], f)

    print("[Data] Generating synthetic preference data...")
    sub_seqs = all_sequences[:args.subset_size]
    prefs    = generate_synthetic_preference(sub_seqs, tokenizer, n_pairs=400)
    pref_path = os.path.join(args.subset_dir, "preference_data.json")
    with open(pref_path, "w") as f:
        json.dump(prefs, f)
    print(f"[Data] Generated {len(prefs)} preference pairs -> {pref_path}")

    tokenizer.save_vocab(os.path.join(args.output_dir, "vocab.json"))

    print(f"""
[Data] Data preparation complete.
       Train tokens : {n_train}
       Val tokens   : {n_val}
       Audio pairs  : {len(pair_records)}
       Pref pairs   : {len(prefs)}
       Output dir   : {args.output_dir}
    """)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--midi_dirs",   nargs="+", required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--subset_dir",  required=True)
    p.add_argument("--soundfont",   required=True)
    p.add_argument("--max_files",   type=int, default=2000)
    p.add_argument("--subset_size", type=int, default=500)
    main(p.parse_args())
