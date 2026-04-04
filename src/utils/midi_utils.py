"""MIDI utility functions."""

from __future__ import annotations
import os
import subprocess
from typing import List

import numpy as np
import pretty_midi


def heuristic_score(token_ids: List[int], tokenizer) -> float:
    """
    Heuristic quality score for a token sequence.
    Rewards structural variety, moderate note density, chord variety,
    multi-track content, and phrase regularity.
    Returns a float in [0, 1].
    """
    from src.data.tokenizer import ID2TOKEN, TOKEN2ID

    tokens = [ID2TOKEN.get(i, "[UNK]") for i in token_ids]

    structure_tokens = {"[INTRO]", "[VERSE]", "[CHORUS]", "[BRIDGE]", "[OUTRO]"}
    n_struct    = sum(1 for t in tokens if t in structure_tokens)
    n_notes     = sum(1 for t in tokens if t.startswith("NOTE_ON_"))
    n_chords    = sum(1 for t in tokens if t.startswith("CHORD_") and t != "CHORD_NONE")
    n_phrase    = sum(1 for t in tokens if t in ("[PHRASE_START]", "[PHRASE_END]"))
    n_bars      = sum(1 for t in tokens if t == "BAR")
    n_tracks    = len(set(t for t in tokens if t.startswith("[TRACK_")))

    if n_bars == 0:
        return 0.0

    note_density  = n_notes / max(n_bars, 1)
    chord_density = n_chords / max(n_bars, 1)

    score = 0.0
    score += min(n_struct / 3.0, 1.0) * 0.3
    density_score = 1.0 - abs(note_density - 8.0) / 8.0
    score += max(0.0, density_score) * 0.25
    score += min(chord_density, 1.0) * 0.2
    score += min(n_tracks / 4.0, 1.0) * 0.15
    score += min(n_phrase / 4.0, 1.0) * 0.1

    return float(np.clip(score, 0.0, 1.0))


def midi_to_wav(midi_path: str, wav_path: str, soundfont: str,
                sample_rate: int = 22050, gain: float = 1.0) -> bool:
    """Render MIDI to WAV using FluidSynth."""
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    cmd = ["fluidsynth", "-ni", "-g", str(gain),
           "-F", wav_path, "-r", str(sample_rate),
           soundfont, midi_path]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return os.path.exists(wav_path)
    except Exception as e:
        print(f"[MIDI->WAV] Failed: {e}")
        return False


def merge_midi_files(paths: List[str], output_path: str) -> str:
    """Concatenate multiple MIDI files into one."""
    merged = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    time_offset = 0.0
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(path)
            for inst in pm.instruments:
                new_inst = pretty_midi.Instrument(
                    program=inst.program,
                    is_drum=inst.is_drum,
                    name=inst.name
                )
                for n in inst.notes:
                    new_inst.notes.append(pretty_midi.Note(
                        n.velocity, n.pitch,
                        n.start + time_offset, n.end + time_offset
                    ))
                merged.instruments.append(new_inst)
            time_offset += pm.get_end_time() + 1.0
        except Exception:
            pass
    merged.write(output_path)
    return output_path
