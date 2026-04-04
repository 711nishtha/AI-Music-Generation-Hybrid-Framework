"""
REMI+ style MIDI tokeniser with hierarchical structure tokens.
Vocabulary:
  - BOS / EOS / PAD
  - STRUCTURE tokens: [VERSE], [CHORUS], [BRIDGE], [INTRO], [OUTRO]
  - PHRASE tokens: [PHRASE_START], [PHRASE_END]
  - TRACK tokens: [TRACK_0..7]
  - Time: BAR, POSITION_0..95 (1/32-note resolution, 3 octaves of sub-beats)
  - Pitch: NOTE_ON_0..127, NOTE_OFF_0..127
  - Duration: DURATION_1..48  (multiples of 1/32 note)
  - Velocity: VELOCITY_0..31  (4-bit quantised, 32 bins)
  - CHORD tokens: CHORD_C_MAJ, CHORD_C_MIN, ... (12 roots x 4 qualities = 48)
  - TEMPO: TEMPO_40..200 (in 10 BPM steps = 17 tokens)
  - EMOTION: EMO_HAPPY, EMO_SAD, EMO_TENSE, EMO_PEACEFUL
  - STYLE: STYLE_POP, STYLE_CLASSICAL, STYLE_JAZZ, STYLE_FOLK
  Total vocab ~512 tokens
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pretty_midi


# ── Vocabulary construction ───────────────────────────────────────────────────

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[MASK]"]
STRUCTURE_TOKENS = ["[INTRO]", "[VERSE]", "[CHORUS]", "[BRIDGE]", "[OUTRO]",
                    "[SECTION_END]"]
PHRASE_TOKENS = ["[PHRASE_START]", "[PHRASE_END]"]
TRACK_TOKENS = [f"[TRACK_{i}]" for i in range(8)]
EMOTION_TOKENS = ["[EMO_HAPPY]", "[EMO_SAD]", "[EMO_TENSE]", "[EMO_PEACEFUL]"]
STYLE_TOKENS = ["[STYLE_POP]", "[STYLE_CLASSICAL]", "[STYLE_JAZZ]", "[STYLE_FOLK]",
                "[STYLE_ELECTRONIC]", "[STYLE_AMBIENT]"]

BAR_TOKEN = ["BAR"]
POSITION_TOKENS = [f"POSITION_{i}" for i in range(96)]
NOTE_ON_TOKENS = [f"NOTE_ON_{i}" for i in range(128)]
NOTE_OFF_TOKENS = [f"NOTE_OFF_{i}" for i in range(128)]
DURATION_TOKENS = [f"DURATION_{i}" for i in range(1, 49)]
VELOCITY_TOKENS = [f"VELOCITY_{i}" for i in range(32)]

_ROOTS = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
_QUALITIES = ["MAJ", "MIN", "DOM7", "DIM"]
CHORD_TOKENS = [f"CHORD_{r}_{q}" for r in _ROOTS for q in _QUALITIES]
NO_CHORD_TOKEN = ["CHORD_NONE"]

TEMPO_TOKENS = [f"TEMPO_{t}" for t in range(40, 210, 10)]

ALL_TOKENS = (
    SPECIAL_TOKENS + STRUCTURE_TOKENS + PHRASE_TOKENS + TRACK_TOKENS +
    EMOTION_TOKENS + STYLE_TOKENS + BAR_TOKEN + POSITION_TOKENS +
    NOTE_ON_TOKENS + NOTE_OFF_TOKENS + DURATION_TOKENS + VELOCITY_TOKENS +
    CHORD_TOKENS + NO_CHORD_TOKEN + TEMPO_TOKENS
)

TOKEN2ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(ALL_TOKENS)}
ID2TOKEN: Dict[int, str] = {idx: tok for tok, idx in TOKEN2ID.items()}

VOCAB_SIZE = len(ALL_TOKENS)
PAD_ID     = TOKEN2ID["[PAD]"]
BOS_ID     = TOKEN2ID["[BOS]"]
EOS_ID     = TOKEN2ID["[EOS]"]


def _velocity_bin(vel: int) -> int:
    return min(int(vel * 32 / 128), 31)


def _tempo_token(bpm: float) -> str:
    snapped = max(40, min(200, round(bpm / 10) * 10))
    return f"TEMPO_{snapped}"


_CHORD_TEMPLATES = {}
for _root_idx, _root in enumerate(_ROOTS):
    for _qual in _QUALITIES:
        if _qual == "MAJ":
            intervals = [0, 4, 7]
        elif _qual == "MIN":
            intervals = [0, 3, 7]
        elif _qual == "DOM7":
            intervals = [0, 4, 7, 10]
        else:
            intervals = [0, 3, 6]
        chroma = np.zeros(12, dtype=float)
        for iv in intervals:
            chroma[(_root_idx + iv) % 12] = 1.0
        _CHORD_TEMPLATES[f"{_root}_{_qual}"] = chroma / (np.linalg.norm(chroma) + 1e-8)


def detect_chord(notes_in_bar: List[pretty_midi.Note]) -> str:
    if not notes_in_bar:
        return "CHORD_NONE"
    chroma = np.zeros(12, dtype=float)
    for n in notes_in_bar:
        chroma[n.pitch % 12] += 1.0
    norm = np.linalg.norm(chroma)
    if norm < 1e-8:
        return "CHORD_NONE"
    chroma /= norm
    best, best_score = "CHORD_NONE", -1.0
    for chord_name, template in _CHORD_TEMPLATES.items():
        score = float(np.dot(chroma, template))
        if score > best_score:
            best_score = score
            best = f"CHORD_{chord_name}"
    return best if best_score > 0.6 else "CHORD_NONE"


class MusicTokenizer:
    """
    Converts pretty_midi objects <-> token-id sequences.
    Supports multi-track MIDI with hierarchical structure annotations.
    """

    TICKS_PER_BEAT = 480
    BEATS_PER_BAR  = 4
    SUBDIVISIONS   = 8
    MAX_TRACKS     = 8
    MAX_SEQ_LEN    = 4096

    vocab_size = VOCAB_SIZE
    pad_id     = PAD_ID
    bos_id     = BOS_ID
    eos_id     = EOS_ID

    def encode(
        self,
        midi: pretty_midi.PrettyMIDI,
        style: str = "POP",
        emotion: str = "HAPPY",
        add_structure: bool = True,
        max_bars: int = 64,
    ) -> List[int]:
        tokens: List[str] = ["[BOS]"]

        style_tok = f"[STYLE_{style.upper()}]"
        emo_tok   = f"[EMO_{emotion.upper()}]"
        if style_tok in TOKEN2ID:
            tokens.append(style_tok)
        if emo_tok in TOKEN2ID:
            tokens.append(emo_tok)

        tempos = midi.get_tempo_change_times()
        bpm = 120.0
        if len(midi.get_tempo_change_times()[1]) > 0:
            bpm = float(midi.get_tempo_change_times()[1][0])
        tokens.append(_tempo_token(bpm))

        spb = 60.0 / bpm
        sps = spb / self.SUBDIVISIONS
        bar_duration = spb * self.BEATS_PER_BAR

        end_time = midi.get_end_time()
        n_bars = min(max_bars, max(1, int(end_time / bar_duration) + 1))

        tracks = [t for t in midi.instruments if not t.is_drum][:self.MAX_TRACKS]

        def _section_label(bar_idx: int, total: int) -> Optional[str]:
            frac = bar_idx / max(total, 1)
            if frac < 0.05:
                return "[INTRO]"
            elif frac < 0.15:
                return "[VERSE]"
            elif 0.3 < frac < 0.5:
                return "[CHORUS]"
            elif 0.5 < frac < 0.6:
                return "[BRIDGE]"
            elif frac > 0.9:
                return "[OUTRO]"
            return None

        prev_section = None
        for bar_idx in range(n_bars):
            bar_start = bar_idx * bar_duration
            bar_end   = bar_start + bar_duration

            if add_structure:
                sec = _section_label(bar_idx, n_bars)
                if sec and sec != prev_section:
                    tokens.append(sec)
                    prev_section = sec

            tokens.append("BAR")

            bar_notes_all = []
            for tr in tracks:
                for n in tr.notes:
                    if bar_start <= n.start < bar_end:
                        bar_notes_all.append(n)
            chord_tok = detect_chord(bar_notes_all)
            tokens.append(chord_tok)

            if bar_idx % 4 == 0:
                tokens.append("[PHRASE_START]")
            if bar_idx % 4 == 3:
                tokens.append("[PHRASE_END]")

            for track_idx, track in enumerate(tracks):
                track_notes = sorted(
                    [n for n in track.notes if bar_start <= n.start < bar_end],
                    key=lambda n: n.start
                )
                if not track_notes:
                    continue
                tokens.append(f"[TRACK_{track_idx}]")
                for note in track_notes:
                    pos = int((note.start - bar_start) / sps)
                    pos = max(0, min(95, pos))
                    dur = max(1, min(48, int(note.get_duration() / sps)))
                    vel = _velocity_bin(note.velocity)
                    tokens.append(f"POSITION_{pos}")
                    tokens.append(f"NOTE_ON_{note.pitch}")
                    tokens.append(f"DURATION_{dur}")
                    tokens.append(f"VELOCITY_{vel}")

            if len(tokens) > self.MAX_SEQ_LEN - 10:
                break

        tokens.append("[EOS]")
        ids = [TOKEN2ID.get(t, TOKEN2ID["[UNK]"]) for t in tokens]
        return ids

    def decode(self, ids: List[int]) -> Tuple[pretty_midi.PrettyMIDI, dict]:
        tokens = [ID2TOKEN.get(i, "[UNK]") for i in ids]

        midi_out    = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        instruments = [pretty_midi.Instrument(program=p, name=f"Track_{i}")
                       for i, p in enumerate([0, 24, 33, 40, 48, 56, 64, 73])]

        bpm = 120.0
        meta = {"style": "POP", "emotion": "HAPPY", "chords": [], "structure": []}

        spb = 60.0 / bpm
        sps = spb / self.SUBDIVISIONS
        bar_duration = spb * self.BEATS_PER_BAR

        cur_bar = 0
        cur_track = 0
        cur_pos   = 0
        cur_pitch = 60
        cur_dur   = 8
        cur_vel   = 80

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            if tok in ("[PAD]", "[UNK]", "[BOS]", "[SEP]", "[MASK]"):
                pass
            elif tok == "[EOS]":
                break
            elif tok.startswith("[STYLE_"):
                meta["style"] = tok[7:-1]
            elif tok.startswith("[EMO_"):
                meta["emotion"] = tok[5:-1]
            elif tok.startswith("TEMPO_"):
                try:
                    bpm = float(tok.split("_")[1])
                    spb = 60.0 / bpm
                    sps = spb / self.SUBDIVISIONS
                    bar_duration = spb * self.BEATS_PER_BAR
                except Exception:
                    pass
            elif tok in ("[INTRO]", "[VERSE]", "[CHORUS]", "[BRIDGE]", "[OUTRO]"):
                meta["structure"].append((cur_bar, tok[1:-1]))
            elif tok == "BAR":
                cur_bar += 1
            elif tok.startswith("CHORD_"):
                meta["chords"].append((cur_bar, tok))
            elif tok.startswith("[TRACK_"):
                try:
                    cur_track = int(tok[7:-1])
                except Exception:
                    cur_track = 0
            elif tok.startswith("POSITION_"):
                try:
                    cur_pos = int(tok.split("_")[1])
                except Exception:
                    cur_pos = 0
            elif tok.startswith("NOTE_ON_"):
                try:
                    cur_pitch = int(tok.split("_")[2])
                except Exception:
                    cur_pitch = 60
            elif tok.startswith("DURATION_"):
                try:
                    cur_dur = int(tok.split("_")[1])
                except Exception:
                    cur_dur = 8
            elif tok.startswith("VELOCITY_"):
                try:
                    cur_vel = min(127, int(tok.split("_")[1]) * 4)
                except Exception:
                    cur_vel = 80
                note_start = (cur_bar - 1) * bar_duration + cur_pos * sps
                note_end   = note_start + cur_dur * sps
                if cur_track < len(instruments) and 0 <= cur_pitch <= 127:
                    instruments[cur_track].notes.append(
                        pretty_midi.Note(
                            velocity=cur_vel,
                            pitch=cur_pitch,
                            start=note_start,
                            end=note_end
                        )
                    )

            i += 1

        for inst in instruments:
            if inst.notes:
                midi_out.instruments.append(inst)

        if not midi_out.instruments:
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(60, 64, 0.0, 0.5))
            midi_out.instruments.append(inst)

        return midi_out, meta

    def pad(self, ids: List[int], max_len: int) -> Tuple[List[int], List[int]]:
        ids = ids[:max_len]
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        ids  = ids + [PAD_ID] * (max_len - len(ids))
        return ids, mask

    def save_vocab(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"token2id": TOKEN2ID, "id2token":
                       {str(k): v for k, v in ID2TOKEN.items()}}, f, indent=2)

    @classmethod
    def load_vocab(cls, path: str) -> "MusicTokenizer":
        tokenizer = cls()
        return tokenizer
