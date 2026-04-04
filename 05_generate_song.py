"""
Stage 5: Full inference pipeline — text prompt -> MIDI + WAV.
Usage:
    python 05_generate_song.py \
        --prompt "upbeat pop song with piano and strings, happy mood" \
        --emotion HAPPY --style POP \
        --duration_bars 64 \
        --output_name my_song

Outputs:
    outputs/my_song.mid   -- Full multi-track MIDI
    outputs/my_song.wav   -- High-fidelity WAV (via neural renderer or FluidSynth)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).parent))
from src.data.tokenizer import MusicTokenizer, VOCAB_SIZE, TOKEN2ID, BOS_ID
from src.models.symbolic_planner import HierarchicalMusicTransformer
from src.models.audio_renderer import MelDiffusionRenderer
from src.utils.audio_utils import mel_to_wav, save_wav, concatenate_wavs
from src.utils.midi_utils import midi_to_wav


def build_prompt_tokens(
    tokenizer: MusicTokenizer,
    style:   str = "POP",
    emotion: str = "HAPPY",
    tempo:   int = 120,
) -> list:
    tokens = []
    tokens.append(BOS_ID)

    style_tok = f"[STYLE_{style.upper()}]"
    emo_tok   = f"[EMO_{emotion.upper()}]"
    tempo_tok = f"TEMPO_{max(40, min(200, round(tempo / 10) * 10))}"

    for tok in [style_tok, emo_tok, tempo_tok,
                "[INTRO]", "BAR", "CHORD_C_MAJ"]:
        if tok in TOKEN2ID:
            tokens.append(TOKEN2ID[tok])
    return tokens


def stitch_sections(
    model:     HierarchicalMusicTransformer,
    tokenizer: MusicTokenizer,
    style:     str,
    emotion:   str,
    tempo:     int,
    bars_per_section: int = 16,
    n_sections:       int = 4,
    device:    str = "cpu",
    temperature:  float = 0.95,
    top_k:     int = 50,
    top_p:     float = 0.92,
) -> list:
    """
    Generate a full song by stitching multiple section generations together.
    Sections: INTRO(8), VERSE(16), CHORUS(16), VERSE(16), CHORUS(16), OUTRO(8)
    Each section seeds the next with the last N tokens for continuity.
    """
    STRUCTURE = [
        ("[INTRO]",  8),
        ("[VERSE]",  bars_per_section),
        ("[CHORUS]", bars_per_section),
        ("[VERSE]",  bars_per_section),
        ("[CHORUS]", bars_per_section),
        ("[BRIDGE]", 8),
        ("[OUTRO]",  8),
    ]

    full_ids = build_prompt_tokens(tokenizer, style, emotion, tempo)
    full_ids_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)

    for sec_tok, n_bars in STRUCTURE:
        if sec_tok in TOKEN2ID:
            sec_id = TOKEN2ID[sec_tok]
            full_ids_tensor = torch.cat([
                full_ids_tensor,
                torch.tensor([[sec_id]], device=device)
            ], dim=1)

        tokens_to_gen = n_bars * 18

        print(f"  Generating {sec_tok} ({n_bars} bars, ~{tokens_to_gen} tokens)...")
        full_ids_tensor = model.generate(
            prompt_ids     = full_ids_tensor,
            max_new_tokens = tokens_to_gen,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
        )

        full_ids_tensor = full_ids_tensor[:, -model.max_len + 128:]

    return full_ids_tensor[0].tolist()


def load_symbolic_model(ckpt_path: str, device: str) -> HierarchicalMusicTransformer:
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt.get("config", {})
        mc   = cfg.get("model", {})
        print(f"[Generate] Loaded symbolic model from {ckpt_path}")
    else:
        print(f"[Generate] Checkpoint {ckpt_path} not found -- "
              f"using randomly initialised model (train first for good results!)")
        mc = {}
        ckpt = None

    model = HierarchicalMusicTransformer(
        vocab_size    = VOCAB_SIZE,
        d_model       = mc.get("d_model",       256),
        n_heads       = mc.get("n_heads",        8),
        n_layers_enc  = mc.get("n_layers_enc",   4),
        n_layers_dec  = mc.get("n_layers_dec",   6),
        d_ff          = mc.get("d_ff",           1024),
        max_len       = mc.get("max_len",        2048),
        dropout       = 0.0,
    ).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_renderer(ckpt_path: str, device: str) -> MelDiffusionRenderer:
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt.get("config", {})
        mc   = cfg.get("model", {})
        print(f"[Generate] Loaded renderer from {ckpt_path}")
    else:
        print(f"[Generate] No renderer checkpoint found -- skipping neural audio rendering.")
        return None

    renderer = MelDiffusionRenderer(
        vocab_size     = VOCAB_SIZE,
        d_model        = mc.get("d_model",      128),
        channel_mults  = tuple(mc.get("channel_mults", [1,2,2,4])),
        attn_resolutions = tuple(mc.get("attn_resolutions", [False,True,True,True])),
        n_heads        = mc.get("n_heads",       4),
        d_context      = mc.get("d_context",     256),
        T_steps        = mc.get("T_steps",       1000),
        dropout        = 0.0,
    ).to(device)

    if ckpt is not None:
        renderer.load_state_dict(ckpt["model"])
    renderer.eval()
    return renderer


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Generate] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = MusicTokenizer()

    sym_ckpt = args.symbolic_ckpt
    if os.path.exists("checkpoints/aligned/dpo_aligned.pt"):
        sym_ckpt = "checkpoints/aligned/dpo_aligned.pt"
        print("[Generate] Using DPO-aligned model.")

    # ── Symbolic generation ───────────────────────────────────────────────────
    model = load_symbolic_model(sym_ckpt, device)

    print(f"\n[Generate] Generating '{args.output_name}' "
          f"({args.style}, {args.emotion}, {args.tempo} BPM)...")
    print(f"[Generate] Prompt: {args.prompt}\n")

    token_ids = stitch_sections(
        model        = model,
        tokenizer    = tokenizer,
        style        = args.style,
        emotion      = args.emotion,
        tempo        = args.tempo,
        bars_per_section = args.bars_per_section,
        n_sections   = 4,
        device       = device,
        temperature  = args.temperature,
        top_k        = args.top_k,
        top_p        = args.top_p,
    )

    print(f"\n[Generate] Generated {len(token_ids)} tokens. Decoding to MIDI...")

    midi_obj, meta = tokenizer.decode(token_ids)

    midi_path = os.path.join(args.output_dir, f"{args.output_name}.mid")
    midi_obj.write(midi_path)
    print(f"[Generate] MIDI saved: {midi_path}")
    print(f"           Duration: {midi_obj.get_end_time():.1f}s | "
          f"Instruments: {len(midi_obj.instruments)} | "
          f"Structure: {meta.get('structure', [])}")

    # ── Audio rendering ────────────────────────────────────────────────────────
    wav_path = os.path.join(args.output_dir, f"{args.output_name}.wav")

    renderer = None
    if not args.skip_neural_render:
        renderer = load_renderer(args.renderer_ckpt, device)

    if renderer is not None:
        print("[Generate] Running neural diffusion renderer...")
        ids_tensor = torch.tensor([token_ids[:1024]], dtype=torch.long, device=device)
        mask_tensor = (ids_tensor != 0).long()
        with torch.no_grad():
            mel = renderer.ddim_sample(
                sym_ids  = ids_tensor,
                sym_mask = mask_tensor,
                n_steps  = args.ddim_steps,
            )
        mel_np = mel[0, 0].cpu().numpy()
        audio  = mel_to_wav(mel_np)
        save_wav(audio, wav_path)
        print(f"[Generate] Neural WAV saved: {wav_path}")

        duration_estimate = midi_obj.get_end_time()
        if duration_estimate > 8.0:
            print("[Generate] Song is long -- stitching additional audio segments...")
            seg_paths = [wav_path]
            for seg_i, window_start in enumerate(range(256, len(token_ids), 512)):
                seg_ids = token_ids[window_start: window_start + 1024]
                if len(seg_ids) < 32:
                    break
                seg_tensor = torch.tensor([seg_ids], dtype=torch.long, device=device)
                seg_mask   = (seg_tensor != 0).long()
                with torch.no_grad():
                    seg_mel = renderer.ddim_sample(
                        sym_ids=seg_tensor, sym_mask=seg_mask, n_steps=args.ddim_steps)
                seg_mel_np = seg_mel[0, 0].cpu().numpy()
                seg_audio  = mel_to_wav(seg_mel_np)
                seg_path   = os.path.join(args.output_dir,
                                          f"{args.output_name}_seg{seg_i+1}.wav")
                save_wav(seg_audio, seg_path)
                seg_paths.append(seg_path)
                if len(seg_paths) >= 6:
                    break

            final_wav = os.path.join(args.output_dir, f"{args.output_name}_full.wav")
            concatenate_wavs(seg_paths, final_wav)
            print(f"[Generate] Full WAV (stitched): {final_wav}")
            os.replace(final_wav, wav_path)

    else:
        print("[Generate] Using FluidSynth for audio rendering (no neural renderer)...")
        sf_candidates = [
            "data/soundfonts/GeneralUser.sf2",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/TimGM6mb.sf2",
        ]
        soundfont = None
        for sf in sf_candidates:
            if os.path.exists(sf):
                soundfont = sf
                break

        if soundfont:
            success = midi_to_wav(midi_path, wav_path, soundfont)
            if success:
                print(f"[Generate] FluidSynth WAV saved: {wav_path}")
            else:
                print("[Generate] FluidSynth rendering failed.")
        else:
            print("[Generate] No SoundFont found. Install fluidsynth + sound fonts.")
            print(f"           MIDI is still saved at: {midi_path}")
            print("           You can open it in GarageBand, Ableton, MuseScore, etc.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
+------------------------------------------------------+
|           GENERATION COMPLETE                        |
+------------------------------------------------------+
|  MIDI  : {midi_path:<42}|
|  WAV   : {wav_path:<42}|
|  Style : {args.style:<10} Emotion: {args.emotion:<12} BPM: {args.tempo:<5}|
+------------------------------------------------------+
To listen: play {wav_path}
           (or open the .mid in any DAW / MuseScore)
    """)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",           type=str,   default="upbeat pop song with piano")
    p.add_argument("--style",            type=str,   default="POP",
                   choices=["POP","CLASSICAL","JAZZ","FOLK","ELECTRONIC","AMBIENT"])
    p.add_argument("--emotion",          type=str,   default="HAPPY",
                   choices=["HAPPY","SAD","TENSE","PEACEFUL"])
    p.add_argument("--tempo",            type=int,   default=120)
    p.add_argument("--bars_per_section", type=int,   default=16)
    p.add_argument("--output_name",      type=str,   default="generated_song")
    p.add_argument("--output_dir",       type=str,   default="outputs/")
    p.add_argument("--symbolic_ckpt",    type=str,   default="checkpoints/symbolic/latest.pt")
    p.add_argument("--renderer_ckpt",    type=str,   default="checkpoints/renderer/latest.pt")
    p.add_argument("--temperature",      type=float, default=0.95)
    p.add_argument("--top_k",            type=int,   default=50)
    p.add_argument("--top_p",            type=float, default=0.92)
    p.add_argument("--ddim_steps",       type=int,   default=50)
    p.add_argument("--skip_neural_render", action="store_true")
    main(p.parse_args())
