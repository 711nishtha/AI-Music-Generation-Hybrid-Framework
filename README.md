# AI Music Generation — Research Implementation

Implementation of the hybrid three-layer AI music generation framework from the paper
**"AI in Music Generation"**, exactly as proposed:

| Layer | Component | Implementation |
|---|---|---|
| Symbolic Planning | Hierarchical Transformer | `src/models/symbolic_planner.py` |
| Audio Rendering | Mel-Spectrogram Diffusion U-Net | `src/models/audio_renderer.py` |
| Alignment | Direct Preference Optimisation (DPO) | `src/models/alignment.py` |

---

## Quick Start (2 commands)

```bash
# 1. Run setup (installs everything — ~5-10 minutes first time)
bash setup.sh

# 2. Run full pipeline (data + train + generate)
bash run_full_pipeline.sh
```

After completion, find your generated music in `outputs/`:
```
outputs/
  song_01_pop_happy.mid
  song_01_pop_happy.wav
  song_02_classical_sad.mid
  song_02_classical_sad.wav
  song_03_ambient_peaceful.mid
  song_03_ambient_peaceful.wav
  song_04_jazz_tense.mid
  song_04_jazz_tense.wav
```

---

## How to Listen

```bash
# Linux (command line)
aplay outputs/song_01_pop_happy.wav
ffplay outputs/song_01_pop_happy.wav      # needs ffmpeg
vlc outputs/song_01_pop_happy.wav         # needs VLC

# Open MIDI in MuseScore (view score + notation)
musescore3 outputs/song_01_pop_happy.mid

# Convert to MP3
ffmpeg -i outputs/song_01_pop_happy.wav outputs/song_01_pop_happy.mp3
```

---

## Generating Custom Songs

```bash
conda activate ai-music-gen

python 05_generate_song.py \
    --prompt   "epic orchestral battle theme with brass and percussion" \
    --style    CLASSICAL \
    --emotion  TENSE \
    --tempo    140 \
    --bars_per_section 16 \
    --output_name my_custom_song

# Additional options:
#   --temperature 0.8      # lower = more conservative
#   --top_p 0.85           # nucleus sampling p
#   --skip_neural_render   # force FluidSynth output
```

**Style options:** `POP | CLASSICAL | JAZZ | FOLK | ELECTRONIC | AMBIENT`
**Emotion options:** `HAPPY | SAD | TENSE | PEACEFUL`

---

## Pipeline Details

### Stage 1: Data Preparation
```bash
bash 01_prepare_data.sh
```
Downloads POP909, Lakh MIDI, MAESTRO. Tokenises with REMI+ vocabulary (512 tokens).
Renders MIDI to WAV via FluidSynth. Generates synthetic preference pairs.

### Stage 2: Symbolic Model Training
```bash
python 02_train_symbolic.py --config config/symbolic_small.yaml
```
Trains the Hierarchical Music Transformer from random initialisation.
- Small config: ~15M params, ~1-3 hours single GPU
- Full config: ~80M params, ~48-72 hours 8xA100

### Stage 3: Audio Renderer Training
```bash
python 03_train_renderer.py --config config/renderer_small.yaml
```
Trains the Mel-Spectrogram Diffusion U-Net conditioned on symbolic tokens.
Requires FluidSynth-rendered audio pairs from Stage 1.

### Stage 4: DPO Alignment
```bash
python 04_preference_alignment.py \
    --symbolic_ckpt checkpoints/symbolic/latest.pt \
    --pref_data     data/subsets/preference_data.json
```
Fine-tunes the symbolic model with Direct Preference Optimisation using
heuristic-scored synthetic preference pairs (or human-labelled data if available).

---

## Monitor Training

```bash
# TensorBoard
tensorboard --logdir runs/

# Check checkpoints
ls -lh checkpoints/symbolic/
ls -lh checkpoints/renderer/
ls -lh checkpoints/aligned/
```

---

## Full-Scale 8xA100 Training

```bash
FULL_SCALE=1 bash run_full_pipeline.sh
```

Or manually:
```bash
# Multi-GPU symbolic training
torchrun --nproc_per_node=8 02_train_symbolic.py --config config/symbolic_full.yaml

# Multi-GPU renderer training
torchrun --nproc_per_node=8 03_train_renderer.py --config config/renderer_full.yaml
```

Resources needed (as per paper):
- 8x NVIDIA A100/A800 GPUs (80GB VRAM each)
- ~500GB storage for full datasets
- Estimated training: 48-72h (symbolic), 72-96h (renderer)

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `fluidsynth` not found | `sudo apt install fluidsynth fluid-soundfont-gm` |
| No SoundFont | `sudo apt install fluid-soundfont-gm` |
| CUDA OOM | Reduce `batch_size` in config YAML |
| No GPU, slow training | Normal — CPU training is ~10x slower but works |
| Poor generation quality | Train longer or use more data (increase `max_steps`) |
| Empty MIDI | Training hasn't converged yet — run more steps |

---

## Architecture Summary (as per paper)

```
Text Prompt / Emotion / Style
         |
    +----v--------------------------+
    |  Symbolic Planning Layer       |  <- Hierarchical Transformer
    |  (REMI+ tokenised MIDI)        |     Structure encoder (4L)
    |  Multi-track MIDI output       |   + Detail decoder (6L)
    +----+---------------------------+     Cross-attention between layers
         | symbolic tokens
    +----v--------------------------+
    |  Audio Rendering Layer         |  <- Diffusion U-Net
    |  Cross-attention on symbolic   |     DDPM training, DDIM 50-step sampling
    |  Mel -> WAV (Griffin-Lim)      |     ~25M parameters
    +----+--------------------------+
         |
    +----v--------------------------+
    |  Alignment Layer (DPO)         |  <- Direct Preference Optimisation
    |  Policy vs Reference           |     Synthetic heuristic preferences
    |  beta = 0.1 KL penalty         |   + supports human-labelled data
    +-------------------------------+
         |
    MIDI + WAV output
```

---

## Datasets Used

| Dataset | Size | Use |
|---|---|---|
| POP909 | 909 MIDI files | Pop music symbolic training |
| Lakh MIDI (clean) | ~17k files | Diverse genre symbolic training |
| MAESTRO v3 | ~200h piano | Classical symbolic + audio pairs |
| MusicCaps | Metadata only | Text-to-music evaluation captions |

---

## Project Structure

```
ai-music-gen/
├── setup.sh                         <- System setup (CUDA, Conda, packages)
├── run_full_pipeline.sh             <- ONE command to run everything
├── 01_prepare_data.sh               <- Download + preprocess datasets
├── 02_train_symbolic.py             <- Train Hierarchical Music Transformer
├── 03_train_renderer.py             <- Train Diffusion Audio Renderer
├── 04_preference_alignment.py       <- DPO alignment training
├── 05_generate_song.py              <- Generate MIDI + WAV songs
├── README.md
├── config/
│   ├── symbolic_small.yaml          <- Single-GPU prototype config
│   ├── symbolic_full.yaml           <- 8xA100 full config
│   ├── renderer_small.yaml
│   └── renderer_full.yaml
└── src/
    ├── data/
    │   ├── tokenizer.py             <- REMI+ 512-token vocabulary
    │   ├── dataset.py               <- 3 PyTorch Dataset classes
    │   └── prepare_data.py          <- Full data pipeline
    ├── models/
    │   ├── symbolic_planner.py      <- Hierarchical Transformer (~15M)
    │   ├── audio_renderer.py        <- Diffusion U-Net (~25M)
    │   └── alignment.py             <- DPO Trainer
    └── utils/
        ├── midi_utils.py
        └── audio_utils.py
```

---

## Citation

Based on the research proposal:
> "AI in Music Generation" — Hybrid Framework with Symbolic Planning,
> Diffusion Audio Rendering, and Preference Alignment (2026)
