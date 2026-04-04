#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
# One-command runner: setup -> data -> train -> generate songs
# Usage:
#   bash run_full_pipeline.sh                    # full pipeline
#   bash run_full_pipeline.sh --skip-setup       # skip setup (already done)
#   bash run_full_pipeline.sh --skip-data        # skip data download
#   bash run_full_pipeline.sh --generate-only    # only generate songs
#   FULL_SCALE=1 bash run_full_pipeline.sh       # 8-GPU full-scale mode
# =============================================================================
set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_SETUP=0; SKIP_DATA=0; GEN_ONLY=0; FULL_SCALE=${FULL_SCALE:-0}
for arg in "$@"; do
    case $arg in
        --skip-setup)    SKIP_SETUP=1 ;;
        --skip-data)     SKIP_DATA=1  ;;
        --generate-only) GEN_ONLY=1   ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

banner() {
    echo -e "\n${BLUE}${BOLD}+================================================+${NC}"
    echo -e "${BLUE}${BOLD}|  $1${NC}"
    echo -e "${BLUE}${BOLD}+================================================+${NC}\n"
}

log()  { echo -e "${GREEN}[PIPELINE]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}    $*"; }
err()  { echo -e "${RED}[ERROR]${NC}   $*"; exit 1; }

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJ_DIR}"

banner "AI Music Generation Pipeline -- Research Prototype"
log "Project directory: ${PROJ_DIR}"
log "Full-scale mode:   ${FULL_SCALE} (set FULL_SCALE=1 for 8-GPU training)"

# ── Step 0: Activate conda environment ───────────────────────────────────────
eval "$(conda shell.bash hook)" 2>/dev/null || \
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || \
    { warn "Cannot activate conda -- ensure setup.sh has been run"; }
conda activate ai-music-gen 2>/dev/null || warn "Could not activate ai-music-gen env"

# ── Step 1: Setup ─────────────────────────────────────────────────────────────
if [ "${SKIP_SETUP}" -eq 0 ] && [ "${GEN_ONLY}" -eq 0 ]; then
    banner "Step 1/5: System Setup"
    bash setup.sh
fi

# ── Step 2: Data preparation ──────────────────────────────────────────────────
if [ "${SKIP_DATA}" -eq 0 ] && [ "${GEN_ONLY}" -eq 0 ]; then
    banner "Step 2/5: Data Download & Preprocessing"
    bash 01_prepare_data.sh
fi

# ── Step 3: Train symbolic planning model ────────────────────────────────────
if [ "${GEN_ONLY}" -eq 0 ]; then
    banner "Step 3/5: Training Symbolic Planning Model"

    if [ "${FULL_SCALE}" -eq 1 ]; then
        SYM_CFG="config/symbolic_full.yaml"
        log "Full-scale training -- using config: ${SYM_CFG}"
        log "Expected: ~48-72 hours on 8xA100"
    else
        SYM_CFG="config/symbolic_small.yaml"
        log "Small-scale training -- using config: ${SYM_CFG}"
        log "Expected: ~1-3 hours on a single GPU (or ~6 hours CPU)"
    fi

    python 02_train_symbolic.py --config "${SYM_CFG}"
    log "Symbolic model trained."
fi

# ── Step 4: Train audio renderer ─────────────────────────────────────────────
if [ "${GEN_ONLY}" -eq 0 ]; then
    banner "Step 4/5: Training Audio Renderer"

    if [ "${FULL_SCALE}" -eq 1 ]; then
        REN_CFG="config/renderer_full.yaml"
    else
        REN_CFG="config/renderer_small.yaml"
    fi

    if [ -f "data/subsets/train_pairs.json" ]; then
        PAIR_COUNT=$(python -c "import json; d=json.load(open('data/subsets/train_pairs.json')); print(len(d))")
        if [ "${PAIR_COUNT}" -gt 10 ]; then
            python 03_train_renderer.py --config "${REN_CFG}"
            log "Audio renderer trained."
        else
            warn "Fewer than 10 audio pairs -- skipping neural renderer training."
            warn "Will use FluidSynth for audio output."
        fi
    else
        warn "No train_pairs.json found -- skipping neural renderer."
        warn "Audio output will use FluidSynth (still sounds good!)."
    fi
fi

# ── Step 4b: DPO Alignment ────────────────────────────────────────────────────
if [ "${GEN_ONLY}" -eq 0 ] && [ -f "checkpoints/symbolic/latest.pt" ]; then
    banner "Step 4b/5: DPO Preference Alignment"
    if [ -f "data/subsets/preference_data.json" ]; then
        PREF_COUNT=$(python -c "import json; d=json.load(open('data/subsets/preference_data.json')); print(len(d))")
        if [ "${PREF_COUNT}" -gt 20 ]; then
            python 04_preference_alignment.py \
                --symbolic_ckpt checkpoints/symbolic/latest.pt \
                --pref_data     data/subsets/preference_data.json \
                --max_steps     1000
            log "DPO alignment complete."
        else
            warn "Too few preference pairs (${PREF_COUNT}) -- skipping DPO."
        fi
    fi
fi

# ── Step 5: Generate sample songs ────────────────────────────────────────────
banner "Step 5/5: Generating Sample Songs"

mkdir -p outputs/

log "Generating Song 1: Upbeat Pop..."
python 05_generate_song.py \
    --prompt "upbeat pop song with catchy melody, piano and strings" \
    --style POP --emotion HAPPY --tempo 128 \
    --output_name song_01_pop_happy \
    --bars_per_section 16

log "Generating Song 2: Melancholic Classical..."
python 05_generate_song.py \
    --prompt "melancholic classical piano piece with gentle dynamics" \
    --style CLASSICAL --emotion SAD --tempo 72 \
    --output_name song_02_classical_sad \
    --bars_per_section 16

log "Generating Song 3: Peaceful Ambient..."
python 05_generate_song.py \
    --prompt "peaceful ambient soundscape with slow evolving pads" \
    --style AMBIENT --emotion PEACEFUL --tempo 80 \
    --output_name song_03_ambient_peaceful \
    --bars_per_section 12

log "Generating Song 4: Tense Jazz..."
python 05_generate_song.py \
    --prompt "tense jazz with dissonant chords and syncopated rhythm" \
    --style JAZZ --emotion TENSE --tempo 140 \
    --output_name song_04_jazz_tense \
    --bars_per_section 16

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "+============================================================+"
echo "|           PIPELINE COMPLETE                                |"
echo "+============================================================+"
echo "|  Generated songs are in: outputs/                         |"
echo "+============================================================+"
echo "|  To listen (Linux):                                        |"
echo "|    aplay outputs/song_01_pop_happy.wav                     |"
echo "|    vlc outputs/song_01_pop_happy.wav                       |"
echo "|    ffplay outputs/song_01_pop_happy.wav                    |"
echo "|                                                            |"
echo "|  To open MIDI in MuseScore:                                |"
echo "|    musescore3 outputs/song_01_pop_happy.mid                |"
echo "+============================================================+"
echo "|  Training logs: tensorboard --logdir runs/                 |"
echo "+============================================================+"
echo -e "${NC}"

echo "Generated files:"
ls -lh outputs/*.wav outputs/*.mid 2>/dev/null || true
