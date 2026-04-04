#!/usr/bin/env bash
# =============================================================================
# 01_prepare_data.sh — Download & preprocess datasets
# Downloads small subsets suitable for single-GPU prototype training
# =============================================================================
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate ai-music-gen 2>/dev/null || true

PROJ="$(pwd)"
DATA_RAW="${PROJ}/data/raw"
DATA_PROC="${PROJ}/data/processed"
SUBSET="${PROJ}/data/subsets"

mkdir -p "${DATA_RAW}/midi" "${DATA_RAW}/audio" "${DATA_PROC}" "${SUBSET}"

log() { echo -e "\033[0;32m[DATA]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }

# ── POP909 (909 MIDI files, pop music, ~25 MB) ────────────────────────────────
log "Downloading POP909 dataset..."
if [ ! -d "${DATA_RAW}/midi/POP909" ]; then
    git clone --depth 1 https://github.com/music-x-lab/POP909-Dataset.git \
        "${DATA_RAW}/midi/POP909" 2>/dev/null || \
    wget -q "https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip" \
        -O /tmp/pop909.zip && unzip -q /tmp/pop909.zip -d "${DATA_RAW}/midi/" && \
        mv "${DATA_RAW}/midi/POP909-Dataset-master" "${DATA_RAW}/midi/POP909"
    log "POP909 downloaded."
else
    log "POP909 already exists, skipping."
fi

# ── Lakh MIDI Dataset — clean_midi subset (~17k MIDI files) ──────────────────
log "Downloading Lakh MIDI clean subset (this may take a few minutes)..."
if [ ! -d "${DATA_RAW}/midi/lakh_clean" ]; then
    wget -q "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz" \
        -O /tmp/clean_midi.tar.gz || \
    { warn "Lakh mirror failed — trying alternative..."; \
      wget -q "https://github.com/craffel/midi-dataset/raw/master/README.md" -O /dev/null; }
    mkdir -p "${DATA_RAW}/midi/lakh_clean"
    if [ -f /tmp/clean_midi.tar.gz ]; then
        tar -xzf /tmp/clean_midi.tar.gz -C "${DATA_RAW}/midi/" 2>/dev/null || true
        mv "${DATA_RAW}/midi/clean_midi" "${DATA_RAW}/midi/lakh_clean" 2>/dev/null || true
        rm -f /tmp/clean_midi.tar.gz
        log "Lakh MIDI downloaded."
    else
        log "Lakh MIDI unavailable — will use POP909 + MAESTRO only."
    fi
fi

# ── MAESTRO v3 — piano performances + MIDI (small subset) ────────────────────
log "Downloading MAESTRO v3 subset..."
if [ ! -d "${DATA_RAW}/midi/maestro" ]; then
    mkdir -p "${DATA_RAW}/midi/maestro"
    wget -q "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json" \
        -O "${DATA_RAW}/midi/maestro/maestro-v3.0.0.json" || \
        warn "MAESTRO metadata unavailable — using POP909 only."
    wget -q "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" \
        -O /tmp/maestro_midi.zip && \
        unzip -q /tmp/maestro_midi.zip -d "${DATA_RAW}/midi/maestro/" && \
        rm -f /tmp/maestro_midi.zip && \
        log "MAESTRO MIDI downloaded." || \
        warn "MAESTRO download failed — using POP909 only."
fi

# ── Download GeneralUser GS SoundFont for MIDI→WAV rendering ─────────────────
log "Downloading SoundFont for MIDI rendering..."
SF_DIR="${PROJ}/data/soundfonts"
mkdir -p "${SF_DIR}"
if [ ! -f "${SF_DIR}/GeneralUser.sf2" ]; then
    wget -q "https://schristiancollins.com/sounds/GeneralUser_GS_1.471.zip" \
        -O /tmp/gu_sf.zip 2>/dev/null || \
    { log "Using system soundfont..."; \
      cp /usr/share/sounds/sf2/FluidR3_GM.sf2 "${SF_DIR}/GeneralUser.sf2" 2>/dev/null || \
      cp /usr/share/sounds/sf2/TimGM6mb.sf2 "${SF_DIR}/GeneralUser.sf2" 2>/dev/null || \
      find /usr -name "*.sf2" 2>/dev/null | head -1 | xargs -I{} cp {} "${SF_DIR}/GeneralUser.sf2"; }
    if [ -f /tmp/gu_sf.zip ]; then
        unzip -q /tmp/gu_sf.zip -d /tmp/gu_sf/
        find /tmp/gu_sf -name "*.sf2" | head -1 | xargs -I{} cp {} "${SF_DIR}/GeneralUser.sf2"
        rm -rf /tmp/gu_sf /tmp/gu_sf.zip
    fi
fi
log "SoundFont ready: ${SF_DIR}/GeneralUser.sf2"

# ── Run Python data pipeline ──────────────────────────────────────────────────
log "Running Python data preprocessing pipeline..."
python src/data/prepare_data.py \
    --midi_dirs "${DATA_RAW}/midi/POP909" \
                "${DATA_RAW}/midi/lakh_clean" \
                "${DATA_RAW}/midi/maestro" \
    --output_dir "${DATA_PROC}" \
    --subset_dir "${SUBSET}" \
    --soundfont   "${SF_DIR}/GeneralUser.sf2" \
    --max_files   2000 \
    --subset_size 500

log "Data preparation complete. Subsets saved to: ${SUBSET}/"
