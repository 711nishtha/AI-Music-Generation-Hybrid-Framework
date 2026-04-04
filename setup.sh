#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for AI Music Generation System
# Tested on Ubuntu 22.04 / 24.04 LTS
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. System packages ────────────────────────────────────────────────────────
log "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip ca-certificates \
    libsndfile1 libsndfile1-dev ffmpeg \
    fluidsynth fluid-soundfont-gm \
    libasound2-dev portaudio19-dev \
    python3-dev pkg-config \
    musescore3 || true   # MuseScore for score rendering (non-fatal)

# ── 2. CUDA 12.4 (skip if no GPU or already installed) ───────────────────────
if command -v nvidia-smi &>/dev/null; then
    log "NVIDIA GPU detected. Checking CUDA..."
    if ! command -v nvcc &>/dev/null; then
        log "Installing CUDA 12.4 toolkit..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        sudo apt-get install -y cuda-toolkit-12-4
        echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        export PATH=/usr/local/cuda-12.4/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
        rm -f cuda-keyring_1.1-1_all.deb
    else
        log "CUDA already installed: $(nvcc --version | head -1)"
    fi
else
    warn "No NVIDIA GPU detected — will run in CPU mode (slower but functional)."
fi

# ── 3. Miniconda ──────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    log "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
else
    log "Conda already installed: $(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# ── 4. Conda environment ──────────────────────────────────────────────────────
ENV_NAME="ai-music-gen"
if conda env list | grep -q "^${ENV_NAME}"; then
    warn "Conda env '${ENV_NAME}' already exists — skipping creation."
else
    log "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

# ── 5. PyTorch + torchaudio ───────────────────────────────────────────────────
log "Installing PyTorch 2.3 + torchaudio..."
if command -v nvcc &>/dev/null; then
    pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 -q
else
    pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── 6. Core Python packages ───────────────────────────────────────────────────
log "Installing Python packages..."
pip install -q \
    transformers==4.41.2 \
    accelerate==0.30.1 \
    datasets==2.19.2 \
    diffusers==0.27.2 \
    einops==0.8.0 \
    miditoolkit==1.0.1 \
    pretty_midi==0.2.10 \
    music21==9.1.0 \
    muspy==0.5.0 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    scipy==1.13.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    wandb==0.17.0 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    tensorboard==2.17.0 \
    scikit-learn==1.5.0 \
    mir_eval==0.7 \
    pyfluidsynth==1.3.3 \
    gradio==4.37.2 \
    requests==2.32.3 \
    gdown==5.2.0

# EnCodec for audio tokenisation reference
pip install -q encodec==0.1.1

log "Installing miditok for advanced MIDI tokenisation..."
pip install -q miditok==3.0.4

log "All packages installed."

# ── 7. Create project directory structure ─────────────────────────────────────
log "Creating project structure..."
PROJ_DIR="$(pwd)/ai-music-gen"
mkdir -p "${PROJ_DIR}"/{src/{data,models,utils},config,data/{raw/{midi,audio},processed,subsets},runs,outputs,checkpoints}

log "
=============================================================================
  Setup complete!  Activate your environment with:
    conda activate ${ENV_NAME}
  Then run the full pipeline:
    bash run_full_pipeline.sh
============================================================================="
