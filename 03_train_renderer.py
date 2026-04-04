"""
Stage 3: Train the Mel-Spectrogram Diffusion Audio Renderer from scratch.
Usage:
    python 03_train_renderer.py --config config/renderer_small.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.data.tokenizer import MusicTokenizer, VOCAB_SIZE
from src.data.dataset import AudioRendererDataset
from src.models.audio_renderer import MelDiffusionRenderer


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    tc = cfg["training"]
    pc = cfg["paths"]

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.get("amp", True) and device == "cuda"
    print(f"[Train-Ren] Using device: {device}")

    tokenizer = MusicTokenizer()

    # ── Data ──────────────────────────────────────────────────────────────────
    if not os.path.exists(pc["train_pairs"]):
        print(f"[Train-Ren] No pair manifest at {pc['train_pairs']}.")
        print("[Train-Ren] Falling back to symbolic-only WAV generation mode.")
        print("[Train-Ren] Skipping renderer training.")
        sys.exit(0)

    train_ds = AudioRendererDataset(pc["train_pairs"], tc["max_seq_len"],
                                     tokenizer, augment=True)
    val_ds   = AudioRendererDataset(pc["val_pairs"],   tc["max_seq_len"],
                                     tokenizer, augment=False)

    if len(train_ds) == 0:
        print("[Train-Ren] No audio pairs found. Skipping renderer training.")
        sys.exit(0)

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"],
                               shuffle=True,  num_workers=2,
                               pin_memory=device == "cuda", drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=tc["batch_size"],
                               shuffle=False, num_workers=2)

    print(f"[Train-Ren] Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MelDiffusionRenderer(
        vocab_size     = VOCAB_SIZE,
        d_model        = mc["d_model"],
        channel_mults  = tuple(mc["channel_mults"]),
        attn_resolutions = tuple(mc["attn_resolutions"]),
        n_heads        = mc["n_heads"],
        d_context      = mc["d_context"],
        T_steps        = mc["T_steps"],
        dropout        = mc["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train-Ren] Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=tc["lr"])
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(pc["checkpoint"], exist_ok=True)
    os.makedirs(pc["runs"],       exist_ok=True)
    writer = SummaryWriter(pc["runs"])

    ckpt_path = os.path.join(pc["checkpoint"], "latest.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[Train-Ren] Resumed from step {start_step}")

    model.train()
    step      = start_step
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    t0        = time.time()
    accum     = 0

    print(f"[Train-Ren] Training for {tc['max_steps']} steps...")

    while step < tc["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch     = next(data_iter)

        mel      = batch["mel"].to(device)
        sym_ids  = batch["input_ids"].to(device)
        sym_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model.training_loss(mel, sym_ids, sym_mask)
            loss = loss / tc["grad_accum"]

        scaler.scale(loss).backward()
        accum += 1

        if accum % tc["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accum = 0
            step += 1

            if step % tc["log_every"] == 0:
                print(f"  Step {step:6d} | diffusion_loss {loss.item() * tc['grad_accum']:.5f}"
                      f" | {time.time() - t0:.1f}s")
                writer.add_scalar("train/diffusion_loss",
                                   loss.item() * tc["grad_accum"], step)

            if step % tc["save_every"] == 0:
                torch.save({
                    "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "config": cfg
                }, ckpt_path)
                print(f"  -- Checkpoint saved at step {step}")

    writer.close()
    print(f"\n[Train-Ren] Renderer training complete. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/renderer_small.yaml")
    main(p.parse_args())
