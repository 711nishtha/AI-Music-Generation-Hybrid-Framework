"""
Stage 2: Train the Hierarchical Symbolic Planning Transformer from scratch.
Usage:
    python 02_train_symbolic.py --config config/symbolic_small.yaml
    python 02_train_symbolic.py --config config/symbolic_full.yaml
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
from src.data.dataset import SymbolicMusicDataset
from src.models.symbolic_planner import HierarchicalMusicTransformer


def cosine_lr(optimizer, step: int, warmup: int, max_steps: int,
              lr: float, lr_min: float):
    import math
    if step < warmup:
        scale = step / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, max_steps - warmup)
        scale = lr_min / lr + 0.5 * (1 - lr_min / lr) * (
            1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * scale


def main(args):
    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc   = cfg["model"]
    tc   = cfg["training"]
    pc   = cfg["paths"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("device") == "cpu":
        device = "cpu"
    print(f"[Train-Sym] Using device: {device}")

    use_amp = cfg.get("amp", True) and device == "cuda"

    # ── Tokenizer & Data ──────────────────────────────────────────────────────
    tokenizer = MusicTokenizer()

    with open(pc["train_tokens"]) as f:
        train_files = json.load(f)
    with open(pc["val_tokens"]) as f:
        val_files = json.load(f)

    train_ds = SymbolicMusicDataset(train_files, tc["max_seq_len"], tokenizer)
    val_ds   = SymbolicMusicDataset(val_files,   tc["max_seq_len"], tokenizer)

    if len(train_ds) == 0:
        print("[Train-Sym] No training data found. Run 01_prepare_data.sh first.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"],
                               shuffle=True,  num_workers=2,
                               pin_memory=device == "cuda", drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=tc["batch_size"],
                               shuffle=False, num_workers=2)

    print(f"[Train-Sym] Train: {len(train_ds)} seqs | Val: {len(val_ds)} seqs")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HierarchicalMusicTransformer(
        vocab_size   = VOCAB_SIZE,
        d_model      = mc["d_model"],
        n_heads      = mc["n_heads"],
        n_layers_enc = mc["n_layers_enc"],
        n_layers_dec = mc["n_layers_dec"],
        d_ff         = mc["d_ff"],
        max_len      = mc["max_len"],
        dropout      = mc["dropout"],
    ).to(device)

    if cfg.get("multi_gpu") and torch.cuda.device_count() > 1:
        print(f"[Train-Sym] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train-Sym] Model parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(pc["checkpoint"], exist_ok=True)
    os.makedirs(pc["runs"],       exist_ok=True)
    writer = SummaryWriter(pc["runs"])

    # Resume if checkpoint exists
    start_step = 0
    ckpt_path = os.path.join(pc["checkpoint"], "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[Train-Sym] Resumed from step {start_step}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    model.train()
    step          = start_step
    accum_loss    = 0.0
    accum_count   = 0
    optimizer.zero_grad()
    data_iter     = iter(train_loader)
    t0            = time.time()

    print(f"[Train-Sym] Starting training for {tc['max_steps']} steps...")

    while step < tc["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch     = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        cosine_lr(optimizer, step, tc["warmup_steps"],
                  tc["max_steps"], tc["lr"], tc["lr_min"])

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss    = loss / tc["grad_accum"]

        scaler.scale(loss).backward()
        accum_loss  += loss.item() * tc["grad_accum"]
        accum_count += 1

        if accum_count % tc["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1

            if step % tc["log_every"] == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / tc["grad_accum"]
                lr_now   = optimizer.param_groups[0]["lr"]
                print(f"  Step {step:6d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | {elapsed:.1f}s")
                writer.add_scalar("train/loss", avg_loss,    step)
                writer.add_scalar("train/lr",   lr_now,      step)
                accum_loss = 0.0

            # Validation
            if step % tc["eval_every"] == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        vi = vbatch["input_ids"].to(device)
                        vl = vbatch["labels"].to(device)
                        vm = vbatch["attention_mask"].to(device)
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            _, vl_loss = model(vi, attention_mask=vm, labels=vl)
                        val_losses.append(vl_loss.item())
                val_loss = sum(val_losses) / max(1, len(val_losses))
                print(f"  -- Val loss: {val_loss:.4f}")
                writer.add_scalar("val/loss", val_loss, step)
                model.train()

            # Checkpoint
            if step % tc["save_every"] == 0 or step == tc["max_steps"]:
                m_state = (model.module.state_dict()
                           if hasattr(model, "module") else model.state_dict())
                torch.save({
                    "step":      step,
                    "model":     m_state,
                    "optimizer": optimizer.state_dict(),
                    "config":    cfg,
                }, ckpt_path)
                torch.save(m_state,
                           os.path.join(pc["checkpoint"], f"step_{step}.pt"))
                print(f"  -- Saved checkpoint at step {step}")

    writer.close()
    print(f"\n[Train-Sym] Training complete. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/symbolic_small.yaml")
    main(p.parse_args())
