"""
Stage 4: DPO Preference Alignment of the Symbolic Planning Model.
Usage:
    python 04_preference_alignment.py \
        --symbolic_ckpt checkpoints/symbolic/latest.pt \
        --pref_data     data/subsets/preference_data.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from src.data.tokenizer import MusicTokenizer, VOCAB_SIZE
from src.data.dataset import PreferenceDataset
from src.models.symbolic_planner import HierarchicalMusicTransformer
from src.models.alignment import DPOTrainer


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DPO] Using device: {device}")

    tokenizer = MusicTokenizer()

    # ── Load base symbolic model ──────────────────────────────────────────────
    print(f"[DPO] Loading symbolic model from {args.symbolic_ckpt}...")
    ckpt = torch.load(args.symbolic_ckpt, map_location=device)
    cfg  = ckpt.get("config", {})
    mc   = cfg.get("model", {})

    base_model = HierarchicalMusicTransformer(
        vocab_size    = VOCAB_SIZE,
        d_model       = mc.get("d_model",       256),
        n_heads       = mc.get("n_heads",        8),
        n_layers_enc  = mc.get("n_layers_enc",   4),
        n_layers_dec  = mc.get("n_layers_dec",   6),
        d_ff          = mc.get("d_ff",           1024),
        max_len       = mc.get("max_len",        2048),
        dropout       = 0.05,
    ).to(device)

    base_model.load_state_dict(ckpt["model"])
    print(f"[DPO] Base model loaded ({base_model.num_parameters:,} params).")

    # ── DPO Trainer ───────────────────────────────────────────────────────────
    trainer = DPOTrainer(base_model, beta=args.beta).to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    if not os.path.exists(args.pref_data):
        print(f"[DPO] Preference data not found at {args.pref_data}")
        sys.exit(0)

    full_ds = PreferenceDataset(args.pref_data, max_seq_len=512, tokenizer=tokenizer)
    n_val   = max(1, int(len(full_ds) * 0.1))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    if n_train == 0:
        print("[DPO] Not enough preference data. Skipping alignment.")
        sys.exit(0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=2)

    print(f"[DPO] Preference pairs -- train: {n_train}, val: {n_val}")

    optimizer = optim.AdamW(
        trainer.policy.parameters(),
        lr=args.lr, weight_decay=0.01
    )

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "runs"))
    ckpt_path = os.path.join(args.output_dir, "dpo_aligned.pt")

    step = 0
    t0   = time.time()
    print(f"[DPO] Training for {args.max_steps} steps...")

    for epoch in range(1000):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            trainer.train()
            loss, metrics = trainer(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 20 == 0:
                print(f"  Step {step:4d} | loss {metrics['dpo_loss']:.4f} | "
                      f"margin {metrics['reward_margin']:.3f} | "
                      f"acc {metrics['accuracy']:.2%} | {time.time()-t0:.1f}s")
                for k, v in metrics.items():
                    writer.add_scalar(f"dpo/{k}", v, step)

            if step % 200 == 0:
                trainer.sync_reference()
                print(f"  -- Reference model updated at step {step}")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    torch.save({
        "step":  step,
        "model": trainer.policy.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"\n[DPO] Alignment complete. Saved: {ckpt_path}")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbolic_ckpt", default="checkpoints/symbolic/latest.pt")
    p.add_argument("--pref_data",     default="data/subsets/preference_data.json")
    p.add_argument("--output_dir",    default="checkpoints/aligned/")
    p.add_argument("--beta",          type=float, default=0.1)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--max_steps",     type=int,   default=1000)
    main(p.parse_args())
