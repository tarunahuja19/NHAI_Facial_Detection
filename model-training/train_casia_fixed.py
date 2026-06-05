"""
train_casia.py  --  Iris on CASIA-WebFace (HuggingFace, JAX/Flax)
========================================================================

The script will:
  1. Load the SaffalPoosh/casia_web_face dataset from HuggingFace
  2. Pre-process images (align + resize to 112x112, normalize)
  3. Train Iris with AdaFace loss
  4. Validate on a held-out split (rank-1 retrieval accuracy on the val split)
  5. Save checkpoints to /home/jemin/Desktop/Code/nhai/checkpoints/

Requirements:
    jax, flax, optax, opencv-python, tqdm, numpy, pillow, datasets, pyarrow, orbax-checkpoint
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import time
import math
import random
import logging
import argparse
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ─── third-party ──────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image
import cv2

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import flax.linen as nn
from flax.training import train_state
from flax import struct
import orbax.checkpoint as ocp
from tqdm import tqdm

# ─── local ────────────────────────────────────────────────────────────────────
from model import Iris, count_params, estimate_int8_size_mb
from losses import AdaFaceLoss, ArcFaceLoss

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class TrainConfig:
    # ── paths ──
    output_dir:       str   = "./output"
    checkpoint_dir:   str   = "./checkpoints"
    log_file:         str   = "./train.log"

    # ── data ──
    image_size:       int   = 112
    val_fraction:     float = 0.05      # 5% of identities held out for val (~528 identities)
    num_workers:      int   = 4
    prefetch:         int   = 2

    # ── model ──
    num_classes:      int   = 10572     # CASIA-WebFace identities
    embedding_dim:    int   = 512
    drop_path_rate:   float = 0.1
    use_remat:        bool  = True       # gradient checkpointing — saves VRAM

    # ── loss ──
    loss_type:        str   = "adaface"  # "arcface" | "adaface"
    arc_scale:        float = 64.0
    arc_margin:       float = 0.4        # AdaFace default
    ada_h:            float = 0.333

    # ── optimiser ──
    batch_size:       int   = 128
    num_epochs:       int   = 25
    warmup_epochs:    int   = 1
    base_lr:          float = 0.03       # SGD with momentum
    momentum:         float = 0.9
    weight_decay:     float = 5e-4
    # LR milestones (epoch numbers at which to multiply lr by 0.1)
    lr_milestones:    Tuple = (10, 18, 22)

    # ── regularisation ──
    label_smoothing:  float = 0.0

    # ── logging ──
    log_every:        int   = 50         # steps
    val_every:        int   = 1          # epochs
    save_every:       int   = 1          # epochs
    keep_last_n:      int   = 3


CFG = TrainConfig()

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers when rerunning.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    
    # Ensure parent directory of log file exists.
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset splits
# ──────────────────────────────────────────────────────────────────────────────

def train_val_split_hf(
    dataset,
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Hold out `val_fraction` of identities entirely for validation."""
    labels = np.array(dataset["label"])
    unique_labels = np.unique(labels)
    
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(unique_labels) * val_fraction))
    val_labels = set(rng.choice(unique_labels, size=n_val, replace=False))
    
    train_indices = [i for i, l in enumerate(labels) if l not in val_labels]
    val_indices = [i for i, l in enumerate(labels) if l in val_labels]
    
    return train_indices, val_indices


# ──────────────────────────────────────────────────────────────────────────────
#  Data loader
# ──────────────────────────────────────────────────────────────────────────────

def batch_generator_hf(
    dataset,
    batch_size: int,
    image_size: int,
    augment: bool,
    shuffle: bool,
    seed: int = 0,
    drop_last: bool = True,
):
    """Generator that yields (images, labels) numpy arrays from HF dataset."""
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    
    _MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    while True:
        if shuffle:
            rng.shuffle(indices)

        end = len(indices) if not drop_last else len(indices) - batch_size + 1
        for start in range(0, max(end, 0), batch_size):
            batch_idx = indices[start : start + batch_size]
            if drop_last and len(batch_idx) < batch_size:
                continue

            imgs, lbls = [], []
            for i in batch_idx:
                try:
                    row = dataset[i]
                    pil_img = row["image"]
                    label = row["label"]
                    
                    # Convert PIL to numpy BGR array
                    img = np.array(pil_img)
                    if img.ndim == 2:  # grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:  # RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    
                    # Ensure it is resized to image_size
                    if img.shape[0] != image_size or img.shape[1] != image_size:
                        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                    
                    img = img.astype(np.float32) / 255.0

                    if augment:
                        # Horizontal flip
                        if random.random() < 0.5:
                            img = img[:, ::-1, :]
                        # Brightness jitter: ±10%
                        img = np.clip(img * (1.0 + random.uniform(-0.1, 0.1)), 0.0, 1.0)

                    img = (img - _MEAN) / _STD                      # normalize to [-1, 1]
                    imgs.append(img.transpose(2, 0, 1))             # HWC → CHW
                    lbls.append(label)
                except Exception:
                    continue

            if not imgs:
                continue

            yield (
                np.stack(imgs, axis=0),        # (B, 3, H, W)
                np.array(lbls, dtype=np.int32), # (B,)
            )


# ──────────────────────────────────────────────────────────────────────────────
#  Learning-rate schedule
# ──────────────────────────────────────────────────────────────────────────────

def make_lr_schedule(
    base_lr: float,
    warmup_steps: int,
    milestones_steps: List[int],
    gamma: float = 0.1,
):
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    decay = optax.piecewise_constant_schedule(
        init_value=base_lr,
        boundaries_and_scales={b: gamma for b in milestones_steps},
    )
    return optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps],
    )


# ──────────────────────────────────────────────────────────────────────────────
#  TrainState
# ──────────────────────────────────────────────────────────────────────────────

class FaceTrainState(train_state.TrainState):
    batch_stats:  any
    loss_params:  any


# ──────────────────────────────────────────────────────────────────────────────
#  Initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_train_state(
    cfg: TrainConfig,
    steps_per_epoch: int,
    key: jnp.ndarray,
) -> FaceTrainState:
    k1, k2, k3 = jr.split(key, 3)

    # ── backbone ──
    model = Iris(
        embedding_dim  = cfg.embedding_dim,
        drop_path_rate = cfg.drop_path_rate,
        use_remat      = cfg.use_remat,
    )
    dummy = jr.normal(k1, (2, 3, cfg.image_size, cfg.image_size))
    backbone_vars = model.init(
        {"params": k2, "droppath": k2},
        dummy,
        training=False,
    )

    # ── loss head ──
    if cfg.loss_type == "adaface":
        loss_module = AdaFaceLoss(
            num_classes     = cfg.num_classes,
            embedding_dim   = cfg.embedding_dim,
            scale           = cfg.arc_scale,
            margin          = cfg.arc_margin,
            h               = cfg.ada_h,
            label_smoothing = cfg.label_smoothing,
        )
    else:
        loss_module = ArcFaceLoss(
            num_classes     = cfg.num_classes,
            embedding_dim   = cfg.embedding_dim,
            scale           = cfg.arc_scale,
            margin          = cfg.arc_margin,
            label_smoothing = cfg.label_smoothing,
        )

    dummy_emb = jnp.ones((2, cfg.embedding_dim)) / math.sqrt(cfg.embedding_dim)
    dummy_lbl = jnp.zeros((2,), dtype=jnp.int32)
    
    if cfg.loss_type == "adaface":
        loss_vars  = loss_module.init({"params": k3}, dummy_emb, dummy_lbl, jnp.ones((2,)))
    else:
        loss_vars  = loss_module.init({"params": k3}, dummy_emb, dummy_lbl)

    # ── optimiser ──
    warmup_steps     = cfg.warmup_epochs * steps_per_epoch
    milestone_steps  = [m * steps_per_epoch for m in cfg.lr_milestones]
    schedule         = make_lr_schedule(cfg.base_lr, warmup_steps, milestone_steps)

    tx = optax.chain(
        optax.add_decayed_weights(cfg.weight_decay),
        optax.clip_by_global_norm(5.0),
        optax.sgd(
            learning_rate = schedule,
            momentum      = cfg.momentum,
            nesterov      = True,
        ),
    )

    all_params = {
        "backbone": backbone_vars["params"],
        "loss_head": loss_vars["params"],
    }

    return FaceTrainState.create(
        apply_fn    = model.apply,
        params      = all_params,
        tx          = tx,
        batch_stats = backbone_vars.get("batch_stats", {}),
        loss_params = {},
    ), model, loss_module


# ──────────────────────────────────────────────────────────────────────────────
#  Train step
# ──────────────────────────────────────────────────────────────────────────────

def make_train_step(model, loss_module, cfg: TrainConfig):

    @jax.jit
    def train_step(
        state: FaceTrainState,
        images:  jnp.ndarray,
        labels:  jnp.ndarray,
        rng:     jnp.ndarray,
    ):
        def loss_fn(params):
            backbone_vars = {
                "params":      params["backbone"],
                "batch_stats": state.batch_stats,
            }
            (embeddings, feat_norm), updates = model.apply(
                backbone_vars,
                images,
                training    = True,
                rngs        = {"droppath": rng},
                mutable     = ["batch_stats"],
            )
            new_batch_stats = updates["batch_stats"]

            # ── margin loss ──
            if cfg.loss_type == "adaface":
                margin_loss = loss_module.apply(
                    {"params": params["loss_head"]},
                    embeddings,
                    labels,
                    feat_norm,
                )
            else:
                margin_loss = loss_module.apply(
                    {"params": params["loss_head"]},
                    embeddings,
                    labels,
                )

            total = margin_loss
            return total, (margin_loss, new_batch_stats, embeddings)

        (loss, (margin_loss, new_bs, embeddings)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_bs)

        metrics = {
            "loss":         loss,
            "margin_loss":  margin_loss,
            "grad_norm":    optax.global_norm(grads),
        }
        return state, metrics, embeddings

    return train_step


# ──────────────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────────────

def make_val_step(model):
    @jax.jit
    def val_step(state: FaceTrainState, images: jnp.ndarray):
        backbone_vars = {
            "params":      state.params["backbone"],
            "batch_stats": state.batch_stats,
        }
        embeddings, _ = model.apply(
            backbone_vars,
            images,
            training = False,
            rngs     = {"droppath": jr.PRNGKey(0)},
        )
        return embeddings
    return val_step


def evaluate_rank1(
    val_ds,
    state: FaceTrainState,
    model,
    cfg: TrainConfig,
    logger: logging.Logger,
) -> float:
    if len(val_ds) == 0:
        return 0.0

    val_step = make_val_step(model)

    all_embs   = []
    all_labels = []

    eval_bs = min(64, len(val_ds))
    gen = batch_generator_hf(
        val_ds,
        batch_size=eval_bs,
        image_size=cfg.image_size,
        augment=False,
        shuffle=False,
        drop_last=False,
    )
    n_batches = math.ceil(len(val_ds) / eval_bs)
    for _ in range(n_batches):
        imgs, lbls = next(gen)
        embs = val_step(state, jnp.array(imgs))
        all_embs.append(np.array(embs))
        all_labels.extend(lbls.tolist())

    E = np.concatenate(all_embs, axis=0)[:len(val_ds)]  # trim padding
    L = np.array(all_labels[:len(val_ds)])

    # Cosine similarity matrix
    sim   = E @ E.T                              # already L2-normalised
    np.fill_diagonal(sim, -2.0)                  # exclude self
    nn_idx = np.argmax(sim, axis=1)              # nearest neighbour indices
    correct = (L[nn_idx] == L).sum()
    acc     = correct / len(L)

    logger.info(f"  Val rank-1: {acc * 100:.2f}%  ({correct}/{len(L)})")
    return acc


# ──────────────────────────────────────────────────────────────────────────────
#  Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    ckpt_dir: str,
    state: FaceTrainState,
    epoch: int,
    metrics: dict,
    keep: int = 3,
):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=keep),
    )
    mngr.save(
        epoch,
        args=ocp.args.StandardSave(state),
    )
    mngr.wait_until_finished()

    import json
    mpath = Path(ckpt_dir) / f"metrics_epoch{epoch:03d}.json"
    with open(mpath, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


def restore_checkpoint(
    ckpt_dir: str,
    state: FaceTrainState,
) -> Tuple[FaceTrainState, int]:
    mngr = ocp.CheckpointManager(ckpt_dir)
    latest = mngr.latest_step()
    if latest is None:
        return state, 0
    state = mngr.restore(
        latest,
        args=ocp.args.StandardRestore(state),
    )
    return state, latest


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig = CFG):
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    devices = jax.devices()
    logger.info(f"JAX devices: {devices}")
    logger.info(f"JAX backend: {jax.default_backend()}")

    # ── dataset ──────────────────────────────────────────────────────────────
    logger.info("Loading CASIA-WebFace dataset from Hugging Face...")
    from datasets import load_dataset
    hf_ds = load_dataset("SaffalPoosh/casia_web_face", split="train")
    logger.info(f"  Total samples in dataset: {len(hf_ds):,}")

    logger.info("Splitting dataset into train/val by identities...")
    train_indices, val_indices = train_val_split_hf(hf_ds, cfg.val_fraction, seed=42)
    
    train_ds = hf_ds.select(train_indices)
    val_ds = hf_ds.select(val_indices)

    logger.info(
        f"  Train: {len(train_ds):,} samples "
        f"| Val: {len(val_ds):,} samples "
        f"({len(set(val_ds['label']))} identities)"
    )

    steps_per_epoch = len(train_ds) // cfg.batch_size
    total_steps     = steps_per_epoch * cfg.num_epochs
    logger.info(
        f"  Batch size: {cfg.batch_size} | Steps/epoch: {steps_per_epoch} "
        f"| Total steps: {total_steps:,}"
    )

    # ── init ─────────────────────────────────────────────────────────────────
    key   = jr.PRNGKey(0)
    state, model, loss_module = init_train_state(cfg, steps_per_epoch, key)

    # ── param count ──────────────────────────────────────────────────────────
    dummy_vars = {
        "params":      state.params["backbone"],
        "batch_stats": state.batch_stats,
    }
    n_params = count_params(dummy_vars, exclude_classifier=True)
    int8_mb  = estimate_int8_size_mb(dummy_vars, exclude_classifier=True)
    logger.info(f"Backbone deploy params: {n_params:,} ({int8_mb:.3f} MB int8)")
    assert int8_mb < 4.0, f"Model exceeds 4 MB budget: {int8_mb:.3f} MB"

    # ── maybe resume ─────────────────────────────────────────────────────────
    state, start_epoch = restore_checkpoint(cfg.checkpoint_dir, state)
    if start_epoch > 0:
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    # ── compiled steps ───────────────────────────────────────────────────────
    train_step = make_train_step(model, loss_module, cfg)

    # ── data generator ───────────────────────────────────────────────────────
    train_gen = batch_generator_hf(
        train_ds,
        batch_size  = cfg.batch_size,
        image_size  = cfg.image_size,
        augment     = True,
        shuffle     = True,
        seed        = 42,
    )

    # ── epoch loop ───────────────────────────────────────────────────────────
    global_step = start_epoch * steps_per_epoch
    best_val    = 0.0

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        rng = jr.PRNGKey(epoch)

        # ─ step loop ──────────────────────────────────────────────────────
        pbar = tqdm(
            range(steps_per_epoch),
            desc  = f"Epoch {epoch+1:02d}/{cfg.num_epochs}",
            leave = False,
        )
        for step in pbar:
            imgs, lbls = next(train_gen)
            rng, step_rng = jr.split(rng)

            state, metrics, _ = train_step(
                state,
                jnp.array(imgs),
                jnp.array(lbls),
                step_rng,
            )

            loss_val = float(metrics["loss"])
            epoch_losses.append(loss_val)
            global_step += 1

            if global_step % cfg.log_every == 0:
                logger.info(
                    f"Ep {epoch+1:02d} | Step {global_step:6d} "
                    f"| loss {loss_val:.4f} "
                    f"| margin {float(metrics['margin_loss']):.4f} "
                    f"| gnorm {float(metrics['grad_norm']):.3f}"
                )
                pbar.set_postfix(loss=f"{loss_val:.4f}")

        # ─ epoch summary ──────────────────────────────────────────────────
        epoch_time  = time.time() - epoch_start
        mean_loss   = float(np.mean(epoch_losses))
        logger.info(
            f"── Epoch {epoch+1:02d} done | "
            f"avg loss {mean_loss:.4f} | "
            f"time {epoch_time:.0f}s"
        )

        # ─ validation ─────────────────────────────────────────────────────
        if (epoch + 1) % cfg.val_every == 0:
            val_acc = evaluate_rank1(
                val_ds, state, model, cfg, logger
            )
            if val_acc > best_val:
                best_val = val_acc
                logger.info(f"  ★ New best val rank-1: {best_val * 100:.2f}%")

        # ─ checkpoint ─────────────────────────────────────────────────────
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_metrics = {
                "epoch":     epoch + 1,
                "mean_loss": mean_loss,
                "val_rank1": best_val,
            }
            save_checkpoint(
                cfg.checkpoint_dir,
                state,
                epoch + 1,
                ckpt_metrics,
                keep = cfg.keep_last_n,
            )
            logger.info(f"  Checkpoint saved → {cfg.checkpoint_dir}/epoch{epoch+1:03d}")

    logger.info(
        f"Training complete. "
        f"Best val rank-1: {best_val * 100:.2f}%"
    )
    return state


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris on CASIA-WebFace")
    parser.add_argument("--output_dir",     default=CFG.output_dir)
    parser.add_argument("--checkpoint_dir", default=CFG.checkpoint_dir)
    parser.add_argument("--loss_type",      default=CFG.loss_type,
                        choices=["arcface", "adaface"])
    parser.add_argument("--batch_size",     default=CFG.batch_size, type=int)
    parser.add_argument("--num_epochs",     default=CFG.num_epochs,  type=int)
    parser.add_argument("--base_lr",        default=CFG.base_lr,     type=float)
    parser.add_argument("--no_remat",       action="store_true",
                        help="Disable gradient checkpointing (more VRAM, faster)")
    args = parser.parse_args()

    cfg = TrainConfig(
        output_dir     = args.output_dir,
        checkpoint_dir = args.checkpoint_dir,
        loss_type      = args.loss_type,
        batch_size     = args.batch_size,
        num_epochs     = args.num_epochs,
        base_lr        = args.base_lr,
        use_remat      = not args.no_remat,
    )

    train(cfg)
