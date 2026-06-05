"""
train_digiface_cluster.py  --  FaceLiVTv2-Lite on DigiFace-1M (GPU Cluster/Multi-Host Optimized)
==============================================================================================

This script is optimized for training on GPU clusters (e.g., SLURM, multi-node, or multi-GPU environments).
It uses the DigiFace-1M synthetic dataset, with support for automatic downloading and extraction of parts 
P1-P5 (720K subset, 10K identities with 72 images per identity), sharded local filesystem data loading,
and robust multi-host checkpointing via Orbax.

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
import urllib.request
import zipfile
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
from model import FaceLiVTv2Lite, count_params, estimate_int8_size_mb
from losses import AdaFaceLoss, ArcFaceLoss

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class TrainConfig:
    # ── paths ──
    output_dir:       str   = "./output_digiface_cluster"
    checkpoint_dir:   str   = "./checkpoints_digiface_cluster"
    log_file:         str   = "./train_digiface_cluster.log"

    # ── dataset ──
    image_dir:        str   = "./digiface_data" # Folder containing DigiFace-1M images
    download_data:    bool  = False             # Whether to download dataset if not present
    image_size:       int   = 112
    val_fraction:     float = 0.05      # fraction of identities held out for val
    num_workers:      int   = 4
    prefetch:         int   = 2

    # ── model ──
    num_classes:      int   = 10000     # Default DigiFace-1M total identities (overwritten dynamically)
    embedding_dim:    int   = 512
    drop_path_rate:   float = 0.1
    use_remat:        bool  = True       # gradient checkpointing — saves VRAM

    # ── loss ──
    loss_type:        str   = "adaface"  # "arcface" | "adaface"
    arc_scale:        float = 64.0
    arc_margin:       float = 0.4        # AdaFace default
    ada_h:            float = 0.333

    # ── optimiser ──
    batch_size_per_gpu: int = 384        # Local batch size per GPU/device
    num_epochs:       int   = 10         # DigiFace-1M standard training epochs
    warmup_epochs:    int   = 1
    base_lr:          float = 0.03       # SGD with momentum base learning rate
    momentum:         float = 0.9
    weight_decay:     float = 5e-4
    # LR milestones (epoch numbers at which to multiply lr by 0.1)
    lr_milestones:    Tuple = (4, 7, 9)
    scale_lr:         bool  = True       # Auto-scale LR linearly with global batch size

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
    logger = logging.getLogger("train_digiface")
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    logger.propagate = False

    # Avoid duplicate handlers when rerunning.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    if jax.process_index() == 0:
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                datefmt="%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


# ──────────────────────────────────────────────────────────────────────────────
#  DigiFace-1M Auto-Downloader
# ──────────────────────────────────────────────────────────────────────────────

def download_and_extract_digiface(download_dir: str, logger: logging.Logger):
    """Downloads and extracts parts P1-P5 of DigiFace-1M (720K subset)."""
    urls = [
        "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip",
        "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_2000-3999_72_imgs.zip",
        "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_4000-5999_72_imgs.zip",
        "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_6000-7999_72_imgs.zip",
        "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_8000-9999_72_imgs.zip"
    ]
    
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    # Process 0 downloads and extracts; other processes block at the barrier
    if jax.process_index() == 0:
        for idx, url in enumerate(urls):
            zip_name = url.split('/')[-1]
            zip_path = os.path.join(download_dir, zip_name)
            
            # Check if this part has already been extracted
            start_subject = idx * 2000
            check_folder = os.path.join(download_dir, str(start_subject))
            if os.path.exists(check_folder):
                logger.info(f"Part P{idx+1} ({zip_name}) already extracted. Skipping download.")
                continue
                
            if not os.path.exists(zip_path):
                logger.info(f"Downloading P{idx+1}/P5 from {url}...")
                start_time = time.time()
                urllib.request.urlretrieve(url, zip_path)
                logger.info(f"Downloaded P{idx+1} in {time.time() - start_time:.1f}s.")
            
            logger.info(f"Extracting P{idx+1} to {download_dir}...")
            start_time = time.time()
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            logger.info(f"Extracted P{idx+1} in {time.time() - start_time:.1f}s.")
            
            # Clean up the zip file to conserve disk space
            try:
                os.remove(zip_path)
            except OSError:
                pass
                
    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu
        logger.info("Waiting for data download to synchronize across processes...")
        mhu.barrier()


# ──────────────────────────────────────────────────────────────────────────────
#  Local file scanner
# ──────────────────────────────────────────────────────────────────────────────

def find_digiface_images(
    data_dir: str,
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], int]:
    """Scans DigiFace directory for subject subfolders and splits them into train/val."""
    subj_dirs = []
    for entry in os.scandir(data_dir):
        if entry.is_dir() and entry.name.isdigit():
            subj_dirs.append(entry.name)
            
    subj_dirs = sorted(subj_dirs, key=int)
    num_classes = len(subj_dirs)
    if num_classes == 0:
        raise ValueError(f"No subject ID directories (integers) found in {data_dir}. Please verify dataset extraction.")
        
    subj_to_label = {s: i for i, s in enumerate(subj_dirs)}
    
    rng = np.random.default_rng(seed)
    n_val = max(1, int(num_classes * val_fraction))
    val_subjs = set(rng.choice(subj_dirs, size=n_val, replace=False))
    
    train_items = []
    val_items = []
    
    for subj in subj_dirs:
        label = subj_to_label[subj]
        subj_path = os.path.join(data_dir, subj)
        
        img_names = []
        for img_entry in os.scandir(subj_path):
            if img_entry.is_file() and img_entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_names.append(img_entry.name)
                
        img_names = sorted(img_names)
        
        # User requested: "use only 72 images per identity"
        img_names = img_names[:72]
        
        for name in img_names:
            full_path = os.path.join(subj_path, name)
            if subj in val_subjs:
                val_items.append((full_path, label))
            else:
                train_items.append((full_path, label))
                
    return train_items, val_items, num_classes


# ──────────────────────────────────────────────────────────────────────────────
#  Data generator
# ──────────────────────────────────────────────────────────────────────────────

def batch_generator_local(
    items: List[Tuple[str, int]],
    batch_size: int,
    image_size: int,
    augment: bool,
    shuffle: bool,
    seed: int = 0,
    drop_last: bool = True,
):
    """Generator that yields (images, labels) numpy arrays from local image paths."""
    indices = list(range(len(items)))
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
            for idx in batch_idx:
                try:
                    img_path, label = items[idx]
                    
                    pil_img = Image.open(img_path)
                    img = np.array(pil_img)
                    if img.ndim == 2:  # grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:  # RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    
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
    global_batch_size: int,
    key: jnp.ndarray,
) -> Tuple[FaceTrainState, FaceLiVTv2Lite, nn.Module]:
    k1, k2, k3 = jr.split(key, 3)

    # ── backbone ──
    model = FaceLiVTv2Lite(
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
    
    # Scale learning rate if configured (linear scaling w.r.t default batch size 128)
    lr = cfg.base_lr
    if cfg.scale_lr:
        lr = cfg.base_lr * (global_batch_size / 128.0)

    schedule         = make_lr_schedule(lr, warmup_steps, milestone_steps)

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
    val_items: List[Tuple[str, int]],
    state: FaceTrainState,
    model,
    cfg: TrainConfig,
    logger: logging.Logger,
) -> float:
    if len(val_items) == 0:
        return 0.0

    val_step = make_val_step(model)

    all_embs   = []
    all_labels = []

    eval_bs = min(64, len(val_items))
    # Note: Validation set is evaluated completely on each host to ensure multi-host agreement
    gen = batch_generator_local(
        val_items,
        batch_size=eval_bs,
        image_size=cfg.image_size,
        augment=False,
        shuffle=False,
        drop_last=False,
    )
    n_batches = math.ceil(len(val_items) / eval_bs)
    for _ in range(n_batches):
        imgs, lbls = next(gen)
        embs = val_step(state, jnp.array(imgs))
        all_embs.append(np.array(embs))
        all_labels.extend(lbls.tolist())

    E = np.concatenate(all_embs, axis=0)[:len(val_items)]  # trim padding
    L = np.array(all_labels[:len(val_items)])

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
    # Orbax checkpoint save needs to be run on all hosts in multi-host setups
    abs_ckpt_dir = os.path.abspath(ckpt_dir)
    mngr = ocp.CheckpointManager(
        abs_ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=keep),
    )
    mngr.save(
        epoch,
        args=ocp.args.StandardSave(state),
    )
    mngr.wait_until_finished()

    if jax.process_index() == 0:
        import json
        mpath = Path(abs_ckpt_dir) / f"metrics_epoch{epoch:03d}.json"
        with open(mpath, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


def restore_checkpoint(
    ckpt_dir: str,
    state: FaceTrainState,
) -> Tuple[FaceTrainState, int]:
    abs_ckpt_dir = os.path.abspath(ckpt_dir)
    mngr = ocp.CheckpointManager(abs_ckpt_dir)
    latest = mngr.latest_step()
    if latest is None:
        return state, 0
        
    # Orbax restore needs to run on all hosts
    # To handle multi-host restores correctly, we provide the abstract pytree shape/dtypes
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    state = mngr.restore(
        latest,
        args=ocp.args.PyTreeRestore(item=abstract_state, partial_restore=True),
    )
    return state, latest


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig = CFG):
    # ── JAX Distributed Initialization ────────────────────────────────────────
    # Try initializing distributed training if running in multi-host environment
    if jax.process_count() > 1 or "MASTER_ADDR" in os.environ or "SLURM_PROCID" in os.environ:
        try:
            jax.distributed.initialize()
        except Exception as e:
            # Print to stdout directly since logger is not setup yet
            print(f"Warning: Failed to initialize JAX distributed: {e}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    # ── Auto-Download DigiFace-1M ─────────────────────────────────────────────
    # If folder doesn't exist or is empty, download the dataset
    is_empty = False
    if os.path.exists(cfg.image_dir):
        # check if it contains any subdirectory
        subdirs = [n for n in os.listdir(cfg.image_dir) if os.path.isdir(os.path.join(cfg.image_dir, n))]
        if len(subdirs) == 0:
            is_empty = True
    else:
        is_empty = True

    if cfg.download_data or is_empty:
        logger.info(f"DigiFace-1M images not found in {cfg.image_dir} or --download_data is set. Downloading P1-P5...")
        download_and_extract_digiface(cfg.image_dir, logger)

    # ── Device & Cluster Info ──────────────────────────────────────────────────
    global_devices = jax.devices()
    num_global_devices = len(global_devices)
    num_local_devices = jax.local_device_count()
    
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"Process count: {jax.process_count()} (current rank: {jax.process_index()})")
    logger.info(f"Global JAX devices count: {num_global_devices}")
    logger.info(f"Local JAX devices count: {num_local_devices}")
    
    # Establish a 1D parallel mesh along the 'data' dimension for Data Parallelism
    logger.info(f"Initialising GPU 1D Mesh of size {num_global_devices} for multi-GPU training...")
    mesh = jax.sharding.Mesh(global_devices, ('data',))
    
    # Partition specs:
    # Model parameters replicated across all GPUs
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Images sharded along the batch axis (dim 0), replicated along other axes
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None, None, None))
    # Labels sharded along the batch axis
    label_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))

    # ── Batch Size calculation ────────────────────────────────────────────────
    global_batch_size = cfg.batch_size_per_gpu * num_global_devices
    local_batch_size = cfg.batch_size_per_gpu * num_local_devices
    logger.info(
        f"Batch Size configuration:\n"
        f"  - Batch size per GPU: {cfg.batch_size_per_gpu}\n"
        f"  - Global batch size (across all processes): {global_batch_size}\n"
        f"  - Local batch size (this host): {local_batch_size}"
    )

    # ── dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Scanning local DigiFace-1M dataset in {cfg.image_dir}...")
    train_items, val_items, num_classes = find_digiface_images(cfg.image_dir, cfg.val_fraction, seed=42)
    
    # Dynamically overwrite the number of classes from dataset
    cfg = cfg.replace(num_classes=num_classes)
    logger.info(f"  Detected {cfg.num_classes} unique identities in dataset.")
    logger.info(f"  Total Train samples: {len(train_items):,} | Total Val samples: {len(val_items):,}")

    # Shard the local training items list across global processes (hosts)
    if jax.process_count() > 1:
        logger.info(f"Sharding training list: process {jax.process_index()+1}/{jax.process_count()}...")
        # Slice index::process_count
        train_items = train_items[jax.process_index()::jax.process_count()]
        logger.info(f"  Local process training list size: {len(train_items):,} samples")

    steps_per_epoch = len(train_items) // local_batch_size
    total_steps     = steps_per_epoch * cfg.num_epochs
    logger.info(
        f"  Local batch size: {local_batch_size} "
        f"| Steps/epoch (this process): {steps_per_epoch} "
        f"| Total steps: {total_steps:,}"
    )

    # ── init ─────────────────────────────────────────────────────────────────
    key   = jr.PRNGKey(0)
    state, model, loss_module = init_train_state(cfg, steps_per_epoch, global_batch_size, key)

    # Replicate/Shard parameters & optimizer state to the GPU mesh
    state = jax.device_put(state, replicated_sharding)

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
    # We use a unique seed per rank so each host shuffles/augments differently
    local_seed = 42 + jax.process_index()
    train_gen = batch_generator_local(
        train_items,
        batch_size   = local_batch_size,
        image_size   = cfg.image_size,
        augment      = True,
        shuffle      = True,
        seed         = local_seed,
    )

    # ── epoch loop ───────────────────────────────────────────────────────────
    global_step = start_epoch * steps_per_epoch
    best_val    = 0.0

    import jax.experimental.multihost_utils as mhu

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        rng = jr.PRNGKey(epoch)

        # ─ step loop ──────────────────────────────────────────────────────
        if jax.process_index() == 0:
            pbar = tqdm(
                range(steps_per_epoch),
                desc  = f"Epoch {epoch+1:02d}/{cfg.num_epochs}",
                leave = False,
            )
        else:
            pbar = range(steps_per_epoch)

        for step in pbar:
            imgs, lbls = next(train_gen)
            rng, step_rng = jr.split(rng)

            # Use multihost_utils to distribute our host-local arrays into globally sharded jax.Arrays
            imgs_sharded = mhu.host_local_array_to_global_array(
                jnp.array(imgs), mesh, jax.sharding.PartitionSpec('data', None, None, None)
            )
            lbls_sharded = mhu.host_local_array_to_global_array(
                jnp.array(lbls), mesh, jax.sharding.PartitionSpec('data')
            )
            step_rng_replicated = jax.device_put(step_rng, replicated_sharding)

            state, metrics, _ = train_step(
                state,
                imgs_sharded,
                lbls_sharded,
                step_rng_replicated,
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
                if jax.process_index() == 0:
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
                val_items, state, model, cfg, logger
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
    parser = argparse.ArgumentParser(description="Train FaceLiVTv2-Lite on DigiFace-1M (GPU Cluster optimized)")
    parser.add_argument("--image_dir",      default=CFG.image_dir,
                        help="Path to local folder containing extracted DigiFace-1M images")
    parser.add_argument("--download_data",   action="store_true",
                        help="Download DigiFace-1M dataset P1-P5 to --image_dir if not present")
    parser.add_argument("--output_dir",     default=CFG.output_dir)
    parser.add_argument("--checkpoint_dir", default=CFG.checkpoint_dir)
    parser.add_argument("--loss_type",      default=CFG.loss_type,
                        choices=["arcface", "adaface"])
    parser.add_argument("--batch_size_per_gpu", default=CFG.batch_size_per_gpu, type=int)
    parser.add_argument("--num_epochs",     default=CFG.num_epochs,  type=int)
    parser.add_argument("--base_lr",        default=CFG.base_lr,     type=float)
    parser.add_argument("--scale_lr",       action="store_true", default=True,
                        help="Auto-scale LR linearly with global batch size (global_bs / 128)")
    parser.add_argument("--no_scale_lr",    action="store_false", dest="scale_lr",
                        help="Disable automatic learning rate scaling")
    parser.add_argument("--no_remat",       action="store_true",
                        help="Disable gradient checkpointing (more VRAM, faster)")
    args = parser.parse_args()

    cfg = TrainConfig(
        image_dir          = args.image_dir,
        download_data      = args.download_data,
        output_dir         = args.output_dir,
        checkpoint_dir     = args.checkpoint_dir,
        loss_type          = args.loss_type,
        batch_size_per_gpu = args.batch_size_per_gpu,
        num_epochs         = args.num_epochs,
        base_lr            = args.base_lr,
        scale_lr           = args.scale_lr,
        use_remat          = not args.no_remat,
    )

    train(cfg)
