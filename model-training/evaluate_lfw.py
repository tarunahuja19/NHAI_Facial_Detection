import os
import sys
import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, auc

# Import local models and configuration
from model import FaceLiVTv2Lite
from losses import AdaFaceLoss
from train_digiface_cluster import FaceTrainState, TrainConfig

def preprocess_image(img, image_size=112):
    # img is shape (H, W, 3) in range [0, 1] or [0, 255]
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        
    # Resize to image_size x image_size
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        
    # Normalize to [-1, 1]
    _MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - _MEAN) / _STD
    
    # Transpose to (3, H, W)
    return img.transpose(2, 0, 1)

def main():
    checkpoint_dir = "./checkpoints_digiface_cluster"
    print("Initializing FaceLiVTv2-Lite model structure...")
    
    cfg = TrainConfig(
        num_classes=10000,
        embedding_dim=512,
        image_size=112,
        drop_path_rate=0.1,
        use_remat=True
    )
    
    # Initialize JAX model variables to get abstract shape matching checkpoint
    key = jr.PRNGKey(0)
    k1, k2 = jr.split(key)
    
    # We call train_digiface_cluster's init_train_state to recreate the exact TrainState structure (with optimizer)
    from train_digiface_cluster import init_train_state
    state, model, loss_module = init_train_state(
        cfg,
        steps_per_epoch=2671,
        global_batch_size=256,
        key=key
    )
    
    # Populate the template batch_stats with the correct structure from model.init(training=True)
    dummy = jr.normal(k1, (2, 3, cfg.image_size, cfg.image_size))
    vars_true = model.init({"params": k2, "droppath": k2}, dummy, training=True)
    state = state.replace(batch_stats=vars_true["batch_stats"])
    
    # Set up sharding on current devices to allow restoring a sharded checkpoint onto the current layout
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ('data',))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    state = jax.device_put(state, replicated_sharding)
    
    # Restore model checkpoint
    print(f"Restoring checkpoint from '{checkpoint_dir}'...")
    abs_ckpt_dir = os.path.abspath(checkpoint_dir)
    if not os.path.exists(abs_ckpt_dir):
        print(f"Error: Checkpoint directory '{abs_ckpt_dir}' does not exist. Did you unzip the zip file?", file=sys.stderr)
        sys.exit(1)
        
    mngr = ocp.CheckpointManager(abs_ckpt_dir)
    latest = mngr.latest_step()
    if latest is None:
        print(f"Error: No checkpoints found in '{abs_ckpt_dir}'.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Latest checkpoint found at step/epoch: {latest}")
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    restored = mngr.restore(
        latest,
        args=ocp.args.StandardRestore(abstract_state),
    )
    print("Checkpoint restored successfully!")
    
    restored_params = restored.params
    restored_batch_stats = restored.batch_stats
    
    # Setup jitted validation function
    @jax.jit
    def get_embeddings(images):
        backbone_vars = {
            "params": restored_params["backbone"],
            "batch_stats": restored_batch_stats,
        }
        embeddings, _ = model.apply(
            backbone_vars,
            images,
            training=False,
            rngs={"droppath": jr.PRNGKey(0)},
        )
        return embeddings

    # Load LFW pairs dataset
    print("Loading LFW pairs dataset (10_folds subset)...")
    lfw_10folds = fetch_lfw_pairs(subset='10_folds', color=True)
    pairs = lfw_10folds.pairs          # Shape: (6000, 2, H, W, 3)
    targets = lfw_10folds.target        # Shape: (6000,)
    print(f"Successfully loaded {len(pairs)} pairs.")
    
    # Extract embeddings for all images in LFW pairs
    print("Extracting embeddings for LFW images...")
    n_pairs = len(pairs)
    batch_size = 64
    
    all_emb1 = []
    all_emb2 = []
    
    for i in tqdm(range(0, n_pairs, batch_size), desc="Computing embeddings"):
        batch_pairs = pairs[i : i + batch_size]
        
        # Preprocess images
        imgs1 = np.stack([preprocess_image(pair[0], cfg.image_size) for pair in batch_pairs], axis=0)
        imgs2 = np.stack([preprocess_image(pair[1], cfg.image_size) for pair in batch_pairs], axis=0)
        
        # Run JAX model
        emb1 = get_embeddings(jnp.array(imgs1))
        emb2 = get_embeddings(jnp.array(imgs2))
        
        all_emb1.append(np.array(emb1))
        all_emb2.append(np.array(emb2))
        
    emb1_all = np.concatenate(all_emb1, axis=0)
    emb2_all = np.concatenate(all_emb2, axis=0)
    
    # Calculate cosine similarity for all pairs
    # Since embeddings are L2-normalized, the dot product is exactly the cosine similarity
    print("Computing similarity scores...")
    similarities = np.sum(emb1_all * emb2_all, axis=1)
    
    # Perform 10-Fold Cross-Validation
    print("Running 10-fold cross validation...")
    fold_size = 600
    n_folds = 10
    
    test_accuracies = []
    best_thresholds = []
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        
        # Split into train/test similarity scores and targets
        test_sim = similarities[test_start:test_end]
        test_lbl = targets[test_start:test_end]
        
        train_sim = np.concatenate([similarities[:test_start], similarities[test_end:]], axis=0)
        train_lbl = np.concatenate([targets[:test_start], targets[test_end:]], axis=0)
        
        # Search for best threshold on training set of current fold
        thresholds = np.arange(-1.0, 1.0, 0.001)
        best_acc = 0.0
        best_thresh = 0.0
        
        for t in thresholds:
            acc = np.mean((train_sim >= t) == train_lbl)
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
                
        # Evaluate on test set of current fold
        test_acc = np.mean((test_sim >= best_thresh) == test_lbl)
        test_accuracies.append(test_acc)
        best_thresholds.append(best_thresh)
        print(f"  Fold {fold+1}/{n_folds}: threshold = {best_thresh:.4f}, test accuracy = {test_acc * 100:.2f}%")
        
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    mean_threshold = np.mean(best_thresholds)
    
    print("\n" + "="*40)
    print("LFW 10-Fold Validation Results:")
    print(f"Mean Accuracy : {mean_acc * 100:.3f}% ± {std_acc * 100:.3f}%")
    print(f"Mean Threshold: {mean_threshold:.4f}")
    print("="*40 + "\n")
    
    # Calculate ROC metrics on the entire set
    fpr, tpr, roc_thresholds = roc_curve(targets, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Equal Error Rate (EER)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = roc_thresholds[eer_idx]
    
    print(f"ROC Area Under Curve (AUC): {roc_auc:.5f}")
    print(f"Equal Error Rate (EER)    : {eer * 100:.2f}% (at threshold {eer_threshold:.4f})")
    
    # Save the numbers to a JSON file
    import json
    results = {
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_threshold": float(mean_threshold),
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "test_accuracies": [float(x) for x in test_accuracies],
        "best_thresholds": [float(x) for x in best_thresholds]
    }
    
    os.makedirs("./scratch", exist_ok=True)
    results_path = "./scratch/lfw_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved numerical results to '{results_path}'")
    
    # Create Plots
    print("Generating performance plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: ROC Curve
    ax1.plot(fpr, tpr, color='#1d3557', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='#e63946', lw=1.5, linestyle='--', label='Random Guess')
    ax1.plot([eer], [1 - eer], marker='o', markersize=8, color='#ffb703', label=f'EER = {eer * 100:.2f}%')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax1.set_title('LFW ROC Curve', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Cosine Similarity Distribution
    pos_sims = similarities[targets == 1]
    neg_sims = similarities[targets == 0]
    
    ax2.hist(pos_sims, bins=50, density=True, alpha=0.6, label='Same Person (Positive)', color='#2ec4b6', edgecolor='none')
    ax2.hist(neg_sims, bins=50, density=True, alpha=0.6, label='Different Person (Negative)', color='#e71d36', edgecolor='none')
    ax2.axvline(mean_threshold, color='#1d3557', linestyle='--', lw=2, label=f'Threshold = {mean_threshold:.3f}')
    ax2.set_xlabel('Cosine Similarity', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('LFW Pair Similarity Distribution', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Define artifact path to save the plot directly
    artifact_dir = "/home/jemin/.gemini/antigravity-ide/brain/ac2097b2-c9d5-427c-acfe-c9669e68f5a6"
    plot_path = os.path.join(artifact_dir, "lfw_metrics_plots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved ROC and distribution plots to '{plot_path}'")

if __name__ == "__main__":
    main()
