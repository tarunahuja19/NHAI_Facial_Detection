import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
from jax2onnx import to_onnx

from model import Iris
from train_digiface_cluster import TrainConfig, init_train_state

def main():
    checkpoint_dir = "./checkpoints_digiface_cluster"
    print("Initializing model...")
    cfg = TrainConfig(
        num_classes=10000,
        embedding_dim=512,
        image_size=112,
        drop_path_rate=0.1,
        use_remat=True
    )
    
    key = jr.PRNGKey(0)
    k1, k2 = jr.split(key)
    
    state, model, loss_module = init_train_state(
        cfg,
        steps_per_epoch=2671,
        global_batch_size=256,
        key=key
    )
    
    dummy = jr.normal(k1, (2, 3, cfg.image_size, cfg.image_size))
    vars_true = model.init({"params": k2, "droppath": k2}, dummy, training=True)
    state = state.replace(batch_stats=vars_true["batch_stats"])
    
    # Restore model checkpoint
    print(f"Restoring checkpoint from '{checkpoint_dir}'...")
    abs_ckpt_dir = os.path.abspath(checkpoint_dir)
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
    
    # Define the forward function to export using the restored model structure
    def predict_fn(images):
        backbone_vars = {
            "params": restored_params["backbone"],
            "batch_stats": restored_batch_stats,
        }
        # Run inference
        embeddings, _ = model.apply(
            backbone_vars,
            images,
            training=False,
            rngs={"droppath": jr.PRNGKey(0)},
        )
        return embeddings
        
    print("Tracing and exporting model to ONNX...")
    os.makedirs("./scratch", exist_ok=True)
    output_path = "./scratch/face_model.onnx"
    
    to_onnx(
        predict_fn,
        inputs=[(1, 3, 112, 112)], # Fixed batch size of 1 for mobile deployment
        model_name="Iris",
        return_mode="file",
        output_path=output_path,
        input_names=["input_image"],
        output_names=["embeddings"]
    )
    print(f"Export completed successfully! Saved to '{output_path}'")

if __name__ == "__main__":
    main()
