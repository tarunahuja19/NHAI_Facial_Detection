"""
ArcFace & AdaFace losses  --  JAX/Flax implementation
=====================================================
Margin-based softmax losses for face recognition training.

ArcFace (Deng et al., 2019)
    Adds a fixed angular margin ``m`` to the target class angle, pushing
    embeddings apart on the unit hypersphere.

AdaFace (Kim et al., 2022)
    Extends ArcFace with adaptive margins.  Uses the embedding norm (before
    L2 normalisation) as an image-quality proxy:
      - High-quality (large norm)  ->  larger angular margin   (push apart)
      - Low-quality  (small norm)  ->  smaller angular margin  (don't harm)

Both losses operate on L2-normalised embeddings and a learnable L2-normalised
weight matrix (the "class centres" on the hypersphere).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


__all__ = ["ArcFaceLoss", "AdaFaceLoss"]


# ─────────────────────────────────────────────────────────────
#  ArcFace Loss
# ─────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """ArcFace angular-margin softmax loss.

    Args:
        num_classes: Number of identity classes.
        embedding_dim: Dimensionality of input embeddings (must be L2-normalised).
        scale: Scaling factor ``s`` (temperature).  Default 64.0.
        margin: Angular margin ``m`` in radians.  Default 0.5.
        label_smoothing: Smoothing factor for cross-entropy. Default 0.0.
    """
    num_classes  : int
    embedding_dim: int
    scale        : float = 64.0
    margin       : float = 0.5
    label_smoothing: float = 0.0

    @nn.compact
    def __call__(self, embeddings, labels):
        """Compute ArcFace loss.

        Args:
            embeddings: L2-normalised embeddings, shape ``(B, embedding_dim)``.
            labels: Integer class labels, shape ``(B,)``.

        Returns:
            Scalar loss value (mean cross-entropy over the batch).
        """
        # Learnable class centres on the hypersphere
        W = self.param(
            "class_centres",
            nn.initializers.normal(0.01),
            (self.num_classes, self.embedding_dim),
        )
        # L2-normalise weights
        W_norm = W / jnp.maximum(jnp.linalg.norm(W, axis=1, keepdims=True), 1e-6)

        # Cosine similarity: (B, num_classes)
        cos_theta = jnp.dot(embeddings, W_norm.T)
        cos_theta = jnp.clip(cos_theta, -1.0 + 1e-5, 1.0 - 1e-5)

        # Add angular margin to target class (clamped to prevent wrap-around)
        theta = jnp.arccos(cos_theta)
        one_hot = jax.nn.one_hot(labels, self.num_classes)
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes

        theta_m = jnp.clip(theta + self.margin, 1e-5, jnp.pi - 1e-5)
        target_cos = jnp.cos(theta_m)

        # Scale and compute cross-entropy
        logits = cos_theta * (1.0 - one_hot) + target_cos * one_hot
        logits = logits * self.scale
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        return jnp.mean(loss)


# ─────────────────────────────────────────────────────────────
#  AdaFace Loss
# ─────────────────────────────────────────────────────────────

class AdaFaceLoss(nn.Module):
    """AdaFace adaptive-margin softmax loss.

    Extends ArcFace with image-quality-adaptive margins.  The embedding norm
    (before L2 normalisation) is used as a proxy for image quality:

    - **High-quality** samples (large norm) receive the full angular margin,
      pushing them further from decision boundaries.
    - **Low-quality** samples (small norm) receive a reduced margin, preventing
      noisy gradients from corrupting the embedding space.

    Args:
        num_classes: Number of identity classes.
        embedding_dim: Dimensionality of input embeddings.
        scale: Scaling factor ``s``.  Default 64.0.
        margin: Base angular margin ``m``.  Default 0.4.
        h: Concentration parameter controlling margin adaptation strength.
           Default 0.333.
        label_smoothing: Smoothing factor for cross-entropy. Default 0.0.
    """
    num_classes  : int
    embedding_dim: int
    scale        : float = 64.0
    margin       : float = 0.4
    h            : float = 0.333
    label_smoothing: float = 0.0

    @nn.compact
    def __call__(self, embeddings, labels, embedding_norms=None):
        """Compute AdaFace loss.

        Args:
            embeddings: L2-normalised embeddings, shape ``(B, embedding_dim)``.
            labels: Integer class labels, shape ``(B,)``.
            embedding_norms: Pre-normalisation norms, shape ``(B,)``.

        Returns:
            Scalar loss value (mean cross-entropy over the batch).
        """
        # Learnable class centres
        W = self.param(
            "class_centres",
            nn.initializers.normal(0.01),
            (self.num_classes, self.embedding_dim),
        )
        W_norm = W / jnp.maximum(jnp.linalg.norm(W, axis=1, keepdims=True), 1e-6)

        # Cosine similarity
        cos_theta = jnp.dot(embeddings, W_norm.T)
        cos_theta = jnp.clip(cos_theta, -1.0 + 1e-5, 1.0 - 1e-5)

        one_hot = jax.nn.one_hot(labels, self.num_classes)
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes

        # Compute adaptive margin per sample
        if embedding_norms is not None:
            # Stop gradient to prevent loss scaling signals from feeding back
            safe_norms = jax.lax.stop_gradient(jnp.clip(embedding_norms, 0.001, 100.0))

            mean = jnp.mean(safe_norms)
            std = jnp.std(safe_norms)

            # Normalize norms: (B,)
            margin_scaler = (safe_norms - mean) / (std + 1e-3)
            margin_scaler = margin_scaler * self.h
            margin_scaler = jnp.clip(margin_scaler, -1.0, 1.0)
            margin_scaler = margin_scaler[:, None]  # (B, 1)

            # g_angular: -m * margin_scaler
            g_angular = -self.margin * margin_scaler

            # Add angular margin to target class (clamped to prevent wrap-around)
            theta = jnp.arccos(cos_theta)
            theta_m = jnp.clip(theta - self.margin * margin_scaler, 1e-5, jnp.pi - 1e-5)
            cos_theta_m = jnp.cos(theta_m)

            # Apply additive margin to target class
            g_add = self.margin + (self.margin * margin_scaler)
            target_logit = cos_theta_m - g_add

            logits = cos_theta * (1.0 - one_hot) + target_logit * one_hot
        else:
            # Degrade to standard ArcFace if norms are not provided
            theta = jnp.arccos(cos_theta)
            theta_m = jnp.clip(theta + self.margin, 1e-5, jnp.pi - 1e-5)
            target_cos = jnp.cos(theta_m)
            logits = cos_theta * (1.0 - one_hot) + target_cos * one_hot

        # Scale and cross-entropy
        logits = logits * self.scale
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        return jnp.mean(loss)
