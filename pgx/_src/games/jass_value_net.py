"""
Value network for Jass.

Architecture: per-card MLP (weights shared across all 36 rows) → mean-pool
→ concat header → dense head → scalar differential in [-157, 157].

The per-card trunk is permutation-invariant over cards and naturally extends to
a per-card policy head (one logit per card) without rearchitecting, making the
transition to a joint value+policy network straightforward.

Usage:
    model = ValueNet()
    params = model.init(key, jnp.zeros((B, 36, 12)), jnp.zeros((B, 20)))
    pred = model.apply(params, cm, hd)   # (B,) predicted differential / scale
"""

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from pgx._src.games.jass_selfplay import collect_batch

# Scale target into roughly [-1, 1] for stable training.
# The network outputs pred ≈ differential / SCALE; multiply back at inference.
TARGET_SCALE = 100.0


class ValueNet(nn.Module):
    hidden: int = 128

    @nn.compact
    def __call__(self, cm: jnp.ndarray, hd: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            cm: (B, 36, 12) bool  card matrix
            hd: (B, 20)     bool  header

        Returns:
            (B,) float32  predicted differential / TARGET_SCALE
        """
        x = cm.astype(jnp.float32)          # (B, 36, 12)

        # Per-card MLP — Dense applies independently to each of the 36 rows
        # because the last axis is the feature axis.
        x = nn.Dense(self.hidden)(x)         # (B, 36, hidden)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden)(x)         # (B, 36, hidden)
        x = nn.gelu(x)

        x = x.mean(axis=-2)                  # (B, hidden)  — pool over cards

        h = hd.astype(jnp.float32)           # (B, 20)
        x = jnp.concatenate([x, h], axis=-1) # (B, hidden + 20)

        x = nn.Dense(self.hidden)(x)
        x = nn.gelu(x)
        x = nn.Dense(1)(x).squeeze(-1)       # (B,)
        return x


def make_train_step(model: ValueNet, optimizer: optax.GradientTransformation):
    """Return a jit-compiled training step function."""

    @jax.jit
    def train_step(params, opt_state, cm, hd, y, mask):
        """
        Args:
            params, opt_state: model + optimiser state
            cm   : (N, 36, 12) bool
            hd   : (N, 20)     bool
            y    : (N,)        float32  raw differential targets
            mask : (N,)        float32  1.0 for alive steps, 0.0 for padding

        Returns:
            updated params, opt_state, scalar loss
        """
        def loss_fn(p):
            pred = model.apply(p, cm, hd)                 # (N,)
            sq   = (pred - y / TARGET_SCALE) ** 2         # (N,)
            return (sq * mask).sum() / mask.sum().clip(1)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    return train_step


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    *,
    batch_size: int = 4096,
    num_epochs: int = 200,
    lr: float = 3e-4,
    print_every: int = 10,
    seed: int = 0,
) -> tuple:
    """Train a ValueNet from scratch on random self-play.

    Each epoch generates a fresh batch of random games as training data.
    A fixed holdout set (same size) is collected once up front for eval.

    Args:
        batch_size: Number of games per training batch and holdout set.
        num_epochs: Total training epochs.
        lr: Adam learning rate.
        print_every: Print train/eval loss every N epochs.
        seed: PRNG seed for reproducibility.

    Returns:
        (params, model) — trained Flax parameters and the ValueNet instance.
    """
    key = jax.random.PRNGKey(seed)

    model = ValueNet()
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    step_fn = make_train_step(model, optimizer)

    print("Collecting holdout batch for eval ...")
    key, k_eval = jax.random.split(key)
    cm_eval, hd_eval, y_eval, alive_eval = collect_batch(k_eval, batch_size)
    cm_eval = cm_eval.reshape(-1, 36, 12)
    hd_eval = hd_eval.reshape(-1, 20)
    y_eval = y_eval.reshape(-1)
    mask_eval = alive_eval.reshape(-1).astype(jnp.float32)
    print(f"  {int(mask_eval.sum())} labeled positions\n")

    t0 = time.perf_counter()
    for epoch in range(num_epochs):
        key, k1 = jax.random.split(key)
        cm, hd, y, alive = collect_batch(k1, batch_size)

        cm = cm.reshape(-1, 36, 12)
        hd = hd.reshape(-1, 20)
        y = y.reshape(-1)
        mask = alive.reshape(-1).astype(jnp.float32)

        params, opt_state, train_loss = step_fn(params, opt_state, cm, hd, y, mask)

        if epoch % print_every == 0:
            _, _, eval_loss = step_fn(params, opt_state, cm_eval, hd_eval, y_eval, mask_eval)
            elapsed = time.perf_counter() - t0
            print(f"[{epoch:4d}]  train={float(train_loss):.4f}"
                  f"  eval={float(eval_loss):.4f}"
                  f"  ({elapsed:.0f}s)")

    return params, model


# ── CLI Driver ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import flax.serialization

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 1 epoch, batch=64")
    parser.add_argument("--save", default="jass_v_weights.msgpack",
                        help="Path to write final weights")
    args = parser.parse_args()

    params, model = train_model(
        batch_size=64 if args.smoke else 4096,
        num_epochs=1 if args.smoke else 200,
        print_every=1 if args.smoke else 10,
    )

    with open(args.save, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"\nWeights saved to {args.save}")
