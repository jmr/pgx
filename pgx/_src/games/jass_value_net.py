"""
Value and policy+value networks for Jass.

ValueNet: per-card MLP (weights shared across all 36 rows) → mean-pool
→ concat header → dense head → scalar differential in [-157, 157].

PolicyValueNet (docs/jass_plan.md Step 2): same per-card trunk, plus
- card logits (36): Dense(1) on each card row before pooling;
- trump logits (7): actions 36–42, from pooled features + header;
- value head structurally identical to ValueNet's.

Usage:
    model = ValueNet()
    params = model.init(key, jnp.zeros((B, 36, 12)), jnp.zeros((B, 20)))
    pred = model.apply(params, cm, hd)   # (B,) predicted differential / scale

    pv = PolicyValueNet()
    params = pv.init(key, jnp.zeros((B, 36, 12)), jnp.zeros((B, 20)))
    logits, value = pv.apply(params, cm, hd)   # (B, 43), (B,)
"""

import time

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import optax

from pgx._src.games.jass import NUM_ACTIONS
from pgx._src.games.jass_selfplay import collect_batch, collect_pv_batch

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


class PolicyValueNet(nn.Module):
    """Joint policy+value network over the full-information value features.

    The trunk and the value path mirror ValueNet exactly (same shapes, same
    layer structure), so a PolicyValueNet trained to convergence should be at
    least as good a leaf evaluator as a ValueNet. The policy covers all 43
    actions: card plays 0–35 (one logit per card row, computed before
    pooling so each card's logit sees that card's own features) and trump
    declarations 36–42 (from the pooled summary + header, where the hand
    composition and Schiebe context live).

    Logits are unmasked — mask with the legal_action_mask at the loss and
    at sampling time.
    """
    hidden: int = 128

    @nn.compact
    def __call__(self, cm: jnp.ndarray, hd: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            cm: (B, 36, 12) bool  card matrix
            hd: (B, 20)     bool  header

        Returns:
            logits: (B, 43) float32  unmasked action logits
            value : (B,)    float32  predicted differential / TARGET_SCALE
        """
        x = cm.astype(jnp.float32)           # (B, 36, 12)

        x = nn.Dense(self.hidden)(x)          # (B, 36, hidden)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden)(x)          # (B, 36, hidden)
        x = nn.gelu(x)

        card_logits = nn.Dense(1)(x).squeeze(-1)   # (B, 36)

        pooled = x.mean(axis=-2)               # (B, hidden)
        h = hd.astype(jnp.float32)             # (B, 20)
        y = jnp.concatenate([pooled, h], axis=-1)  # (B, hidden + 20)

        v = nn.Dense(self.hidden)(y)
        v = nn.gelu(v)
        value = nn.Dense(1)(v).squeeze(-1)     # (B,)

        t = nn.Dense(self.hidden)(y)
        t = nn.gelu(t)
        trump_logits = nn.Dense(NUM_ACTIONS - 36)(t)  # (B, 7)

        logits = jnp.concatenate([card_logits, trump_logits], axis=-1)  # (B, 43)
        return logits, value


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


def make_pv_train_step(model: PolicyValueNet,
                       optimizer: optax.GradientTransformation,
                       policy_weight: float = 1.0):
    """Return a jit-compiled training step for the joint policy+value net."""

    @jax.jit
    def train_step(params, opt_state, cm, hd, y, pi, legal, mask):
        """
        Args:
            params, opt_state: model + optimiser state
            cm   : (N, 36, 12) bool
            hd   : (N, 20)     bool
            y    : (N,)        float32  raw differential targets
            pi   : (N, 43)     float32  policy targets (one-hot or visit
                   distribution); must be zero on illegal actions
            legal: (N, 43)     bool     legal_action_mask per step
            mask : (N,)        float32  1.0 for alive steps, 0.0 for padding

        Returns:
            updated params, opt_state, (total, value, policy) losses.
            Policy cross-entropy is computed over legal actions only
            (illegal logits forced to -1e9 before log_softmax).
        """
        def loss_fn(p):
            logits, v = model.apply(p, cm, hd)            # (N, 43), (N,)
            v_sq = (v - y / TARGET_SCALE) ** 2            # (N,)
            masked_logits = jnp.where(legal, logits, jnp.float32(-1e9))
            logp = jax.nn.log_softmax(masked_logits, axis=-1)
            ce = -(pi * logp).sum(axis=-1)                # (N,)
            denom  = mask.sum().clip(1)
            v_loss = (v_sq * mask).sum() / denom
            p_loss = (ce * mask).sum() / denom
            return v_loss + policy_weight * p_loss, (v_loss, p_loss)

        (loss, (v_loss, p_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state,
                loss, v_loss, p_loss)

    return train_step


# ── Training ──────────────────────────────────────────────────────────────────


# Checkpoint I/O uses only open(): some environments (e.g. colab with
# mounted/remote storage) monkey-patch open() for special paths but not
# os.replace / os.path.exists, so atomic-rename tricks are unavailable.
# Instead, writes alternate between two slot files — an interrupted write
# can corrupt at most the slot being written, never the previous good one —
# and the loader picks the newest slot that deserializes.
_CKPT_SLOTS = (".a", ".b")


def _save_checkpoint(path, params, opt_state, next_epoch: int, slot: int):
    """Write (params, opt_state, next_epoch) to one of the two slot files."""
    with open(path + _CKPT_SLOTS[slot % 2], "wb") as f:
        f.write(flax.serialization.to_bytes((params, opt_state, next_epoch)))


def _load_checkpoint(path, template):
    """Return the newest readable checkpoint among the slot files, or None.

    Also tries the bare path ("" suffix) for checkpoints written by older
    single-file code or by hand.
    """
    best = None
    for suffix in _CKPT_SLOTS + ("",):
        try:
            with open(path + suffix, "rb") as f:
                cand = flax.serialization.from_bytes(template, f.read())
        except Exception:  # missing, partially written, or corrupt slot
            continue
        if best is None or cand[2] > best[2]:
            best = cand
    return best


def train_model(
    *,
    collect_fn=None,
    batch_size: int = 8192,
    num_epochs: int = 1000,
    lr: float = 3e-4,
    print_every: int = 100,
    seed: int = 0,
    checkpoint_path: str = None,
    checkpoint_every: int = 100,
) -> tuple:
    """Train a ValueNet from scratch on self-play data.

    Each epoch generates a fresh batch of games as training data.
    A fixed holdout set (same size) is collected once up front for eval.

    The defaults are the canonical V0 settings (eval loss plateaus around
    epoch 500); see docs/jass_plan.md Step 0.

    Args:
        collect_fn: Data generator with the same signature and return as
            jass_selfplay.collect_batch: (key, batch_size) → (cm, hd,
            labels, alive). Defaults to collect_batch (uniform-random
            play). For generation ≥1 pass a V-guided generator, e.g.
            jass_selfplay.make_v_collect_fn(model.apply, prev_params).
        batch_size: Number of games per training batch and holdout set.
        num_epochs: Total training epochs.
        lr: Adam learning rate.
        print_every: Print train/eval loss every N epochs.
        seed: PRNG seed for reproducibility.
        checkpoint_path: If given, (params, opt_state, epoch) is written
            to checkpoint_path + ".a"/".b" (alternating) every
            checkpoint_every epochs; put it on Drive in colab. If a
            readable checkpoint exists there, training RESUMES from the
            newest one: the RNG stream is fast-forwarded so the resumed
            run consumes the same data sequence as an uninterrupted one
            and produces identical final weights. Resume assumes the same
            collect_fn / batch_size / lr / seed as the interrupted run.
        checkpoint_every: Checkpoint interval in epochs.

    Returns:
        (params, model) — trained Flax parameters and the ValueNet instance.
    """
    if collect_fn is None:
        collect_fn = collect_batch
    key = jax.random.PRNGKey(seed)

    model = ValueNet()
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    step_fn = make_train_step(model, optimizer)

    start_epoch = 0
    if checkpoint_path is not None:
        loaded = _load_checkpoint(checkpoint_path, (params, opt_state, 0))
        if loaded is not None:
            params, opt_state, start_epoch = loaded
            print(f"Resuming from {checkpoint_path} at epoch {start_epoch}\n")

    print("Collecting holdout batch for eval ...")
    key, k_eval = jax.random.split(key)
    cm_eval, hd_eval, y_eval, alive_eval = collect_fn(k_eval, batch_size)
    cm_eval = cm_eval.reshape(-1, 36, 12)
    hd_eval = hd_eval.reshape(-1, 20)
    y_eval = y_eval.reshape(-1)
    mask_eval = alive_eval.reshape(-1).astype(jnp.float32)
    print(f"  {int(mask_eval.sum())} labeled positions\n")

    t0 = time.perf_counter()
    for epoch in range(num_epochs):
        key, k1 = jax.random.split(key)
        if epoch < start_epoch:
            continue  # replay the RNG stream up to the checkpoint

        cm, hd, y, alive = collect_fn(k1, batch_size)

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

        if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, params, opt_state, epoch + 1,
                             slot=(epoch + 1) // checkpoint_every)

    return params, model


def _flatten_pv(batch):
    """Flatten a (cm, hd, y, pi, legal, alive) batch for the train step."""
    cm, hd, y, pi, legal, alive = batch
    return (cm.reshape(-1, 36, 12), hd.reshape(-1, 20), y.reshape(-1),
            pi.reshape(-1, NUM_ACTIONS), legal.reshape(-1, NUM_ACTIONS),
            alive.reshape(-1).astype(jnp.float32))


def train_pv_model(
    *,
    collect_fn=None,
    batch_size: int = 8192,
    num_epochs: int = 1000,
    lr: float = 3e-4,
    policy_weight: float = 1.0,
    print_every: int = 100,
    seed: int = 0,
    checkpoint_path: str = None,
    checkpoint_every: int = 100,
) -> tuple:
    """Train a PolicyValueNet from scratch on self-play data.

    Same loop shape as train_model (fresh batch per epoch, fixed holdout
    for eval, slot-file checkpointing with RNG fast-forward on resume),
    but for the joint net with the PV collect contract.

    Args:
        collect_fn: Data generator with the PV contract: (key, batch_size)
            → (cm, hd, labels, pi, legal, alive); e.g.
            jass_selfplay.make_search_collect_fn(...) for Step 2 data.
            Defaults to collect_pv_batch (uniform-random play — smoke
            tests / value-only pretraining; its pi targets carry no
            signal).
        batch_size: Number of games per training batch and holdout set.
        num_epochs: Total training epochs.
        lr: Adam learning rate.
        policy_weight: Weight of the policy cross-entropy in the loss
            (value MSE has weight 1).
        print_every: Print train/eval losses every N epochs.
        seed: PRNG seed for reproducibility.
        checkpoint_path: As in train_model (slot files, resume replays
            the RNG stream; same collect_fn / hyperparameters assumed).
        checkpoint_every: Checkpoint interval in epochs.

    Returns:
        (params, model) — trained Flax parameters and the PolicyValueNet
        instance.
    """
    if collect_fn is None:
        collect_fn = collect_pv_batch
    key = jax.random.PRNGKey(seed)

    model = PolicyValueNet()
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    step_fn = make_pv_train_step(model, optimizer, policy_weight)

    start_epoch = 0
    if checkpoint_path is not None:
        loaded = _load_checkpoint(checkpoint_path, (params, opt_state, 0))
        if loaded is not None:
            params, opt_state, start_epoch = loaded
            print(f"Resuming from {checkpoint_path} at epoch {start_epoch}\n")

    print("Collecting holdout batch for eval ...")
    key, k_eval = jax.random.split(key)
    eval_batch = _flatten_pv(collect_fn(k_eval, batch_size))
    print(f"  {int(eval_batch[-1].sum())} labeled positions\n")

    t0 = time.perf_counter()
    for epoch in range(num_epochs):
        key, k1 = jax.random.split(key)
        if epoch < start_epoch:
            continue  # replay the RNG stream up to the checkpoint

        batch = _flatten_pv(collect_fn(k1, batch_size))
        params, opt_state, train_loss, _, _ = step_fn(params, opt_state, *batch)

        if epoch % print_every == 0:
            _, _, e_loss, e_v, e_p = step_fn(params, opt_state, *eval_batch)
            elapsed = time.perf_counter() - t0
            print(f"[{epoch:4d}]  train={float(train_loss):.4f}"
                  f"  eval={float(e_loss):.4f}"
                  f"  (v={float(e_v):.4f}  p={float(e_p):.4f})"
                  f"  ({elapsed:.0f}s)")

        if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, params, opt_state, epoch + 1,
                             slot=(epoch + 1) // checkpoint_every)

    return params, model


# ── CLI Driver ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 1 epoch, batch=64")
    parser.add_argument("--save", default="jass_v_weights.msgpack",
                        help="Path to write final weights")
    args = parser.parse_args()

    if args.smoke:
        params, model = train_model(batch_size=64, num_epochs=1, print_every=1)
    else:
        params, model = train_model()

    with open(args.save, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"\nWeights saved to {args.save}")
