"""
Vmapped random self-play for Jass value-network training.

collect_batch(key, batch_size) → (cm, hd, labels, alive)
    cm    : (B, T, 36, 12) bool  card matrix per timestep
    hd    : (B, T, 20)     bool  header per timestep
    labels: (B, T)         f32   acting player's terminal differential
    alive : (B, T)         bool  False once game is terminal (label mask)

Flatten and filter by alive before feeding the trainer:
    cm    = cm.reshape(-1, 36, 12)[alive.reshape(-1)]
    hd    = hd.reshape(-1, 20)   [alive.reshape(-1)]
    labels= labels.reshape(-1)   [alive.reshape(-1)]

Or pass alive.reshape(-1).astype(jnp.float32) as a sample-weight mask
to avoid dynamic shapes inside jit.
"""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from pgx._src.games.jass import Game, NUM_ACTIONS, value_features

_game = Game()
_MAX_STEPS = 38   # 2 trump-selection + 9*4 card-play steps


def _play_one(key: Array):
    """Run one full random game. Returns per-step features and terminal rewards."""
    init_key, play_key = jax.random.split(key)
    s0 = _game.init(init_key)

    def step_fn(carry, _):
        s, k = carry
        done = s.trick_num >= 9

        k, sk = jax.random.split(k)
        mask   = _game.legal_action_mask(s)
        logits = jnp.where(mask, 0.0, -1e9)
        action = jax.random.categorical(sk, logits).astype(jnp.int32)

        cm, hd = value_features(s, s.current_player)
        out = (cm, hd, s.current_player, ~done)

        ns = _game.step(s, action)
        # Hold state fixed once terminal so the scan stays well-defined.
        ns = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), s, ns)
        return (ns, k), out

    (final, _), (cm, hd, actor, alive) = jax.lax.scan(
        step_fn, (s0, play_key), None, length=_MAX_STEPS
    )
    # cm:    (T, 36, 12)
    # hd:    (T, 20)
    # actor: (T,) int32
    # alive: (T,) bool
    # rew:   (4,) float32

    rew = _game.rewards(final)  # (4,)
    return cm, hd, actor, alive, rew


@functools.partial(jax.jit, static_argnames=("batch_size",))
def collect_batch(key: Array, batch_size: int):
    """Run batch_size random games in parallel; label every step.

    Returns:
        cm    : (B, T, 36, 12) bool
        hd    : (B, T, 20)     bool
        labels: (B, T)         float32 — acting player's terminal differential
        alive : (B, T)         bool    — mask out post-terminal padding
    """
    keys = jax.random.split(key, batch_size)
    cm, hd, actor, alive, rew = jax.vmap(_play_one)(keys)
    # rew:   (B, 4); actor: (B, T)
    # For each (b, t): labels[b, t] = rew[b, actor[b, t]]
    labels = jnp.take_along_axis(
        rew[:, jnp.newaxis, :],   # (B, 1, 4)
        actor[..., jnp.newaxis],  # (B, T, 1)
        axis=-1,
    ).squeeze(-1)                 # (B, T)
    return cm, hd, labels, alive
