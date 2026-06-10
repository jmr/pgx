"""
Vmapped self-play for Jass value-network training.

Two data generators with the same contract:

collect_batch(key, batch_size)
    Uniform-random play (generation 0).

make_v_collect_fn(v_apply, v_params, ...)(key, batch_size)
    V-greedy softmax play (generation >= 1): each legal action is scored by
    V(state after action) from the acting player's perspective and sampled
    with probability softmax(score / temperature).

Both return (cm, hd, labels, alive):
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


def _random_action(s, key: Array):
    """Uniform-random legal action."""
    mask   = _game.legal_action_mask(s)
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def _play_one(action_fn, key: Array):
    """Run one full game with action_fn(state, key) selecting moves.

    Returns per-step features and terminal rewards.
    """
    init_key, play_key = jax.random.split(key)
    s0 = _game.init(init_key)

    def step_fn(carry, _):
        s, k = carry
        done = s.trick_num >= 9

        k, sk = jax.random.split(k)
        action = action_fn(s, sk)

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


def _collect(action_fn, key: Array, batch_size: int):
    """Run batch_size games in parallel with action_fn; label every step."""
    keys = jax.random.split(key, batch_size)
    cm, hd, actor, alive, rew = jax.vmap(
        functools.partial(_play_one, action_fn)
    )(keys)
    # rew:   (B, 4); actor: (B, T)
    # For each (b, t): labels[b, t] = rew[b, actor[b, t]]
    labels = jnp.take_along_axis(
        rew[:, jnp.newaxis, :],   # (B, 1, 4)
        actor[..., jnp.newaxis],  # (B, T, 1)
        axis=-1,
    ).squeeze(-1)                 # (B, T)
    return cm, hd, labels, alive


@functools.partial(jax.jit, static_argnames=("batch_size",))
def collect_batch(key: Array, batch_size: int):
    """Run batch_size uniform-random games in parallel; label every step.

    Returns:
        cm    : (B, T, 36, 12) bool
        hd    : (B, T, 20)     bool
        labels: (B, T)         float32 — acting player's terminal differential
        alive : (B, T)         bool    — mask out post-terminal padding
    """
    return _collect(_random_action, key, batch_size)


def make_v_collect_fn(v_apply, v_params, *, v_scale: float = 100.0,
                      temperature: float = 10.0):
    """Build a collect_fn(key, batch_size) that plays with a value network.

    All four seats select moves the same way: every action's successor state
    is scored by V from the acting player's perspective (same evaluation as
    the V-MCTS leaf, but without determinization — self-play states are
    fully known), and a legal action is sampled with probability
    softmax(score / temperature).

    Args:
        v_apply: Network apply function, (params, cm, hd) → scaled value.
        v_params: Network parameters (passed as a traced argument, so one
            compilation serves all generations of weights).
        v_scale: Multiplier from network output to points (TARGET_SCALE).
        temperature: Softmax temperature in points. Must be > 0; lower is
            greedier. At 10.0, actions within ~10 points of the best keep
            meaningful probability.

    Returns:
        collect_fn(key, batch_size) with the same contract as collect_batch.
    """
    all_actions = jnp.arange(NUM_ACTIONS, dtype=jnp.int32)

    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _v_collect(params, key: Array, batch_size: int):
        def action_fn(s, k):
            next_states = jax.vmap(_game.step, in_axes=(None, 0))(s, all_actions)
            cm, hd = jax.vmap(value_features, in_axes=(0, None))(
                next_states, s.current_player
            )
            vals   = v_apply(params, cm, hd) * v_scale       # (A,) points
            mask   = _game.legal_action_mask(s)
            logits = jnp.where(mask, vals / temperature, -1e9)
            return jax.random.categorical(k, logits).astype(jnp.int32)

        return _collect(action_fn, key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _v_collect(v_params, key, batch_size)

    return collect_fn
