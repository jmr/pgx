"""Determinized PUCT for Jass via mctx (Option B) — the AlphaZero search.

For each of K determinizations of the current information state, run a
batched tree search — `mctx.gumbel_muzero_policy` by default, or classical
full-width PUCT via `mctx.muzero_policy` (`search_variant="muzero"`, the
JTR-style operator; A′ probe 2026-07-06) — with the PolicyValueNet
supplying priors and leaf values, and the real game engine (`Game.step`)
as the dynamics. The K trees are aggregated by SUMMING ROOT
VISIT COUNTS and acting on the summed counts — the load-bearing choice from
docs/jass_plan.md (Q-sum aggregation neutralizes the tree policy).

Grounded-teacher knobs (2026-07-06, the gen-9 fixed-point escape —
see the plan's NEXT block): `prior_mix_uniform` mixes the net's priors
toward uniform-over-legal at every node; `rollout_value_weight` blends
leaf values toward a uniform-random playout-to-terminal return (real
points). At 1.0/1.0 the search is a classical, net-free ISMCTS.

Sign conventions (mctx backs up q(parent, a) = reward + discount * v(child)):
- every node's value is from the perspective of the player to move there
  (the net's value is already acting-player-relative);
- reward on an edge is from the parent mover's perspective (nonzero only
  on the step that ends the game);
- discount is +1 when the child mover is on the parent mover's team,
  -1 otherwise (teams are seat parity: {0,2} vs {1,3}), and 0 once the
  game is over.

Everything is pure JAX: usable under jit and vmapped over games (the inner
mctx batch dimension is the K determinizations).
"""

import functools

import jax
import jax.numpy as jnp
import mctx
from jax import Array

from pgx._src.games.jass import Game, GameState, NUM_ACTIONS, value_features
from pgx._src.games.jass_mcts import sample_determinization

_game = Game()

_ILLEGAL = jnp.float32(-1e9)


def _hold_if(done: Array, old: GameState, new: GameState) -> GameState:
    """Per-leaf where(done, old, new) with done broadcast over leading dim."""
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
        old, new,
    )


def _rollout_value(states: GameState, key: Array) -> Array:
    """Uniform-random playout to terminal; points for each state's mover.

    The classical (grounded) leaf evaluation: play random legal moves to
    the end of the game and score the real outcome from the perspective
    of the player to move at `states` — no learned function involved.
    Already-terminal states are held in place and scored as they stand
    (callers zero terminal values, matching `_pv_eval`'s convention).
    """
    movers = states.current_player                       # (B,)

    def cond(carry):
        s, _ = carry
        return jnp.any(s.trick_num < 9)

    def body(carry):
        s, k = carry
        k, k_act = jax.random.split(k)
        legal = jax.vmap(_game.legal_action_mask)(s)     # (B, 43)
        action = jax.random.categorical(
            k_act, jnp.where(legal, 0.0, _ILLEGAL))      # (B,)
        done = s.trick_num >= 9
        next_s = jax.vmap(_game.step)(s, action.astype(jnp.int32))
        return _hold_if(done, s, next_s), k

    final, _ = jax.lax.while_loop(cond, body, (states, key))
    rewards = jax.vmap(_game.rewards)(final)             # (B, 4)
    return jnp.take_along_axis(
        rewards, movers[:, None], axis=1).squeeze(-1)


def _pv_eval(
    pv_apply,
    pv_params,
    states: GameState,
    v_scale: float,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    rollout_key: Array = None,
):
    """Evaluate the net on a batch of states from each mover's perspective.

    prior_mix_uniform / rollout_value_weight are the grounded-teacher
    knobs (2026-07-06 fixed-point escape; see docs/jass_plan.md). Both
    must be trace-time Python floats: they gate computation with `if`,
    so a value of 0.0 costs nothing.

    Returns:
        logits: (B, 43) masked to the legal actions of each state.
        value : (B,) in points, 0 for terminal states.
        legal : (B, 43) bool.
    """
    cm, hd = jax.vmap(value_features)(states, states.current_player)
    logits, value = pv_apply(pv_params, cm, hd)
    legal = jax.vmap(_game.legal_action_mask)(states)
    logits = jnp.where(legal, logits, _ILLEGAL)
    if prior_mix_uniform:
        lam = prior_mix_uniform
        probs = jax.nn.softmax(logits, axis=-1)
        n_legal = legal.sum(axis=-1, keepdims=True).clip(1)
        uniform = jnp.where(legal, 1.0 / n_legal, 0.0)
        logits = jnp.where(
            legal,
            jnp.log(((1.0 - lam) * probs + lam * uniform).clip(1e-9)),
            _ILLEGAL)
    value = value * v_scale
    if rollout_value_weight:
        w = rollout_value_weight
        value = (1.0 - w) * value + w * _rollout_value(states, rollout_key)
    done = states.trick_num >= 9
    value = jnp.where(done, 0.0, value)
    return logits, value, legal


def _make_recurrent_fn(pv_apply, v_scale: float,
                       prior_mix_uniform: float = 0.0,
                       rollout_value_weight: float = 0.0):
    """Build the mctx recurrent_fn over batched GameState embeddings."""

    def recurrent_fn(params, rng_key, action, states: GameState):
        prev_player = states.current_player           # (B,)
        prev_done = states.trick_num >= 9             # (B,)

        next_states = jax.vmap(_game.step)(states, action.astype(jnp.int32))
        next_states = _hold_if(prev_done, states, next_states)

        # Terminal reward from the parent mover's perspective; zero if the
        # game was already over before this edge (no double counting).
        rewards = jax.vmap(_game.rewards)(next_states)    # (B, 4)
        reward = jnp.take_along_axis(
            rewards, prev_player[:, None], axis=1).squeeze(-1)
        reward = jnp.where(prev_done, 0.0, reward)

        logits, value, _ = _pv_eval(
            pv_apply, params, next_states, v_scale,
            prior_mix_uniform=prior_mix_uniform,
            rollout_value_weight=rollout_value_weight,
            rollout_key=rng_key)

        done = next_states.trick_num >= 9
        same_team = (next_states.current_player % 2) == (prev_player % 2)
        discount = jnp.where(same_team, 1.0, -1.0)
        discount = jnp.where(done, 0.0, discount)

        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return output, next_states

    return recurrent_fn


def puct_search(
    state: GameState,
    player_id: Array,
    key: Array,
    pv_params,
    pv_apply,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "gumbel",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
) -> tuple[Array, Array]:
    """Run K determinized tree searches and sum root visit counts.

    Args:
        state: Current (true or self-play) game state.
        player_id: Acting player; determinizations keep this hand fixed.
        pv_params / pv_apply: PolicyValueNet weights and apply function
            ((params, cm, hd) → (logits (B,43), value (B,))).
        num_determinizations: K sampled worlds (the mctx batch dimension).
        num_simulations: Tree simulations per determinization.
        v_scale: Net output → points (TARGET_SCALE of the training run).
        max_num_considered_actions: Gumbel sequential-halving width at the
            root. Gumbel variant only.
        search_variant: "gumbel" (default, `mctx.gumbel_muzero_policy` —
            the standing collector) or "muzero" (`mctx.muzero_policy`,
            classical full-width PUCT — the JTR-style operator, A′ probe
            2026-07-06).
        pb_c_init: Classical-PUCT exploration constant on mctx's
            per-node-normalized Q. Muzero variant only. JTR's c=100 on
            the raw 0–157 point scale ≈ 0.64 here.
        dirichlet_fraction: Root Dirichlet-noise share. Muzero variant
            only; 0.0 (default) = deterministic teacher, AlphaZero
            self-play uses 0.25.
        prior_mix_uniform: λ share of uniform-over-legal mixed into the
            net's priors at the root and every tree node; 1.0 = flat
            (uniform-prior) PUCT. Grounded-teacher knob (2026-07-06):
            breaks the prior half of the search(π)≈π self-confirmation.
        rollout_value_weight: w share of a uniform-random playout-to-
            terminal return (real points) blended into every leaf value;
            1.0 replaces the value head entirely (classical evaluation).
            Breaks the value half of the self-confirmation. Costs up to
            38 extra env steps per node expansion. Both knobs must be
            Python floats at trace time (0.0 compiles to a no-op).

    Returns:
        (visit_counts, legal): (43,) float32 root visit counts summed over
        the K trees (zero on illegal actions), and the (43,) bool legal
        mask of the information state.
    """
    K = num_determinizations
    det_key, search_key, root_key = jax.random.split(key, 3)
    det_states = jax.vmap(
        lambda k: sample_determinization(state, player_id, k)
    )(jax.random.split(det_key, K))                      # (K,) GameState

    logits, value, _ = _pv_eval(
        pv_apply, pv_params, det_states, v_scale,
        prior_mix_uniform=prior_mix_uniform,
        rollout_value_weight=rollout_value_weight,
        rollout_key=root_key)
    root = mctx.RootFnOutput(
        prior_logits=logits, value=value, embedding=det_states)

    # The legal mask is an information-state property: identical across
    # determinizations (hands of others don't constrain the mover's moves).
    legal = _game.legal_action_mask(state)               # (43,)
    invalid = jnp.broadcast_to(~legal, (K, NUM_ACTIONS))

    recurrent_fn = _make_recurrent_fn(
        pv_apply, v_scale, prior_mix_uniform, rollout_value_weight)
    if search_variant == "gumbel":
        out = mctx.gumbel_muzero_policy(
            params=pv_params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid.astype(jnp.float32),
            max_num_considered_actions=max_num_considered_actions,
        )
    elif search_variant == "muzero":
        out = mctx.muzero_policy(
            params=pv_params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid.astype(jnp.float32),
            pb_c_init=pb_c_init,
            dirichlet_fraction=dirichlet_fraction,
        )
    else:
        raise ValueError(f"unknown search_variant: {search_variant!r}")

    visits = out.search_tree.summary().visit_counts      # (K, 43)
    visits = jnp.where(legal, visits.sum(axis=0), 0.0)   # (43,)
    return visits, legal


@functools.partial(jax.jit, static_argnames=(
    "pv_apply", "num_determinizations", "num_simulations",
    "max_num_considered_actions", "search_variant",
    "prior_mix_uniform", "rollout_value_weight"))
def puct_action(
    state: GameState,
    player_id: Array,
    key: Array,
    pv_params,
    pv_apply,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "gumbel",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
) -> Array:
    """Greedy PUCT move: argmax of summed root visits (ties → first legal)."""
    visits, legal = puct_search(
        state, player_id, key, pv_params, pv_apply,
        num_determinizations, num_simulations, v_scale,
        max_num_considered_actions, search_variant, pb_c_init,
        dirichlet_fraction, prior_mix_uniform, rollout_value_weight)
    scored = jnp.where(legal, visits, -jnp.inf)
    return jnp.argmax(scored).astype(jnp.int32)


def make_puct_policy_fn(
    pv_apply,
    pv_params,
    *,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "gumbel",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    temperature: float = None,
):
    """Build a policy_fn(state, key) → (action, pi) for jass_selfplay.

    pi is the normalized summed visit distribution — the Step 3 policy
    training target. The executed action is the visit argmax when
    temperature is None, otherwise sampled ∝ visits^(1/temperature)
    (AlphaZero-style exploration; temperature=1 samples the visit
    distribution itself).
    """

    def policy_fn(state: GameState, key: Array):
        k_search, k_sample = jax.random.split(key)
        visits, legal = puct_search(
            state, state.current_player, k_search, pv_params, pv_apply,
            num_determinizations, num_simulations, v_scale,
            max_num_considered_actions, search_variant, pb_c_init,
            dirichlet_fraction, prior_mix_uniform, rollout_value_weight)
        pi = visits / visits.sum().clip(1.0)
        if temperature is None:
            action = jnp.argmax(jnp.where(legal, visits, -jnp.inf))
        else:
            logits = jnp.where(
                legal, jnp.log(visits.clip(1e-9)) / temperature, _ILLEGAL)
            action = jax.random.categorical(k_sample, logits)
        return action.astype(jnp.int32), pi

    return policy_fn


def make_puct_action_fn(pv_apply, pv_params, **kwargs):
    """action_fn(state, key) → action wrapper (for policy_match / arenas)."""
    policy_fn = make_puct_policy_fn(pv_apply, pv_params, **kwargs)

    def action_fn(state: GameState, key: Array) -> Array:
        action, _ = policy_fn(state, key)
        return action

    return action_fn


def make_puct_collect_fn(
    pv_apply,
    pv_params,
    *,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "gumbel",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    temperature: float = 1.0,
):
    """Build a collect_fn(key, batch_size) generating PUCT self-play data.

    The Step 3 generator: every seat plays PUCT; policy targets are the
    aggregated root visit distributions; value labels are the acting
    player's terminal differential. Same contract as
    jass_selfplay.make_search_collect_fn:
    (cm, hd, labels, pi, legal, alive).
    """
    from pgx._src.games.jass_selfplay import _collect_pv

    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _puct_collect(params, key: Array, batch_size: int):
        policy_fn = make_puct_policy_fn(
            pv_apply, params,
            num_determinizations=num_determinizations,
            num_simulations=num_simulations,
            v_scale=v_scale,
            max_num_considered_actions=max_num_considered_actions,
            search_variant=search_variant,
            pb_c_init=pb_c_init,
            dirichlet_fraction=dirichlet_fraction,
            prior_mix_uniform=prior_mix_uniform,
            rollout_value_weight=rollout_value_weight,
            temperature=temperature)
        return _collect_pv(policy_fn, key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _puct_collect(pv_params, key, batch_size)

    return collect_fn
