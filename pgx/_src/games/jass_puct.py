"""Determinized PUCT for Jass via mctx (Option B) — the AlphaZero search.

For each of K determinizations of the current information state, run a
batched Gumbel-MuZero tree search (`mctx.gumbel_muzero_policy`) with the
PolicyValueNet supplying priors and leaf values, and the real game engine
(`Game.step`) as the dynamics. The K trees are aggregated by SUMMING ROOT
VISIT COUNTS and acting on the summed counts — the load-bearing choice from
docs/jass_plan.md (Q-sum aggregation neutralizes the tree policy).

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


def _pv_eval(pv_apply, pv_params, states: GameState, v_scale: float):
    """Evaluate the net on a batch of states from each mover's perspective.

    Returns:
        logits: (B, 43) masked to the legal actions of each state.
        value : (B,) in points, 0 for terminal states.
        legal : (B, 43) bool.
    """
    cm, hd = jax.vmap(value_features)(states, states.current_player)
    logits, value = pv_apply(pv_params, cm, hd)
    legal = jax.vmap(_game.legal_action_mask)(states)
    logits = jnp.where(legal, logits, _ILLEGAL)
    done = states.trick_num >= 9
    value = jnp.where(done, 0.0, value * v_scale)
    return logits, value, legal


def _make_recurrent_fn(pv_apply, v_scale: float):
    """Build the mctx recurrent_fn over batched GameState embeddings."""

    def recurrent_fn(params, rng_key, action, states: GameState):
        del rng_key
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

        logits, value, _ = _pv_eval(pv_apply, params, next_states, v_scale)

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
) -> tuple[Array, Array]:
    """Run K determinized Gumbel-MuZero searches and sum root visit counts.

    Args:
        state: Current (true or self-play) game state.
        player_id: Acting player; determinizations keep this hand fixed.
        pv_params / pv_apply: PolicyValueNet weights and apply function
            ((params, cm, hd) → (logits (B,43), value (B,))).
        num_determinizations: K sampled worlds (the mctx batch dimension).
        num_simulations: Tree simulations per determinization.
        v_scale: Net output → points (TARGET_SCALE of the training run).
        max_num_considered_actions: Gumbel sequential-halving width at the
            root.

    Returns:
        (visit_counts, legal): (43,) float32 root visit counts summed over
        the K trees (zero on illegal actions), and the (43,) bool legal
        mask of the information state.
    """
    K = num_determinizations
    det_key, search_key = jax.random.split(key)
    det_states = jax.vmap(
        lambda k: sample_determinization(state, player_id, k)
    )(jax.random.split(det_key, K))                      # (K,) GameState

    logits, value, _ = _pv_eval(pv_apply, pv_params, det_states, v_scale)
    root = mctx.RootFnOutput(
        prior_logits=logits, value=value, embedding=det_states)

    # The legal mask is an information-state property: identical across
    # determinizations (hands of others don't constrain the mover's moves).
    legal = _game.legal_action_mask(state)               # (43,)
    invalid = jnp.broadcast_to(~legal, (K, NUM_ACTIONS))

    out = mctx.gumbel_muzero_policy(
        params=pv_params,
        rng_key=search_key,
        root=root,
        recurrent_fn=_make_recurrent_fn(pv_apply, v_scale),
        num_simulations=num_simulations,
        invalid_actions=invalid.astype(jnp.float32),
        max_num_considered_actions=max_num_considered_actions,
    )

    visits = out.search_tree.summary().visit_counts      # (K, 43)
    visits = jnp.where(legal, visits.sum(axis=0), 0.0)   # (43,)
    return visits, legal


@functools.partial(jax.jit, static_argnames=(
    "pv_apply", "num_determinizations", "num_simulations",
    "max_num_considered_actions"))
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
) -> Array:
    """Greedy PUCT move: argmax of summed root visits (ties → first legal)."""
    visits, legal = puct_search(
        state, player_id, key, pv_params, pv_apply,
        num_determinizations, num_simulations, v_scale,
        max_num_considered_actions)
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
            max_num_considered_actions)
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
            temperature=temperature)
        return _collect_pv(policy_fn, key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _puct_collect(pv_params, key, batch_size)

    return collect_fn
