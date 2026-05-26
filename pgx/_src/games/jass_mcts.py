"""Determinized rollout policy for Jass (Option A).

For each of K determinizations:
  - Sample a consistent assignment of unknown cards to opponents, respecting
    known suit-void constraints (void_in_suit tracked in GameState).
  - For each legal action, apply the action and run N random rollouts to game end.
  - Average rewards over rollouts.
Average over determinizations, then pick the action with the best score.

See docs/jass_mcts.md for design rationale and Option B (UCT via mctx).
"""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from pgx._src.games.jass import CARD_SUIT, Game, GameState, NUM_ACTIONS

_game = Game()


# ──────────────────────────────────────────────────────────────────────────────
# Card assignment with void constraints


def _assign_cards_void_aware(
    unknown_mask: Array,
    player_needs: Array,
    void_in_suit: Array,
    player_id: Array,
    key: Array,
) -> Array:
    """Distribute unknown cards to opponents respecting void-in-suit constraints.

    Processes players in descending order of void-suit count (most constrained
    first). Each player draws their required number of cards uniformly at random
    from the remaining unknown cards that are valid for them (not in a void suit).
    Processing constrained players first guarantees they can always fill their
    hand from the full pool before unconstrained players deplete valid cards.

    Args:
        unknown_mask: (36,) bool — cards not visible to the current player.
        player_needs: (4,) int32 — cards each player still holds.
        void_in_suit: (4, 4) bool — void_in_suit[p, s] from GameState.
        player_id: scalar int32 — excluded from assignment.
        key: JAX PRNG key.

    Returns:
        (4, 36) bool assignment array; player_id's row is all-False.
    """
    # Process order: most void-constrained opponent first; player_id last.
    void_count   = void_in_suit.sum(axis=1).astype(jnp.int32)   # (4,)
    priority     = jnp.where(jnp.arange(4) == player_id, jnp.int32(-999), void_count)
    process_order = jnp.argsort(-priority).astype(jnp.int32)    # (4,) descending

    keys         = jax.random.split(key, 4)                     # one key per player
    ordered_keys = keys[process_order]                           # (4, 2) reordered

    def assign_one_player(carry, inp):
        remaining, assigned = carry
        player_p, key_p = inp
        is_opp   = player_p != player_id
        needs_p  = jnp.where(is_opp, player_needs[player_p], jnp.int32(0))

        # Cards valid for player_p: remaining unknown, not in a void suit.
        not_void_p  = ~void_in_suit[player_p][CARD_SUIT]        # (36,) bool
        valid_for_p = remaining & not_void_p & is_opp            # (36,)

        # Take the first needs_p valid cards in a random order (cumsum approach).
        perm_p     = jax.random.permutation(key_p, 36)
        cumcount   = jnp.cumsum(valid_for_p[perm_p].astype(jnp.int32))  # (36,)
        card_pos   = cumcount[jnp.argsort(perm_p)]              # position per card
        goes_to_p  = valid_for_p & (card_pos <= needs_p)        # (36,)

        new_remaining = remaining & ~goes_to_p
        new_assigned  = assigned.at[player_p].set(goes_to_p)
        return (new_remaining, new_assigned), None

    init_assigned = jnp.zeros((4, 36), dtype=jnp.bool_)
    (_, final_assigned), _ = jax.lax.scan(
        assign_one_player,
        (unknown_mask, init_assigned),
        (process_order, ordered_keys),
    )
    return final_assigned


# ──────────────────────────────────────────────────────────────────────────────
# Determinization sampling


def sample_determinization(state: GameState, player_id: Array, key: Array) -> GameState:
    """Return a GameState identical to *state* but with opponents' hands resampled.

    The current player's hand is kept fixed. Unknown cards (not in the current
    player's hand, not collected, not in the current trick) are redistributed
    among opponents. Known suit-void constraints (``state.void_in_suit``) are
    respected: no opponent receives a card of a suit they are known to be void in.

    Args:
        state: Full game state (as stored by the pgx environment).
        player_id: The player whose perspective we are taking (0–3).
        key: JAX PRNG key consumed by this call.

    Returns:
        A new GameState with resampled opponent hands.
    """
    my_hand       = state.hands[player_id]                        # (36,)
    collected_any = state.cards_collected.any(axis=0)             # (36,)
    valid_trick   = state.trick_cards >= 0                        # (4,)
    safe_trick    = jnp.where(valid_trick, state.trick_cards, 0)
    # Use .max() not .set(): invalid slots map to index 0, and .set() last-wins
    # semantics would overwrite a True at card 0 with False.
    in_trick      = jnp.zeros(36, dtype=jnp.bool_).at[safe_trick].max(valid_trick)  # (36,)

    unknown_mask = ~my_hand & ~collected_any & ~in_trick          # (36,)

    # Each player holds 9 − completed_tricks − (1 if already played this trick).
    played_in_trick = (state.trick_cards >= 0).astype(jnp.int32) # (4,)
    player_needs    = jnp.int32(9) - state.trick_num - played_in_trick  # (4,)

    opponent_hands = _assign_cards_void_aware(
        unknown_mask, player_needs, state.void_in_suit, player_id, key
    )  # (4, 36)

    new_hands = jnp.where(
        (jnp.arange(4) == player_id)[:, None],  # (4, 1)
        state.hands,                             # keep my real hand
        opponent_hands,
    )
    return state._replace(hands=new_hands)


# ──────────────────────────────────────────────────────────────────────────────
# Random rollout


def _random_rollout(state: GameState, key: Array) -> Array:
    """Play to game completion with uniformly random legal moves.

    Args:
        state: Starting game state (may be mid-game).
        key: JAX PRNG key.

    Returns:
        Reward array of shape (4,) from ``Game.rewards``.
    """
    def step_fn(carry, _):
        s, k = carry
        is_done = s.trick_num >= 9
        k, subkey = jax.random.split(k)
        mask   = _game.legal_action_mask(s)
        logits = jnp.where(mask, jnp.float32(0), jnp.float32(-1e9))
        action = jax.random.categorical(subkey, logits).astype(jnp.int32)
        next_s = _game.step(s, action)
        # No-op once terminal so the state is stable for remaining scan steps.
        out_s = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_done, a, b), s, next_s
        )
        return (out_s, k), None

    # Upper bound: 2 trump steps + 9 tricks × 4 cards = 38 steps.
    (final_state, _), _ = jax.lax.scan(step_fn, (state, key), None, length=38)
    return _game.rewards(final_state)


# ──────────────────────────────────────────────────────────────────────────────
# Main policy


@functools.partial(jax.jit, static_argnames=("num_determinizations", "num_rollouts"))
def best_action(
    state: GameState,
    player_id: Array,
    key: Array,
    num_determinizations: int = 32,
    num_rollouts: int = 8,
) -> Array:
    """Return the action with the best estimated score via determinized rollouts.

    For each of ``num_determinizations`` sampled consistent worlds:
      - Apply each legal action.
      - Run ``num_rollouts`` random rollouts from the resulting state.
      - Record the mean reward for ``player_id``.
    Average scores across determinizations; return the action with the highest
    mean score (illegal actions receive −∞).

    JIT-compiled: ``num_determinizations`` and ``num_rollouts`` are static
    (they determine array shapes). Subsequent calls with the same values reuse
    the compiled kernel.

    Args:
        state: Current game state as returned by the pgx environment.
        player_id: The acting player (should equal ``state.current_player``).
        key: JAX PRNG key consumed by this call.
        num_determinizations: Number of sampled worlds (K).
        num_rollouts: Number of random rollouts per (world, action) pair (N).

    Returns:
        Scalar int32 action index.
    """
    K = num_determinizations
    N = num_rollouts
    A = NUM_ACTIONS

    mask        = _game.legal_action_mask(state)                 # (A,) same for all dets
    first_legal = jnp.argmax(mask).astype(jnp.int32)            # fallback for illegal slots

    split_keys   = jax.random.split(key, 1 + K)
    det_keys     = split_keys[1:]                                # (K, 2)
    rollout_keys = jax.random.split(split_keys[0], K * A * N).reshape(K, A, N, 2)

    # Sample K determinizations.
    det_states = jax.vmap(
        lambda k: sample_determinization(state, player_id, k)
    )(det_keys)                                                  # (K,) GameState

    actions = jnp.arange(A, dtype=jnp.int32)                    # (A,)

    def eval_det_action(det_state, action, rollout_ks):
        """Score one action in one determinization by averaging N rollouts."""
        safe_action = jnp.where(mask[action], action, first_legal)
        next_state  = _game.step(det_state, safe_action)
        scores      = jax.vmap(lambda k: _random_rollout(next_state, k))(rollout_ks)  # (N, 4)
        return scores[:, player_id].mean()

    # Inner vmap: over A actions.
    eval_over_actions = jax.vmap(eval_det_action, in_axes=(None, 0, 0))
    # (GameState, (A,), (A, N, 2)) → (A,)

    # Outer vmap: over K determinizations.
    eval_over_dets = jax.vmap(eval_over_actions, in_axes=(0, None, 0))
    # ((K,) GameState, (A,), (K, A, N, 2)) → (K, A)

    det_action_scores = eval_over_dets(det_states, actions, rollout_keys)  # (K, A)
    mean_scores       = det_action_scores.mean(axis=0)                     # (A,)

    masked_scores = jnp.where(mask, mean_scores, jnp.float32(-jnp.inf))
    return jnp.argmax(masked_scores).astype(jnp.int32)
