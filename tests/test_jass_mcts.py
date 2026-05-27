import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass import Game, GameState, NUM_ACTIONS, DECLARE_OFFSET
from pgx._src.games.jass_mcts import (
    sample_determinization,
    _random_rollout,
    best_action,
)

game = Game()


# ──────────────────────────────────────────────────
# sample_determinization


def _deal(key):
    """Return a freshly dealt GameState."""
    return game.init(key)


def test_determinization_my_hand_unchanged():
    state = _deal(jax.random.PRNGKey(0))
    player_id = jnp.int32(0)
    det = sample_determinization(state, player_id, jax.random.PRNGKey(42))
    assert (det.hands[player_id] == state.hands[player_id]).all()


def test_determinization_no_card_duplication():
    state = _deal(jax.random.PRNGKey(1))
    player_id = jnp.int32(2)
    det = sample_determinization(state, player_id, jax.random.PRNGKey(7))
    # Each card appears in exactly one hand.
    card_counts = det.hands.sum(axis=0)  # (36,)
    assert (card_counts == 1).all(), f"card appears in multiple hands: {card_counts}"


def test_determinization_hand_sizes():
    state = _deal(jax.random.PRNGKey(2))
    player_id = jnp.int32(1)
    det = sample_determinization(state, player_id, jax.random.PRNGKey(99))
    # All 9 cards per player at game start.
    assert (det.hands.sum(axis=1) == 9).all()


def test_determinization_mid_trick():
    """Mid-trick: only player 0 has played; others still hold correct counts."""
    state = _deal(jax.random.PRNGKey(3))
    # Declare trump (mode 0 = ♦) and play one card from player 0's hand.
    state = game.step(state, jnp.int32(DECLARE_OFFSET))   # declare ♦ trump
    card0 = int(jnp.argmax(state.hands[0]))
    state = game.step(state, jnp.int32(card0))            # player 0 plays

    # Now player 1 is to play; determinize from player 1's perspective.
    player_id = jnp.int32(1)
    det = sample_determinization(state, player_id, jax.random.PRNGKey(11))

    # Player 1's hand unchanged.
    assert (det.hands[player_id] == state.hands[player_id]).all()
    # No duplicates.
    card_counts = det.hands.sum(axis=0)
    assert (card_counts <= 1).all()
    # Player 0 has 8 cards (played one); player 1 has 9; players 2,3 have 9.
    expected = jnp.int32([8, 9, 9, 9])
    assert (det.hands.sum(axis=1) == expected).all()


def test_determinization_after_two_tricks():
    """After 2 complete tricks: hands shrink by 2 each."""
    state = _deal(jax.random.PRNGKey(5))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))  # declare ♦ trump
    for _ in range(8):  # 2 tricks × 4 cards
        mask = game.legal_action_mask(state)
        action = int(jnp.argmax(mask))
        state = game.step(state, jnp.int32(action))

    assert int(state.trick_num) == 2
    player_id = jnp.int32(state.current_player)
    det = sample_determinization(state, player_id, jax.random.PRNGKey(55))
    assert (det.hands[player_id] == state.hands[player_id]).all()
    assert (det.hands.sum(axis=1) == 7).all()


def test_determinization_uniform_across_samples():
    """Different keys should produce different opponent hands."""
    state = _deal(jax.random.PRNGKey(9))
    player_id = jnp.int32(0)
    det1 = sample_determinization(state, player_id, jax.random.PRNGKey(1))
    det2 = sample_determinization(state, player_id, jax.random.PRNGKey(2))
    # With overwhelming probability two random deals differ.
    assert not (det1.hands == det2.hands).all()


def test_determinization_respects_void_constraint():
    """Opponent known void in a suit must not receive cards of that suit."""
    from pgx._src.games.jass import GameState, DECLARE_OFFSET

    # Build a state where player 1 is known void in ♦ (suit 0).
    state = _deal(jax.random.PRNGKey(50))
    state = game.step(state, jnp.int32(DECLARE_OFFSET + 1))  # ♥ trump

    # Manually mark player 1 as void in ♦.
    void_in_suit = state.void_in_suit.at[1, 0].set(True)
    state = state._replace(void_in_suit=void_in_suit)

    player_id = jnp.int32(0)
    # Check 20 different determinizations — none should give player 1 a ♦ card.
    for i in range(20):
        det = sample_determinization(state, player_id, jax.random.PRNGKey(i))
        # ♦ cards are indices 0–8.
        player1_diamond = det.hands[1, :9]
        assert not player1_diamond.any(), \
            f"player 1 (void in ♦) received a ♦ card in determinization {i}"


# ──────────────────────────────────────────────────
# _random_rollout


def test_rollout_terminates():
    state = _deal(jax.random.PRNGKey(10))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))  # declare trump
    rewards = _random_rollout(state, jax.random.PRNGKey(0))
    assert rewards.shape == (4,)
    # Zero-sum across teams.
    assert abs(float(rewards[0] + rewards[1] + rewards[2] + rewards[3])) < 1e-3


def test_rollout_zero_sum():
    """Team rewards always sum to zero."""
    state = _deal(jax.random.PRNGKey(11))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    for key_int in range(20):
        rewards = _random_rollout(state, jax.random.PRNGKey(key_int))
        team_a = float(rewards[0] + rewards[2])
        team_b = float(rewards[1] + rewards[3])
        assert abs(team_a + team_b) < 1e-3, f"not zero-sum: {rewards}"


def test_rollout_from_trump_selection():
    """Rollout starting at trump selection should also work."""
    state = _deal(jax.random.PRNGKey(12))
    assert int(state.phase) == 0
    rewards = _random_rollout(state, jax.random.PRNGKey(5))
    assert rewards.shape == (4,)
    assert abs(float(rewards.sum())) < 1e-3


def test_rollout_reward_range():
    """Rewards should fall in [-157, 157]."""
    state = _deal(jax.random.PRNGKey(13))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    for key_int in range(10):
        rewards = _random_rollout(state, jax.random.PRNGKey(key_int))
        assert (jnp.abs(rewards) <= 157).all(), f"reward out of range: {rewards}"


# ──────────────────────────────────────────────────
# best_action


def test_best_action_is_legal():
    """best_action must return a legal action."""
    state = _deal(jax.random.PRNGKey(20))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))  # declare trump
    player_id = state.current_player
    action = best_action(state, player_id, jax.random.PRNGKey(0),
                         num_determinizations=4, num_rollouts=2)
    mask = game.legal_action_mask(state)
    assert bool(mask[action]), f"action {int(action)} is not legal"


def test_best_action_trump_selection():
    """best_action during trump selection returns a mode declaration."""
    state = _deal(jax.random.PRNGKey(21))
    assert int(state.phase) == 0
    player_id = state.current_player
    action = best_action(state, player_id, jax.random.PRNGKey(1),
                         num_determinizations=4, num_rollouts=2)
    mask = game.legal_action_mask(state)
    assert bool(mask[action])
    # Must be a trump or schiebe action (>= 36).
    assert int(action) >= DECLARE_OFFSET


def test_best_action_all_players():
    """best_action works for all four player perspectives."""
    state = _deal(jax.random.PRNGKey(30))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    for p in range(4):
        action = best_action(state, jnp.int32(p), jax.random.PRNGKey(p),
                             num_determinizations=4, num_rollouts=2)
        # If we're asking for a non-current player the mask is for current_player,
        # but the action should still be legal from the current player's mask.
        mask = game.legal_action_mask(state)
        assert bool(mask[action])


def test_best_action_with_constant_v():
    """V-MCTS returns a legal action when V is a constant function."""
    state = _deal(jax.random.PRNGKey(20))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))  # enter card-play phase
    player_id = state.current_player

    v_apply = lambda params, cm, hd: jnp.zeros(cm.shape[0])
    action = best_action(state, player_id, jax.random.PRNGKey(1),
                         num_determinizations=4, num_rollouts=1,
                         v_apply=v_apply, v_params={})
    assert bool(game.legal_action_mask(state)[action]), \
        f"V-MCTS returned illegal action {int(action)}"


def test_best_action_full_game():
    """Play a full game using best_action; game should terminate."""
    state = _deal(jax.random.PRNGKey(40))
    key = jax.random.PRNGKey(0)
    for _ in range(40):  # at most 38 steps
        if bool(game.is_terminal(state)):
            break
        player_id = state.current_player
        key, subkey = jax.random.split(key)
        action = best_action(state, player_id, subkey,
                             num_determinizations=4, num_rollouts=2)
        state = game.step(state, action)

    assert bool(game.is_terminal(state))
    rewards = game.rewards(state)
    assert abs(float(rewards.sum())) < 1e-3
