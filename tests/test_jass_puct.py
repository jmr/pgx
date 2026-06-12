import jax
import jax.numpy as jnp

from pgx._src.games.jass import DECLARE_OFFSET, Game, MODE_SCORES
from pgx._src.games.jass_puct import (
    make_puct_action_fn,
    make_puct_collect_fn,
    puct_action,
)
from pgx._src.games.jass_selfplay import policy_match, random_action_fn
from pgx._src.games.jass_value_net import PolicyValueNet

game = Game()


def _pv():
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    return model.apply, params


def _greedy_points_pv(params, cm, hd):
    """Stand-in net: value = collected-points differential so far (scaled).

    Gives the tree genuine intermediate signal with known sign semantics
    (positive = the player whose features these are is ahead), uniform
    priors. Used to validate the mctx sign conventions end to end.
    """
    mode = jnp.argmax(hd[:, :6].astype(jnp.int32), axis=-1)       # (B,)
    scores = MODE_SCORES[mode]                                    # (B, 36)
    my = (cm[:, :, 4] * scores).sum(-1)
    opp = (cm[:, :, 5] * scores).sum(-1)
    value = (my - opp) / 100.0
    logits = jnp.zeros((cm.shape[0], 43), dtype=jnp.float32)
    return logits, value


def test_puct_action_is_legal_card_phase():
    pv_apply, params = _pv()
    state = game.init(jax.random.PRNGKey(0))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    action = puct_action(state, state.current_player, jax.random.PRNGKey(0),
                         params, pv_apply,
                         num_determinizations=2, num_simulations=8)
    assert bool(game.legal_action_mask(state)[action])


def test_puct_action_trump_phase():
    pv_apply, params = _pv()
    state = game.init(jax.random.PRNGKey(3))
    assert int(state.phase) == 0
    action = puct_action(state, state.current_player, jax.random.PRNGKey(0),
                         params, pv_apply,
                         num_determinizations=2, num_simulations=8)
    assert bool(game.legal_action_mask(state)[action])
    assert int(action) >= DECLARE_OFFSET


def test_puct_action_deterministic():
    pv_apply, params = _pv()
    state = game.init(jax.random.PRNGKey(4))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    a = puct_action(state, state.current_player, jax.random.PRNGKey(7),
                    params, pv_apply,
                    num_determinizations=2, num_simulations=8)
    b = puct_action(state, state.current_player, jax.random.PRNGKey(7),
                    params, pv_apply,
                    num_determinizations=2, num_simulations=8)
    assert int(a) == int(b)


def test_puct_full_game():
    pv_apply, params = _pv()
    fn = make_puct_action_fn(pv_apply, params,
                             num_determinizations=2, num_simulations=8)
    state = game.init(jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(0)
    for _ in range(40):
        if bool(game.is_terminal(state)):
            break
        key, sk = jax.random.split(key)
        action = fn(state, sk)
        assert bool(game.legal_action_mask(state)[action])
        state = game.step(state, action)
    assert bool(game.is_terminal(state))
    assert abs(float(game.rewards(state).sum())) < 1e-3


def test_puct_collect_fn_contract():
    pv_apply, params = _pv()
    collect_fn = make_puct_collect_fn(pv_apply, params,
                                      num_determinizations=2,
                                      num_simulations=4)
    B, T = 2, 38
    cm, hd, labels, pi, legal, alive = collect_fn(jax.random.PRNGKey(0), B)
    assert cm.shape == (B, T, 36, 12)
    assert hd.shape == (B, T, 20)
    assert labels.shape == (B, T)
    assert pi.shape == (B, T, 43)
    assert legal.shape == (B, T, 43)
    assert alive.shape == (B, T)
    # pi is a distribution over legal actions on alive steps.
    assert jnp.allclose(jnp.where(alive, pi.sum(-1), 1.0), 1.0, atol=1e-5)
    assert not jnp.any((pi > 0) & ~legal & alive[..., None])
    assert jnp.all(jnp.abs(labels) <= 157)


def test_puct_sign_conventions_beat_random():
    """PUCT with a greedy points-collected value must clearly beat random.

    This is the end-to-end check on the reward/discount sign conventions:
    if perspectives were flipped anywhere, the agent would minimize its
    own points and lose badly instead.
    """
    fn = make_puct_action_fn(_greedy_points_pv, {},
                             num_determinizations=2, num_simulations=16)
    scores = policy_match(fn, random_action_fn, jax.random.PRNGKey(0), 8)
    mean = float(scores.mean())
    # Seed 0 measures ≈ +24 (and ≈ +14 over 64 games); a perspective flip
    # anywhere makes the agent dump points and score around -30 or worse.
    # Threshold sits between the regimes with margin for numeric drift.
    assert mean > 5.0, f"PUCT vs random mean {mean:+.1f}; sign error likely"
