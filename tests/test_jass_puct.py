import jax
import jax.numpy as jnp
import pytest

from pgx._src.games.jass import DECLARE_OFFSET, Game, MODE_SCORES
from pgx._src.games.jass_puct import (
    _pv_eval,
    _qsum_scores,
    _rollout_value,
    make_puct_action_fn,
    make_puct_collect_fn,
    make_puct_policy_fn,
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


def _batch_states(n, seed=0, past_trump=True):
    states = jax.vmap(game.init)(
        jax.random.split(jax.random.PRNGKey(seed), n))
    if past_trump:
        states = jax.vmap(game.step)(
            states, jnp.full((n,), DECLARE_OFFSET, jnp.int32))
    return states


def test_rollout_value_terminates_and_is_bounded():
    states = _batch_states(4)
    vals = jax.jit(_rollout_value)(states, jax.random.PRNGKey(0))
    assert vals.shape == (4,)
    # Real point differentials: bounded by the 157-point game total.
    assert jnp.all(jnp.abs(vals) <= 157)


def test_prior_mix_uniform_flattens_priors():
    pv_apply, params = _pv()
    states = _batch_states(2)
    _, _, legal = _pv_eval(pv_apply, params, states, 100.0)
    logits_flat, value_flat, _ = _pv_eval(
        pv_apply, params, states, 100.0, prior_mix_uniform=1.0)
    # λ=1: uniform over legal — all legal logits equal, illegal masked.
    probs = jax.nn.softmax(logits_flat, axis=-1)
    expected = legal / legal.sum(-1, keepdims=True)
    assert jnp.allclose(probs, expected, atol=1e-5)
    # Value path untouched by the prior knob.
    _, value_ref, _ = _pv_eval(pv_apply, params, states, 100.0)
    assert jnp.allclose(value_flat, value_ref)


def test_rollout_value_weight_full_replaces_value_head():
    pv_apply, params = _pv()
    states = _batch_states(2)
    key = jax.random.PRNGKey(3)
    logits_ref, _, _ = _pv_eval(pv_apply, params, states, 100.0)
    logits, value, _ = _pv_eval(
        pv_apply, params, states, 100.0,
        rollout_value_weight=1.0, rollout_key=key)
    # w=1: value is exactly the rollout return; priors untouched.
    assert jnp.allclose(value, _rollout_value(states, key))
    assert jnp.allclose(logits, logits_ref)


def test_puct_action_grounded_knobs_legal():
    """The classical (net-free) configuration searches and moves legally."""
    pv_apply, params = _pv()
    state = game.init(jax.random.PRNGKey(0))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    action = puct_action(state, state.current_player, jax.random.PRNGKey(0),
                         params, pv_apply,
                         num_determinizations=2, num_simulations=8,
                         search_variant="muzero",
                         prior_mix_uniform=1.0, rollout_value_weight=1.0)
    assert bool(game.legal_action_mask(state)[action])


def test_qsum_scores_mean_not_sum_under_negative_q():
    """From a losing position the well-searched least-bad move must win.

    Raw score-sum Σ N·Q prefers the lightly-visited terrible action
    (−40·2 = −80 beats −1·40 = −40): the bug behind the −61 probe
    reading of 2026-07-06. Visit-weighted mean Q must rank them right.
    """
    visits = jnp.array([[20.0, 1.0, 0.0], [20.0, 1.0, 0.0]])   # (K=2, A=3)
    qvalues = jnp.array([[-1.0, -40.0, 0.0], [-1.0, -40.0, 0.0]])
    legal = jnp.array([True, True, True])
    scores = _qsum_scores(visits, qvalues, legal)
    assert int(jnp.argmax(scores)) == 0
    assert jnp.isclose(scores[0], -1.0) and jnp.isclose(scores[1], -40.0)
    assert scores[2] == -jnp.inf                # unvisited, never argmax


def test_qsum_readout_action_legal_and_one_hot_pi():
    pv_apply, params = _pv()
    state = game.init(jax.random.PRNGKey(0))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))
    policy_fn = make_puct_policy_fn(
        pv_apply, params,
        num_determinizations=2, num_simulations=8,
        search_variant="muzero", readout="qsum")
    action, pi = jax.jit(policy_fn)(state, jax.random.PRNGKey(1))
    legal = game.legal_action_mask(state)
    assert bool(legal[action])
    assert float(pi.sum()) == 1.0 and float(pi[action]) == 1.0


def test_qsum_readout_rejects_temperature():
    pv_apply, params = _pv()
    with pytest.raises(ValueError, match="qsum"):
        make_puct_policy_fn(pv_apply, params,
                            readout="qsum", temperature=1.0)


def test_classical_qsum_sign_conventions_beat_random():
    """Net-free classical search read by Q-sum must clearly beat random.

    The load-bearing semantic check for the qsum readout: qvalues are
    aggregated in the root mover's perspective across the K trees, so a
    perspective or masking error would pick point-minimizing moves.
    Random-init net params: flat priors + rollout values use no net.
    """
    pv_apply, params = _pv()
    fn = make_puct_action_fn(pv_apply, params,
                             num_determinizations=2, num_simulations=16,
                             search_variant="muzero",
                             prior_mix_uniform=1.0, rollout_value_weight=1.0,
                             readout="qsum")
    scores = policy_match(fn, random_action_fn, jax.random.PRNGKey(0), 8)
    mean = float(scores.mean())
    assert mean > 5.0, f"classical qsum vs random {mean:+.1f}; sign error likely"


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
