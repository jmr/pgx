import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgx._src.games.jass import Game, value_features
from pgx._src.games.jass_belief import (
    as_traj_action_fn,
    belief_policy_match,
    empty_trajectory,
    make_belief_fair_raw_action_fn,
    make_belief_puct_action_fn,
    make_belief_puct_collect_fn,
    make_belief_puct_policy_fn,
    make_belief_world_fn,
    record_step,
    run_belief_arena,
    world_log_likelihoods,
)
from pgx._src.games.jass_selfplay import random_action_fn
from pgx._src.games.jass_value_net import PolicyValueNet

game = Game()


def _pv(seed=1):
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(seed),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    return model.apply, params


def _play_traj(n_steps, seed=0):
    """Random-play n_steps of a game, recording the public trajectory."""
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    s = game.init(init_key)
    traj = empty_trajectory(s)
    for t in range(n_steps):
        key, ak = jax.random.split(key)
        mask = game.legal_action_mask(s)
        action = jax.random.categorical(
            ak, jnp.where(mask, 0.0, -1e9)).astype(jnp.int32)
        traj = record_step(traj, t, s, action)
        s = game.step(s, action)
    return s, traj


def test_trajectory_record():
    s, traj = _play_traj(6)
    assert traj.actions.shape == (38,)
    assert traj.valid.shape == (38,)
    assert bool(traj.valid[:6].all()) and not bool(traj.valid[6:].any())
    # Slot 0 holds the pre-move state of the first step: a fresh deal.
    assert int(traj.states.trick_num[0]) == 0
    assert int(traj.states.hands[0].sum()) == 36


def test_true_world_likelihood_matches_direct_eval():
    """Reconstruction check: under the TRUE world, the core's score equals
    evaluating the net directly on the recorded true states."""
    hc_apply, hc_params = _pv()
    s, traj = _play_traj(10)
    me = s.current_player
    mask = traj.valid & (traj.states.current_player != me)
    assert bool(mask.any())
    logl = world_log_likelihoods(hc_apply, hc_params, traj.states,
                                 traj.actions, mask, s.hands, s.hands[None])
    total = 0.0
    for t in range(10):
        if not bool(mask[t]):
            continue
        st = jax.tree_util.tree_map(lambda x: x[t], traj.states)
        cm, hd = value_features(st, st.current_player)
        logits, _ = hc_apply(hc_params, cm[None], hd[None])
        lm = game.legal_action_mask(st)
        lp = jax.nn.log_softmax(jnp.where(lm, logits[0], jnp.float32(-1e9)))
        total += float(lp[traj.actions[t]])
    assert np.isclose(float(logl[0]), total, atol=1e-4)


def test_world_fn_uniform_without_history():
    hc_apply, hc_params = _pv()
    s0 = game.init(jax.random.PRNGKey(0))
    world_fn = make_belief_world_fn(hc_apply, hc_params, num_particles=4)
    worlds, w = jax.jit(world_fn)(s0, empty_trajectory(s0),
                                  jax.random.PRNGKey(1))
    assert worlds.shape == (4, 4, 36) and w.shape == (4,)
    # No observed moves yet → exactly uniform weights.
    assert np.allclose(np.asarray(w), 0.25, atol=1e-5)
    # Each world keeps the mover's hand and is a valid deal.
    me = int(s0.current_player)
    assert np.all(np.asarray(worlds[:, me] == s0.hands[me]))
    assert np.all(np.asarray(worlds.sum(1)) == 1)
    assert np.all(np.asarray(worlds.sum(-1)) == 9)


def test_world_fn_mid_game_weights_normalized():
    hc_apply, hc_params = _pv()
    s, traj = _play_traj(10)
    world_fn = make_belief_world_fn(hc_apply, hc_params, num_particles=4)
    worlds, w = jax.jit(world_fn)(s, traj, jax.random.PRNGKey(1))
    w = np.asarray(w)
    assert np.all(np.isfinite(w)) and np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    # λ=1 overrides any evidence: exactly uniform.
    flat_fn = make_belief_world_fn(hc_apply, hc_params, num_particles=4,
                                   mix_uniform=1.0)
    _, wf = jax.jit(flat_fn)(s, traj, jax.random.PRNGKey(1))
    assert np.allclose(np.asarray(wf), 0.25, atol=1e-5)


def test_belief_fair_raw_action_legal_and_deterministic():
    pv_apply, pv_params = _pv()
    hc_apply, hc_params = _pv(seed=2)
    s, traj = _play_traj(10)
    fn = jax.jit(make_belief_fair_raw_action_fn(
        pv_apply, pv_params, hc_apply, hc_params, num_particles=2))
    a = fn(s, traj, jax.random.PRNGKey(3))
    b = fn(s, traj, jax.random.PRNGKey(3))
    assert bool(game.legal_action_mask(s)[a])
    assert int(a) == int(b)


def test_belief_puct_action_legal():
    pv_apply, pv_params = _pv()
    hc_apply, hc_params = _pv(seed=2)
    s, traj = _play_traj(10)
    fn = jax.jit(make_belief_puct_action_fn(
        pv_apply, pv_params, hc_apply, hc_params,
        num_particles=2, num_determinizations=2, num_simulations=8))
    a = fn(s, traj, jax.random.PRNGKey(3))
    assert bool(game.legal_action_mask(s)[a])


def test_belief_puct_qsum_rejects_temperature():
    pv_apply, pv_params = _pv()
    with pytest.raises(ValueError, match="qsum"):
        make_belief_puct_action_fn(pv_apply, pv_params, pv_apply, pv_params,
                                   readout="qsum", temperature=1.0)


def test_belief_puct_policy_fn_contract():
    pv_apply, pv_params = _pv()
    hc_apply, hc_params = _pv(seed=2)
    s, traj = _play_traj(10)
    fn = jax.jit(make_belief_puct_policy_fn(
        pv_apply, pv_params, hc_apply, hc_params,
        num_particles=2, num_determinizations=2, num_simulations=8,
        temperature=1.0))
    action, pi = fn(s, traj, jax.random.PRNGKey(3))
    legal = np.asarray(game.legal_action_mask(s))
    assert bool(legal[action])
    pi = np.asarray(pi)
    assert pi.shape == (43,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert np.all(pi[~legal] == 0)


def test_belief_collect_contract():
    pv_apply, pv_params = _pv()
    hc_apply, hc_params = _pv(seed=2)
    collect_fn = make_belief_puct_collect_fn(
        pv_apply, pv_params, hc_apply, hc_params,
        num_particles=2, num_determinizations=2, num_simulations=4)
    cm, hd, labels, pi, legal, alive = collect_fn(jax.random.PRNGKey(0), 2)
    B, T = 2, 38
    assert cm.shape == (B, T, 36, 12) and hd.shape == (B, T, 20)
    assert labels.shape == (B, T) and pi.shape == (B, T, 43)
    assert legal.shape == (B, T, 43) and alive.shape == (B, T)
    alive = np.asarray(alive)
    pi = np.asarray(pi)
    legal = np.asarray(legal)
    # Every game runs 37 or 38 steps (1–2 trump + 36 card plays).
    assert np.all(alive.sum(axis=1) >= 37)
    # Alive steps: pi is a distribution over the legal actions.
    assert np.allclose(pi[alive].sum(-1), 1.0, atol=1e-5)
    assert np.all(pi[alive][~legal[alive]] == 0)
    # Labels are the acting player's terminal differential: within a
    # game, every step's label is ± one magnitude (seat parity).
    labels = np.asarray(labels)
    for b in range(B):
        assert len(np.unique(np.abs(labels[b][alive[b]]))) == 1
    # Same key replays the same games.
    again = collect_fn(jax.random.PRNGKey(0), 2)
    assert np.array_equal(np.asarray(again[2]), labels)


def test_belief_policy_match_contract():
    pv_apply, pv_params = _pv()
    challenger = make_belief_fair_raw_action_fn(
        pv_apply, pv_params, pv_apply, pv_params, num_particles=2)
    baseline = as_traj_action_fn(random_action_fn)
    scores = belief_policy_match(challenger, baseline,
                                 jax.random.PRNGKey(0), 2)
    assert scores.shape == (4,)
    assert np.all(np.abs(np.asarray(scores)) <= 157)
    again = belief_policy_match(challenger, baseline,
                                jax.random.PRNGKey(0), 2)
    assert np.array_equal(np.asarray(scores), np.asarray(again))


def test_run_belief_arena_smoke(capsys):
    a = as_traj_action_fn(random_action_fn)
    scores = run_belief_arena(a, a, pairs=2, chunk_pairs=1, seed=0,
                              label_c="rand", label_b="rand")
    assert scores.shape == (4,)
    out = capsys.readouterr().out
    assert "Challenger" in out and "t-test" in out
