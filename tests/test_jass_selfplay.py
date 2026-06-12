import jax
import jax.numpy as jnp

from pgx._src.games.jass_selfplay import (
    collect_batch,
    make_policy_action_fn,
    make_policy_collect_fn,
    make_search_collect_fn,
    make_v_action_fn,
    make_v_collect_fn,
    policy_match,
    random_action_fn,
)
from pgx._src.games.jass_value_net import TARGET_SCALE, PolicyValueNet, ValueNet


B = 4
T = 38  # _MAX_STEPS


def _check_batch(cm, hd, labels, alive):
    assert cm.shape == (B, T, 36, 12)
    assert hd.shape == (B, T, 20)
    assert labels.shape == (B, T)
    assert alive.shape == (B, T)
    assert cm.dtype == jnp.bool_
    assert hd.dtype == jnp.bool_

    for b in range(B):
        n_alive = int(alive[b].sum())
        # 36 card plays + 1 or 2 trump-selection steps (Schiebe).
        assert n_alive in (37, 38)
        # alive is a prefix: no revival after the game ends.
        assert bool(alive[b, :n_alive].all())
        assert not bool(alive[b, n_alive:].any())

        y = labels[b][alive[b]]
        # Every step is labeled with the acting player's terminal
        # differential: same magnitude all game, range [-157, 157].
        assert jnp.all(jnp.abs(y) == jnp.abs(y[0]))
        assert jnp.abs(y[0]) <= 157


def test_collect_batch_shapes_and_labels():
    cm, hd, labels, alive = collect_batch(jax.random.PRNGKey(0), B)
    _check_batch(cm, hd, labels, alive)


def test_v_collect_fn_shapes_and_labels():
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_v_collect_fn(model.apply, params, v_scale=TARGET_SCALE)
    cm, hd, labels, alive = collect_fn(jax.random.PRNGKey(0), B)
    _check_batch(cm, hd, labels, alive)


def test_v_collect_fn_deterministic_and_param_sensitive():
    model = ValueNet()
    p1 = model.init(jax.random.PRNGKey(1),
                    jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    p2 = model.init(jax.random.PRNGKey(2),
                    jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    f1 = make_v_collect_fn(model.apply, p1, v_scale=TARGET_SCALE,
                           temperature=1.0)
    f2 = make_v_collect_fn(model.apply, p2, v_scale=TARGET_SCALE,
                           temperature=1.0)

    a = f1(jax.random.PRNGKey(0), B)
    b = f1(jax.random.PRNGKey(0), B)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))

    # Different weights play differently (low temperature, same key/deals).
    c = f2(jax.random.PRNGKey(0), B)
    assert not all(jnp.array_equal(x, y) for x, y in zip(a, c))


def test_policy_match_random_vs_random_is_balanced():
    scores = policy_match(random_action_fn, random_action_fn,
                          jax.random.PRNGKey(0), 128)
    assert scores.shape == (256,)
    # Same policy on both sides: no edge beyond pair-cancelled noise.
    pair_means = scores.reshape(-1, 2).mean(axis=1)
    assert abs(float(pair_means.mean())) < 15.0
    # Scores are valid differentials.
    assert jnp.all(jnp.abs(scores) <= 157)


def test_policy_match_v_policy_runs_and_is_deterministic():
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    v_fn = make_v_action_fn(model.apply, params, v_scale=TARGET_SCALE,
                            temperature=1.0)
    a = policy_match(v_fn, random_action_fn, jax.random.PRNGKey(0), 4)
    b = policy_match(v_fn, random_action_fn, jax.random.PRNGKey(0), 4)
    assert a.shape == (8,)
    assert jnp.array_equal(a, b)


def _check_pv_batch(cm, hd, labels, pi, legal, alive):
    _check_batch(cm, hd, labels, alive)
    assert pi.shape == (B, T, 43)
    assert legal.shape == (B, T, 43)
    assert pi.dtype == jnp.float32
    assert legal.dtype == jnp.bool_
    # pi is a distribution supported on legal actions (alive steps only).
    live = alive[..., None]
    assert jnp.allclose(jnp.where(alive, pi.sum(-1), 1.0), 1.0)
    assert not jnp.any((pi > 0) & ~legal & live)
    # Every alive step has at least one legal action.
    assert jnp.all(legal.any(-1) | ~alive)


def _pv_params(seed=1):
    model = PolicyValueNet()
    return model, model.init(jax.random.PRNGKey(seed),
                             jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))


def test_search_collect_fn_contract():
    collect_fn = make_search_collect_fn(num_determinizations=2,
                                        num_rollouts=1, temperature=10.0)
    out = collect_fn(jax.random.PRNGKey(0), B)
    _check_pv_batch(*out)
    # Greedy search variant too (one-hot pi by construction).
    greedy_fn = make_search_collect_fn(num_determinizations=2, num_rollouts=1)
    out = greedy_fn(jax.random.PRNGKey(0), B)
    _check_pv_batch(*out)


def test_search_collect_fn_with_v_leaf():
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_search_collect_fn(model.apply, params,
                                        num_determinizations=2,
                                        num_rollouts=1,
                                        v_scale=TARGET_SCALE)
    _check_pv_batch(*collect_fn(jax.random.PRNGKey(0), B))


def test_policy_collect_fn_contract_and_determinism():
    model, params = _pv_params()
    collect_fn = make_policy_collect_fn(model.apply, params)
    a = collect_fn(jax.random.PRNGKey(0), B)
    _check_pv_batch(*a)
    b = collect_fn(jax.random.PRNGKey(0), B)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))


def test_policy_action_fn_in_policy_match():
    model, params = _pv_params()
    p_fn = make_policy_action_fn(model.apply, params, temperature=1.0)
    scores = policy_match(p_fn, random_action_fn, jax.random.PRNGKey(0), 4)
    assert scores.shape == (8,)
    assert jnp.all(jnp.abs(scores) <= 157)


def test_v_collect_matches_random_play_distribution_contract():
    # The two generators must be drop-in interchangeable for train_model:
    # identical pytree structure, shapes, and dtypes.
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_v_collect_fn(model.apply, params, v_scale=TARGET_SCALE)
    rand = collect_batch(jax.random.PRNGKey(0), B)
    vsel = collect_fn(jax.random.PRNGKey(0), B)
    for r, v in zip(rand, vsel):
        assert r.shape == v.shape
        assert r.dtype == v.dtype
