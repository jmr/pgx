import jax
import jax.numpy as jnp

from pgx._src.games.jass_selfplay import collect_batch, make_v_collect_fn
from pgx._src.games.jass_value_net import TARGET_SCALE, ValueNet


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
