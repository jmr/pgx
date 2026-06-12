import os

import jax
import jax.numpy as jnp
import optax

from pgx._src.games.jass_value_net import (
    PolicyValueNet,
    make_pv_train_step,
    train_model,
    train_pv_model,
)


def _params_equal(a, b):
    return all(jnp.array_equal(x, y)
               for x, y in zip(jax.tree_util.tree_leaves(a),
                               jax.tree_util.tree_leaves(b)))


def test_pv_net_shapes():
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    cm = jnp.zeros((5, 36, 12), dtype=jnp.bool_)
    hd = jnp.zeros((5, 20), dtype=jnp.bool_)
    logits, value = model.apply(params, cm, hd)
    assert logits.shape == (5, 43)
    assert value.shape == (5,)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.all(jnp.isfinite(value))


def _synthetic_pv_batch(key, n=64):
    """Random features with a learnable (input-dependent) action/value target."""
    k1, k2, k3 = jax.random.split(key, 3)
    cm = jax.random.bernoulli(k1, 0.2, (n, 36, 12))
    hd = jax.random.bernoulli(k2, 0.3, (n, 20))
    legal = jnp.zeros((n, 43), dtype=jnp.bool_).at[:, :36].set(True)
    # Target action: a deterministic function of the input features.
    target = cm[:, :, 0].argmax(axis=-1)                       # (n,) in [0, 36)
    pi = jax.nn.one_hot(target, 43)
    y = jax.random.uniform(k3, (n,), minval=-157, maxval=157)
    mask = jnp.ones(n, dtype=jnp.float32)
    return cm, hd, y, pi, legal, mask


def test_pv_train_step_learns():
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    step = make_pv_train_step(model, optimizer)

    batch = _synthetic_pv_batch(jax.random.PRNGKey(1))
    params, opt_state, loss0, v0, p0 = step(params, opt_state, *batch)
    for _ in range(200):
        params, opt_state, loss, v_loss, p_loss = step(params, opt_state, *batch)

    # Overfitting one fixed batch must drive both heads' losses down hard.
    assert float(p_loss) < 0.5 * float(p0)
    assert float(v_loss) < 0.5 * float(v0)
    assert float(loss) < float(loss0)


def test_pv_card_logits_use_global_context():
    """Card logits must see global context, not just their own card's row.

    Target: the FIRST held card when header bit 0 is set, else the LAST
    held card. Per-row-only card logits (the Step 2 run 2 architecture)
    provably cannot separate these — both targets look identical from the
    single card's features — and stall at CE ≈ ln 2. With pooled context
    fed back into the card head this is learnable to near zero.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    n = 256
    cm = jnp.zeros((n, 36, 12), dtype=jnp.bool_).at[:, :, 0].set(
        jax.random.bernoulli(k1, 0.25, (n, 36)))
    cm = cm.at[:, 0, 0].set(True).at[:, 35, 0].set(True)  # ≥2 held cards
    hd = jnp.zeros((n, 20), dtype=jnp.bool_).at[:, 0].set(
        jax.random.bernoulli(k2, 0.5, (n,)))

    held = cm[:, :, 0]
    first = jnp.argmax(held, axis=-1)
    last = 35 - jnp.argmax(held[:, ::-1], axis=-1)
    target = jnp.where(hd[:, 0], first, last)

    pi = jax.nn.one_hot(target, 43)
    legal = jnp.zeros((n, 43), dtype=jnp.bool_).at[:, :36].set(held)
    y = jnp.zeros(n)
    mask = jnp.ones(n)

    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    step = make_pv_train_step(model, optimizer)
    for _ in range(400):
        params, opt_state, _, _, p_loss = step(
            params, opt_state, cm, hd, y, pi, legal, mask)

    # Context-free card logits bottom out at ~ln 2 ≈ 0.69 on this task.
    assert float(p_loss) < 0.35, f"policy CE stuck at {float(p_loss):.3f}"


def test_pv_train_step_mask_zeroes_padding():
    """Padding steps (mask=0) must not contribute to the loss."""
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step = make_pv_train_step(model, optimizer)

    cm, hd, y, pi, legal, mask = _synthetic_pv_batch(jax.random.PRNGKey(2), n=32)
    # Corrupt the second half of the batch and mask it out.
    y_bad = y.at[16:].set(1e6)
    mask_half = mask.at[16:].set(0.0)
    _, _, loss_a, _, _ = step(params, opt_state, cm, hd, y, pi, legal, mask_half)
    _, _, loss_b, _, _ = step(params, opt_state, cm, hd, y_bad, pi, legal, mask_half)
    assert jnp.allclose(loss_a, loss_b)


def test_checkpoint_resume_is_equivalent(tmp_path):
    ckpt = str(tmp_path / "ckpt.msgpack")

    # Uninterrupted run: 4 epochs, no checkpointing.
    p_full, _ = train_model(batch_size=4, num_epochs=4, print_every=100)

    # Interrupted run: stop after 2 epochs (checkpoint written), then
    # resume from the checkpoint and finish.
    p_half, _ = train_model(batch_size=4, num_epochs=2, print_every=100,
                            checkpoint_path=ckpt, checkpoint_every=2)
    assert os.path.exists(ckpt + ".a") or os.path.exists(ckpt + ".b")
    p_resumed, _ = train_model(batch_size=4, num_epochs=4, print_every=100,
                               checkpoint_path=ckpt, checkpoint_every=2)

    assert _params_equal(p_full, p_resumed)
    # And the resumed run actually trained further than the checkpoint.
    assert not _params_equal(p_half, p_resumed)


def test_checkpoint_corrupt_slot_falls_back(tmp_path):
    ckpt = str(tmp_path / "ckpt.msgpack")

    # 4 epochs with checkpoints at 2 (slot .b) and 4 (slot .a).
    p4, _ = train_model(batch_size=4, num_epochs=4, print_every=100,
                        checkpoint_path=ckpt, checkpoint_every=2)
    assert os.path.exists(ckpt + ".a") and os.path.exists(ckpt + ".b")

    # Corrupt the newest slot (epoch 4, .a) as if preempted mid-write;
    # resume must fall back to the epoch-2 slot and retrain to the same
    # final weights.
    with open(ckpt + ".a", "wb") as f:
        f.write(b"truncated garbage")
    p_resumed, _ = train_model(batch_size=4, num_epochs=4, print_every=100,
                               checkpoint_path=ckpt, checkpoint_every=2)
    assert _params_equal(p4, p_resumed)


def test_pv_train_model_smoke():
    params, model = train_pv_model(batch_size=4, num_epochs=2, print_every=100)
    logits, value = model.apply(params, jnp.zeros((2, 36, 12)), jnp.zeros((2, 20)))
    assert logits.shape == (2, 43)
    assert value.shape == (2,)


def test_pv_train_model_round_robins_collect_fns():
    from pgx._src.games.jass_selfplay import collect_pv_batch

    calls = []

    def gen(name):
        def fn(key, batch_size):
            calls.append(name)
            return collect_pv_batch(key, batch_size)
        return fn

    train_pv_model(collect_fn=[gen("a"), gen("b")],
                   batch_size=2, num_epochs=4, print_every=100)
    # First call is the eval holdout (from the first = newest generator),
    # then epochs alternate a, b, a, b.
    assert calls == ["a", "a", "b", "a", "b"]

    calls.clear()
    train_pv_model(collect_fn=[gen("a"), gen("b")],
                   eval_collect_fn=gen("ev"),
                   batch_size=2, num_epochs=2, print_every=100)
    # Dedicated holdout generator; rotation untouched.
    assert calls == ["ev", "a", "b"]


def test_pv_checkpoint_resume_is_equivalent(tmp_path):
    ckpt = str(tmp_path / "pv_ckpt.msgpack")

    p_full, _ = train_pv_model(batch_size=4, num_epochs=4, print_every=100)
    train_pv_model(batch_size=4, num_epochs=2, print_every=100,
                   checkpoint_path=ckpt, checkpoint_every=2)
    p_resumed, _ = train_pv_model(batch_size=4, num_epochs=4, print_every=100,
                                  checkpoint_path=ckpt, checkpoint_every=2)
    assert _params_equal(p_full, p_resumed)


def test_checkpoint_legacy_bare_file_is_read(tmp_path):
    ckpt = str(tmp_path / "ckpt.msgpack")

    # Write a checkpoint, then move the newest slot to the bare path
    # (single-file layout written by older code or a manual stopgap).
    train_model(batch_size=4, num_epochs=2, print_every=100,
                checkpoint_path=ckpt, checkpoint_every=2)
    os.rename(ckpt + ".b", ckpt)

    p_resumed, _ = train_model(batch_size=4, num_epochs=4, print_every=100,
                               checkpoint_path=ckpt, checkpoint_every=2)
    p_full, _ = train_model(batch_size=4, num_epochs=4, print_every=100)
    assert _params_equal(p_full, p_resumed)
