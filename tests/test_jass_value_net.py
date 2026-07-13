import os

import jax
import jax.numpy as jnp
import optax
import pytest

from pgx._src.games.jass_value_net import (
    PolicyValueNet,
    PolicyValueNetAttn,
    _replicate,
    _shard_pv,
    decay_mask,
    make_pv_train_step,
    train_model,
    train_pv_model,
)

# Both classes share the (cm, hd) → (logits, value) contract.
PV_MODELS = [PolicyValueNet, PolicyValueNetAttn]


def _params_equal(a, b):
    return all(jnp.array_equal(x, y)
               for x, y in zip(jax.tree_util.tree_leaves(a),
                               jax.tree_util.tree_leaves(b)))


@pytest.mark.parametrize("model_cls", PV_MODELS)
def test_pv_net_shapes(model_cls):
    model = model_cls()
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


@pytest.mark.parametrize("model_cls", PV_MODELS)
def test_pv_train_step_learns(model_cls):
    model = model_cls()
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


@pytest.mark.parametrize("model_cls", PV_MODELS)
def test_pv_card_logits_use_global_context(model_cls):
    """Card logits must see global context, not just their own card's row.

    Target: the FIRST held card when header bit 0 is set, else the LAST
    held card. Per-row-only card logits (the Step 2 run 2 architecture)
    provably cannot separate these — both targets look identical from the
    single card's features — and stall at CE ≈ ln 2. With pooled context
    fed back into the card head (PolicyValueNet) or self-attention over
    the rows (PolicyValueNetAttn) this is learnable to near zero.
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

    model = model_cls()
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


@pytest.mark.parametrize("n", [64, 30])  # 30: exercises the zero-pad path
def test_pv_train_step_accum_matches_plain(n):
    """accum_steps>1 must produce the same losses and updates as plain."""
    model = PolicyValueNetAttn(num_layers=1)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    # SGD: params move by -lr*grad, so this compares the two paths'
    # gradients directly. (Adam would amplify float-noise on exactly-zero
    # gradients — e.g. the card-head output bias, softmax-invariant — to
    # ±lr updates with implementation-dependent sign.)
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    plain = make_pv_train_step(model, optimizer)
    accum = make_pv_train_step(model, optimizer, accum_steps=4)

    batch = _synthetic_pv_batch(jax.random.PRNGKey(1), n=n)
    p_a, o_a, loss_a, v_a, pl_a = plain(params, opt_state, *batch)
    p_b, o_b, loss_b, v_b, pl_b = accum(params, opt_state, *batch)

    assert jnp.allclose(loss_a, loss_b, rtol=1e-5)
    assert jnp.allclose(v_a, v_b, rtol=1e-5)
    assert jnp.allclose(pl_a, pl_b, rtol=1e-5)
    for x, y in zip(jax.tree_util.tree_leaves(p_a),
                    jax.tree_util.tree_leaves(p_b)):
        assert jnp.allclose(x, y, rtol=1e-4, atol=1e-6)


def test_pv_train_step_pmap_matches_plain():
    """The pmap'd step must produce the same update as the plain step.

    Runs on however many local devices exist (1 on CPU here — the psum /
    shard / replicate plumbing is still exercised; the multi-device case
    is covered by the subprocess test below).
    """
    model = PolicyValueNetAttn(num_layers=1)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.sgd(0.1)  # linear in grads — see the accum test
    opt_state = optimizer.init(params)
    plain = make_pv_train_step(model, optimizer)
    pstep = make_pv_train_step(model, optimizer, pmap_axis="dp")

    n_dev = jax.local_device_count()
    batch = _synthetic_pv_batch(jax.random.PRNGKey(1), n=64)
    r_params, r_opt = _replicate((params, opt_state), n_dev)

    p_a, _, loss_a, v_a, pl_a = plain(params, opt_state, *batch)
    p_b, _, loss_b, v_b, pl_b = pstep(r_params, r_opt,
                                      *_shard_pv(batch, n_dev))

    assert jnp.allclose(loss_a, loss_b[0], rtol=1e-5)
    assert jnp.allclose(v_a, v_b[0], rtol=1e-5)
    assert jnp.allclose(pl_a, pl_b[0], rtol=1e-5)
    for x, y in zip(jax.tree_util.tree_leaves(p_a),
                    jax.tree_util.tree_leaves(p_b)):
        assert jnp.allclose(x, y[0], rtol=1e-4, atol=1e-6)
        # params must stay replicated: every device applied the same update
        assert all(jnp.array_equal(y[0], y[d]) for d in range(n_dev))


def test_pv_data_parallel_multi_device():
    """Real multi-device equivalence + train_pv_model(data_parallel=True).

    XLA's device count is fixed at backend init, so a fresh interpreter
    with --xla_force_host_platform_device_count is the only way to get
    multiple devices on CPU.
    """
    import subprocess
    import sys
    import tempfile
    import textwrap

    code = textwrap.dedent("""
        import os
        os.environ["XLA_FLAGS"] = ("--xla_force_host_platform_device_count=4 "
                                   + os.environ.get("XLA_FLAGS", ""))
        os.environ["JAX_PLATFORMS"] = "cpu"
        import sys
        import jax
        import jax.numpy as jnp
        import optax
        from pgx._src.games.jass_value_net import (
            PolicyValueNetAttn, _replicate, _shard_pv,
            make_pv_train_step, train_pv_model)

        assert jax.local_device_count() == 4

        # 1) step equivalence across 4 devices (incl. pad: 30 % 4 != 0)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        n = 30
        cm = jax.random.bernoulli(k1, 0.2, (n, 36, 12))
        hd = jax.random.bernoulli(k2, 0.3, (n, 20))
        legal = jnp.zeros((n, 43), dtype=jnp.bool_).at[:, :36].set(True)
        pi = jax.nn.one_hot(cm[:, :, 0].argmax(axis=-1), 43)
        y = jax.random.uniform(k3, (n,), minval=-157, maxval=157)
        batch = (cm, hd, y, pi, legal, jnp.ones(n, jnp.float32))

        model = PolicyValueNetAttn(num_layers=1)
        params = model.init(jax.random.PRNGKey(0),
                            jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
        optimizer = optax.sgd(0.1)
        opt_state = optimizer.init(params)
        plain = make_pv_train_step(model, optimizer)
        pstep = make_pv_train_step(model, optimizer, pmap_axis="dp")
        r_params, r_opt = _replicate((params, opt_state), 4)

        p_a, _, loss_a, _, _ = plain(params, opt_state, *batch)
        p_b, _, loss_b, _, _ = pstep(r_params, r_opt, *_shard_pv(batch, 4))
        assert jnp.allclose(loss_a, loss_b[0], rtol=1e-5)
        for x, ys in zip(jax.tree_util.tree_leaves(p_a),
                         jax.tree_util.tree_leaves(p_b)):
            assert jnp.allclose(x, ys[0], rtol=1e-4, atol=1e-6)

        # 2) train_pv_model(data_parallel=True): runs, resumes, and its
        # checkpoints are single-device (loadable by a plain run).
        ckpt = sys.argv[1] + "/pv_dp_ckpt.msgpack"
        p_full, _ = train_pv_model(data_parallel=True, batch_size=4,
                                   num_epochs=4, print_every=100)
        train_pv_model(data_parallel=True, batch_size=4, num_epochs=2,
                       print_every=100, checkpoint_path=ckpt,
                       checkpoint_every=2)
        p_res, _ = train_pv_model(data_parallel=True, batch_size=4,
                                  num_epochs=4, print_every=100,
                                  checkpoint_path=ckpt, checkpoint_every=2)
        for x, y in zip(jax.tree_util.tree_leaves(p_full),
                        jax.tree_util.tree_leaves(p_res)):
            assert jnp.array_equal(x, y)
        assert jax.tree_util.tree_leaves(p_full)[0].ndim <= 2  # unreplicated

        print("MULTI_DEVICE_OK")
    """)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-c", code, tmp],
                           capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"stderr:\n{r.stderr[-3000:]}"
    assert "MULTI_DEVICE_OK" in r.stdout


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


def test_pv_train_step_head_masks_matches_legacy():
    """v_mask = p_mask = mask must reproduce the single-mask step exactly."""
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.sgd(0.1)  # linear in grads — see the accum test
    opt_state = optimizer.init(params)
    plain = make_pv_train_step(model, optimizer)
    hc = make_pv_train_step(model, optimizer, head_masks=True)

    cm, hd, y, pi, legal, mask = _synthetic_pv_batch(jax.random.PRNGKey(1))
    p_a, _, loss_a, v_a, pl_a = plain(params, opt_state,
                                      cm, hd, y, pi, legal, mask)
    p_b, _, loss_b, v_b, pl_b = hc(params, opt_state,
                                   cm, hd, y, pi, legal, mask, mask)
    assert jnp.allclose(loss_a, loss_b, rtol=1e-6)
    assert jnp.allclose(v_a, v_b, rtol=1e-6)
    assert jnp.allclose(pl_a, pl_b, rtol=1e-6)
    for x, z in zip(jax.tree_util.tree_leaves(p_a),
                    jax.tree_util.tree_leaves(p_b)):
        assert jnp.allclose(x, z, rtol=1e-5, atol=1e-7)


def test_pv_train_step_head_masks_isolate_heads():
    """Value-only rows ignore pi; policy-only rows ignore y (sop gen-11:
    world rows carry no valid outcome, true rows must not train the
    policy)."""
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step = make_pv_train_step(model, optimizer, head_masks=True)

    cm, hd, y, pi, legal, mask = _synthetic_pv_batch(jax.random.PRNGKey(2),
                                                     n=32)
    v_mask = mask.at[16:].set(0.0)   # first half: value rows
    p_mask = mask.at[:16].set(0.0)   # second half: policy rows
    p_ref, _, loss_ref, v_ref, pl_ref = step(
        params, opt_state, cm, hd, y, pi, legal, v_mask, p_mask)

    y_bad = y.at[16:].set(1e6)              # garbage outcome on policy rows
    pi_bad = pi.at[:16].set(jnp.roll(pi[:16], 1, axis=-1))  # on value rows
    p_b, _, loss_b, v_b, pl_b = step(
        params, opt_state, cm, hd, y_bad, pi_bad, legal, v_mask, p_mask)

    assert jnp.array_equal(loss_ref, loss_b)
    assert jnp.array_equal(v_ref, v_b) and jnp.array_equal(pl_ref, pl_b)
    for x, z in zip(jax.tree_util.tree_leaves(p_ref),
                    jax.tree_util.tree_leaves(p_b)):
        assert jnp.array_equal(x, z)


def _hc_synthetic_batch(key, n=64):
    """Synthetic hc rows: overlapping, unequal v/p masks."""
    cm, hd, y, pi, legal, mask = _synthetic_pv_batch(key, n=n)
    rows = jnp.arange(n)
    v_mask = mask * (rows % 3 == 0)          # every 3rd row: value
    p_mask = mask * (rows % 3 != 0)          # the rest: policy
    return cm, hd, y, pi, legal, v_mask, p_mask


@pytest.mark.parametrize("n", [64, 30])  # 30: exercises the zero-pad path
def test_pv_train_step_hc_accum_matches_plain(n):
    """head_masks accum_steps>1 must match the plain head_masks step."""
    model = PolicyValueNetAttn(num_layers=1)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    plain = make_pv_train_step(model, optimizer, head_masks=True)
    accum = make_pv_train_step(model, optimizer, accum_steps=4,
                               head_masks=True)

    batch = _hc_synthetic_batch(jax.random.PRNGKey(1), n=n)
    p_a, o_a, loss_a, v_a, pl_a = plain(params, opt_state, *batch)
    p_b, o_b, loss_b, v_b, pl_b = accum(params, opt_state, *batch)

    assert jnp.allclose(loss_a, loss_b, rtol=1e-5)
    assert jnp.allclose(v_a, v_b, rtol=1e-5)
    assert jnp.allclose(pl_a, pl_b, rtol=1e-5)
    for x, y_ in zip(jax.tree_util.tree_leaves(p_a),
                     jax.tree_util.tree_leaves(p_b)):
        assert jnp.allclose(x, y_, rtol=1e-4, atol=1e-6)


def test_pv_train_step_hc_pmap_matches_plain():
    """The head_masks pmap step must produce the same update as plain."""
    model = PolicyValueNetAttn(num_layers=1)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    plain = make_pv_train_step(model, optimizer, head_masks=True)
    pstep = make_pv_train_step(model, optimizer, pmap_axis="dp",
                               head_masks=True)

    n_dev = jax.local_device_count()
    batch = _hc_synthetic_batch(jax.random.PRNGKey(1), n=64)
    r_params, r_opt = _replicate((params, opt_state), n_dev)

    p_a, _, loss_a, v_a, pl_a = plain(params, opt_state, *batch)
    p_b, _, loss_b, v_b, pl_b = pstep(r_params, r_opt,
                                      *_shard_pv(batch, n_dev))

    assert jnp.allclose(loss_a, loss_b[0], rtol=1e-5)
    assert jnp.allclose(v_a, v_b[0], rtol=1e-5)
    assert jnp.allclose(pl_a, pl_b[0], rtol=1e-5)
    for x, y_ in zip(jax.tree_util.tree_leaves(p_a),
                     jax.tree_util.tree_leaves(p_b)):
        assert jnp.allclose(x, y_[0], rtol=1e-4, atol=1e-6)


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


def test_pv_train_model_head_masks_smoke():
    """train_pv_model end to end on the hc contract (sop gen-11)."""
    from pgx._src.games.jass_puct import make_puct_hc_collect_fn

    gen = PolicyValueNet()
    gen_params = gen.init(jax.random.PRNGKey(3),
                          jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_puct_hc_collect_fn(gen.apply, gen_params,
                                         num_world_rows=2,
                                         num_determinizations=2,
                                         num_simulations=4)
    params, model = train_pv_model(collect_fn=collect_fn, head_masks=True,
                                   batch_size=2, num_epochs=2,
                                   print_every=100)
    logits, value = model.apply(params, jnp.zeros((2, 36, 12)),
                                jnp.zeros((2, 20)))
    assert jnp.all(jnp.isfinite(logits)) and jnp.all(jnp.isfinite(value))


def test_pv_train_model_head_masks_requires_hc_collect():
    with pytest.raises(ValueError, match="head_masks"):
        train_pv_model(head_masks=True, batch_size=2, num_epochs=1)


def test_pv_train_model_optimizer_passthrough():
    params, model = train_pv_model(optimizer=optax.adamw(3e-4,
                                                         weight_decay=1e-2),
                                   batch_size=4, num_epochs=2,
                                   print_every=100)
    logits, value = model.apply(params, jnp.zeros((2, 36, 12)),
                                jnp.zeros((2, 20)))
    assert logits.shape == (2, 43)


def test_decay_mask_kernels_only():
    """decay_mask marks exactly the Dense/attention kernels for decay —
    biases, LayerNorm scale/bias, and pool_query are excluded."""
    model = PolicyValueNetAttn()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    flat, _ = jax.tree_util.tree_flatten_with_path(decay_mask(params))
    seen = set()
    for path, decayed in flat:
        name = path[-1].key
        seen.add(name)
        assert decayed == (name == "kernel"), (path, decayed)
    assert {"kernel", "bias", "scale", "pool_query"} <= seen


def test_pv_train_model_weight_decay(tmp_path):
    """weight_decay trains, shrinks kernels vs plain adam on the same
    stream, and its adamw opt_state checkpoints/resumes."""
    ckpt = str(tmp_path / "pv_wd_ckpt.msgpack")

    p_adam, _ = train_pv_model(batch_size=4, num_epochs=4, print_every=100)
    p_wd, model = train_pv_model(weight_decay=0.5, batch_size=4,
                                 num_epochs=4, print_every=100)
    logits, value = model.apply(p_wd, jnp.zeros((2, 36, 12)),
                                jnp.zeros((2, 20)))
    assert logits.shape == (2, 43)
    # Same data/RNG stream, so any kernel-norm drop is the decay term.
    def kernel_norm(p):
        flat, _ = jax.tree_util.tree_flatten_with_path(p)
        return sum(float(jnp.sum(v ** 2))
                   for path, v in flat if path[-1].key == "kernel")
    assert kernel_norm(p_wd) < kernel_norm(p_adam)

    train_pv_model(weight_decay=0.5, batch_size=4, num_epochs=2,
                   print_every=100, checkpoint_path=ckpt, checkpoint_every=2)
    p_resumed, _ = train_pv_model(weight_decay=0.5, batch_size=4,
                                  num_epochs=4, print_every=100,
                                  checkpoint_path=ckpt, checkpoint_every=2)
    assert _params_equal(p_wd, p_resumed)


def test_pv_train_model_weight_decay_and_optimizer_conflict():
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_pv_model(weight_decay=1e-2, optimizer=optax.adam(3e-4),
                       batch_size=4, num_epochs=1, print_every=100)


def test_pv_train_model_snapshots(tmp_path):
    """snapshot_every keeps params-only .ep{N} files; the epoch-N
    snapshot equals a fresh N-epoch run (same RNG stream), so an
    early-stopped net is a file copy, not a retrain."""
    import flax.serialization

    ckpt = str(tmp_path / "pv_snap_ckpt.msgpack")
    p4, _ = train_pv_model(batch_size=4, num_epochs=4, print_every=100,
                           checkpoint_path=ckpt, checkpoint_every=100,
                           snapshot_every=2)
    with open(ckpt + ".ep4", "rb") as f:
        snap4 = flax.serialization.from_bytes(p4, f.read())
    assert _params_equal(snap4, p4)

    p2, _ = train_pv_model(batch_size=4, num_epochs=2, print_every=100)
    with open(ckpt + ".ep2", "rb") as f:
        snap2 = flax.serialization.from_bytes(p2, f.read())
    assert _params_equal(snap2, p2)


def test_pv_snapshot_requires_checkpoint_path():
    with pytest.raises(ValueError, match="snapshot_every"):
        train_pv_model(batch_size=4, num_epochs=2, print_every=100,
                       snapshot_every=1)


def test_pv_train_model_attn(tmp_path):
    """train_pv_model(model=PolicyValueNetAttn()) trains and resumes."""
    ckpt = str(tmp_path / "pv_attn_ckpt.msgpack")

    p_full, model = train_pv_model(model=PolicyValueNetAttn(),
                                   batch_size=4, num_epochs=4, print_every=100)
    assert isinstance(model, PolicyValueNetAttn)
    logits, value = model.apply(p_full, jnp.zeros((2, 36, 12)),
                                jnp.zeros((2, 20)))
    assert logits.shape == (2, 43)
    assert value.shape == (2,)

    # Checkpoint resume must replay to the same weights (the checkpoint
    # template comes from the passed model, not the default architecture).
    train_pv_model(model=PolicyValueNetAttn(), batch_size=4, num_epochs=2,
                   print_every=100, checkpoint_path=ckpt, checkpoint_every=2)
    p_resumed, _ = train_pv_model(model=PolicyValueNetAttn(), batch_size=4,
                                  num_epochs=4, print_every=100,
                                  checkpoint_path=ckpt, checkpoint_every=2)
    assert _params_equal(p_full, p_resumed)


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
