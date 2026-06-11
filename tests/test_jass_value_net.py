import os

import jax
import jax.numpy as jnp

from pgx._src.games.jass_value_net import train_model


def _params_equal(a, b):
    return all(jnp.array_equal(x, y)
               for x, y in zip(jax.tree_util.tree_leaves(a),
                               jax.tree_util.tree_leaves(b)))


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
