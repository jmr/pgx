import os

import jax
import jax.numpy as jnp

from pgx._src.games.jass_value_net import train_model


def test_checkpoint_resume_is_equivalent(tmp_path):
    ckpt = str(tmp_path / "ckpt.msgpack")

    # Uninterrupted run: 4 epochs, no checkpointing.
    p_full, _ = train_model(batch_size=4, num_epochs=4, print_every=100)

    # Interrupted run: stop after 2 epochs (checkpoint written), then
    # resume from the checkpoint and finish.
    p_half, _ = train_model(batch_size=4, num_epochs=2, print_every=100,
                            checkpoint_path=ckpt, checkpoint_every=2)
    assert os.path.exists(ckpt)
    p_resumed, _ = train_model(batch_size=4, num_epochs=4, print_every=100,
                               checkpoint_path=ckpt, checkpoint_every=2)

    full = jax.tree_util.tree_leaves(p_full)
    resumed = jax.tree_util.tree_leaves(p_resumed)
    assert all(jnp.array_equal(a, b) for a, b in zip(full, resumed))

    # And the resumed run actually trained further than the checkpoint.
    half = jax.tree_util.tree_leaves(p_half)
    assert not all(jnp.array_equal(a, b) for a, b in zip(half, resumed))
