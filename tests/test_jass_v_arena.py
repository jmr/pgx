import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass_v_arena import run_batched_arena
from pgx._src.games.jass_value_net import ValueNet


def _init_params(seed):
    model = ValueNet()
    return model.init(jax.random.PRNGKey(seed),
                      jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))


def test_run_batched_arena_smoke():
    """Tiny end-to-end run: V-MCTS vs rollout baseline, 2 swapped-deal pairs."""
    params = _init_params(0)
    scores = run_batched_arena(params, k_v=2, k_base=2, n_base=1,
                               games=4, chunk_pairs=1, seed=0)
    assert scores.shape == (4,)
    assert np.all(np.abs(scores) <= 157)
    assert np.all(np.isfinite(scores))


def test_run_batched_arena_v_vs_v_deterministic():
    """V-vs-V gating mode runs and is reproducible for a fixed seed."""
    params = _init_params(0)
    baseline = _init_params(1)
    a = run_batched_arena(params, baseline_params=baseline, k_v=2, k_base=2,
                          games=2, chunk_pairs=1, seed=3)
    b = run_batched_arena(params, baseline_params=baseline, k_v=2, k_base=2,
                          games=2, chunk_pairs=1, seed=3)
    assert a.shape == (2,)
    assert np.array_equal(a, b)
