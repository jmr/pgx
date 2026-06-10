"""
Arena: V-MCTS challenger vs random-rollout baseline (CLI wrapper).

Usage:
    python scripts/jass_v_arena.py --weights jass_v_weights.msgpack

Options:
    --weights   Path to weights file saved by jass_value_net.py  [required]
    --k-v       Determinizations for V-MCTS challenger           [default: 64]
    --k-base    Determinizations for random-rollout baseline      [default: 8]
    --n-base    Rollouts per action for baseline                  [default: 8]
    --games     Max games to play                                 [default: 1000]
    --hours     Time budget in hours                              [default: 4]
    --seed      PRNG seed                                         [default: 0]
"""

import argparse

import flax.serialization
import jax
import jax.numpy as jnp

from pgx._src.games.jass_v_arena import run_arena
from pgx._src.games.jass_value_net import ValueNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True,
                        help="Path to .msgpack weights file from jass_value_net.py")
    parser.add_argument("--baseline-weights", default=None,
                        help="Optional .msgpack weights for a V-MCTS baseline "
                             "(K=k_base); omit for the random-rollout baseline")
    parser.add_argument("--k-v",    type=int, default=64)
    parser.add_argument("--k-base", type=int, default=8)
    parser.add_argument("--n-base", type=int, default=8)
    parser.add_argument("--games",  type=int, default=1000)
    parser.add_argument("--hours",  type=float, default=4.0)
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    # ── Load weights ──────────────────────────────────────────────────────
    print("Loading weights ...", flush=True)
    model = ValueNet()
    dummy = model.init(jax.random.PRNGKey(0),
                       jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    with open(args.weights, "rb") as f:
        params = flax.serialization.from_bytes(dummy, f.read())
    print(f"  Loaded from {args.weights}\n", flush=True)

    baseline_params = None
    if args.baseline_weights is not None:
        with open(args.baseline_weights, "rb") as f:
            baseline_params = flax.serialization.from_bytes(dummy, f.read())
        print(f"  Baseline weights from {args.baseline_weights}\n", flush=True)

    run_arena(params, baseline_params=baseline_params,
              k_v=args.k_v, k_base=args.k_base, n_base=args.n_base,
              games=args.games, hours=args.hours, seed=args.seed)
