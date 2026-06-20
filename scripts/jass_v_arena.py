"""
Arena: V-MCTS challenger vs random-rollout baseline (CLI wrapper).

Usage:
    # ValueNet weights (gen0/gen1):
    python scripts/jass_v_arena.py --weights jass_v_weights.msgpack

    # PolicyValueNet weights (gen2/gen3) vs random baseline:
    python scripts/jass_v_arena.py --weights pv_gen3_s128.msgpack --model-type pv

    # gen3 vs gen2 (direct model comparison):
    python scripts/jass_v_arena.py \
        --weights pv_gen3_s128.msgpack \
        --baseline-weights pv_gen2_s128.msgpack \
        --model-type pv --k-v 64 --k-base 64 --games 1000

Options:
    --weights           Path to challenger weights file (.msgpack)   [required]
    --baseline-weights  Optional .msgpack weights for V-MCTS baseline
                        (omit for random-rollout baseline)
    --model-type        v (ValueNet) or pv (PolicyValueNet)          [default: v]
    --k-v               Determinizations for V-MCTS challenger       [default: 64]
    --k-base            Determinizations for random-rollout baseline  [default: 8]
    --n-base            Rollouts per action for baseline              [default: 8]
    --games             Max games to play                             [default: 1000]
    --hours             Time budget in hours                          [default: 4]
    --seed              PRNG seed                                     [default: 0]
"""

import argparse

import flax.serialization
import jax
import jax.numpy as jnp

from pgx._src.games.jass_v_arena import run_arena
from pgx._src.games.jass_value_net import PolicyValueNet, ValueNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True,
                        help="Path to challenger .msgpack weights file")
    parser.add_argument("--baseline-weights", default=None,
                        help="Optional .msgpack weights for a V-MCTS baseline "
                             "(K=k_base); omit for the random-rollout baseline")
    parser.add_argument("--model-type", choices=["v", "pv"], default="v",
                        help="Network class: v=ValueNet (gen0/gen1), "
                             "pv=PolicyValueNet (gen2+)  [default: v]")
    parser.add_argument("--k-v",    type=int, default=64)
    parser.add_argument("--k-base", type=int, default=8)
    parser.add_argument("--n-base", type=int, default=8)
    parser.add_argument("--games",  type=int, default=1000)
    parser.add_argument("--hours",  type=float, default=4.0)
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    # ── Load weights ──────────────────────────────────────────────────────
    print(f"Model type: {args.model_type.upper()}  ({args.weights})", flush=True)

    if args.model_type == "pv":
        model = PolicyValueNet()
        dummy = model.init(jax.random.PRNGKey(0),
                           jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
        # Extract value head from the (logits, value) pair
        v_apply = lambda p, cm, hd: model.apply(p, cm, hd)[1]
    else:
        model = ValueNet()
        dummy = model.init(jax.random.PRNGKey(0),
                           jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
        v_apply = model.apply

    with open(args.weights, "rb") as f:
        params = flax.serialization.from_bytes(dummy, f.read())
    print(f"  Challenger loaded from {args.weights}\n", flush=True)

    baseline_params = None
    if args.baseline_weights is not None:
        with open(args.baseline_weights, "rb") as f:
            baseline_params = flax.serialization.from_bytes(dummy, f.read())
        print(f"  Baseline loaded from {args.baseline_weights}\n", flush=True)

    run_arena(params, baseline_params=baseline_params,
              k_v=args.k_v, k_base=args.k_base, n_base=args.n_base,
              games=args.games, hours=args.hours, seed=args.seed,
              v_apply=v_apply, baseline_v_apply=v_apply)
