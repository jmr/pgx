"""
Belief-weighted determinization arena: belief challenger vs uniform baseline.

The measurement gate of SOP "Belief-weighted determinization —
integration" (2026-07-17). Two arms:

    # fair PUCT (muzero K=16x64 both sides), belief- vs uniform-sampled:
    python scripts/jass_belief_arena.py --arm puct \
        --pv-weights pv_gen10.msgpack --hc-weights pv_gen11hc.msgpack \
        --pairs 300 --seed 0

    # belief-weighted fair raw (argmax) vs baseline raw tau=0.05:
    python scripts/jass_belief_arena.py --arm raw \
        --pv-weights pv_gen10.msgpack --hc-weights pv_gen11hc.msgpack

Thin CLI wrapper around pgx._src.games.jass_belief — see that module
for the filter and the fairness argument. Both sides play the
--pv-weights net unless --baseline-weights is given; --hc-weights is
the likelihood net (gen-11hc) used only by the challenger's filter.
Pre-register --particles / --mix-uniform before running the gate.
"""

import argparse

import flax.serialization
import jax
import jax.numpy as jnp

from pgx._src.games.jass_belief import (
    as_traj_action_fn,
    make_belief_fair_raw_action_fn,
    make_belief_puct_action_fn,
    run_belief_arena,
)
from pgx._src.games.jass_selfplay import make_policy_action_fn
from pgx._src.games.jass_value_net import PolicyValueNetAttn


def _load(path):
    with open(path, "rb") as f:
        variables = flax.serialization.msgpack_restore(f.read())
    fp = float(sum(jnp.abs(x).sum()
                   for x in jax.tree_util.tree_leaves(variables)))
    print(f"loaded {path}  fingerprint={fp:.2f}")
    return variables


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=("puct", "raw"), required=True)
    parser.add_argument("--pv-weights", required=True,
                        help="Challenger's PolicyValueNetAttn .msgpack "
                             "(search/policy net)")
    parser.add_argument("--hc-weights", required=True,
                        help="Likelihood net (gen-11hc) .msgpack")
    parser.add_argument("--baseline-weights", default=None,
                        help="Baseline net .msgpack (default: --pv-weights)")
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--chunk-pairs", type=int, default=10,
                        help="Pairs per jitted call — the HBM knob "
                             "(default: 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--particles", type=int, default=32,
                        help="Belief filter worlds N (default: 32)")
    parser.add_argument("--mix-uniform", type=float, default=0.0,
                        help="λ uniform share on the belief weights "
                             "(default: 0)")
    parser.add_argument("--dets", type=int, default=16,
                        help="puct arm: K determinizations (default: 16)")
    parser.add_argument("--sims", type=int, default=64,
                        help="puct arm: simulations per tree (default: 64)")
    parser.add_argument("--raw-temperature", type=float, default=0.05,
                        help="raw arm: baseline raw τ (default: 0.05; "
                             "challenger fair raw plays argmax)")
    args = parser.parse_args()

    model = PolicyValueNetAttn()
    pv_params = _load(args.pv_weights)
    hc_params = _load(args.hc_weights)
    base_params = (_load(args.baseline_weights)
                   if args.baseline_weights else pv_params)

    if args.arm == "puct":
        from pgx._src.games.jass_puct import make_puct_action_fn
        challenger = make_belief_puct_action_fn(
            model.apply, pv_params, model.apply, hc_params,
            num_particles=args.particles, mix_uniform=args.mix_uniform,
            num_determinizations=args.dets, num_simulations=args.sims)
        baseline = as_traj_action_fn(make_puct_action_fn(
            model.apply, base_params,
            num_determinizations=args.dets, num_simulations=args.sims))
        label_c = (f"belief PUCT K={args.dets}x{args.sims} "
                   f"N={args.particles} λ={args.mix_uniform}")
        label_b = f"uniform PUCT K={args.dets}x{args.sims}"
    else:
        challenger = make_belief_fair_raw_action_fn(
            model.apply, pv_params, model.apply, hc_params,
            num_particles=args.particles, mix_uniform=args.mix_uniform)
        baseline = as_traj_action_fn(make_policy_action_fn(
            model.apply, base_params, temperature=args.raw_temperature))
        label_c = (f"belief fair-raw argmax N={args.particles} "
                   f"λ={args.mix_uniform}")
        label_b = f"raw τ={args.raw_temperature}"

    run_belief_arena(challenger, baseline, pairs=args.pairs,
                     chunk_pairs=args.chunk_pairs, seed=args.seed,
                     label_c=label_c, label_b=label_b)


if __name__ == "__main__":
    main()
