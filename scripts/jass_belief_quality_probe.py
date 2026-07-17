"""
Belief-quality probe: how much world mass does Bayes-inverting the
hc policy buy? (The gate on belief-weighted determinization.)

Usage:
    python scripts/jass_belief_quality_probe.py --weights pv_gen11hc.msgpack
    python scripts/jass_belief_quality_probe.py --weights ... \
        --games 128 --particles 32 [--blind]

Thin CLI wrapper around pgx._src.games.jass_probes.belief_quality_probe —
see that module for the method and the pre-registered bar
(SOP "Belief-quality probe", 2026-07-15; payoff = 12.6·q̄).
"""

import argparse

import flax.serialization
import jax
import jax.numpy as jnp

from pgx._src.games.jass_probes import (
    belief_quality_probe,
    print_belief_quality_report,
    uniform_pv_apply,
)
from pgx._src.games.jass_selfplay import make_policy_action_fn
from pgx._src.games.jass_value_net import PolicyValueNetAttn


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to the likelihood net (gen-11hc) "
                             "PolicyValueNetAttn .msgpack checkpoint")
    parser.add_argument("--games", type=int, default=64,
                        help="Self-play games to probe (default: 64)")
    parser.add_argument("--particles", type=int, default=16,
                        help="Sampled worlds per decision, on top of the "
                             "injected true world (default: 16)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game-chunk", type=int, default=16,
                        help="Games per jitted call — the HBM knob "
                             "(default: 16)")
    parser.add_argument("--blind", action="store_true",
                        help="Legality-only baseline: same games (actor = "
                             "the loaded net, raw τ=1), but score worlds "
                             "with constant logits — isolates the world "
                             "mass bought by legal-move constraints alone")
    args = parser.parse_args()

    model = PolicyValueNetAttn()
    with open(args.weights, "rb") as f:
        variables = flax.serialization.msgpack_restore(f.read())
    fp = float(sum(jnp.abs(x).sum()
                   for x in jax.tree_util.tree_leaves(variables)))
    print(f"loaded {args.weights}  fingerprint={fp:.2f}")

    hc_apply, hc_params = model.apply, variables
    actor_action_fn = None
    if args.blind:
        actor_action_fn = make_policy_action_fn(model.apply, variables,
                                                temperature=1.0)
        hc_apply, hc_params = uniform_pv_apply, None

    res = belief_quality_probe(hc_apply, hc_params,
                               games=args.games, particles=args.particles,
                               seed=args.seed,
                               actor_action_fn=actor_action_fn,
                               game_chunk=args.game_chunk)
    print_belief_quality_report(res)


if __name__ == "__main__":
    main()
