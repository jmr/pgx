"""
Hidden-hand sensitivity probe: does the net USE the hidden-hand inputs?

Usage:
    python scripts/jass_hidden_hand_probe.py --weights pv_gen9_s128.msgpack
    python scripts/jass_hidden_hand_probe.py --weights ... --games 256 --worlds 16

Thin CLI wrapper around pgx._src.games.jass_probes.hidden_hand_probe —
see that module for the method and the gen-9 reference numbers
(log entry 2026-07-12).
"""

import argparse

import flax.serialization
import jax
import jax.numpy as jnp

from pgx._src.games.jass_probes import (
    hidden_hand_probe,
    print_hidden_hand_report,
)
from pgx._src.games.jass_value_net import PolicyValueNetAttn


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to PolicyValueNetAttn .msgpack checkpoint")
    parser.add_argument("--games", type=int, default=128,
                        help="On-policy games to probe (default: 128)")
    parser.add_argument("--worlds", type=int, default=8,
                        help="Resampled worlds per position (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PolicyValueNetAttn()
    with open(args.weights, "rb") as f:
        variables = flax.serialization.msgpack_restore(f.read())
    fp = float(sum(jnp.abs(x).sum()
                   for x in jax.tree_util.tree_leaves(variables)))
    print(f"loaded {args.weights}  fingerprint={fp:.2f}")

    res = hidden_hand_probe(model.apply, variables,
                            games=args.games, worlds=args.worlds,
                            seed=args.seed)
    print_hidden_hand_report(res)


if __name__ == "__main__":
    main()
