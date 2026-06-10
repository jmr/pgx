"""
Value-network diagnostics (CLI wrapper).

Usage:
    python scripts/jass_v_diagnostic.py --weights jass_v_weights.msgpack
    python scripts/jass_v_diagnostic.py --weights jass_v_weights.msgpack --save diag.png

Diagnostics:
    1. Calibration:  scatter predicted vs actual on random holdout games.
    2. Perspective:   V from all 4 seats on a mid-game state (sanity check).
"""

import argparse

import flax.serialization
import jax
import jax.numpy as jnp

from pgx._src.games.jass_v_diagnostic import run_calibration, perspective_probe
from pgx._src.games.jass_value_net import ValueNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to .msgpack weights file")
    parser.add_argument("--save", default=None,
                        help="Path to save calibration plot (default: /tmp/jass_v_calibration.png)")
    parser.add_argument("--n-games", type=int, default=2048,
                        help="Holdout games for calibration (default: 2048)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load weights ──────────────────────────────────────────────────────
    print("Loading weights ...", flush=True)
    model = ValueNet()
    dummy = model.init(jax.random.PRNGKey(0),
                       jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    with open(args.weights, "rb") as f:
        params = flax.serialization.from_bytes(dummy, f.read())
    print(f"  Loaded from {args.weights}\n", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, k1, k2 = jax.random.split(key, 3)

    r2, fig = run_calibration(params, model, k1, n_games=args.n_games)
    path = args.save or "/tmp/jass_v_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved calibration plot to {path}")

    perspective_probe(params, model, k2)
