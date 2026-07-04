#!/usr/bin/env python3
"""
Extract PolicyValueNet / PolicyValueNetAttn parameters from a Flax msgpack
checkpoint to a numpy .npz file.

Runs in the pgx venv (JAX 0.10.1 + Flax 0.12.7).  TensorFlow is NOT required.
The output .npz is then loaded by export_pv_savedmodel.py (TF-only env).

Checkpoints are loaded template-free via msgpack_restore, so no architecture
flag is needed for real weights — from_bytes with the wrong template would
silently reshape the tree (see the Dense_4 trap).  --arch only selects the
random-init architecture when --weights is omitted.

Usage:
    # Random-init weights (for pipeline testing):
    python scripts/extract_pv_weights.py --arch attn --out /tmp/pv_random.npz

    # Real weights:
    python scripts/extract_pv_weights.py \
        --weights ~/Downloads/pv_gen7b_es_s128.msgpack \
        --out /tmp/pv_gen7b.npz

Output keys are the flattened Flax param paths joined with "_":
    MLP:  Dense_{0..8}_kernel, Dense_{0..8}_bias
    Attn: additionally LayerNorm_{i}_{scale,bias},
          MultiHeadDotProductAttention_{i}_{query,key,value,out}_{kernel,bias},
          pool_query
"""
import argparse
import sys
import os

# Run from the pgx repo root so pgx package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def flatten(tree, prefix="") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, value in sorted(tree.items()):
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            out.update(flatten(value, prefix=f"{key}_"))
        else:
            out[key] = np.array(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        metavar="MSGPACK",
        help="Path to .msgpack checkpoint (omit for random init)",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "attn"],
        default="mlp",
        help="Architecture for random init (ignored when --weights is given)",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="NPZ",
        help="Output .npz path",
    )
    args = parser.parse_args()

    if args.weights:
        import flax.serialization
        with open(args.weights, "rb") as f:
            params = flax.serialization.msgpack_restore(f.read())
        print(f"Loaded weights from {args.weights}")
    else:
        import jax
        import jax.numpy as jnp
        from pgx._src.games.jass_value_net import (
            PolicyValueNet,
            PolicyValueNetAttn,
        )
        model = PolicyValueNetAttn() if args.arch == "attn" else PolicyValueNet()
        params = model.init(
            jax.random.PRNGKey(0), jnp.zeros((1, 36, 12)), jnp.zeros((1, 20))
        )
        print(f"Using random-init {args.arch} weights (no --weights supplied)")

    arrays = flatten(params["params"])
    arch = "attn" if "pool_query" in arrays else "mlp"
    print(f"Detected architecture: {arch}")
    print("Extracted layers:")
    for key_name, np_array in arrays.items():
        print(f"  {key_name:55s}  {np_array.shape}")

    np.savez(args.out, **arrays)
    print(f"\nSaved {len(arrays)} arrays to {args.out}")


if __name__ == "__main__":
    main()
