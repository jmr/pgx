#!/usr/bin/env python3
"""
Extract PolicyValueNet parameters from a Flax msgpack checkpoint to a numpy .npz file.

Runs in the pgx venv (JAX 0.10.1 + Flax 0.12.7).  TensorFlow is NOT required.
The output .npz is then loaded by export_pv_savedmodel.py (TF-only env).

Usage:
    # Random-init weights (for pipeline testing):
    python scripts/extract_pv_weights.py --out /tmp/pv_random.npz

    # Real weights from Google Drive:
    python scripts/extract_pv_weights.py \
        --weights ~/Downloads/pv_gen3_s128.msgpack \
        --out /tmp/pv_gen3.npz

Output keys (one per Dense layer):
    Dense_{0..8}_kernel, Dense_{0..8}_bias
"""
import argparse
import sys
import os

# Run from the pgx repo root so pgx package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp
from pgx._src.games.jass_value_net import PolicyValueNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        metavar="MSGPACK",
        help="Path to .msgpack checkpoint (omit for random init)",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="NPZ",
        help="Output .npz path",
    )
    args = parser.parse_args()

    model = PolicyValueNet()
    key = jax.random.PRNGKey(0)
    # Build param template via init
    params = model.init(key, jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))

    if args.weights:
        import flax.serialization
        with open(args.weights, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
        print(f"Loaded weights from {args.weights}")
    else:
        print("Using random-init weights (no --weights supplied)")

    # The Flax parameter tree is: {'params': {'Dense_0': {'kernel': ..., 'bias': ...}, ...}}
    dense_params = params["params"]
    arrays: dict[str, np.ndarray] = {}
    print("Extracted layers:")
    for layer_name, layer_weights in sorted(dense_params.items()):
        for weight_name, array in layer_weights.items():
            key_name = f"{layer_name}_{weight_name}"
            np_array = np.array(array)
            arrays[key_name] = np_array
            print(f"  {key_name:25s}  {np_array.shape}")

    np.savez(args.out, **arrays)
    print(f"\nSaved {len(arrays)} arrays to {args.out}")


if __name__ == "__main__":
    main()
