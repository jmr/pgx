#!/usr/bin/env python3
"""
Export PolicyValueNet / PolicyValueNetAttn as a TF2 SavedModel for
TensorFlow-Java.

Run this in the .venv_tf environment (Python 3.13, tensorflow==2.20.*).
See scripts/README.md for setup instructions.
It does NOT require JAX or Flax.  Weights come from a .npz file produced by
extract_pv_weights.py (or are random-initialised when --weights is omitted).

The Keras models are faithful reimplementations of
    pgx/_src/games/jass_value_net.py :: PolicyValueNet / PolicyValueNetAttn
using tf.nn.gelu(approximate=True) to match Flax's default approximate GELU
and LayerNormalization(epsilon=1e-6) to match Flax's nn.LayerNorm default.
The architecture is inferred from the .npz keys (pool_query ⇒ attn);
--arch only selects the random-init architecture when --weights is omitted.

Usage:
    # Random-init (pipeline smoke-test — no real weights needed):
    .venv_tf/bin/python scripts/export_pv_savedmodel.py \
        --out ../JassTheRipper/src/main/resources/models/pgx_pv/export

    # Real weights:
    .venv_tf/bin/python scripts/export_pv_savedmodel.py \
        --weights /tmp/pv_gen7b.npz \
        --out ../JassTheRipper/src/main/resources/models/pgx_pv/export

Serving signature:
    Inputs:  cm  (None, 36, 12) float32  — card matrix
             hd  (None, 20)     float32  — header
    Outputs: logits  (None, 43) float32  — unmasked action logits (cards 0-35, trump 36-42)
             value   (None,)    float32  — predicted score differential / 100.0
"""
import argparse
import os
import numpy as np
import tensorflow as tf

# ── Architecture constants ────────────────────────────────────────────────────

HIDDEN = 128
NUM_CARD_LOGITS = 36
NUM_TRUMP_LOGITS = 7   # modes ♦ ♥ ♠ ♣ Obenabe Undeufe Schiebe
NUM_LOGITS = NUM_CARD_LOGITS + NUM_TRUMP_LOGITS  # 43

# Static card identity: (36, 13) = suit one-hot (4) + rank one-hot (9).
# Suit order: ♦=0 ♥=1 ♠=2 ♣=3  (matches pgx and Java Color.getValue()).
# Rank idx:  6→0  7→1  8→2  9→3  10→4  J→5  Q→6  K→7  A→8
_suits = np.repeat(np.arange(4), 9).astype(np.int32)   # (36,)
_ranks = np.tile(np.arange(9), 4).astype(np.int32)      # (36,)
CARD_IDENTITY = np.concatenate(
    [np.eye(4, dtype=np.float32)[_suits],
     np.eye(9, dtype=np.float32)[_ranks]],
    axis=-1,
)  # (36, 13)


# ── Model ─────────────────────────────────────────────────────────────────────

class PolicyValueNet(tf.keras.Model):
    """TF/Keras reimplementation of pgx PolicyValueNet.

    call() takes a single dict {"cm": ..., "hd": ...} so that Keras 3's
    auto-generated serving_default (which calls self(inputs) with one arg)
    works correctly, while still allowing named cm/hd inputs in the export.

    Forward pass mirrors jass_value_net.py::PolicyValueNet.__call__ exactly:
      1. Prepend static card identity (36,13) to each row → (B,36,25).
      2. Two-layer per-card trunk (Dense→gelu)×2 → (B,36,128).
      3. Mean-pool over cards + concat header → y (B,148).
      4. Card head: dense global context, concat per-card, dense→card_logits (B,36).
      5. Value head: dense→gelu→Dense(1) → value (B,).
      6. Trump head: dense→gelu→Dense(7) → trump_logits (B,7).
      7. Concat card_logits + trump_logits → logits (B,43).
    """

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self._card_identity = tf.constant(CARD_IDENTITY)   # (36,13)
        h = hidden
        # Trunk (per-card MLP, shared over all 36 rows via leading batch dims)
        self.d0 = tf.keras.layers.Dense(h, name="Dense_0")
        self.d1 = tf.keras.layers.Dense(h, name="Dense_1")
        # Card head
        self.d2 = tf.keras.layers.Dense(h, name="Dense_2")   # global context
        self.d3 = tf.keras.layers.Dense(h, name="Dense_3")   # per-card + context
        self.d4 = tf.keras.layers.Dense(1, name="Dense_4")   # card logit
        # Value head
        self.d5 = tf.keras.layers.Dense(h, name="Dense_5")
        self.d6 = tf.keras.layers.Dense(1, name="Dense_6")
        # Trump head
        self.d7 = tf.keras.layers.Dense(h, name="Dense_7")
        self.d8 = tf.keras.layers.Dense(NUM_TRUMP_LOGITS, name="Dense_8")

    def call(self, inputs: dict) -> dict[str, tf.Tensor]:  # type: ignore[override]
        x = tf.cast(inputs["cm"], tf.float32)   # (B, 36, 12)
        h = tf.cast(inputs["hd"], tf.float32)   # (B, 20)

        # Prepend static card identity (36, 13) → broadcast over batch
        B = tf.shape(x)[0]
        ident = tf.broadcast_to(self._card_identity[tf.newaxis], [B, 36, 13])
        x = tf.concat([x, ident], axis=-1)          # (B, 36, 25)

        # Trunk
        x = tf.nn.gelu(self.d0(x), approximate=True)   # (B, 36, 128)
        x = tf.nn.gelu(self.d1(x), approximate=True)   # (B, 36, 128)

        pooled = tf.reduce_mean(x, axis=1)              # (B, 128)
        y = tf.concat([pooled, h], axis=-1)             # (B, 148)

        # Card head: each card row sees its trunk features + broadcast global ctx
        ctx = tf.nn.gelu(self.d2(y), approximate=True)  # (B, 128)
        ctx_rows = tf.broadcast_to(
            ctx[:, tf.newaxis, :], [B, 36, HIDDEN]
        )                                                 # (B, 36, 128)
        c = tf.concat([x, ctx_rows], axis=-1)            # (B, 36, 256)
        c = tf.nn.gelu(self.d3(c), approximate=True)     # (B, 36, 128)
        card_logits = tf.squeeze(self.d4(c), axis=-1)    # (B, 36)

        # Value head
        v = tf.nn.gelu(self.d5(y), approximate=True)
        value = tf.squeeze(self.d6(v), axis=-1)           # (B,)

        # Trump head
        t = tf.nn.gelu(self.d7(y), approximate=True)
        trump_logits = self.d8(t)                         # (B, 7)

        logits = tf.concat([card_logits, trump_logits], axis=-1)  # (B, 43)
        return {"logits": logits, "value": value}


class PolicyValueNetAttn(tf.keras.Model):
    """TF/Keras reimplementation of pgx PolicyValueNetAttn.

    Same dict-input call() convention as PolicyValueNet above.  Forward pass
    mirrors jass_value_net.py::PolicyValueNetAttn.__call__ exactly:
      1. Concat card matrix + static identity + broadcast header → (B,36,45).
      2. Dense embed → (B,36,128).
      3. num_layers pre-LN transformer blocks (MHA + gelu MLP, residual).
      4. Final LayerNorm → rows (B,36,128).
      5. Learned-query attention pool over rows → pooled (B,128).
      6. y = concat(pooled, header) (B,148).
      7. Card head: dense→gelu on rows, dense→card_logits (B,36).
      8. Value head: dense→gelu→Dense(1) → value (B,).
      9. Trump head: dense→gelu→Dense(7) → trump_logits (B,7).

    Flax parity notes: LayerNormalization uses epsilon=1e-6 (Flax nn.LayerNorm
    default; Keras default is 1e-3).  Keras MultiHeadAttention and Flax
    MultiHeadDotProductAttention share kernel layouts — q/k/v (in, heads,
    head_dim), out (heads, head_dim, out) — and both scale scores by
    1/sqrt(head_dim).
    """

    def __init__(self, hidden: int = HIDDEN, num_heads: int = 4,
                 num_layers: int = 2) -> None:
        super().__init__()
        self._card_identity = tf.constant(CARD_IDENTITY)   # (36,13)
        self._hidden = hidden
        h = hidden
        self.embed = tf.keras.layers.Dense(h, name="Dense_0")
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append({
                "ln_attn": tf.keras.layers.LayerNormalization(
                    epsilon=1e-6, name=f"LayerNorm_{2 * i}"),
                "attn": tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=h // num_heads,
                    name=f"MultiHeadDotProductAttention_{i}"),
                "ln_mlp": tf.keras.layers.LayerNormalization(
                    epsilon=1e-6, name=f"LayerNorm_{2 * i + 1}"),
                "mlp_up": tf.keras.layers.Dense(
                    2 * h, name=f"Dense_{2 * i + 1}"),
                "mlp_down": tf.keras.layers.Dense(
                    h, name=f"Dense_{2 * i + 2}"),
            })
        self.ln_final = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"LayerNorm_{2 * num_layers}")
        self.pool_query = self.add_weight(
            name="pool_query", shape=(1, h), trainable=True)
        self.pool_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=h // num_heads,
            name=f"MultiHeadDotProductAttention_{num_layers}")
        d = 2 * num_layers  # Dense numbering continues after the blocks
        # Card head
        self.card_hidden = tf.keras.layers.Dense(h, name=f"Dense_{d + 1}")
        self.card_out = tf.keras.layers.Dense(1, name=f"Dense_{d + 2}")
        # Value head
        self.value_hidden = tf.keras.layers.Dense(h, name=f"Dense_{d + 3}")
        self.value_out = tf.keras.layers.Dense(1, name=f"Dense_{d + 4}")
        # Trump head
        self.trump_hidden = tf.keras.layers.Dense(h, name=f"Dense_{d + 5}")
        self.trump_out = tf.keras.layers.Dense(
            NUM_TRUMP_LOGITS, name=f"Dense_{d + 6}")

    def call(self, inputs: dict) -> dict[str, tf.Tensor]:  # type: ignore[override]
        x = tf.cast(inputs["cm"], tf.float32)   # (B, 36, 12)
        h = tf.cast(inputs["hd"], tf.float32)   # (B, 20)

        B = tf.shape(x)[0]
        ident = tf.broadcast_to(self._card_identity[tf.newaxis], [B, 36, 13])
        h_rows = tf.broadcast_to(h[:, tf.newaxis, :], [B, 36, 20])
        x = tf.concat([x, ident, h_rows], axis=-1)      # (B, 36, 45)

        x = self.embed(x)                               # (B, 36, 128)

        for blk in self.blocks:
            y = blk["ln_attn"](x)
            y = blk["attn"](query=y, value=y)
            x = x + y
            y = blk["ln_mlp"](x)
            y = tf.nn.gelu(blk["mlp_up"](y), approximate=True)
            y = blk["mlp_down"](y)
            x = x + y

        rows = self.ln_final(x)                         # (B, 36, 128)

        q = tf.broadcast_to(
            self.pool_query[tf.newaxis], [B, 1, self._hidden])
        pooled = tf.squeeze(
            self.pool_attn(query=q, value=rows), axis=1)  # (B, 128)

        y = tf.concat([pooled, h], axis=-1)             # (B, 148)

        c = tf.nn.gelu(self.card_hidden(rows), approximate=True)
        card_logits = tf.squeeze(self.card_out(c), axis=-1)   # (B, 36)

        v = tf.nn.gelu(self.value_hidden(y), approximate=True)
        value = tf.squeeze(self.value_out(v), axis=-1)         # (B,)

        t = tf.nn.gelu(self.trump_hidden(y), approximate=True)
        trump_logits = self.trump_out(t)                       # (B, 7)

        logits = tf.concat([card_logits, trump_logits], axis=-1)  # (B, 43)
        return {"logits": logits, "value": value}


# ── Weight loading ────────────────────────────────────────────────────────────

_LAYER_MAP = {
    "Dense_0": "d0", "Dense_1": "d1",
    "Dense_2": "d2", "Dense_3": "d3", "Dense_4": "d4",
    "Dense_5": "d5", "Dense_6": "d6",
    "Dense_7": "d7", "Dense_8": "d8",
}


def load_weights_from_npz(model: PolicyValueNet, npz_path: str) -> None:
    data = np.load(npz_path)
    print("Loading weights:")
    for layer_name, attr_name in _LAYER_MAP.items():
        layer: tf.keras.layers.Dense = getattr(model, attr_name)
        kernel_key = f"{layer_name}_kernel"
        bias_key = f"{layer_name}_bias"
        if kernel_key not in data or bias_key not in data:
            print(f"  WARNING: {layer_name} not found in npz — using random init")
            continue
        kernel = data[kernel_key]
        bias = data[bias_key]
        layer.set_weights([kernel, bias])
        print(f"  {layer_name:10s}  kernel {kernel.shape}  bias {bias.shape}")


# Keras path segment / variable name → Flax npz key segment
_SEGMENT_MAP = {
    "attention_output": "out",  # MHA output projection sublayer
    "gamma": "scale",           # LayerNormalization scale
    "beta": "bias",             # LayerNormalization bias
}


def load_attn_weights_from_npz(model: PolicyValueNetAttn, npz_path: str) -> None:
    """Assign every model variable from the npz by its (mapped) weight path.

    Keras paths like 'policy_value_net_attn/MultiHeadDotProductAttention_0/
    query/kernel' map to npz keys like 'MultiHeadDotProductAttention_0_query_
    kernel' (segments joined with '_', _SEGMENT_MAP applied, model-name prefix
    dropped).  Assigning by path instead of set_weights() makes the load
    independent of Keras's internal weight ordering.
    """
    data = np.load(npz_path)
    unused = set(data.keys())
    print("Loading weights:")
    for w in model.weights:
        segments = [_SEGMENT_MAP.get(s, s) for s in w.path.split("/")]
        # Drop leading segments (model name, etc.) until the key matches.
        key = None
        for start in range(len(segments)):
            candidate = "_".join(segments[start:])
            if candidate in data:
                key = candidate
                break
        if key is None:
            raise KeyError(f"No npz key found for model weight '{w.path}'")
        value = data[key]
        if tuple(w.shape) != value.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': model {tuple(w.shape)} "
                f"vs npz {value.shape}"
            )
        w.assign(value)
        unused.discard(key)
        print(f"  {key:55s}  {value.shape}")
    if unused:
        raise ValueError(f"Unused npz keys (not mapped to any weight): {sorted(unused)}")


# ── Export ────────────────────────────────────────────────────────────────────

def export(weights_npz: str | None, out_dir: str, arch: str = "mlp") -> None:
    if weights_npz:
        # Infer architecture from the checkpoint, not the flag.
        arch = "attn" if "pool_query" in np.load(weights_npz) else "mlp"
    print(f"Architecture: {arch}")
    model = PolicyValueNetAttn() if arch == "attn" else PolicyValueNet()

    # Build Keras layer weights via a dummy forward pass (required before set_weights)
    dummy = {"cm": tf.zeros([1, 36, 12]), "hd": tf.zeros([1, 20])}
    model(dummy)

    if weights_npz:
        if arch == "attn":
            load_attn_weights_from_npz(model, weights_npz)
        else:
            load_weights_from_npz(model, weights_npz)
    else:
        print("No --weights given — exporting random-init model")

    os.makedirs(out_dir, exist_ok=True)
    # model.export() is the Keras 3 way to produce a SavedModel for serving.
    # input_signature as a dict gives named cm/hd inputs in the serving signature.
    model.export(out_dir, input_signature=[{
        "cm": tf.TensorSpec([None, 36, 12], tf.float32, name="cm"),
        "hd": tf.TensorSpec([None, 20],     tf.float32, name="hd"),
    }])
    print(f"\nSaved model to: {out_dir}")

    _print_signature(out_dir)


def _print_signature(out_dir: str) -> None:
    from tensorflow.python.saved_model import loader_impl  # type: ignore[import]
    saved = loader_impl.parse_saved_model(out_dir)
    sd = saved.meta_graphs[0].signature_def["serving_default"]

    print("\nserving_default inputs:")
    for k, v in sd.inputs.items():
        dims = [d.size for d in v.tensor_shape.dim]
        print(f"  '{k}' → '{v.name}'  shape={dims}")
    print("serving_default outputs:")
    for k, v in sd.outputs.items():
        dims = [d.size for d in v.tensor_shape.dim]
        print(f"  '{k}' → '{v.name}'  shape={dims}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        metavar="NPZ",
        help="Path to .npz weights file from extract_pv_weights.py (omit for random init)",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "attn"],
        default="mlp",
        help="Architecture for random init (inferred from npz when --weights is given)",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for the TF2 SavedModel",
    )
    args = parser.parse_args()
    export(args.weights, args.out, args.arch)
