# pgx/scripts — Jass model export pipeline

Scripts for converting trained JAX/Flax models into TF2 SavedModels that can be
loaded by the Java engine (`JassTheRipper`) via TensorFlow-Java.

---

## Python environments

Two separate venvs are needed because JAX 0.10.1 requires Python 3.14 (no TF wheel
exists for 3.14) and TensorFlow requires Python ≤ 3.13.

### `.venv` — JAX/Flax (already present)
Used for training, evaluation, and weight extraction.

```bash
# Already exists. To recreate:
/opt/local/bin/python3.14 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### TF venv — for SavedModel export (create once)
Used only for exporting SavedModels. Does **not** need JAX.

Two options — either works:

**Option A: use the TF venv already in JassTheRipper** (already set up):
```bash
# Located at ../JassTheRipper/src/main/java/to/joeli/jass/client/strategy/training/python/.venv
# Has tensorflow==2.21.x, Python 3.13
```

**Option B: the dedicated `.venv_tf` in this repo** (created for this purpose):
```bash
# Already exists at .venv_tf (tensorflow==2.20.x, Python 3.13)
# To recreate:
/opt/local/bin/python3.13 -m venv .venv_tf
.venv_tf/bin/pip install "tensorflow==2.20.*" numpy
```

**Why Python 3.13?** TF 2.20 is the first version with Python 3.13 wheels. Python 3.14
(used by the JAX venv) has no TF wheel at all. Python 3.12 would give TF ≤ 2.18 but
only requires `sudo port install python312`.

**Why not worry about TF version matching Java exactly?** The Java runtime bundles
TF 2.18 (in `tensorflow-core-platform:1.1.0`), but SavedModels exported with TF 2.20/2.21
load fine because our model only uses basic Dense+GELU ops that haven't changed.

---

## Key export details (lessons learned)

**Use `model.export()`, not `tf.saved_model.save()`.**
Keras 3 (bundled with TF 2.20+) auto-generates a `serving_default` that calls
`self(inputs)` with a *single* argument. Any model whose `call()` takes two positional
args (`cm, hd`) will fail to serialize. Fix: make `call()` accept a single dict input
and use `model.export()` with a dict `input_signature`.

```python
def call(self, inputs: dict) -> dict:
    cm, hd = inputs["cm"], inputs["hd"]
    ...

model.export(out_dir, input_signature=[{
    "cm": tf.TensorSpec([None, 36, 12], tf.float32, name="cm"),
    "hd": tf.TensorSpec([None, 20],     tf.float32, name="hd"),
}])
```

This produces serving tensor names `serving_default_cm:0` and `serving_default_hd:0`,
which TF-Java's `SessionFunction.call(Map.of("cm", ..., "hd", ...))` uses directly.

---

## Generating a SavedModel

### Quick start — random-init model (smoke-test, no weights needed)

Useful for testing the Java pipeline end-to-end before real weights are available.

```bash
cd /path/to/pgx
.venv_tf/bin/python scripts/export_pv_savedmodel.py \
    --out ../JassTheRipper/src/main/resources/models/pgx_pv/export
```

### With real weights

**Step 1** — extract Flax params to `.npz` (runs in JAX venv):
```bash
.venv/bin/python scripts/extract_pv_weights.py \
    --weights pv_gen3_s128.msgpack \
    --out /tmp/pv_weights.npz
```

**Step 2** — export SavedModel with those weights (runs in TF venv):
```bash
.venv_tf/bin/python scripts/export_pv_savedmodel.py \
    --weights /tmp/pv_weights.npz \
    --out ../JassTheRipper/src/main/resources/models/pgx_pv/export
```

The script prints the serving signature tensor names after export.

---

## Running in Java

```bash
cd /path/to/JassTheRipper
JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-21-macports.jdk/Contents/Home \
./gradlew run -Pmyargs="--url=ws://127.0.0.1:3000,--name=MyBot,--team=1,\
--pgx-model=src/main/resources/models/pgx_pv/export,--pgx-value"
```

Flags:
- `--pgx-model[=path]` — load SavedModel (default: `src/main/resources/models/pgx_pv/export`)
- `--pgx-value` — use value head as MCTS leaf evaluator
- `--pgx-policy` — use policy head as PUCT prior (also needs `--puct --puct-prior=pgx`)

Tests (skip automatically if model not present):
```bash
JAVA_HOME=... ./gradlew test --tests "*.PgxPolicyValueEstimatorTest"
```

---

## Version reference

| Component | Version | Reason |
|-----------|---------|--------|
| Python (JAX venv) | 3.14 | MacPorts default; matches JAX 0.10.1 wheel |
| Python (TF venv) | 3.13 | Newest Python with a TF wheel (2.20+) |
| TensorFlow (export) | 2.20.x or 2.21.x | First TF releases with Python 3.13 wheels |
| TF-Java runtime | 2.18.0 | Bundled in `tensorflow-core-platform:1.1.0`; loads 2.20/2.21 SavedModels for basic ops |
| JAX | 0.10.1 | Matches training environment |
| Flax | 0.12.7 | Matches training environment |
