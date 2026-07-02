# Jass PUCT loop — Standard Operating Procedure (per generation)

Runbook for one generation of the Step-3 PUCT expert-iteration loop
(gen `SRC` → gen `GEN`). Strategy and current state live in
`docs/jass_plan.md`, dated results and diagnostics in
`docs/jass_experiment_log.md`; this file is the *how*. Update it when the
procedure changes (e.g. the gate change of 2026-06-21).

## The single anchor (do this first, every cell)

Derive **every** path from one integer so you can't load a stale generation:

```python
GEN = 5            # the generation you PRODUCE this run (bump this, nothing else)
SRC = GEN - 1      # champion: generator, corpus name, gate opponent
```

Then all filenames are f-strings: `pv_gen{SRC}_s128.msgpack` (champion),
`corpus_puct_gen{SRC}_16x2048_s128k8.pickle` (named by its *generator*). The
new generation has **two distinct files — don't confuse them**:
`pv_gen{GEN}_s128_ckpt.msgpack` is the resumable **training checkpoint**
(`checkpoint_path=`, written every 500 epochs), and `pv_gen{GEN}_s128.msgpack`
is the **final net saved right after training completes** — *that* is the one
you **gate**, and keep as champion on a win. So the gate/diagnostic cells load
`new_params` from `pv_gen{GEN}_s128.msgpack`, **not** the `_ckpt` file (the
`_ckpt` may not even survive to gate time). Params are **role-named**:
`src_params`, `new_params` — not `gen3_params`.

> Two bugs this prevents (both hit on 2026-06-20): a 9 s "training" run that
> silently resumed a *completed* `pv_gen{SRC}` checkpoint because the path
> wasn't bumped; and a gate against the wrong opponent file
> (`pv_gen2.msgpack` ≠ the champion `pv_gen2_s128.msgpack`).

## Stage 1 — Collect (TPU 2×4, ~48 min)

Generate the PUCT corpus with the champion as generator. Since 2026-07-02
this is the **entire** training set: **the step2 anchor is RETIRED** (its
50% share caused the gen-5 plateau — see the experiment log). Keep
`corpus_k8_v1_24x4096` on Drive; never regenerate it, never train on it.

- `make_puct_policy_fn(pv_model.apply, src_params, num_determinizations=8,`
  `num_simulations=128, temperature=1.0)`, pmap'd over the 8 chips.
- **Per-chip batch = 64 games** (the profiled VMEM optimum; ~5× faster than
  the old 256/chip — see jass_plan "HOW TO SCALE STAGE 1"). Keep the *saved*
  batch at 2048: collect each 2048-game batch as **4 pmap calls of 512**
  (64/chip) and concat, so training (which makes each saved batch one
  2048-game step) is unchanged.
- 16 saved batches × 2048 = 32k games. Write each shard once
  (`corpus_gen{SRC}_s128_batch_{i:03d}.pickle`) and probe with `open()` for
  **restart-safe resume**; assemble the list-of-per-batch-6-tuples
  `(cm, hd, labels, pi, legal, alive)` at the end. Steady ~180 s/batch.
- Re-profile the per-chip optimum with `jass_selfplay.profile_collect_fn`
  **if K or sims change** (the optimum tracks the tree working set, B×K≈512).

## Stage 2 — Train (1×1 if TPU budget-limited, else 2×4; ~1 h)

Restart to a **1×1** runtime to spend the scarce 2×4 budget only on
collection (training uses 1 chip regardless; corpus is on Drive). Then:

- Fresh `PolicyValueNet`, `train_pv_model(collect_fn=[puct_fn], …)` —
  **100% newest-PUCT** (recipe changed 2026-07-02: the old 50% step2 anchor
  absorbed the distillation gradient; gen-5b climbed +12/+16 without it).
  20k epochs, `policy_weight=1.0`, `augment=True`.
- Corpus on **host** (numpy, never `jnp.asarray` the whole thing).
  `make_cached_collect_fn`: `split=1` for the 2048-game PUCT batches →
  ~2048 games/step (the retired step2 corpus needed `split=2` for its
  4096-game batches). Hold out the last PUCT batch for eval.
- `checkpoint_path=…f"pv_gen{GEN}_s128_ckpt.msgpack"` (NEW path — a
  pre-existing file makes it resume and finish in seconds), `checkpoint_every=500`.
  Resume = rerun the same call (RNG fast-forwarded).
- **Confirm it actually reached 20k** (the `training done` print / final eval
  at epoch 20000) before gating — don't read stale scrollback.
- **Training-health check:** eval **value loss ≈ 0.13–0.14** at the end
  (gen-5b: 0.1332). Do NOT
  gate on value loss (a wash every gen) or raw policy CE (floored at the soft
  target's entropy) — diagnose policy by top-1 agreement.

## Stage 3 — Gate (CPU, `JAX_PLATFORMS=cpu`, minutes)

**PROGRESS gate = raw-vs-raw** (changed 2026-06-21; PUCT@64 went blind to
policy gains — see the gen-4 REFRAME entry in jass_experiment_log.md):

```python
new_raw = make_policy_action_fn(pv_model.apply, new_params, temperature=0.05)
src_raw = make_policy_action_fn(pv_model.apply, src_params, temperature=0.05)
# A = new (gen GEN), B = src (gen SRC); +mean = new policy better
print_stats(f"gen-{GEN} raw", f"gen-{SRC} raw",
            run_gate(new_raw, src_raw, seed=0, total_pairs=300))   # + seed 2
```

- **Fingerprint both nets before trusting any number** (`new_params` fp ≠
  `src_params` fp) — paths/vars have been mixed up before. A fast, ~0-mean,
  ~50% result usually means you fed the *same* net twice.
- **Promote** on a significant raw-vs-raw win (two seeds, p<0.05) →
  `pv_gen{GEN}_s128.msgpack`.
- **PUCT@64 vs the champion** (`make_puct_action_fn`, greedy, K=8/sims=64) is
  now a **deployed-strength check only**, not the climb signal — expect it to
  under-read policy gains.

## Diagnostics (when a gate looks off)

All CPU, `policy_match`, on existing nets — no training/collection.

**Scale PUCT games by SEED-LOOPING, never by raising `num_pairs`.**
`policy_match` vmaps all pairs into one XLA op, so memory scales with
`num_pairs`; a PUCT@128 (K=8) working set OOMs/swaps the CPU colab runtime
above ~80–100 pairs — it manifests as a cell that *runs but never finishes and
won't respond to interrupt* (the interrupt only fires between Python ops; a
monolithic vmap gives no such window). Instead loop seeds at ≤80 pairs and
pool: `np.concatenate([policy_match(a, b, PRNGKey(s), 80) for s in range(3)])`
→ 240 pairs at the memory cost of 80, and interruptible between chunks. (Raw
`policy_match` is cheap and safe at 300.) Diagnostics:

- **Operator (starved?):** gen-`SRC` PUCT vs gen-`SRC` RAW, swept sims. A
  *small/negative* margin = starved → bump corpus sims. A *large* margin
  (gen-3 was +26 @ sims=128) = plenty of fuel → NOT a sims problem.
- **Policy moving but gate flat? (gate masking):** gen-`GEN` raw vs gen-`SRC`
  raw. Large + while PUCT-vs-PUCT is flat = the search is masking real policy
  gains (this is now the standing gate, above).

## Current strategic state (2026-07-02, end of day)

**The sims=128 loop is CONVERGED — for real this time.** gen-6 (the first
generation collected and trained fully on the 100%-PUCT recipe) gated flat
(+2.1/+1.3 ns) with every artifact ruled out, and the operator fuel gauge
collapsed: gen-5b PUCT@128 vs gen-5b raw = **+3.0 ns** (240 pairs), after
+26 (gen-3) → +11 (gen-4). The teacher has nothing left at this search
depth. **CHAMPION stays gen-5b (`pv_gen5b_s128.msgpack`).** Context: the
JTR re-calibration says model gains DO convert to absolute strength (gap
to POWERFUL halved, −22 → ≈−9.5/game), so making the model better again is
worth real effort. In order:

1. **Cheap CPU probe (minutes, no collection):** does the operator re-open
   at sharper search? gen-5b PUCT@256 (and/or K=16) vs gen-5b raw,
   seed-looped at **≤40 pairs/chunk** (the @256 tree ~doubles the @128
   working set that already OOMs above ~80 pairs).
2. **If a real margin re-opens → gen-7 at sims=256/K=16.** Re-profile the
   per-chip collect optimum first (B×K≈512 → expect per-chip batch ~32)
   and budget ~2× Stage-1 wall-clock.
3. **If still ~0 → Step-4 value-head scaling** (attention over the 36 card
   rows vs mean pool) is the only live lever; deeper search on the current
   value head teaches nothing.

Optional confirmatory check (cheap, not decision-relevant): gen-6 top-1
adoption on the held-out batch (the mix-ablation diagnostic). Prediction if
the exhaustion story is right: gen-6 agrees with the gen-5b teacher at a
high rate and the correction share (36.6% at gen-4) has shrunk — few
corrections left to adopt.
