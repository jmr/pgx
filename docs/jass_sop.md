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

## Current strategic state (2026-07-02)

**The loop is RE-OPENED — the gen-5 flat gates were the 50% step2 anchor,
not saturation.** The mix ablation's dose-response (all on the same gen-4
corpus): adoption of teacher corrections rises as the anchor shrinks
(16.5% → 24.7% → 33.5% at 50/20/0%), and the 0% retrain climbed
**+11.8/+16.2 raw vs gen-4**, then **passed the PUCT@64 deployed check**
(+2.2 ns — normal compression, gen-4's +15 raw read +3.5 here; no
value-coverage damage). **CHAMPION: gen-5b (`pv_gen5b_s128.msgpack`).**
The recipe above is already updated (100% newest-PUCT). In order:

1. **gen-6 on the new recipe** (gen-5b generator, sims=128/K=8, 100% PUCT,
   raw-vs-raw gate, two seeds). Re-measure the operator while the corpus
   collects (gen-5b PUCT@128 vs gen-5b raw, seed-looped ≤80 pairs): +26 at
   gen-3 → +11 at gen-4 → ? — the fuel gauge for how long the crank runs.
2. **JTR re-calibration at gen-5b** (owed since gen-4) — the
   self-relative→absolute conversion rate vs gen-3's −22/game. The
   searched agent has moved little since gen-3 (+3.5, then +2.2 ns at
   PUCT@64), so expect the VALUE head to cap the conversion.

Queued: Step-4 value-head scaling (the deployed-strength cap) and target
sharpening (sims 256 / K 16) — revisit at the next deceleration; with the
anchor gone, a flat raw gate will then mean saturation for real.
