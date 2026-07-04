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

- `make_puct_policy_fn(attn_model.apply, src_params, num_determinizations=8,`
  `num_simulations=128, temperature=1.0)`, pmap'd over the 8 chips.
- **Per-chip batch = 8 games with the attn generator** (profiled
  2026-07-04: 615 ms/game at B=8 vs 684 @4 / 753 @16 — the VMEM knee
  moved 8× vs the old net's B=64, but the optimum ms/game barely moved,
  563→615: collection is tree-bound, not model-bound). Keep the *saved*
  batch at 2048: collect each 2048-game batch as **32 pmap calls of 64**
  (8/chip) and concat, so training (which makes each saved batch one
  2048-game step) is unchanged.
- gen-7: 32 saved batches × 2048 = 64k games. Write each shard once
  (`corpus_gen{SRC}_s128_batch_{i:03d}.pickle`) and probe with `open()` for
  **restart-safe resume**; assemble the list-of-per-batch-6-tuples
  `(cm, hd, labels, pi, legal, alive)` at the end. Expect ~160 s/batch
  (+ host dispatch for the 32 pmap calls) → **~1.4 h for 64k**.
- Re-profile the per-chip optimum with `jass_selfplay.profile_collect_fn`
  **if K, sims, or the architecture change** (the optimum tracks the
  tree+activations working set; attn: B×K≈64, old net: B×K≈512).

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
`policy_match` is cheap and safe at 300.)

**TODO — parallelize probe chunks across TPU chips.** With the attn net
the seed-looped probes are no longer "minutes on CPU": the gen-6b_es
K=16 @128 arm ran ~70 s/chunk × 12 chunks (~14 min/arm), single-device —
`policy_match` is plain `jit`+`vmap`, no pmap/sharding. On a 2×4 runtime
it uses 1 of 8 chips; distributing the seed chunks across devices
(`jax.device_put` params per device, round-robin chunks) is ~8× without
touching `policy_match` itself. Worth doing before the next probe sweep.

Diagnostics:

- **Operator (starved?):** gen-`SRC` PUCT vs gen-`SRC` RAW, swept sims. A
  *small/negative* margin = starved → bump corpus sims. A *large* margin
  (gen-3 was +26 @ sims=128) = plenty of fuel → NOT a sims problem.
- **Policy moving but gate flat? (gate masking):** gen-`GEN` raw vs gen-`SRC`
  raw. Large + while PUCT-vs-PUCT is flat = the search is masking real policy
  gains (this is now the standing gate, above).

## Current strategic state (2026-07-03, post-promotion)

**CHAMPION: gen-6b_es (`pv_gen6b_es_s128.msgpack`) — `PolicyValueNetAttn`
early-stopped at 7k epochs.** Promoted 2026-07-03 on raw +10.3/+7.4
(both seeds significant) and PUCT@64 deployed **+5.2 (p=0.01)** — the
first significant deployed gain since gen-3, above the +2–3.5
policy-compression band: the value-head upgrade converts to searched
strength. Full arc in the log (overfit post-mortem → U-curve → early
stop). ⚠ The champion is a DIFFERENT ARCHITECTURE: every cell that loads
it needs `PolicyValueNetAttn().apply`; a `PolicyValueNet` template
silently mangles the params. Gate cells against older generations need
TWO model instances.

- **The attn recipe (until the weight-decay arm lands): 7,000 epochs,
  NOT 20k** — the architecture overfits from ~8k (eval-v U-curve bottoms
  ~0.115 at 6.0–7.6k; 20k lands at 0.147 and gates flat). Everything
  else per Stage 2, plus `model=PolicyValueNetAttn()` and
  `data_parallel=True`. Training-health check for attn: eval v ≈
  0.115 ± 0.002 at 7k (the old 0.13–0.14 band is the *old architecture's*
  level).
**OWED gen-6b_es measurements (in order):**

1. ~~Operator re-probe~~ **DONE 2026-07-04: gauge reads ZERO** — K=16
   @128 −0.2 (p=0.92), K=8 @128 −3.4 (p=0.11). Search adds nothing over
   gen-6b_es raw at any reachable sharpness → gen-7 collects at the
   cheap baseline (K=8/sims=128), NOT sharp. Full entry in the log.
2. ~~Collect re-profile~~ **DONE 2026-07-04: per-chip B=8, 615 ms/game**
   (knee moved 8× to B×K≈64; optimum ms/game only ~9% worse than the old
   net — collection is tree-bound). gen-7 64k ≈ 1.4 h on the 2×4.
   Stage-1 numbers above updated; full entry in the log.
3. **PUCT@64 vs raw on gen-6b_es** (new, from the probe result): if @128
   ≤ raw then the deployed/JTR config @64 presumably is too — would the
   JTR calibration do better submitting the RAW policy (~65× cheaper per
   move)? 240 pairs, same seed-loop method.
4. Optional: top-1 adoption diagnostics (gen-6b_es and gen-6 vs the
   teacher, held-out batch); JTR calibration of gen-6b_es — blocked on
   attn support in the export scripts (they hardcode `PolicyValueNet()`).

**gen-7 DECISION (2026-07-03): collect a LARGER corpus — the principled
fix for the attn overfit, replacing early stopping.** Overfit onset was
~8k epochs ≈ ~530 passes over each of the 15 train batches; doubling to
**32 batches × 2048 = 64k games** halves passes-per-epoch and feeds the
spare capacity — and the value head benefits from more games regardless
of the operator margin (value targets are game outcomes, not
search-dependent). Re-priced by the 2026-07-04 probe: collect at
K=8/sims=128 (sharpness buys nothing); the corpus case is value-half +
capacity, not better policy targets. Train the attn net full-length on
it, full logs: if
the U-curve minimum moves past 20k, early stopping retires. Per-epoch
training cost is unchanged (one batch per step); only Stage-1 cost
scales — hence the re-profile first. Do NOT pad with older generations'
corpora (the gen-2 3-way-mix regression). Fallback arms if a big corpus
still overfits: weight decay
(`optimizer=optax.adamw(3e-4, weight_decay=1e-2)`, passthrough landed
2026-07-03; success = holdout v ≤ 0.115 at 20k with the gap closed),
then dropout in the attention blocks / `num_layers=1`.
- **Training on the 2×4: `train_pv_model(..., data_parallel=True)`**
  (landed 2026-07-03) — pmaps the train step over all local chips
  (batches sharded, grads psum'd; same update math, checkpoints stay
  single-device and interchangeable). Context: PolicyValueNetAttn trains
  ~5× slower than PolicyValueNet (90 s vs 18 s per 100 epochs → ~5 h for
  20k; measured 2026-07-03, 1×1 + `accum_steps=4` — its 2048-game step
  needs ~33 G HBM vs 15.75 G/chip). On the 2×4, sharding 8× (~9.5k
  positions ≈ 4 G/chip) makes `accum_steps` unnecessary; **measured
  2026-07-03: 20 s/100 epochs → 20k in ~67 min** (4.5× the 1×1+accum
  path; not 8× — host-side augment/sharding is serial). A mid-run
  1×1→2×4 checkpoint handoff worked (resume + `data_parallel=True`). **Preference: run everything on one 2×4
  instance** (no 1×1↔2×4 stop/start); the old 1×1-for-training rule only
  mattered when collection competed for the 2×4.
- Optional diagnostics: gen-6b_es top-1 adoption vs gen-5b on the
  held-out batch (how much of the +10 is teacher-correction adoption vs
  better generalization off the same targets); gen-6 top-1 adoption (the
  old exhaustion-story check — now expected to show the corpus held
  signal the old architecture couldn't take).
