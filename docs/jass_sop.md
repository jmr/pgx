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
- **Pipelining note (measured gen-7, 2026-07-04): `np.asarray` inside the
  pmap-call loop costs ~25%** (200 s/batch vs the ~157 s compute bound) —
  it syncs the chips after every call. Dispatch ALL calls first (append
  the raw pmap outputs; JAX queues them async so chips run back-to-back),
  convert to numpy after the loop. Per-call outputs are a few MB on
  device, so in-flight accumulation is safe.
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
- **The PUCT@64 deployed check is RETIRED (2026-07-04): the deployed
  config is RAW** — gen-7 PUCT@64 measured −6.3 (p=0.0033) vs its own
  raw, so searched play now under-reads AND underperforms; the raw gate
  covers deployed strength too. PUCT-vs-PUCT arenas remain diagnostics
  only.

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

**Parallelize probe chunks across TPU chips (recipe, 2026-07-05).**
`policy_match` is plain `jit`+`vmap` — single device. On a 2×4, run one
chunk per chip with `pmap`; the pair chunk plays the role of the old
seed chunk (8 pairs/chip × 8 chips × 5 rounds = 320 pairs at the memory
cost of 8). Build the action fns INSIDE the pmapped fn from its params
argument — closing over host params would bake them in as constants:

```python
def make_chunk_fn(K, sims, pairs_per_chip):
    def run_chunk(params, key):
        puct = make_puct_action_fn(attn_model.apply, params,
                                   num_determinizations=K,
                                   num_simulations=sims)   # greedy
        raw = make_policy_action_fn(attn_model.apply, params,
                                    temperature=0.05)
        return policy_match(puct, raw, key, pairs_per_chip)
    return jax.pmap(run_chunk, in_axes=(None, 0))

# per round: chunk_fn(src_params, jax.random.split(PRNGKey(seed), 8))
# → (8, 2*pairs_per_chip), pair-adjacent per chip; reshape(-1) and pool.
```

(The old CPU guidance — seed-loop at ≤80 pairs, never raise
`num_pairs` — still applies to single-device runs; the gen-6b_es
K=16 @128 arm ran ~70 s/chunk × 12 chunks single-device.)

Diagnostics:

- **Operator (starved?):** gen-`SRC` PUCT vs gen-`SRC` RAW, swept sims. A
  *small/negative* margin = starved → bump corpus sims. A *large* margin
  (gen-3 was +26 @ sims=128) = plenty of fuel → NOT a sims problem.
- **Policy moving but gate flat? (gate masking):** gen-`GEN` raw vs gen-`SRC`
  raw. Large + while PUCT-vs-PUCT is flat = the search is masking real policy
  gains (this is now the standing gate, above).

## Current strategic state (2026-07-04, post-gen-7 promotion)

**CHAMPION: gen-7 (`pv_gen7_s128.msgpack`) — `PolicyValueNetAttn`,
64k corpus, early-stopped at 10k.** Promoted 2026-07-04 on raw
+5.2/+10.2 (both seeds significant); PUCT@64 deployed check FLAT (+1.1
ns) — search no longer converts raw gains at the deployed config.
**The operator fuel gauge is RETIRED as a crank gate**: gen-7 climbed
+5/+10 from a corpus whose generator search measured ZERO margin over
its own raw policy. Plain numeric GEN/SRC anchors work again (gen-8:
SRC=7). ⚠ Champion + generator are ATTN: every cell that loads them
needs `PolicyValueNetAttn().apply`; a `PolicyValueNet` template
silently mangles the params. Gate cells against pre-attn generations
need TWO model instances.

- **The attn recipe (standing; weight decay SHELVED 2026-07-05 — both
  gen-8 arms gated WASH vs gen-7): EARLY-STOP at
  the corpus-size-dependent U-minimum, NOT 20k.** Measured: 15 train
  batches → stop at 7k (flat 6.0–7.6k, floor ~0.115; 20k lands 0.147);
  31 train batches → stop at 10k (flat 7.5–11k, floor ~0.113; 20k lands
  0.121). Onset scales sublinearly with corpus (NOT constant
  passes-per-batch — measured 2026-07-04), so for a new corpus size run
  full logs once and read the flat region — **with `snapshot_every=500`
  (knob landed 2026-07-05): keeps params-only `.ep{N}` files next to
  the checkpoint, so the early-stopped net is a file copy off the
  calibration run instead of a ~36 min retrain** (the gen-8 Arm A run
  predated this and owed exactly that retrain). Everything else per
  Stage 2, plus `model=PolicyValueNetAttn()` and `data_parallel=True`.
  Training-health check for attn: eval v at the floor ≈ 0.113–0.115
  (the old 0.13–0.14 band is the *old architecture's* level).
**Queued work (order decided end-of-session 2026-07-04):**

0. **NEXT (2026-07-06): gen-9 direction — probe DONE, Option A
   CLOSED.** No mctx config beats gen-7 raw (JTR-mirror K=45×64:
   −1.1 ns; depth control K=8×360: −9.8***), yet JTR's searcher gets
   +10.15 from the same net at the same budget. A′ SCOPED by code
   reading (log 2026-07-06): JTR uses the pgx value head at leaves
   (no rollouts in card play) through classical full-width PUCT —
   next action is the `mctx.muzero_policy` swap in `jass_puct.py` +
   re-run of the K=45×64 probe. B (capacity on the existing corpus)
   is the parallel internal lever. Full option table: log 2026-07-06
   entries + plan NEXT block.
1. **DONE 2026-07-05: JTR arena chapter.** Export + PUCT calibration
   (gap to POWERFUL = ZERO) + raw arena (`--pgx-raw`, JTR change
   `qzzrmuqy`): externally PUCT > raw +10.15 and raw < POWERFUL
   −8.5 — deploy-raw is HARNESS-SCOPED (internal gates raw, external
   submissions PUCT; log entries 2026-07-05). Cheating-raw diagnostic
   DONE: perfect-info raw still −7.5, so the gap is the raw policy
   itself (JTR ≈ 2,880 expansions/move vs the internal probes'
   512–2,048 — sims are per-det, correction in the log 2026-07-05),
   not marginalization.
   Original export scope (surveyed 2026-07-04, landed b942b68a):
   - `scripts/extract_pv_weights.py` hardcodes `PolicyValueNet()` at
     `model = PolicyValueNet()` — needs an `--arch` flag or template-free
     msgpack_restore (small).
   - `scripts/export_pv_savedmodel.py` is a **TF/Keras REIMPLEMENTATION
     of the forward pass** (its `PolicyValueNet(tf.keras.Model)` mirrors
     the flax module exactly) — attn support = porting
     `PolicyValueNetAttn` (hidden=128, num_heads=4, num_layers=2;
     header broadcast into all 36 card rows pre-embedding; pre-LN
     transformer blocks of MultiHeadDotProductAttention + gelu MLP with
     residuals; learned-query attention pool `pool_query`; card logits
     read off attended rows) + the flax→keras weight-name mapping, and a
     parity test vs `attn_model.apply` on random inputs.
   - Raw deployment means JTR only needs the POLICY head path verified,
     but export both heads (the SavedModel contract is shared).
2. **DEFERRED to gen-9 (2026-07-05): raw-vs-PUCT corpus A/B** — gen-8
   is scoped to the weight-decay arm alone (item 3); collect gen-8 with
   the standard PUCT recipe. Rationale: pipeline optimizations only pay
   if many rounds remain, and the crank may stall soon — prove the
   training-side fix first, optimize throughput after. Original A/B
   design (gen-7 generator both arms, 64k each, same training recipe,
   ES @10k):
   - **Raw arm FIRST** (~45 min end-to-end: raw τ=1.0 self-play collects
     64k in ~minutes — forward-only, ~65× cheaper; policy targets =
     one-hot own-sampled moves ≈ policy stays put, value channel live).
     Student gates flat vs gen-7 → visit distributions were the live
     channel; collect the standard PUCT arm, nothing lost. Student
     climbs → gen-8 for 45 min and search exits Stage 1.
   - PUCT arm per the standard recipe (~1.8 h collect) if needed, or as
     an optional head-to-head (A-student vs B-student) to measure the
     target-sharpening channel directly.
3. **DONE 2026-07-05 — WASH, WD shelved. The gen-8 arm: weight decay.**
   Both gen-8 nets (8b_es = Arm A-ES, 8c_wd = Arm B) gated WASH vs
   gen-7 (raw gate, two seeds each; log 2026-07-05) — the planned
   B-vs-A-ES head-to-head is moot when neither clears the champion.
   ES stays the recipe. Original procedure kept for the record:
   (`train_pv_model(..., weight_decay=1e-2)` — first-class knob landed
   2026-07-05: masked adamw, decay on Dense/attention kernels ONLY,
   biases/LayerNorm/pool_query excluded per the standard transformer
   recipe; supersedes the raw `optimizer=optax.adamw(...)` passthrough,
   which decays everything). The point of WD is to make the stopping
   epoch a free choice — train to 20k with NO U-bend to hit; do NOT
   combine ES+WD in one run (confounds the comparison, and if WD works
   there is nothing to stop early for). gen-8 procedure:
   - Collect the standard PUCT corpus (gen-7 generator, 32×2048,
     K=8/sims=128).
   - **Arm A (baseline):** full 20k WITHOUT decay, full logs — the
     fresh-corpus overfitting look; its U-min defines the ES baseline
     (floor + stopping epoch for this corpus).
   - **Arm B:** full 20k WITH `weight_decay=1e-2`, full logs. Take the
     20k net — or any epoch past where the holdout flattens, if 67 min
     is worth trimming.
   - **Arm A-ES — the REAL comparison point for WD:** retrain WITHOUT
     decay, early-stopped at Arm A's U-min (the standing recipe's net
     for this corpus).
   - **Compare.** Proxy read: Arm B's holdout v at 20k vs A-ES's floor
     (~0.113 band), seen-vs-holdout gap closed, no upturn. Decider:
     **head-to-head raw arena, B-20k vs A-ES** — WD earns its keep
     only by matching or beating the ES net it would replace. WD
     wins/ties → gen-8 = B-20k, gate vs gen-7 (raw-vs-raw, two seeds,
     as usual); ES + the per-corpus-size calibration run retire from
     the recipe. WD loses → gen-8 = A-ES, gate vs gen-7; escalation:
     dropout in the attn blocks, then num_layers=1.
   ⚠ adamw opt_state ≠ adam's: don't resume an adam checkpoint with
   weight_decay set (or vice versa) — separate GEN= checkpoint names
   per arm.
4. **DEFERRED to gen-9 (2026-07-05, same scoping call as item 2):
   15-batch dose-response arm — upgraded from optional earlier the
   same day: it carries the corpus-size DECISION.** The crank objective is
   points gained per wall-clock hour, and 32×2048 collection is the
   dominant round cost. Round model (attn generator: PUCT collect
   177 s/2048-batch measured 2026-07-05, ES train 20 s/100 epochs,
   gate ≈10 min, human overhead ≈30 min/round):
   **64k ≈ 2.8 h/round, 32k ≈ 1.8 h, 16k ≈ 1.4 h** — so 64k must gain
   ≥ ~1.5× more per round than 32k to pay its way (32k vs 16k:
   ≥ ~1.3×). Cross-round evidence says it doesn't (gen-7's 64k gained
   +5.2/+10.2 vs the 32k generations' +10–16; floor 0.113 vs 0.115)
   but is CONFOUNDED (gen-6b_es's climb included the architecture
   jump). The clean measurement is this arm: train a student on the
   gen-7 corpus `batches[:15]` + the same holdout, gate vs the
   31-batch student — zero new collection, ~1 h. If ≈flat → **drop
   the standard corpus to 16×2048** (~1.5× the rounds/hour);
   optionally bracket with a `batches[:8]` arm (below 16k the ~40 min
   fixed cost dominates — don't shrink further). Interactions: if the
   gen-8 raw arm wins, collection is ~65× cheaper and only the
   train-time term still scales with size; the weight-decay arm
   removes the per-size ES-calibration tax (~1.5 h per NEW size) that
   otherwise penalizes changing corpus size at all.

**gen-7 DECISION (2026-07-03): collect a LARGER corpus — the principled
fix for the attn overfit, replacing early stopping.**
**RESOLVED 2026-07-04: helps but does not replace early stopping** —
the 64k full-log run bottomed at ~0.113 (flat 7.5–11k, onset ~12k, 20k
at 0.121): the U-minimum did not move past 20k, so early stopping stays
(at 10k for 64k) and the weight-decay arm remains queued. Original
rationale kept for the record: overfit onset was
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
(`train_pv_model(..., weight_decay=1e-2)`, masked-adamw knob landed
2026-07-05; success = holdout v ≤ 0.115 at 20k with the gap closed),
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
