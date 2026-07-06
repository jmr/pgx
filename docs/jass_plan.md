# Jass AlphaZero Roadmap

Goal: an AlphaZero-style agent (policy+value network + PUCT search over
determinizations) for Jass, built incrementally on the pgx JAX environment.

This document is the working plan. It is written so that a fresh agent (or a
human returning after a break) can pick up from any step. Update the **Status**
markers as work completes. **Dated experiment results and diagnostics live in
`docs/jass_experiment_log.md`** (append new results there; this file keeps
conclusions and pointers). The per-generation procedure is `docs/jass_sop.md`.

## Status snapshot (2026-07-06)

**CHAMPION: gen-9 (`pv_gen9_s128.msgpack`) — PROMOTED 2026-07-06,
raw +2.8/+4.2 vs gen-8d_mz, both seeds significant (t p=0.0211 /
0.0005).** Same muzero recipe as gen-8d_mz (32×2048, K=16×64,
pb_c=1.25), only the generator advanced to the new champion; full
20k, NO U-curve again, loss floors dropped further (holdout v 0.0655,
policy CE 0.604). **But the step collapsed ~4–5× vs gen-8d_mz's
+13.7/+11.0** — the first same-recipe iteration, so the big jump was
the one-time teacher-swap unlock and iterating the fixed recipe yields
a thin gen-over-gen gain.

**SATURATION CONFIRMED (2026-07-06): muzero search is FLAT vs gen-9
raw at every budget — K=16/32/45×64 = −1.0/−1.6/−0.1 ns.** The
teacher that led gen-7 raw by +10.5 can no longer beat gen-9's raw
even at 2,880 exp/move (POWERFUL's budget). The determinization-PUCT
improvement operator has saturated: search(π) ≈ π, a policy-iteration
fixed point. **The net fully captures its own search, so pure B-capacity
scaling on the current corpus is a WASH** (training was already clean).
The escape is a stronger TEACHER than self-play can generate.
POWERFUL beating our raw ~7.5 at equal compute is the existence proof
that such a teacher exists — but it is **OFF-LIMITS as a teacher: NO
JTR games in the training mix (standing DECISION 2026-07-05,
Step 4b)**; JTR is the preserved external benchmark, and the goal is
strong *general* Jass, not a JTR-exploiter.

**GROUNDED-TEACHER PROBES: ALL FOUR ARMS FAILED (2026-07-06, log).**
Root noise −0.8 ns; rollout leaf value w=1 −1.9 ns (a random playout
≈ the value head as leaf guidance — the PRIORS, not the value, are
the binding self-confirmation); flat priors λ=1 −13.7*** and
classical λ=1/w=1 −24.3*** — but the flat-prior collapses are the
**argmax-summed-visits readout failing on a non-concentrating visit
distribution** (JTR ideas.md predicted exactly this), not clean
evidence on classical search strength. And the existence proof is
STALE: POWERFUL's +8.5 was vs gen-7-era raw; gen-9 is ≈+16 internal
points above that, so POWERFUL's edge over gen-9 may already be ≈0.

**NEXT (2026-07-06): (1) gen-9 JTR re-calibration — DECISION-CRITICAL
(owed anyway).** Export gen-9 → JTR real-PUCT harness vs POWERFUL.
Still clearly below → headroom real: **(2) build the Q-sum-over-
determinizations readout knob** (JTR's aggregation; correct for
uniform-prior search where argmax-visits reads noise) and re-probe
classical λ=1 w=1 at K=45×64 (POWERFUL-parity, in-house) — a win =
the gen-10 teacher. gen-9 ≈/above POWERFUL → the equal-compute
existence proof is gone (we've caught classical search at this budget
class); reframe: above-parity teachers (budget + B-capacity) or
stronger external calibration targets (JTR++, KUS). NO JTR games in
the training mix in any branch (2026-07-05, Step 4b).

## Previous snapshot (2026-07-06 — gen-8d_mz)

**gen-8d_mz (`pv_gen8d_mz.msgpack`) — PROMOTED 2026-07-06,
raw +13.7/+11.0 vs gen-7, both seeds p<0.0001.** The muzero-teacher
retake of gen-8: same net, same 32×2048 corpus size, ONE change vs
the two washed attempts — the collection searcher
(`search_variant="muzero"`, K=16×64, pb_c=1.25). Trained the full
20k with NO U-curve (first attn run that doesn't overfit; holdout v
0.074 vs the old 0.111–0.113 floors), so no `_es` suffix. The
operator, the training curve, and the gate all confirm the same
mechanism: the Gumbel-visits teacher was the binding constraint —
see the 2026-07-06 log arc.

## Previous snapshot (2026-07-05)

**EXTERNAL MILESTONE (2026-07-05): the gap to JTR's classical POWERFUL
has closed to ZERO.** gen-6b_es and gen-7, exported through the new
attn scripts and run through JTR's real-PUCT harness (SWEEP_64, 250
pairs each): gen-6b_es vs gen-5b **+6.75/game p<0.0001** (external
confirmation of the internal climb), both attn gens vs POWERFUL
**flat ns** (trendline −22 → −9.5 → 0). gen-7 vs gen-6b_es washes
under external PUCT too. Full entries in the log. **The raw arena
(same day, via JTR's new `--pgx-raw`) flipped the internal
search-hurts sign: externally PUCT > raw +10.15 and raw loses to
POWERFUL −8.5 — see crank update 2 for the harness-scoped
deployment DECISION.**

**CHAMPION: gen-7 (`pv_gen7_s128.msgpack`) — `PolicyValueNetAttn`
trained on the first gen-6b_es-generated corpus (64k games,
K=8/sims=128, per-chip B=8), early-stopped at 10k** (the 64k U-curve
bottoms ~0.113, flat 7.5–11k, onset ~12k: a bigger corpus deepens the
floor and softens memorization but does NOT retire early stopping).
Gates vs gen-6b_es: **raw +5.2/+10.2 (t=3.08/5.74, both seeds
significant)**; PUCT@64 deployed check **FLAT (+1.1 ns)**.

**Two crank-level updates (2026-07-04, full entries in the log):**

1. **The operator fuel gauge is RETIRED as a crank gate.** gen-6b_es
   search measured ZERO margin over its own raw policy (K=16 −0.2 /
   K=8 −3.4 ns) — and the corpus that very search generated still
   trained a +5/+10 student. Play-strength margin ≠ target information;
   candidate channels: value labels from stronger games, visit
   distributions as variance-reduced self-distillation targets, corpus
   volume (the 15-batch dose-response arm separates these).
2. **Search now HURTS at the deployed config: DEPLOY RAW — re-scoped
   2026-07-05: this holds for PERFECT-INFO harnesses only.** gen-7
   PUCT@64 vs its own raw = **−6.3 (p=0.0033)** internally; the flat
   promotion check (+1.1) was two nets each dragged down by their own
   search masking a +5/+10 raw gap. The PUCT@64 deployed check is
   retired (the raw gate covers deployed strength). **But in JTR's
   imperfect-info harness the sign FLIPS: gen-7 PUCT beats gen-7 raw
   +10.15 (p<0.0001), and raw loses to POWERFUL −8.5 where PUCT
   ties. Cheating-raw diagnostic (same day): perfect-info raw STILL
   loses −7.5 — the gap is the raw policy itself vs an OOD classical
   opponent, not imperfect-info marginalization. (Budget arithmetic
   CORRECTED 2026-07-05, see the log: `num_simulations` is per
   determinization, so internal probes ran 512–2,048
   expansions/move, not 64–128; JTR's SWEEP_64 ≈ 2,880 differs
   mainly in WORLDS — 45 vs 8–16 — and the searcher, not raw
   budget.)** JTR/external submissions stay PUCT (log, 2026-07-05);
   external absolute strength currently requires the search wrapper.
   Open crank question: gen-8's teacher has NEGATIVE play margin
   (internally) — the queued raw-corpus arm tests whether visit
   distributions still carry signal (if not, Stage 1 collapses from
   ~1.8 h to ~2 min); collection is perfect-info, so the external
   flip does not affect it.

**NEXT: gen-8, RETAKEN with the muzero teacher — student
gen-8d_mz_es** (a generation isn't abandoned when attempts fail;
attempts get suffixes — naming entry 2026-07-06). The regularization
attempts are CLOSED (2026-07-05): both arms — 8b_es (early stop
@10k) and 8c_wd (weight decay 1e-2, full 20k) — gated WASH vs gen-7. **A same-size self-distillation round is a FIXED
POINT at gen-7**; weight decay is shelved and early stopping stays
the recipe until something re-opens the climb.

**RESOLVED (2026-07-06, log entries): THE OPERATOR IS RE-OPENED —
the bottleneck was the GUMBEL SEARCHER.** The probe arc, all on
gen-7, all at the JTR budget (K=45×64 ≈ 2,880 expansions/move):
Gumbel reads **−1.1 ns** (K is a noise knob, depth at fixed budget
actively hurts: K=8×360 −9.8***); JTR's classical PUCT extracts
**+10.15** externally from the same net; reading JTR's code showed
its card-play leaf evals are the pgx VALUE HEAD (not rollouts); and
swapping the internal searcher to `mctx.muzero_policy`
(`search_variant="muzero"`, landed `sxznyotm`) reproduces the margin
internally: **+11.8*** vs raw (pb_c plateau 0.64–2.5)**. Net-only
signals — the trump-phase heuristic channel isn't needed. Every
historic "operator exhausted" number (fuel trendline +26→−6.3,
sharpening probe, gauge-ZERO, the gen-8 fixed point) was measured
through Gumbel-read-by-summed-visits and is REINTERPRETED as "the
net outgrew small-sim Gumbel", not "self-play is exhausted".

**The gen-8 retake = the muzero-teacher crank (DECISION 2026-07-06,
log; corpus anchor SRC="7b_es_mz"):**

1. **Pre-collection probe DONE (2026-07-06, log): the margin holds
   cheap** — K=16×64 reads +10.5 (≈91% of the JTR-budget margin at
   36% of the cost; worlds-over-depth again: 8×128 only +7.1).
2. **Collect** with `make_puct_collect_fn(..., search_variant=
   "muzero", pb_c_init=1.25, num_determinizations=16,
   num_simulations=64)`, `dirichlet_fraction=0`, τ=1.0, standard
   32×2048 (per-chip B=8 should stand — same tree working set;
   `profile_collect_fn` sanity check first). Train (ES per recipe),
   gate raw-vs-raw vs gen-7.
3. **Dose-response goes LIVE** (stronger targets may need fewer
   games; it carries the corpus-size DECISION for the new recipe).

Parked: B (capacity scaling — revisit if the muzero crank stalls);
C′ (Gumbel with its NATIVE readout — the visit-count readout was an
impedance mismatch, see the 2026-07-06 searcher post-mortem; the
correct integration averages `out.action_weights` across the K
worlds instead of summing visits. An efficiency play, not a strength
play: probe post-gate whether it matches muzero at ~16 sims/det —
if so, Stage 1 gets ~4× cheaper. Needs a `readout=` knob in
`puct_search`). Ruled out: mctx-Gumbel-by-visits budget/worlds, more
regularization, JTR games as targets (standing DECISION), Step-5
imperfect-info.

## Previous snapshot (2026-07-03)

**CHAMPION: gen-6b_es (`pv_gen6b_es_s128.msgpack`) — the first Step-4
net: `PolicyValueNetAttn` (self-attention over the 36 card rows +
learned-query attention pooling, 393k params), trained on gen-6's corpus
and EARLY-STOPPED at 7k epochs** (the architecture overfits from ~8k —
the full-20k run gated flat; the eval-v U-curve bottoms at ~0.115 vs the
old architecture's 0.133 floor). Gates vs gen-5b: **raw +10.3/+7.4
(t=6.0/4.38, both seeds)** — a generation-class climb from a corpus that
had gated flat twice for the old architecture — and **PUCT@64 deployed
+5.2 (t=2.59, p=0.01), the first significant deployed-strength gain
since gen-3**, clearing the +2–3.5 band where pure-policy gains
compress: the value-head upgrade converts to searched strength. Full arc
(OOM → accum/data_parallel, overfit post-mortem, U-curve, interpretation
corrections — notably: gen-6's "fuel exhaustion" was
architecture-relative, the corpus held +10 the old net couldn't extract)
in the experiment log, 2026-07-03 entries. (Superseded 2026-07-04:
gen-7 promoted over gen-6b_es; the operator re-probe read ZERO and the
fuel gauge is retired as a crank gate — see the current snapshot.)

## Previous snapshot (2026-07-02)

**The gen-5 "saturation" was the RECIPE, not the loop — the 50% step2
anchor was the plateau.** The pre-registered mix ablation (retrains on the
*same* gen-4 corpus, zero collection) resolved it: the 0% arm (100% PUCT
epochs) took the raw gate from flat to **+11.8 (t=6.67) seed 0 / +16.2
(t=8.0) seed 2 vs gen-4** — a gen-4-sized climb. **PROMOTED: gen-5b
(`pv_gen5b_s128.msgpack`) is CHAMPION and the gen-6 generator.** Both
boxes cleared: two-seed raw gate p≈0, and the **PUCT@64 deployed check
passed** (+2.2, t=1.08, ns positive — normal compression, same band as
gen-4's +3.5; no value-coverage damage from training the value head on
champion-self-play only, and eval value loss even improved, 0.1382 →
0.1358 → 0.1332 across 50/20/0%). The 2026-07-01 gen-5 VERDICT ("policy
expert-iteration saturated") is **OVERTURNED** (entries + the
interpretation correction in the experiment log).

**Mechanism (anchor dose-response, all three nets trained on the same
gen-4 corpus; full entries in the log):** as the anchor shrinks
50%→20%→0%, adoption of the teacher's corrections rises
**16.5%→24.7%→33.5%** while the drift off already-agreed moves plateaus
(~15–16%) — at 50% the anchor absorbed the distillation gradient entirely
(the "student at a CE optimum" read was wrong), at 20% adoption gains ≈
drift cost (flat gate), at 0% the corrections' signal wins outright. **The
diffuse sims=128 targets carry real, convertible signal; the anchor was
both suppressing adoption and eating the conversion.**

**Standing recipe CHANGED (2026-07-02): 100% newest-PUCT corpus**
(`collect_fn=[puct_fn]`); the step2 anchor is retired from training (keep
`corpus_k8_v1_24x4096.pkl` on Drive; never regenerate it). Everything else
unchanged: sims=128, K=8, fresh net, 20k epochs — `docs/jass_sop.md`.

The ladder (each vs the prior champion): gen-1 **+4.7** (PUCT@64) →
gen-2(s128) **+11.8–13.1** → gen-3 **+13.9** → gen-4 **+2.6/+4.4** PUCT@64
but **+15 raw** (the PUCT@64 gate had gone blind to policy gains; the
progress gate is **raw-vs-raw** since 2026-06-21) → gen-5 (50/50) **flat,
not promoted** → **gen-5b (0% anchor, same corpus) +11.8/+16.2 raw**
(PUCT@64 +2.2 ns — deployed check passed).
Value loss is irrelevant every time (the climbs are all priors). Step 4's
external benchmark: gen-3 calibrated vs JassTheRipper 2026-06-20 — weak in
absolute terms, **the model is the limiter**; net scaling TODO (queued).

**NEXT STEPS:**

1. **gen-6 on the new recipe:** collect with gen-5b @ sims=128/K=8 (2×4,
   64 games/chip), train 100% PUCT, gate raw-vs-raw (two seeds).
   Re-measure the operator while the corpus collects (gen-5b PUCT@128 vs
   gen-5b raw, seed-looped ≤80 pairs) — the fuel gauge read +26 at gen-3,
   +11 at gen-4; its trajectory decides how long the crank runs.
2. **JTR re-calibration (owed since gen-4):** export gen-5b, run vs
   POWERFUL. The ladder has added ~+27–31 raw points since gen-3 (gen-4
   +15, gen-5b +12–16) against gen-3's −22/game baseline — this measures
   the self-relative→absolute conversion rate. NOTE the *searched* agent
   has moved little since gen-3 (+3.5 at gen-4, +2.2 ns at gen-5b,
   PUCT@64), so expect the VALUE head to cap the conversion — this
   measurement decides when item 3 fires.
3. **Queued:** Step-4 value-head scaling (attention over the 36 card
   rows — the deployed-strength cap) and target sharpening (sims 128→256,
   K 8→16) — revisit at the next raw-gate deceleration; with the anchor
   gone, a flat raw gate will then mean saturation for real. (Warm-start /
   target-sharpening probes from the ablation write-up are likewise
   shelved unless the crank stalls.)

**Operational facts:** TPU quota constraints are gone, but the active
Colab TPU has only **~12.2 G usable** (not 16 G). V-vs-V arenas: 1000
games ≈ 1 min; vs-rollout arenas ≈ 2 s/game (the rollout side dominates).
**PUCT cost vs `num_simulations`** — ⚠ the earlier "super-linear,
multiplier ≈ num_simulations" claim (from the K=8 collector: sims 16/32/64
= 22.7/95.5/504.5 ms/game) was NOT reproduced by the 2026-06-16 arena
sweep: `policy_match` (K=8, vmapped) measured ~250/600/1500 ms/game at
sims=64/128/256 — only ~×2.4 per doubling, roughly *linear*. Treat sims as
roughly linear-cost until re-measured; the *collector* is much heavier than
the arena (see "HOW TO SCALE STAGE 1" under Step 3 — per-chip batch 64 is
the profiled optimum). Larger batch does NOT help (compute-bound, not
utilization-bound). **Memory rules for PV training:** keep the corpus on
HOST (numpy), never `jnp.asarray` the whole thing — that pins ~30 batches on
the TPU and OOMs. Use 2048 games/step (`split=1` for 2048-game PUCT batches,
`split=2` for 4096-game step2 batches); a 4096-game step needs ~14 G. The
eval batch also runs a full grad step, so it must be small too.

**Artifacts:** weights on Drive under `MyDrive/jass/`: `v0.msgpack`,
`v1.msgpack` (canonical ValueNet line, legacy/rank-blind), `pv3_ckpt.msgpack*`
slots + `pv_gen0.msgpack` (Step 2 PolicyValueNet, run 3 @ 20k = gen-0),
`pv_gen1.msgpack`, `pv_gen2_s128.msgpack`, `pv_gen3_s128.msgpack`,
`pv_gen4_s128.msgpack`,
`pv_gen5_s128.msgpack` (50/50 — trained, NOT promoted; keep),
**`pv_gen5b_s128.msgpack` (0% anchor — CHAMPION; the final mix100 net; do
NOT overwrite `pv_gen5_s128.msgpack`)**, `pv_gen6_s128.msgpack` (flat
gate, NOT promoted; keep), `pv_gen6b_s128.msgpack` (attn @ 20k —
overfit; NOT promoted; keep),
**`pv_gen6b_es_s128.msgpack` (attn @ 7k early-stop — CHAMPION
2026-07-03; ⚠ PolicyValueNetAttn architecture, needs
`PolicyValueNetAttn().apply` — a PolicyValueNet template silently
mangles it)**.
Corpora: `corpus_k8_v1_24x4096.pkl` (Step 2 — RETIRED from training
2026-07-02, keep on Drive), `corpus_puct_gen0_8x4096_s16k8.pickle` (gen-1's
sims=16 corpus),
`corpus_puct_gen{SRC}_16x2048_s128k8.pickle` (per-generation sims=128 PUCT
corpora, named by *generator*; latest SRC=4).

**Colab workflow:** train on TPU; arena/diagnostics on CPU runtime
(`JAX_PLATFORMS=cpu` — `run_arena` is dispatch-bound; `policy_match`,
`run_batched_arena`, and the PUCT/search collectors are vmapped and fast
anywhere). `pip install mctx` (needed for `jass_puct`; not preinstalled,
intentionally not in requirements.txt). Update the package with
`pip install --force-reinstall --no-deps git+<fork>@main`, restart the
runtime, and verify a newly added symbol exists before burning quota.
Training survives preemption via `train_model(checkpoint_path=...)`
pointed at Drive (one checkpoint file per generation; resume = rerun the
same call).

## Current state (as of 2026-06-10)

Implemented and tested (65 tests in `tests/test_jass*.py`):

| Component | File | Notes |
|:---|:---|:---|
| Game env (single hand, tournament rules) | `pgx/_src/games/jass.py`, `pgx/jass.py` | 43 actions, 120-bit obs; see `docs/jass.md` |
| Void-aware determinization + flat-rollout MCTS (Option A) | `pgx/_src/games/jass_mcts.py` | `best_action(state, player, key, K, N)`; jitted, vmapped |
| Full-information value features | `value_features()` in `jass.py` | (36,12) card matrix + (20,) header; see `docs/jass.md` |
| Value net + `train_model()` | `pgx/_src/games/jass_value_net.py` | per-card MLP → pool → head; target = differential/100. Defaults = canonical V₀ settings (1000 epochs × batch 8192; loss plateaus ~500) |
| Self-play data collection | `pgx/_src/games/jass_selfplay.py` | **currently uniform-random play** — see "Key insight" below |
| V wired into `best_action` as leaf evaluator | `jass_mcts.py` (`v_apply`/`v_params`/`v_scale`) | replaces N rollouts with one V(next_state) call; K=64, N=1 recommended |
| K/N sweep harness | `scripts/jass_sweep.py` | found K≥8 indistinguishable, K=4 worse → random rollouts are the ceiling |
| V-MCTS vs rollout arena | `pgx/_src/games/jass_v_arena.py` (`run_arena()`, colab-friendly; `scripts/jass_v_arena.py` is a thin CLI wrapper) | swapped-deal pairs; paired t-test + sign test on pair means, Wilson 95% CI |
| Batched arena (TPU/GPU) | `run_batched_arena()` in `jass_v_arena.py`; `make_search_action_fn()` in `jass_mcts.py` | games vmapped in lockstep via `policy_match`; both searches run each ply, seat-parity select |

External reference: `~/Documents/src/JassTheRipper` — a competitive Java DMCTS
Jass agent. Its `IDEAS.md` documents extensive negative results (see below) and
`MSc__Joel_Niklaus.pdf` is the underlying thesis. Its known weaknesses: heuristic
trump selection (never shifts), weak early-round play.

## Key insight driving this plan

**Training V on uniformly random self-play is a dead end.** It reproduces the
thesis negative result (JassTheRipper IDEAS.md, "DNN as value estimator"): a
DNN value ≈ 10 random rollouts; 100 MCTS iterations beats both and keeps
improving while the DNN plateaus. A net trained on random play learns
value-under-random-play — a weak, *fixed* target. No architecture or
hyperparameter change on that data escapes the ceiling.

The missing AlphaZero ingredient is the **iterated loop** (expert iteration):
generate data with the current search agent → retrain → stronger agent →
better data. The two things the thesis never tested — and which JassTheRipper's
own analysis identifies as the real levers — are:

1. **Policy priors guiding the tree (PUCT)** — their tree-stats instrumentation
   found ~13% of moves are "genuine close calls" where PUCT should help, and
   ~63% have plausible room.
2. **Argmax-visits aggregation across determinizations** — their UCB c-sweep
   showed Q-sum aggregation neutralizes the tree policy entirely (c was a wash
   across 7 orders of magnitude). Visit counts must be load-bearing for priors
   to matter.

## Negative results to NOT retry (from JassTheRipper / thesis, ≥1000 games each)

- Heavy / rule-based rollouts: no improvement over random rollouts.
- ISMCTS substrate vs determinized MCTS: ISMCTS underperformed.
- Learned card-belief models for determinization sampling (CardsEstimator,
  auto-regressive variants): no signal vs uniform sampling.
- UCB exploration-constant tuning: wash, because of Q-sum aggregation (above).
- Further flat-rollout K/N tuning in this repo: our own sweep showed K≥8 is
  indistinguishable; rollout quality, not K or N, is the bottleneck.
- V-only 1-ply-greedy expert iteration (this repo, Step 1, 2026-06-12):
  one generation of training on V₀-greedy data left both the MCTS-leaf
  gate and greedy-policy strength neutral, despite the data being far
  better than random. Don't retry without changing the improvement
  operator (search-generated data / PUCT).

---

## Step 0 — Baseline the random-play V₀  [Status: DONE 2026-06-10]

Cheap; do before any new code. Train V₀ with `train_model()` (defaults are
the canonical V₀ settings: 1000 epochs × batch 8192), then run the arena
(`run_arena()` or `scripts/jass_v_arena.py`) against the K=8,N=8 rollout
baseline.

- Expect roughly parity or worse (per the thesis result). That is fine.
- **Do not tune anything here.** The number exists only as the yardstick for
  later generations.

**Result (recorded baseline, 2026-06-10; full entries in the experiment
log):** V-MCTS K=64 with random-play V₀ vs rollout K=8 N=8: **−37.5
pts/game, t=−8.08** (100 games / 50 pairs) — confirms the thesis negative
result in this stack. This is the yardstick Step 1 generations must beat.

## Step 1 — Close the expert-iteration loop, V only  [Status: CLOSED 2026-06-12 — negative result]

Smallest change that adds the missing ingredient. Tasks:

1. **Agent-driven self-play.** DONE (code):
   `jass_selfplay.make_v_collect_fn(v_apply, v_params, v_scale=TARGET_SCALE,
   temperature=10.0)` — V-greedy softmax over V(next_state), vmappable,
   same contract as `collect_batch`; plug into
   `train_model(collect_fn=...)`. Full `best_action` per move inside the
   scan is too slow for data generation at first; optionally mix in a
   small fraction of search-generated games later.
2. **Suit-permutation augmentation.** DONE (code, 2026-06-12):
   `augment_suits` in `jass_selfplay.py`, wired into `train_model`
   (`augment=False` default — canonical V0/V1 RNG stream preserved) and
   `train_pv_model` (`augment=True` default). 3! in trump modes (trump
   suit fixed), 4! in Obenabe/Undeufe and during trump selection. Also
   permutes pi/legal card actions 0–35 AND trump-declare actions 36–39
   (needed for policy targets; not in the original sketch). Verified
   against the engine on directly-relabeled GameStates.
3. **Replay buffer.** DONE (code, 2026-06-12) for the PV loop:
   `train_pv_model(collect_fn=[gen_g, gen_g-1, gen_g-2])` round-robins
   epochs over the listed generators (newest first; eval holdout comes
   from the first). Mixes the last few generations' data evenly to avoid
   catastrophic drift.
4. **Gated promotion.** New V must beat old V in the arena (significant
   at p<0.05 on the paired tests) before becoming the data generator. Keep all
   generation weights (`v0.msgpack`, `v1.msgpack`, ...).
5. **Finish porting the colab training loop.** `train_model()` is now in
   `jass_value_net.py` with defaults matching the canonical V₀ settings
   (1000 epochs × batch 8192) and a `collect_fn` hook for plugging in the
   V-guided generator (DONE). Remaining: verify eval loss plateaus before
   a generation is arena-gated; otherwise gate failures are uninterpretable
   (bad data vs. undertrained net).
6. **Batched arena (infrastructure).** DONE (code, 2026-06-12). The old
   arena plays one game at a time with two host syncs per move —
   dispatch-bound, pathological on accelerators (~8.5 s/game on a colab
   1×1 v5 TPU vs ~0.9 s/game of compute on an M-series CPU). Now batched:
   `make_search_action_fn()` (jass_mcts.py) wraps `best_action` as an
   `action_fn(state, key)` (greedy, or softmax-sampled with
   `temperature=` for self-play exploration), and `run_batched_arena()`
   (jass_v_arena.py) drives it through `policy_match` — games vmapped in
   lockstep, one `lax.scan` over plies, both agents evaluated each ply,
   seat-parity select (2× compute waste, chunk-level parallelism, zero
   per-move dispatch). Same swapped-deal pairing + stats as `run_arena`
   (but not bitwise the same games for a given seed). **VALIDATED on
   colab TPU 2026-06-12:** V₁-vs-V₀ gate (K=64 both, 100 games, seed 0)
   reproduced the sequential arena's neutral result, in ~22 s total
   including compile — vs ~8.5 s/game sequential on the same hardware,
   ≈40× faster. Follow-up at 1000 games: 62 s (≈0.06 s/game, ≈140×
   sequential) — V-vs-V gates default to 1000 games / ±2–3 pt
   resolution from now on. (Rollout-baseline matchups are heavier per
   ply — time a small run before scaling.)
7. Iterate 2–3 generations.

**Success criterion:** monotone improvement across generations AND beating the
rollout baseline. If generation 2 does not beat generation 1, **debug here**
before adding anything (check: data distribution shift, target scale, buffer
staleness, exploration temperature).

**Result (2026-06-12; full entry + Q1/Q2 diagnostics in the experiment
log): gate FAILED (≈neutral; +2.7 ± 2.4 at 1k-game power) — Step 1 CLOSED
as a negative result.** The training data improved a lot (V₀-greedy +30 vs
random), the trained V did not (V₁-greedy ≈ V₀-greedy): the 1-ply-greedy
**improvement operator**, not the data pipeline, is the bottleneck. Do not
run more V-greedy generations; the path forward is search as the improvement
operator (Steps 2–3).

## Step 2 — Add the policy head  [Status: DONE 2026-06-13]

Extend `ValueNet` to a joint policy+value net (`jass_value_net.py` docstring
notes the per-card trunk was designed for this):

- **Card logits (36):** one logit per card. DONE (code, 2026-06-12):
  `PolicyValueNet` in `jass_value_net.py`, returns `(logits (B,43),
  value (B,))`. ⚠ The original sketch here — bare Dense(1) on each trunk
  row before pooling — is exactly the run 2 bug (see the experiment log):
  each logit must also see global context, and the rows need the card
  identity encoding. Don't regress to the sketch.
- **Trump logits (7):** actions 36–42, from the pooled features + header.
  DONE (same).
- Mask illegal actions at the loss and at sampling (mask comes from
  `legal_action_mask`; note the mask is over the *information state*, which is
  identical across determinizations of the same root). DONE at the loss:
  `make_pv_train_step` (masked cross-entropy + value MSE, `policy_weight`
  knob).
- Value head unchanged (structurally identical layers; fresh init).
- Training loop: `train_pv_model()` in `jass_value_net.py` — same shape
  as `train_model` (fresh batch per epoch, fixed eval holdout, slot-file
  checkpointing + RNG-fast-forward resume), PV collect contract,
  `policy_weight` knob, prints v/p loss split. DONE (code, 2026-06-12).
  `collect_pv_batch` (random play, PV contract) is the smoke default.

First uses, before PUCT exists:

- Self-play move sampler (much faster than search-per-move data generation).
  DONE (code, 2026-06-12): `make_policy_action_fn` (policy-head sampler,
  also usable in `policy_match` for the success criterion below) and
  `make_policy_collect_fn` in `jass_selfplay.py`.
- Policy training target for now: the search agent's chosen action
  (cross-entropy on the argmax), upgraded to visit distributions in Step 3.
  DONE (code, 2026-06-12): `make_search_collect_fn` (search-played games,
  one-hot pi targets, optional exploration `temperature=`). PV collect
  contract: `(cm, hd, labels, pi, legal, alive)`; the general
  `policy_fn(state, key) → (action, pi)` plumbing (`as_policy_fn`,
  `_collect_pv`) is what the Step 3 PUCT generator plugs into with
  visit-distribution pi.

**Success criterion:** joint net's V at least matches Step 1's V in the arena;
policy-only player (no search) clearly beats random and is in the ballpark of
small-K rollout MCTS.

**Results (2026-06-12/13, three runs; full entries in the experiment log):**
run 1 found a target bug (the τ-sampled action as one-hot target teaches the
exploration noise — fixed to search-argmax targets). Run 2's policy learned
but greedy play was *worse than random*, and the post-mortem found two
architectural holes — card logits without global context, and no card
identity anywhere (rank/suit live in row position, invisible to the
row-shared trunk; the whole V₀/V₁ line was rank-blind) — both fixed in
`PolicyValueNet`. Run 3 (fixed architecture) **passed both gates**: policy
CE 0.48, policy-only **+9.9 vs random** (teacher: +33.7), value head
**+12.6 over V₁** (first value improvement of the project), and the
rollout-baseline yardstick moved for the first time since Step 0:
**PV-MCTS K=64 vs rollout K=8 N=8 = −20.6 ± ~2.5** (−19.8 at the 20k-epoch
extension = gen-0; policy-only reached +33, matching the teacher). That is
the number generation 1 had to beat. **STEP 2 CLOSED 2026-06-13.**

## Step 3 — PUCT via mctx (Option B) — the actual AlphaZero step  [Status: CLOSED ON THIS NET 2026-07-03 — gen-6 gated flat (+2.1/+1.3 ns, no artifact left to blame) and the sharpening probe found no search axis that re-opens the operator at gen-5b (+3.0/+2.4/+3.8, all ns, vs the +11–26 that drove climbs): the leaf evaluator is the cap. **CHAMPION since 2026-07-03: `pv_gen6b_es_s128.msgpack` (PolicyValueNetAttn — the Step-4 upgrade landed and PROMOTED).** The pre-registered re-open test is NOW DUE: re-run the operator probe on gen-6b_es (K=16 arm first), then decide on gen-7. Procedure → docs/jass_sop.md]

Implemented 2026-06-12 in `pgx/_src/games/jass_puct.py` (`puct_search`,
`puct_action`, `make_puct_action_fn`, `make_puct_policy_fn`,
`make_puct_collect_fn`). Requires `mctx` (`pip install mctx`, not
preinstalled on colab; like flax/optax it is intentionally not in
`requirements/requirements.txt`). Sign conventions validated end to end by
`test_puct_sign_conventions_beat_random`: PUCT (K=2, 16 sims) with a
greedy points-collected stand-in value beats uniform random by ≈ +14
pts/game over 64 games (t≈2.3); a perspective flip anywhere would make it
≈ −30 or worse. Notable implementation points beyond the sketch below:
reward on a tree edge is taken from the *parent mover's* perspective and
the team-aware discount is +1 within team / −1 across teams (consecutive
movers are NOT always opponents in Jass — trick winners lead, Schiebe
passes to partner — so the usual two-player discount=−1 is wrong here);
terminal states are held fixed with reward/discount 0 to prevent double
counting.

See `docs/jass_mcts.md` "Option B" for the integration sketch. Key decisions:

- `recurrent_fn` = `Game.step`; embedding = determinized `GameState`
  (a JAX pytree, so it works directly); root embeddings = K determinized
  states from the existing void-aware sampler.
- Priors = policy head logits (computed on each determinized state);
  leaf value = V head (already in player-relative differential form —
  mind the sign convention: mctx expects value from the perspective of the
  player to move at that node).
- **Aggregate across K determinizations by SUMMING VISIT COUNTS** (argmax of
  summed root visits), not Q values. This is the load-bearing choice — see
  "Key insight" above.
- `mctx.gumbel_muzero_policy` is the recommended entry point (works well at
  low simulation counts).
- Training targets: policy = aggregated root visit distribution (over the
  information-state legal actions), value = terminal score differential.
  Re-run the Step 1 loop with this generator.

This is "determinized AlphaZero": each per-determinization tree is a
perfect-information search, sidestepping AZ's perfect-info assumption.
Strategy fusion / non-locality is an accepted known flaw at this stage.

**Success criterion:** PUCT agent beats the Step 1/2 V-MCTS agent and the
rollout baseline at matched wall-clock.

**Results — the generation ladder (full dated entries, diagnostics, and
operational war stories in `docs/jass_experiment_log.md`):**

| gen (corpus) | gate vs prior champion | verdict |
|:--|:--|:--|
| gen-1 (gen-0 PUCT, s16) | PUCT@64 **+4.9 / +4.6** (p<0.005) | PROMOTED — first search-improved climb |
| gen-2 (3-way ⅓ mix, s16) | PUCT@64 **−5.0 / −4.1** | REGRESSED — 3-way mix drags to old policy |
| gen-2b (2-way, s16) | PUCT@64 −2.0 / +1.1 ns | WASH — s16 operator went NEGATIVE (−9 vs raw); sims sweep → s128 |
| gen-2 (s128) | PUCT@64 **+11.8 / +13.1** | PROMOTED — sims=128 recipe locked |
| gen-3 (s128) | PUCT@64 **+13.9** (t=9.5) | PROMOTED — loop self-sustaining |
| gen-4 (s128) | PUCT@64 +2.6/+4.4; **raw +15.0** | PROMOTED — PUCT@64 gate went blind; progress gate → raw-vs-raw |
| gen-5 (s128, 50/50) | raw **+1.5 / +2.4 ns**; PUCT@16 +0.5 | NOT promoted — the flat gate was the step2 anchor (see gen-5b) |
| gen-5b (s128, **0% step2**, same gen-4 corpus) | raw **+11.8 / +16.2** (t=6.7/8.0); PUCT@64 +2.2 ns | PROMOTED — anchor was the plateau; deployed check passed |
| gen-6 (s128, gen-5b corpus) | raw +2.1 / +1.3 ns | NOT promoted — read then as "sims=128 teacher exhausted" (corrected: architecture-relative, see gen-6b_es) |
| gen-6b_es (**PolicyValueNetAttn @ 7k epochs**, same gen-5b corpus) | raw **+10.3 / +7.4** (t=6.0/4.38); **PUCT@64 +5.2 (p=0.01)** | PROMOTED — first Step-4 net; first significant deployed gain since gen-3 |

Mechanism lessons pinned along the way (details in the log):

- **PUCT-vs-PUCT was the Step-3 gate** (V-MCTS arenas can't see policy
  gains) until gen-4, when PUCT@64 itself went blind to prior quality —
  the progress gate is **raw-vs-raw** since 2026-06-21.
- **Policy CE against soft visit targets is floored at the target's
  entropy** — diagnose with top-1 agreement, never raw CE.
- **The improvement operator must be measured, not assumed** (PUCT-vs-raw,
  at the corpus-generation sims): it went *negative* at sims=16 once the
  policy strengthened (gen-2b), was restored by sims=128, and is decaying
  again (+26 at gen-3 → +11 at gen-4).
- **Mix composition is load-bearing** because each generation retrains from
  scratch: the 3-way mix with an older PUCT corpus regressed a full
  generation.
- **The replay anchor becomes the binding constraint once the marginal
  signal is subtle — ablate the mix before declaring saturation.** At gen-5
  even the standing 50% step2 anchor absorbed the entire distillation
  gradient: flat gates on the same corpus that yields +12–16 without it
  (gen-5b). The anchor was the only never-ablated recipe component.
- **2026-06-20 DECISION and gen-4 planned recipe** are recorded in the log;
  the gen-5 VERDICT superseded their "keep cranking" guidance and was
  itself OVERTURNED by the 2026-07-02 mix ablation (gen-5b).

**HOW TO SCALE STAGE 1 (collection) — the per-generation bottleneck
(~3.3 h, ~75% of a generation; analysis 2026-06-20).** Cost driver is the
PUCT search per move: K=8 determinizations × sims=128 = 1024 sims/move,
each a `Game.step` + a (tiny) net eval. Collection is on the critical path
and CANNOT be pipelined across generations (gen N+1's corpus needs gen N's
champion). It is, however, the one **embarrassingly parallel** stage.

**MEASURED (2026-06-21, 1×1 batch sweep via `profile_collect_fn` at the
production K=8/sims=128 settings) — there is a sharp per-chip OPTIMUM at
B=64 games/chip, and the production collector was running ~5× SLOW.**
ms/game vs games-per-chip: 16→732, 32→656, **64→563 (min)**, 128→2048,
256→2885, 512→3320, 1024→4401. It is a sharp peak, not a plateau:
everything above 64 degrades *superlinearly*. **Not memory** — peak HBM was
0.41 G at B=1024 vs the 16.9 G limit. The cliff is the on-chip-memory
(VMEM) working set: the mctx tree `[B, K, sims]` of `GameState` embeddings
fits fast scratchpad at B=64 and spills to HBM in (64,128), so every sim
step starts streaming. **The optimum couples B,K,sims (≈ a fixed tree
working-set, B×K≈512) — re-profile if `num_determinizations` or
`num_simulations` change** (lower sims → bigger optimal B, higher sims →
smaller). Production ran `16×2048` pmap'd over 8 chips = **256 games/chip**
= 2885 ms/game ≈ 2.9 s/game/chip — exactly the recorded ~3 s/game/chip, so
the gen-2/3 corpus was generated ~5× slower than necessary. **B=64/chip =
0.563 s/game/chip → 32k games on the 2×4 in ~38 min (was 3.3 h), free, no
new hardware.** ⚠ Confirm with one real pmap'd collection at 64/chip (512
total) before rebuilding a corpus on it.

Levers, **REORDERED by the measurement** (batch-fix first, then chips):

1. **SET PER-CHIP BATCH TO ~64 — the free ~5× win, no recipe change, no
   quality risk.** This is now the first move. Find it with
   `jass_selfplay.profile_collect_fn(collect_fn, max_batch=…)` on **1×1**
   (per-chip numbers transfer to each pmap shard). ⚠ Do NOT use the
   "double until OOM" default here: this workload is VMEM-bound with a
   sharp optimum, OOM never comes, and the sweep marched 6 h overnight
   past a knee visible at 128 — bound the sweep and stop once ms/game
   rises past its min.
2. **MORE CHIPS — the clean linear win on top of the batch fix.**
   Collection pmap's ~8× near-perfectly over games, so wall-clock ÷ chips
   (on top of the ~38 min @ 64/chip). Routes: **TPU Research Cloud (TRC)**
   — free Cloud TPU pod slices (v5e/v4) for research, exactly this use
   case; run the SAME pmap collector on a v5e-16/32. Or GCP paid Cloud
   TPU. Or a Kaggle v3-8 as a *second* box — but generations are serial,
   so a second box can't speed one collection; only useful for concurrent
   *different* work (gate, JTR calibration, best-of-N replica).
3. **QUALITY-TRADING KNOBS (gate-validate only).** sims 128 → ~96 (sweep
   was +4@64 / +10@128; there may be a cheaper knee, but lowering sims is
   what starved gen-2 — re-run the PUCT-vs-raw-policy sweep at gen-3's
   strength first, the crossover rises as the policy strengthens). Fewer
   games (16k vs 32k) — halves cost IF corpus size isn't load-bearing;
   needs a one-off ablation.
4. **NOT WORTH IT:** bf16 (net too small for matmuls to bind); mctx
   subtree reuse (re-determinize every move, nothing to carry); any
   data-parallel trick (collection is *already* the parallel stage).

## Step 4 — Scale and benchmark externally  [Status: IN PROGRESS — external benchmark DONE (gen-3 calibrated 2026-06-20, gen-5b re-calibrated 2026-07-02: gap to POWERFUL roughly HALVED, −22/game → ≈−9.5/game; gen-6b_es/gen-7 re-calibrated 2026-07-05: gap CLOSED TO ZERO, both flat ns vs POWERFUL). Net scaling: IN PROGRESS since 2026-07-03 — value head first (attention over the 36 card rows vs mean pooling), per the sharpening-probe DECISION in the log: no search axis re-opens the operator on the current net, the leaf evaluator is the cap. First arm (gen-6b, shared-trunk PolicyValueNetAttn, standing recipe) first arm resolved 2026-07-03: full-20k training OVERFITS (value head fits seen data 0.073 vs holdout 0.147; gen-5b's gap is zero) but **early-stopped at 7k it PROMOTED — gen-6b_es is CHAMPION** (raw +10.3/+7.4, PUCT@64 +5.2 p=0.01, first significant deployed gain since gen-3). Remaining: weight-decay arm to replace hand-picked stopping; attn support in the JTR export scripts; re-profile collection for the attn net — see jass_sop.md "Current strategic state" and the log entries]

- Net scaling: attention over the 36 card rows is the natural upgrade from
  mean pooling; then width/depth, more simulations, larger batches.
  **Code DONE 2026-07-03:** `PolicyValueNetAttn` in `jass_value_net.py`
  (header broadcast into each row, 2 pre-LN self-attention blocks over the
  36 rows, learned-query attention pool replacing the mean; 393k params vs
  111k, ~6.5× CPU forward at B=512 — expect slower collection if it wins).
  Train via `train_pv_model(model=PolicyValueNetAttn(), ...)`; ⚠ checkpoints
  and weight files are architecture-specific — NEW filenames, and never
  restore PolicyValueNet weights into it (flax from_bytes silently mangles
  params restored against the wrong template). Training/gating plan → jass_sop.md
  "Current strategic state".
- **Cross-engine arena vs JassTheRipper.** It has a server/arena setup
  (`compare-strengths-arena.sh`, jass-server protocol — see its README and
  `JassInterface.pdf`). Build a thin bridge so the pgx agent can play it.
  This gives a calibrated external opponent instead of self-relative numbers.
- Trump selection needs no special work — it falls out of the policy/value net
  (apply each legal trump action hypothetically, or just use the policy head).
  It directly targets JassTheRipper's known weakness. Verify with a targeted
  arena where only the trump decision differs.

**DECISION (2026-07-05): NO JTR games in the training mix — self-play
only.** Tempting once the external gap closed to zero, but rejected:
(1) JTR is the only external benchmark; training on its games converts
"ties POWERFUL" from a generalization result into eval-on-train.
(2) The whole bet is AlphaZero-style generality — beating JTR without
ever seeing it should transfer to stronger/other engines (JTR++, KUS),
while a policy tuned on JTR's specific conventions (rule-based trump,
rollout-shaped lines) inherits its blind spots. (3) Data economics:
JTR games cost ~seconds each in Java vs ~ms in pgx self-play.
Revisit only if a dedicated held-out external opponent exists AND
self-play has demonstrably plateaued against it.

**Result (2026-06-20; full entry in the experiment log):** gen-3 exported
(Flax → TF SavedModel) and JTR integrated with real PUCT priors +
summed-visit aggregation; **gen-3 > gen-2 reproduces externally** (the old
argmax-tip + Q-sum setup had washed it). Absolute calibration: gen-3
**−22/game vs POWERFUL** (only −9 even at 256 runs/det, exceeding
POWERFUL's own search), ≈ level with FAST_TEST, the weakest MCTS baseline.
**The model is the limiter, not the integration or search depth** — the
lever is stronger pgx models (per the 2026-06-20 DECISION, in the log).
Not yet competitive.

**Re-calibration (2026-07-02; full entry in the experiment log):** gen-4 and
gen-5b exported and re-run through the same JTR real-PUCT harness. gen-5b >
gen-4 reproduces externally (+8.35/game, p=0.0000), and **the gap to
POWERFUL has roughly HALVED since gen-3: −22/game → ≈−9.5/game** (250
paired games, p=0.0000). The self-relative policy gains since gen-3 convert
substantially into absolute strength — still not competitive with POWERFUL,
but closing. gen-4's own POWERFUL calibration was skipped (deferred
straight to gen-5b), so this is only a two-point trendline.

## Step 5 (optional, research-grade) — imperfect-information refinements  [Status: TODO]

Only if early-game weakness persists after Step 4 and profiling points at
determinization quality:

- Learned belief model to bias determinization sampling. Negative in the
  thesis AND in JassTheRipper's auto-regressive experiments; revisit only with
  strong-play data and a clear hypothesis for why those failed.
- More practical: **distill** the determinized-search policy into a network
  over the 120-bit information-state observation — a fast standalone player
  with no hidden-information leakage, useful for deployment.

---

## Working agreements

- Every step gates on the arena harness (paired t-test, sign test, Wilson CI;
  swapped-deal pairing). A generation that doesn't measurably improve does not
  get promoted.
- Keep all generation weights; record arena numbers as dated entries in
  `docs/jass_experiment_log.md` (this file keeps only conclusions and the
  ladder summary).
- Don't retry the documented negative results without new evidence.
