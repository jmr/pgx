# Jass AlphaZero Roadmap

Goal: an AlphaZero-style agent (policy+value network + PUCT search over
determinizations) for Jass, built incrementally on the pgx JAX environment.

This document is the working plan. It is written so that a fresh agent (or a
human returning after a break) can pick up from any step. Update the **Status**
markers as work completes; record arena results in the step's **Results** slot.

## Status snapshot (2026-06-12)

**Where we are:** Step 0 done (recorded baseline below). Step 1 is **closed
as a negative result** (2026-06-12): generation 1's promotion gate was
neutral, and the Q1/Q2 diagnostics localized the failure — the V₀-greedy
data was genuinely much better than random (Q1: 69%/+30 vs random play),
yet V₁ improved neither as MCTS leaf (gate) nor as greedy policy (Q2:
neutral). 1-ply V-greedy expert iteration is saturated; do not run more
V-greedy generations. **Next: Steps 2–3 (policy head + PUCT), starting
with the batched search self-play infrastructure** — see the Step 1
CONCLUSION block for rationale.

**Artifacts:** weights live on Drive under `MyDrive/jass/`: `v0.msgpack`,
`v1.msgpack` (canonical training, 1000 × 8192), plus `v1_ckpt.msgpack*`
slot files (deletable now that V₁ is finished). Everything through
`policy_match` is pushed (`main`).

**Colab workflow:** train on TPU; arena/diagnostics on CPU runtime
(`JAX_PLATFORMS=cpu` — `run_arena` is dispatch-bound; `policy_match` is
vmapped and fast anywhere). Update the package with
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

**Results:**

- **~2026-05 (preliminary, pre-pairing-fix arena, 100 unpaired games):**
  V-MCTS K=64 with random-play V₀: **31 wins vs 69** for the random-rollout
  K=8 N=8 baseline, t ≈ −6 (decisive, ≈20+ pts/game deficit).
  Confirms the thesis negative result in this stack. V₀ was trained to
  plateau (colab: ~1k gradient steps at batch 8192, loss flat after ~500),
  so this is a clean measurement of the random-play-value ceiling, not an
  undertraining artifact. Below-parity (not just parity) is explained by:
  (a) V₀'s approximation error is *biased* and highly correlated across the
  64 determinizations (similar feature inputs), so it doesn't average out
  the way rollout noise does, and argmax action selection harvests the
  bias; (b) late-game random rollouts are near-exact (tiny remaining tree)
  while V₀'s error is constant across stages; (c) V₀ also picks trump,
  where random-play values are least informative.
- **2026-06-10 (RECORDED BASELINE — swapped-deal-paired arena, 100 games /
  50 pairs, colab 1×1 v5 TPU):** challenger V-MCTS K=64 with random-play V₀
  vs random-rollout K=8 N=8: **win 33%, mean −37.5 pts/game, sd(game)=66.8,
  sd(pair mean)=32.5, t=−8.08** on pair means (p≈1e-10). Consistent with
  the preliminary run, deficit even larger. This is the yardstick Step 1
  generations must beat. Note: sd(pair mean)≈half sd(game) → pairing gives
  ~2× effective sample size here (same-deal games diverge via trump
  choices, so cancellation is partial).

## Step 1 — Close the expert-iteration loop, V only  [Status: TODO]

Smallest change that adds the missing ingredient. Tasks:

1. **Agent-driven self-play.** DONE (code):
   `jass_selfplay.make_v_collect_fn(v_apply, v_params, v_scale=TARGET_SCALE,
   temperature=10.0)` — V-greedy softmax over V(next_state), vmappable,
   same contract as `collect_batch`; plug into
   `train_model(collect_fn=...)`. Full `best_action` per move inside the
   scan is too slow for data generation at first; optionally mix in a
   small fraction of search-generated games later.
2. **Suit-permutation augmentation.** Documented in `docs/jass.md` but NOT yet
   implemented. 3! = 6× in trump modes (non-trump suits permute freely; trump
   suit fixed), 4! = 24× in Obenabe/Undeufe. Apply to (cm, hd, y) batches —
   permuting suits = permuting 9-row blocks of the card matrix + remapping the
   trump one-hot in the header.
3. **Replay buffer.** Mix the last few generations of data (e.g. uniform over
   the most recent 3) to avoid catastrophic drift.
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
   (but not bitwise the same games for a given seed). Colab validation
   (reproduce a known result, e.g. V₁-vs-V₀ neutral, and time it on TPU)
   still pending.
7. Iterate 2–3 generations.

**Success criterion:** monotone improvement across generations AND beating the
rollout baseline. If generation 2 does not beat generation 1, **debug here**
before adding anything (check: data distribution shift, target scale, buffer
staleness, exploration temperature).

**Results:**

- **2026-06-12, generation 1 — GATE FAILED (neutral).** V₁ trained with
  canonical settings (1000 epochs × 8192, ~5.7 s/epoch on 1×1 v5 TPU;
  train/eval loss 0.124/0.127, fairly flat by 1000) on V₀-greedy data
  (`make_v_collect_fn`, temperature=10).
  - V₁ vs rollout K=8 N=8 (yardstick, seed 0, 100 games / 50 pairs):
    win 32%, mean −31.1, sd(game)=65, sd(pair mean)=32.7, t=−6.66.
    Nominally 6.4 pts better than V₀'s −37.5 but within noise.
  - V₁ vs V₀ (gate, K=64 vs K=64, 100 games / 50 pairs): **49 wins,
    mean −2.5, sd(pair mean)=23.8, t=−0.7 — neutral.** Power was ~±9 pts,
    so this is a genuine null, not underpowered.

  **Debugging hypotheses (in test order):**
  1. ~~The V₀-greedy generator barely improves on random play~~
     **ELIMINATED 2026-06-12** by `policy_match` Q1 (256 pairs / 512
     games): V₀-greedy(τ=1) vs uniform-random player: **69% wins,
     +30 pts/game, t=11**; τ=10: 66%, +30, t=11. The generator's play is
     far better than random and nearly temperature-insensitive in
     [1, 10] — V₁'s training data was genuinely improved. (Note the
     uniform-random *player* here is much weaker than the rollout-MCTS
     arena *baseline*; V₀-greedy beating one while V₀-MCTS loses to the
     other is consistent.)
  2. One step of 1-ply-greedy policy iteration saturates.
     **CONFIRMED 2026-06-12 by Q2** (`policy_match`, V₁-greedy vs
     V₀-greedy, τ=1, 256 pairs): **neutral**. So: training data improved
     a lot (Q1), the trained V did not (Q2 + gate). The improvement
     operator — not the data pipeline — is the bottleneck; consistent
     with the Step 0 analysis (correlated net bias dominates at the
     action-gap scale).
  3. Not yet done and relevant regardless: suit-permutation augmentation
     (task 2) and replay-buffer mixing (task 3).

  **CONCLUSION — Step 1 closed as a negative result (2026-06-12).**
  V-only 1-ply-greedy expert iteration does not climb in this setting.
  Do not run more V-greedy generations. The path forward is the
  AlphaZero structure: search as the improvement operator — Step 2
  (policy head) + Step 3 (PUCT via mctx), with search-generated training
  data. Prerequisite infrastructure for both: **batched search self-play**
  (vmap `best_action` over a batch of games in lockstep scan), which also
  delivers the batched arena (Step 1 task 6). With the V leaf (K=64, N=1)
  this is TPU-friendly; rollout leaves are too expensive to batch at
  scale.

## Step 2 — Add the policy head  [Status: IN PROGRESS]

Extend `ValueNet` to a joint policy+value net (`jass_value_net.py` docstring
notes the per-card trunk was designed for this):

- **Card logits (36):** one logit per card from the per-card trunk (Dense(1)
  on each row before pooling). DONE (code, 2026-06-12): `PolicyValueNet`
  in `jass_value_net.py`, returns `(logits (B,43), value (B,))`.
- **Trump logits (7):** actions 36–42, from the pooled features + header.
  DONE (same).
- Mask illegal actions at the loss and at sampling (mask comes from
  `legal_action_mask`; note the mask is over the *information state*, which is
  identical across determinizations of the same root). DONE at the loss:
  `make_pv_train_step` (masked cross-entropy + value MSE, `policy_weight`
  knob).
- Value head unchanged (structurally identical layers; fresh init).

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

**Results:**

## Step 3 — PUCT via mctx (Option B) — the actual AlphaZero step  [Status: TODO]

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

**Results:**

## Step 4 — Scale and benchmark externally  [Status: TODO]

- Net scaling: attention over the 36 card rows is the natural upgrade from
  mean pooling; then width/depth, more simulations, larger batches.
- **Cross-engine arena vs JassTheRipper.** It has a server/arena setup
  (`compare-strengths-arena.sh`, jass-server protocol — see its README and
  `JassInterface.pdf`). Build a thin bridge so the pgx agent can play it.
  This gives a calibrated external opponent instead of self-relative numbers.
- Trump selection needs no special work — it falls out of the policy/value net
  (apply each legal trump action hypothetically, or just use the policy head).
  It directly targets JassTheRipper's known weakness. Verify with a targeted
  arena where only the trump decision differs.

**Results:**

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
- Keep all generation weights and record arena numbers in this file.
- Don't retry the documented negative results without new evidence.
