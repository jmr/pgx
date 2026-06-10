# Jass AlphaZero Roadmap

Goal: an AlphaZero-style agent (policy+value network + PUCT search over
determinizations) for Jass, built incrementally on the pgx JAX environment.

This document is the working plan. It is written so that a fresh agent (or a
human returning after a break) can pick up from any step. Update the **Status**
markers as work completes; record arena results in the step's **Results** slot.

## Current state (as of 2026-06-10)

Implemented and tested (65 tests in `tests/test_jass*.py`):

| Component | File | Notes |
|:---|:---|:---|
| Game env (single hand, tournament rules) | `pgx/_src/games/jass.py`, `pgx/jass.py` | 43 actions, 120-bit obs; see `docs/jass.md` |
| Void-aware determinization + flat-rollout MCTS (Option A) | `pgx/_src/games/jass_mcts.py` | `best_action(state, player, key, K, N)`; jitted, vmapped |
| Full-information value features | `value_features()` in `jass.py` | (36,12) card matrix + (20,) header; see `docs/jass.md` |
| Value net + `train_model()` | `pgx/_src/games/jass_value_net.py` | per-card MLP → pool → head; target = differential/100. NOTE: defaults (200 epochs × batch 4096) are NOT the canonical V₀ settings — use `num_epochs=1000, batch_size=8192` (loss plateaus ~500) |
| Self-play data collection | `pgx/_src/games/jass_selfplay.py` | **currently uniform-random play** — see "Key insight" below |
| V wired into `best_action` as leaf evaluator | `jass_mcts.py` (`v_apply`/`v_params`/`v_scale`) | replaces N rollouts with one V(next_state) call; K=64, N=1 recommended |
| K/N sweep harness | `scripts/jass_sweep.py` | found K≥8 indistinguishable, K=4 worse → random rollouts are the ceiling |
| V-MCTS vs rollout arena | `pgx/_src/games/jass_v_arena.py` (`run_arena()`, colab-friendly; `scripts/jass_v_arena.py` is a thin CLI wrapper) | swapped-deal pairs; paired t-test + sign test on pair means, Wilson 95% CI |

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

---

## Step 0 — Baseline the random-play V₀  [Status: preliminary result; redo paired]

Cheap; do before any new code. Train V₀ with
`train_model(num_epochs=1000, batch_size=8192)` (the canonical V₀ settings —
NOT the CLI defaults), then run the arena (`run_arena()` or
`scripts/jass_v_arena.py`) against the K=8,N=8 rollout baseline.

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
- TODO: redo with the swapped-deal-paired arena so the recorded baseline is
  measured the same way as later Step 1 gates.

## Step 1 — Close the expert-iteration loop, V only  [Status: TODO]

Smallest change that adds the missing ingredient. Tasks:

1. **Agent-driven self-play.** Replace the uniform-random policy in
   `jass_selfplay.collect_batch` with the current best agent. For generation 1,
   V-greedy with exploration (softmax over V(next_state) at temperature τ, or
   ε-greedy) is acceptable and stays cheaply vmappable inside `lax.scan`.
   Full `best_action` per move inside the scan is too slow for data generation
   at first; optionally mix in a small fraction of search-generated games later.
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
   `jass_value_net.py` ("[Jass] Extract train_model() from CLI driver"), so
   colab and CLI share one loop. Remaining: the defaults (200 epochs ×
   batch 4096) still don't reproduce the evaluated V₀ — canonical settings
   are `num_epochs=1000, batch_size=8192` (loss plateaus ~500). Either bump
   the defaults or always pass them explicitly, and verify eval loss
   plateaus before a generation is arena-gated; otherwise gate failures are
   uninterpretable (bad data vs. undertrained net). For generation ≥1,
   `train_model()` needs a data-generator hook (e.g. a `collect_fn`
   argument) instead of the hardcoded random-play `collect_batch` import.
6. Iterate 2–3 generations.

**Success criterion:** monotone improvement across generations AND beating the
rollout baseline. If generation 2 does not beat generation 1, **debug here**
before adding anything (check: data distribution shift, target scale, buffer
staleness, exploration temperature).

**Results:** _(per-generation arena output here)_

## Step 2 — Add the policy head  [Status: TODO]

Extend `ValueNet` to a joint policy+value net (`jass_value_net.py` docstring
notes the per-card trunk was designed for this):

- **Card logits (36):** one logit per card from the per-card trunk (Dense(1)
  on each row before pooling).
- **Trump logits (7):** actions 36–42, from the pooled features + header.
- Mask illegal actions at the loss and at sampling (mask comes from
  `legal_action_mask`; note the mask is over the *information state*, which is
  identical across determinizations of the same root).
- Value head unchanged.

First uses, before PUCT exists:

- Self-play move sampler (much faster than search-per-move data generation).
- Policy training target for now: the search agent's chosen action
  (cross-entropy on the argmax), upgraded to visit distributions in Step 3.

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
