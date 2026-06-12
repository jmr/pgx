# Jass AlphaZero Roadmap

Goal: an AlphaZero-style agent (policy+value network + PUCT search over
determinizations) for Jass, built incrementally on the pgx JAX environment.

This document is the working plan. It is written so that a fresh agent (or a
human returning after a break) can pick up from any step. Update the **Status**
markers as work completes; record arena results in the step's **Results** slot.

## Status snapshot (2026-06-12, evening)

**Where we are:** Step 0 done (recorded baseline below). Step 1 closed as
a negative result (see its CONCLUSION block; do not run more V-greedy
generations). **The Step 2–3 infrastructure is now fully coded and
locally tested (95 tests)**, in atomic commits on `main`: batched search
arena (`run_batched_arena`), `PolicyValueNet` + masked-CE/MSE loss +
`train_pv_model` (checkpointing, suit-permutation augmentation on by
default, generation round-robin replay mixing), search/policy/PUCT data
generators in the PV contract `(cm, hd, labels, pi, legal, alive)`, and
determinized PUCT via mctx (`jass_puct.py`, visit-count-sum
aggregation; sign conventions arena-validated against random play).
Nothing is trained yet — Step 2/3 numbers are all TBD.

**Next colab session (in order):**

1. `pip install mctx` (extra step; not in requirements.txt), update the
   package, verify `from pgx._src.games.jass_puct import puct_action`.
2. ~~Validate the batched arena~~ **DONE 2026-06-12**: V₁-vs-V₀ gate
   (K=64, 100 games) neutral as expected, ~22 s total on TPU incl.
   compile (≈40× the sequential arena). See Step 1 task 6 for details.
3. **Step 2 training run:** search-generated data with the V₁ leaf —
   `make_search_collect_fn(v_apply, v1_params, num_determinizations=8,
   num_rollouts=1, v_scale=TARGET_SCALE, temperature=10.0)` into
   `train_pv_model(...)`. Note search data costs ~8× V-greedy data per
   game (K=8 × 43 evals/move); start with fewer epochs (e.g. 200–300,
   watch for the eval-loss plateau) and/or smaller batch; augmentation
   is on by default. If generation cost dominates, consider adding
   fixed-corpus reuse to the loop before scaling up.
4. **Step 2 gates:** (a) joint net's V head as MCTS leaf vs V₁ in
   `run_batched_arena` (K=64 both) — must not be worse; (b) policy-only
   vs random via `policy_match(make_policy_action_fn(pv_apply, params),
   random_action_fn, key, 256)` — must clearly beat random (V₀-greedy
   managed 69%/+30 in Q1, that's the bar to be in the ballpark of).
5. **Step 3 loop:** swap the generator for
   `make_puct_collect_fn(pv_apply, params, num_determinizations=8,
   num_simulations=64, temperature=1.0)` and re-run the Step 1 loop
   (gates + replay mixing via the collect_fn list). Tune K/sims to the
   TPU budget; PUCT data is far more expensive per game than 1-ply
   search data.

**Artifacts:** weights live on Drive under `MyDrive/jass/`: `v0.msgpack`,
`v1.msgpack` (canonical training, 1000 × 8192), plus `v1_ckpt.msgpack*`
slot files (deletable now that V₁ is finished). Everything through
`policy_match` is pushed (`main`).

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
    so a null at the gate's resolution (but see the 1000-game re-run
    below: there IS a small edge under the ±9 floor).
  - **AMENDMENT (2026-06-12, batched arena, 1000 games / 500 pairs,
    62 s on TPU):** V₁ vs V₀ (K=64 both): **mean +2.7, sd(pair
    mean)=27.2, t=2.2, p=0.028.** So generation 1 was not strictly
    neutral: one V-greedy generation bought ≈ +2.7 ± 2.4 pts/game,
    resolvable only at 1k-game power. Third look at this matchup
    (−2.5, ≈0, +2.7), so treat as "probably a small real edge". This
    does NOT reopen Step 1: ~3 pts/generation with Q2 showing the
    1-ply greedy improvement already saturated cannot close the
    ~37.5-pt gap to the rollout baseline. Conclusion below stands.

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

- **Card logits (36):** one logit per card. DONE (code, 2026-06-12):
  `PolicyValueNet` in `jass_value_net.py`, returns `(logits (B,43),
  value (B,))`. ⚠ The original sketch here — bare Dense(1) on each trunk
  row before pooling — is exactly the run 2 bug (see Results): each
  logit must also see global context, and the rows need the card
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

**Results:**

- **2026-06-12, run 1 — policy head learned nothing (target bug, fixed).**
  First PV training (corpus reuse: 12 × 4096 search games, K=8 V₁-leaf,
  τ=10, ~1.2 s/epoch on TPU vs 73 s/epoch fresh-data): value head healthy
  (eval v 0.30 → 0.14 by epoch 500), but policy CE flat at uniform-over-
  legal (1.32 → 1.31). Root cause: the collector emitted the τ=10-SAMPLED
  action as the one-hot target, i.e. it taught the policy the exploration
  noise, whose optimum is near uniform. Fixed in `make_search_collect_fn`
  (now `make_search_policy_fn`): pi = one-hot of the search ARGMAX,
  played action sampled separately. Search corpora collected before the
  fix have unusable pi targets — re-collect (labels/value data were fine).
  Practical numbers from this run, same hardware: collection 73 s per
  8192 games; batch 8192 training OOMs a 16G TPU (~13.5G train step +
  ~2G pinned corpus) — use batch 4096 and `jax.device_get` the corpus.
- **2026-06-12, run 2 (argmax-target fix) — policy head learns.** Corpus
  24 × 4096 search games (K=8 V₁-leaf, play τ=10, argmax targets),
  600 epochs: **eval policy CE 1.35 → 0.90, plateau by ~epoch 200**
  (uniform-over-legal ≈ 1.3; the floor includes irreducible K=8
  determinization noise in the argmax). Quota constraint lifted
  (2026-06-12), so future runs can use bigger corpora / 1k-game arenas
  freely.
- **2026-06-12, run 2 gates: (a) PASSED, (b) FAILED.**
  - Gate (a), PV value head vs V₁ as K=64 leaf (1000 games): **+1.4,
    t=1.6, p=0.10** — at least V₁'s equal; trunk sharing cost nothing.
  - Gate (b), policy-only vs uniform random (512 games): τ=1 (samples
    the raw policy, ~40% mass on the search argmax per CE 0.90):
    **−0.6, neutral**. Near-greedy τ=0.1: **43.5% wins, −11.2,
    t=−6.2 — significantly WORSE than random.** Sharpening hurts ⇒
    errors are confident and correlated (same pathology as the Step 0
    analysis), not uniform.
  - **Diagnostics (2026-06-12):** D1 — teacher (greedy K=8 V₁-search) vs
    random, 512 games: **72.3% wins, +33.7** — the data source is strong;
    imitation, not data, is the problem. D2 — hybrid splits vs random:
    trump-only **−2.0 (ns)**, cards-only **−7.7 (p=0.008)** — the card
    head is the pathology, the trump head is fine.
  - **ROOT CAUSE (architectural, two layers deep).** (1) Card logits were
    `Dense(1)` per trunk row, and the trunk processes rows independently —
    each card's logit saw only that card's own 12 bits: a context-free
    card priority table, which greedy play executes as a systematic
    (worse-than-random) bias. The trump head, which gets pooled+header
    context, was unaffected — exactly matching D2. (2) Deeper: the net
    had NO card identity — identity lives in row position, invisible to a
    row-shared trunk + mean pool. No rank/suit information (beyond
    is-trump) reached policy OR value. **Both fixed 2026-06-12** in
    `PolicyValueNet`: suit+rank one-hots appended to each row inside the
    module, and the card head now sees per-card features ⊕ pooled global
    context. Regression test pins the context path (first-vs-last-held
    task, unlearnable by the old head). NOTE: ValueNet is left
    identity-blind for V₀/V₁ artifact compatibility — meaning the entire
    Step 0/1 value line was rank-blind; the PV value head no longer is,
    so gate (a) may now beat V₁ rather than match it. Retrain required
    (same corpus is fine — data is architecture-independent).
- **2026-06-13, run 3 (fixed architecture, same 24×4096 corpus) — BOTH
  GATES PASS.** Training: total 1.60 → 0.59, **v 0.29 → 0.12, p 1.3 →
  0.48** (≈62% mass on the search argmax; still slightly falling at
  epoch 600 — old 0.90 "floor" was the architecture, not noise).
  - **Gate (a): +12.6 over V₁** (K=64 both, 1000 games) — decisive, and
    the first clear value-function improvement of the project. The
    rank-blind V line's ceiling was at least partly the missing card
    identity, not only the improvement operator.
  - Gate (b): policy-only (τ=0.1) vs random: **+9.9, p≈0.0000** — from
    −11.2 to clearly-better-than-random. Below the teacher's +33.7;
    CE still falling, so extension may close some of the gap. Good
    enough as PUCT priors regardless.
  - Remaining before closing Step 2: re-measure the rollout-baseline
    yardstick with the PV value head (V₁ measured −31.1 at 100 games;
    expect roughly −20 now — 1000 games, `run_batched_arena(pv_params,
    k_v=64, k_base=8, n_base=8, games=1000, v_apply=pv_value_apply)`),
    and optionally extend training to 1200 epochs via checkpoint resume
    and re-gate.

## Step 3 — PUCT via mctx (Option B) — the actual AlphaZero step  [Status: CODE DONE, untrained]

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
