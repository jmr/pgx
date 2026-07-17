# Jass PUCT loop — Standard Operating Procedure (per generation)

Runbook for one generation of the Step-3 PUCT expert-iteration loop
(gen `SRC` → gen `GEN`). Strategy and current state live in
`docs/jass_plan.md`, dated results and diagnostics in
`docs/jass_experiment_log.md`; this file is the *how*. Update it when the
procedure changes (e.g. the gate change of 2026-06-21).

## The single anchor (do this first, every cell)

Derive **every** path from the anchor tokens so you can't load a stale
generation. **Since 2026-07-06 the anchors are STRINGS, three tokens**
(attempt letters and recipe tags accrete — generations aren't abandoned
on failed attempts):

```python
CHAMP = "9"              # champion label — the params file you LOAD
SRC   = CHAMP + "_mz"    # corpus namespace: champion + teacher-recipe tag
GEN   = "10a"            # the student this run PRODUCES (sweep: 10-ctrl/10a/10b/10c)
```

For a **capacity sweep** (gen-10 onward) CHAMP/SRC are fixed across arms
(one shared corpus) and only GEN changes per arm — see "gen-10 —
capacity sweep" below.

⚠ Never build the champion-load path from `SRC` — the recipe tag makes
it a nonexistent file; and label gate baselines with `CHAMP`, not
`SRC`/`GEN` (a GEN-for-CHAMP print mixup mislabeled the 2026-07-06
gate output). Then all filenames are f-strings: `pv_gen{CHAMP}…`
(champion), `corpus_puct_gen{SRC}…` (corpus, named by generator +
recipe), `pv_gen{GEN}…` (student). The
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

- `make_puct_policy_fn(attn_model.apply, src_params, num_determinizations=16,`
  `num_simulations=64, search_variant="muzero", pb_c_init=1.25,`
  `temperature=1.0)`, pmap'd over the 8 chips. **RECIPE CHANGED
  2026-07-06: the teacher is classical PUCT (muzero_policy), K=16×64**
  — the Gumbel default washed two gen-8 attempts; ⚠ omitting
  `search_variant="muzero"` silently collects a fixed-point corpus.
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

## Belief-weighted determinization — integration (2026-07-17, DONE same day — GATE PASSED on the PUCT arm, config PROMOTED; log 2026-07-17)

**Gate outcome (2026-07-17):** PUCT arm +2.8 (p=0.057) / +5.3
(p=0.0003), pooled +4.1/game over 600 pairs (p<1e-4) — the
pre-registered "any significant +" clause fires; implied realized
q̄ ≈ 0.33 (between the 0.17 legality floor and the 0.56 matched
ceiling, as priced). Raw arm null (−1.3/−0.3 ns): gen-10's
hands-blind policy head has no mechanism to convert q — expected.
**Belief-weighted PUCT (K=16×64, N=32, λ=0, gen-11hc likelihood) is
the internal best play config; champion net stays gen-10; deployed
raw config unchanged. NEXT = external check vs POWERFUL, then decide
belief-weighted COLLECTION (the pass re-opens the gen-12 question —
all plateau negatives were measured at q≈0 search).** External-harness
note: the JTR driver must thread the public record; a mirror
trajectory whose opponent "hands" are their not-yet-played cards is
sufficient — the filter reads hand DIFFS (cards played per player)
and public fields only, all observable from outside.

**External harness BUILT (2026-07-17 late, JassTheRipper-2 commit
`rtmtvklk` "MCTS: belief-weighted determinization"). Ready to run.**
- **Export needed NO changes:** gen-11hc is the standing 128/2/4 attn
  recipe, same (cm, hd) contract; arch auto-inferred from the npz.
  Exported to `JassTheRipper-2/src/main/resources/models/pv_gen11hc/
  export` from `../jass-models/pv_gen11hc.msgpack` (fp 29962.42
  verified = sum |w|); JAX↔TF parity max |Δlogits| 1.6e-5.
  gen-10 export copied to `models/pv_gen10-ctrl_s128/export`.
- **JTR-side port** (`PgxBeliefFilter` + `JassBoard.setBeliefWorlds`):
  N=32 void-consistent worlds via the standard `CardKnowledgeBase`
  sampler, hc log-likelihood by replaying the public record on a
  shadow session (hands at step t = world hands now ∪ played-card
  suffix — no per-world game replay), log-softmax over the LEGAL card
  logits (illegal observed move ⇒ −1e9, the legality channel), root
  determinizations drawn ∝ weights with replacement. Two documented
  divergences from the pgx filter, both priced ~nil: the two
  trump-phase decisions are not scored (JTR's move record is cards
  only; trump-only q̄ 0.039), and trump-phase determinizations stay
  uniform.
- **Run** (from JassTheRipper-2, after `./gradlew installDist`):
  `build/install/JassTheRipper/bin/JassTheRipperArena
  --name1=belief-gen10 --strength1=<level>
  --pgx-model1=src/main/resources/models/pv_gen10-ctrl_s128/export
  --pgx-policy1 --pgx-trump1
  --pgx-belief1=src/main/resources/models/pv_gen11hc/export
  --name2=POWERFUL --strength2=POWERFUL --mode=RUNS --games=<n>`
  (knobs `--belief-particles1`, `--belief-mix1` default to the gate
  config 32 / 0). Baseline arm = same command without `--pgx-belief1`,
  per the standing vs-POWERFUL protocol (net trump on).
- **Smoke receipt** (2 games, FAST): both nets load, filter runs each
  card decision, ESS 32 at the first decision (uniform, empty
  history) → ~1.2–2.2 mid-game — the pgx gate saw ~2.5.

**Decision → plan NEXT (2026-07-17); priced by the probe (q̄ 0.56
ceiling, log 2026-07-17) and the dose-response law (12.6·q, log
2026-07-13). All in pgx — JTR stays the untouched external yardstick.**

**What.** At each decision, run the probe's particle filter at the
root: sample N void-consistent candidate worlds
(`sample_determinization`), weight each by the gen-11hc likelihood of
the OTHER three players' observed moves so far, then draw the K
search determinizations **with replacement ∝ weights** (mid-game ESS
~2.5 — duplicated trees ARE the q mass; do NOT deduplicate, do NOT
inject the true world — that was a measurement-only device). Wire
into `puct_search` (currently uniform at the root) and a
belief-weighted variant of `make_fair_raw_action_fn`.

**Design facts already established (session 2026-07-15/17, code in
`jass_probes.py` — factor the scoring core out of
`belief_quality_probe` rather than rewriting):**
- Past states under a candidate world need NO replay: hands at t =
  world's hands ∪ cards publicly played in [t,T); reconstruct = hands
  swap on the recorded public trajectory, evaluate hc from the
  mover-at-t's seat, sum masked log-probs. Both phases count (trump
  choices are near-uninformative, q̄ 0.039, but free).
- ⚠ **The arena/selfplay `action_fn(state, key)` interface cannot
  express this**: `GameState` does not record who played which card
  (`cards_collected` is by trick-WINNER), so the observed-move
  history is NOT recoverable from the current state. The play loop
  must thread a public-trajectory carry (38-slot stacked
  states/actions/movers buffer, exactly the probe's `play_game`
  record) alongside the state — new driver or extended action-fn
  convention; this is the main plumbing job.
- Fair by construction: the filter reads only own hand + public
  fields/diffs. Keep it that way — the oracle-contamination rule
  (raw TRUE-state arenas) does not bite here.
- The legality channel alone (constant-logits `uniform_pv_apply`) is
  q̄ 0.17 — a net-free fallback arm if hc plumbing is awkward, and
  the ablation to run if the hc arm disappoints.
- Cost per decision ≈ N × (steps so far) hc evals (~32×38 worst
  case) — same order as one K=16×64 search; HBM knob = chunking
  (`game_chunk` pattern).

**Measurement (gate as usual):** fair PUCT (muzero K=16×64 both
sides) belief-weighted vs gen-10 uniform-sampling, 300 pairs, seeds
0/2; plus belief-weighted fair-raw vs gen-10 raw τ=0.05. Ceiling
prediction +7.1/game; realized will be LOWER (deployed opponents ≠
the matched actor — the likelihood is mismatched vs gen-10-style
play). Any significant + at these harnesses promotes per the standing
gate; external check vs POWERFUL after, unchanged.

**Knobs (pre-register before running):** N particles (probe used 32;
baseline 1/33), K=16 dets unchanged, optional uniform mix λ on the
weights (guard against degenerate/misspecified likelihood — analog of
`prior_mix_uniform`; default 0, decide before the arena).

**Tooling (BUILT 2026-07-17, this stack):**
`pgx/_src/games/jass_belief.py` — `world_log_likelihoods` is the
scoring core FACTORED OUT of `belief_quality_probe` (probe re-verified
bit-identical after the refactor); `make_belief_world_fn` runs one
filter pass (N particles + normalized weights, λ mix knob);
`make_belief_puct_action_fn` resamples K dets ∝ weights and feeds them
to `puct_search(det_states=…)` (the new hook — uniform sampler
bypassed, no dedup, no true-world injection);
`make_belief_fair_raw_action_fn` plays the exact weighted policy
mixture over the N particles (no resampling, argmax = standing
fair-raw config). Plumbing: `PublicTrajectory` (38-slot stacked
states/actions/valid, the probe's `play_game` record) threaded by
`belief_policy_match` / `run_belief_arena` to `(state, traj, key)`
action fns; lift plain agents with `as_traj_action_fn`. Thin wrapper
`scripts/jass_belief_arena.py --arm puct|raw` runs both measurement
arms with the standing stats. Smoke-verified on CPU: legality-only
weights move off uniform mid-game (ESS 5.9/8 @ trick 6), belief-PUCT
vs uniform-PUCT arena runs end to end.

## Belief-quality probe — hc-likelihood particle filter (2026-07-15, DONE 2026-07-17 — q̄ = 0.56 ≈ 3× the buy bar → BUY; log 2026-07-17)

**Gate outcome (2026-07-17):** q̄ 0.5639 overall / 0.5848 card-play
(uniform 1/33), predicted +7.1/game at the matched-actor ceiling;
mass front-loaded (peak 0.66 @ trick 4); blind/legality-only arm
q̄ 0.166. The buy branch fires: likelihood-weighted world sampling in
`puct_search`/fair-raw, fair arenas vs gen-10, gate as usual.

**Decision → plan NEXT (2026-07-15). This runs BEFORE any search
integration: it prices the belief lever offline. Cost: an afternoon of
pgx code + CPU/TPU minutes. No collection, no training.**

**Principle.** Strength = 12.6·q per game (dose-response, LINEAR, log
2026-07-13); uniform world sampling has q ≈ 0; gen-11 proved
world-conditional skill without world mass is worth ZERO. The one
artifact that can concentrate mass is gen-11hc
(`pv_gen11hc.msgpack`, fp 29962.42): its policy is world-conditional
at teacher level (probe KL 0.229), so Bayes-invert it —
`P(world | observed opponent moves) ∝ Π P(move | world)` with the
likelihood read off the hc policy evaluated on the candidate world
FROM THE OPPONENT'S SEAT. No card-prediction head, no
marginals→joint sampling (the thesis-era belief-net mini-project
stays sidestepped).

**Design (all pgx, self-play, true world known):** run a particle
filter alongside self-play games — N candidate worlds per player
(sampled by `sample_determinization` at deal... particles must respect
revealed void constraints as the game evolves; simplest correct v0:
RESAMPLE N consistent worlds fresh at each decision and score each on
the FULL history of that game's opponent moves so far). For each
candidate world w and each past opponent decision t: reconstruct the
state at t under w (public history + w's hands), evaluate the hc
policy from the mover's seat, accumulate
`log L(w) += log P(observed move_t | w)`. At every decision of the
probed player report, vs the uniform 1/N baseline:
- **effective q**: normalized weight mass on the true world (and
  mass on worlds within Hamming distance d of it, d = misplaced
  hidden cards — the partial-overlap caveat, log 2026-07-13 pt 3);
- **placement accuracy**: weight-averaged fraction of hidden cards
  placed in the correct opponent hand;
- both **by trick** (constraints accumulate; belief should sharpen
  late — but late is also where worlds matter least, trick-7 JSD dip).

**Bar (pre-registered):** the payoff estimate is 12.6·q̄ per game.
- q̄ (or its overlap-adjusted equivalent) **≥ ~0.2** → predicted
  ≥ ~+2.5/game (one old-generation step) → BUY the integration:
  likelihood-weighted world sampling in `puct_search`/fair-raw,
  re-run the fair arenas vs gen-10, gate as usual.
- q̄ **≲ 0.05** → the likelihood route is dead at this net/budget —
  do NOT build search plumbing; next = exact endgame targets
  (plan queue).
- In between: check the trick profile before deciding (mass early
  beats mass late).

**Tooling (BUILT 2026-07-15, change `wtttptqq`):**
`belief_quality_probe` + `print_belief_quality_report` in
`pgx/_src/games/jass_probes.py`, thin wrapper
`scripts/jass_belief_quality_probe.py` (`--weights pv_gen11hc.msgpack
--games --particles --seed`). Implementation notes: true world is
injected as particle 0, so the uniform baseline is 1/(N+1); past
states under a candidate world need no replay (hands at t = world
hands ∪ cards publicly played in [t,T)); actor defaults to the hc net
raw τ=1 = matched likelihood = the route's CEILING (pass
`actor_action_fn` for mismatched-actor arms). ⚠ Run the `--blind` arm
too (constant-logits likelihood, same actor): the LEGAL-MOVE SET is
world-dependent evidence (1/n_legal + illegal-move exclusion), and
with a random-init net it accounts for ~all concentration — hc's
contribution = hc arm minus blind arm. `make_fair_raw_action_fn`
(world-averaged softmax, the standing fair eval mode for hands-aware
nets) is now in `jass_probes.py` too — the inline-colab TODO is
closed.

## gen-11 — hands-conditional policy targets (2026-07-13, DONE 2026-07-15 — mechanism bound at TEACHER level, fair deployment ZERO, ctrl refresh ZERO; NO promotion, champion stays gen-10; full arc → log 2026-07-15)

**Gate outcomes (2026-07-15):** (1) hidden-hand probe **PASSED AT THE
CEILING** — KL 0.003→0.229 ≈ teacher JSD 0.241, flips 23.3% ≈ 25.4%;
(2) fair arenas **WASH** — fair PUCT +0.5 ns, fair raw (16-world-
averaged softmax) −0.1/−3.9 vs gen-10; ⚠ the pre-registered raw gate
is oracle-contaminated for hands-aware nets (internal raw arenas feed
TRUE states; the +26.9/+24.7*** reading is the oracle track, not
deployed strength); (3) operator did NOT re-open; (4) external moot.
Plus: **ctrl gates −1.0/−1.1 ns vs gen-10 — the ~+2.5/gen corpus
refresh is GONE.** Fork fired: next = belief-weighted determinization
as the hc-likelihood probe (plan NEXT 2026-07-15); gen-11hc
(`pv_gen11hc.msgpack`) is the likelihood model.

**Decision + justification → plan (2026-07-13); pre-registered
gates → log 2026-07-12 (teacher-signal entry). Evidence: log
2026-07-12/13 (hidden-hand probe, teacher-signal probe, oracle arc).**

**Principle.** Collection today pairs TRUE-state features with the
visit sum across K=16 determinized trees — an info-set-marginal
target that trains the policy head to ignore the hidden-hand input
columns (measured: net KL 0.003 across worlds vs teacher JSD 0.24).
gen-11 fixes the pairing: each world's visit distribution becomes a
training row on THAT world's features. The value label stays on the
true-state row only (world features + true-world outcome would be a
mismatched value pair).

**Row types emitted per move (one collect serves BOTH arms):**

| row | features | pi | y | hc-arm masks | ctrl arm |
|:--|:--|:--|:--|:--|:--|
| 1× true-state | `value_features(state)` | aggregate (kept for ctrl) | outcome | v_mask=1, p_mask=0 | uses this row ONLY, legacy single mask |
| W=4× per-world | `value_features(det_states[k])` | tree-k root visits, normalized over legal | ignored | v_mask=0, p_mask=1 | unused |

W=4 of the K=16 trees (they are iid samples given the info set, so
the FIRST four are fine). 32×2048 games ⇒ ~64k value rows (standard)
+ ~256k policy rows (4× standard policy data). Trump-declare steps
(phase 0) get the same treatment — the trump search is determinized
too. ⚠ Do NOT train the hc arm's policy on the true-state row: true
features + marginal target is exactly the blindness bug.

**Code changes (all pgx) — ALL LANDED 2026-07-13:**
1. `jass_puct.py` — `return_visits=True` now returns `(scores, legal,
   visits, det_states)` (det_states = the (K,)-batched determinized
   roots); the single caller (`scripts/jass_teacher_signal_probe.py`)
   unpacks 4. Per-world features are `jax.vmap(value_features,
   in_axes=(0, None))` over the first W det_states.
2. `jass_selfplay.py` — `_play_one_pv_hc`/`_collect_pv_hc` (policy_fn
   returns `(action, pi_agg, world_cm, world_hd, world_pi)`); each
   step emits the (1+W) rows above, all sharing the step's `legal`
   and alive mask, as `(cm, hd, labels, pi, legal, v_mask, p_mask)`
   with a row axis after T. The generator entry point is
   `jass_puct.make_puct_hc_collect_fn(..., num_world_rows=4)`;
   `hc_batch_to_pv` slices the true rows back out as a legacy PV
   batch (the ctrl arm's view of the same corpus). Existing collect
   fns untouched (the ctrl arm and any rollback need them).
3. `jass_value_net.py` — `make_pv_train_step(head_masks=True)` takes
   per-row `v_mask`/`p_mask` (replacing the single `mask`):
   `v_loss = Σ v_sq·v_mask / Σ v_mask`, `p_loss = Σ ce·p_mask /
   Σ p_mask`, total `v_loss + policy_weight·p_loss`. Normalizing each
   head by its OWN mask sum preserves the current head balance
   despite 4× more policy rows. Accum/pmap paths match exactly (the
   heads' different normalizers are folded into each microbatch's
   loss as constants, so summed grads stay exact; verified by test).
   Train via `train_pv_model(collect_fn=<hc>, head_masks=True)`.
   Legacy single-mask signature is untouched code — checkpoint-resume
   streams of old runs are bit-for-bit unaffected.

**Stage 1 — collect ONCE from CHAMP=gen-10 (=10-ctrl, 128/2/4).**
Standing muzero recipe, passed explicitly: `num_determinizations=16,
num_simulations=64, search_variant="muzero", pb_c_init=1.25,
dirichlet_fraction=0, temperature=1.0`; standard 32×2048
(restart-safe per-shard). Re-run `profile_collect_fn` — the extra
per-world featurization is forward-free but the emitted arrays are
~5× per step; check per-chip B=8 still fits.

**Stage 2 — train two arms, same arch (128/2/4), full 20k, ES armed:**
- `gen-11hc` — value on true rows, policy on world rows (new masks).
- `gen-11ctrl` — true rows only, standard recipe: the corpus-refresh
  bar (~+2.5) AND the target-construction control (same generator,
  same games; the ONLY delta is target pairing).
⚠ Holdout split must stay GAME-level (per-shard batches, as now) so
world rows and the true row of the same position never straddle the
split. ⚠ Holdout policy CE is NOT comparable to any previous
generation (different target type AND features); compare hc-vs-ctrl
value floors only. Value channel is unchanged in size — expect the
usual U-curve watch.

**Stage 3 — gates, in the pre-registered order (log 2026-07-12):**
1. Mechanism check BEFORE arena time:
   `python scripts/jass_hidden_hand_probe.py --weights <gen-11hc>` —
   policy KL must move well off 0.003 toward (not necessarily to)
   the teacher's 0.24. Still ~0.003 ⇒ the pairing didn't bind; debug.
2. Raw gate, two seeds: gen-11hc vs gen-11ctrl (DECISIVE — must clear
   the control, capacity-sweep discipline) + vs gen-10 for the record.
3. Operator re-probe: muzero K=16×64 vs gen-11hc raw. Re-opened
   margin = the prize (restart the crank) even if (2) is thin.
4. External: JTR raw + PUCT vs POWERFUL (fair mode — determinization
   averaging is the CORRECT marginalization of a hands-conditional
   policy at inference).
Forks: (1) fails → fix pairing. (1) passes, (2)+(3) wash →
hands-conditional info doesn't convert; next = exact endgame targets
(plan queue). (3) re-opens → iterate the crank as gen-12.

## gen-10 — capacity sweep (2026-07-09, DONE 2026-07-10 — all arms arena-equal, 10-ctrl promoted; see log)

The operator is saturated (both muzero PUCT and JTR's classical MCTS
extract ≈0 over gen-9 raw — plan snapshot 2026-07-07), so capacity is
the lever for a stronger *raw* policy. gen-10 is a **sweep of student
architectures on ONE shared corpus**, not a single net.

**Why one corpus feeds every arm:** collection runs the CHAMPION
(gen-9) as the search generator; the *student* net you train afterward
never touches Stage 1. So each extra arm costs only a 20k train + a
gate — no new collection. This is what makes gen10-ctrl/a/b/c cheap.

**Stage 1 — collect ONCE** (CHAMP=`9`, load `pv_gen9_s128.msgpack`
with `PolicyValueNetAttn().apply`). Standing muzero recipe per Stage 1
above: `make_puct_policy_fn(attn.apply, src_params,
num_determinizations=16, num_simulations=64, search_variant="muzero",
pb_c_init=1.25, dirichlet_fraction=0, temperature=1.0)`. (`"muzero"`
is now the code DEFAULT — landed 2026-07-09 — but pass it explicitly.)
Collect **LARGER than the 64k standard: target ~128k = 64×2048**
(restart-safe per-shard) so the wide arms have data and so a
`batches[:32]` subsample gives a free dose-response check. Re-run
`profile_collect_fn` first (K/sims unchanged from gen-9, so per-chip
B=8 should still hold).

**Stage 2 — train the arms** (all on the shared corpus; full 20k,
`data_parallel=True`, ES armed). One axis at a time off the gen-9
baseline (`hidden 128 / num_layers 2 / num_heads 4`, ~393k params):

| GEN | model kwargs | file | tests |
|:--|:--|:--|:--|
| `10-ctrl` | `PolicyValueNetAttn()` | `pv_gen10-ctrl_h128.msgpack` | corpus refresh alone (control) |
| `10a` | `PolicyValueNetAttn(hidden=256, num_heads=8)` | `pv_gen10a_h256.msgpack` | width (~4× params) — headline |
| `10b` | `PolicyValueNetAttn(num_layers=4)` | `pv_gen10b_h128.msgpack` | depth at current width |
| `10c` *(cond.)* | `PolicyValueNetAttn(hidden=256, num_heads=8, num_layers=4)` | `pv_gen10c_h256.msgpack` | combine — only if a/b clears |

- **Filename tag.** ⚠ The historical `s128` tag means sims=128 (the
  corpus `num_simulations` — see the `corpus_..._s128k8` naming),
  which is coincidentally equal to the net's `hidden`; the muzero
  recipe dropped to sims=64 so `s` is now vestigial and does NOT track
  width. The table above shows the ideal `h{hidden}` convention, but
  **DECISION 2026-07-10: gen-10 keeps the vestigial `s128` in every
  arm's filename** (the `checkpoint_path` var carried it, and a
  mid-sweep rename would break snapshot/resume paths) — so the real
  files are `pv_gen10a_..._s128...`. The GEN token (`10a` vs
  `10-ctrl`) disambiguates arms regardless of the suffix. Actual gen-10
  names also carry per-axis tags (`10a_h256`, `10b_l4`) on TOP of the
  vestigial `s128`, and `s128` is injected in more than one call site —
  so the names are a mild mess (e.g. `pv_gen10b_l4_s128...`), harmless
  but ugly. **TODO gen-11 — full naming-convention review** (⚠ NEEDS
  REVIEW, flagged 2026-07-10): drop `s128`, standardize on
  `gen{N}{arch-tag}` where the tag encodes what varies (`h{hidden}`,
  `l{layers}`), and de-duplicate the tag-injection sites. Do it at the
  gen-11 boundary, not mid-gen-10 (rename breaks snapshot/resume paths).
- `num_heads` must divide `hidden` (keep head_dim=32: 128→4, 256→8).
- **Run ctrl + a + b first, read them, THEN decide on 10c.** Don't
  fan out all four at once — you want to know if capacity moves the
  needle before spending the combine arm.
- **Watch the U-curve.** The muzero teacher is what let 128/2 train
  full 20k with no overfit (holdout v 0.0655); wider nets have more
  room to memorize. Keep `snapshot_every=500`, watch holdout v,
  early-stop (`_es` suffix) only if an upturn appears.
- Separate `checkpoint_path` per arm (distinct GEN token).

**Stage 3 — gate** each arm raw-vs-raw vs **gen-9**
(`pv_gen9_s128.msgpack`, `src_params`), two seeds, p<0.05 (Stage 3
above). Best clearing arm → gen-10 champion. Losing arms keep their
suffix — attempts, not failures.

**Dose-response (post-winner, free):** retrain the winning arch on
`batches[:32]` (64k) and gate vs the full-128k net — separates "needs
more capacity" from "needs more data."

**External (DEFERRED, DECISION 2026-07-09):** the vs-POWERFUL number
(net trump on) needs the export scripts taught the arch dims —
`scripts/export_pv_savedmodel.py` hardcodes `HIDDEN=128` and has no
`--hidden/--heads/--layers` CLI flags (the keras mirror constructor
already accepts them). Add those flags **only once we have a winning
arch worth exporting** — not before.

**SWEEP RESULT (2026-07-10): capacity is NOT the lever at 64k — all
three arms arena-equal, gen-10 = 10-ctrl (128/2) promoted.** Width (10a)
overfit → wash; depth (10b) clean but still washed. Full read in the
experiment log (2026-07-10 entries). One confound left: width's wash
was data-starvation (256-wide, 64k), so it gets re-run well-fed next.

## gen10c — the 128k width re-feed (2026-07-10, DONE 2026-07-11 — capacity dead; see log)

Resolves the one open sweep confound: was 256/2's wash just data
starvation? Still the **gen-10 family** (gen-9-generated corpus), just
the extended 128k version — **NOT gen-11** (that stays reserved for a
gen-10-generated corpus). ⚠ `10c` is REDEFINED here: the abandoned
256/4/8 combine arm (never ran) → now the 128k re-feed.

**Stage 1 — EXTEND the existing corpus, don't recollect.** You have
32×2048 from the sweep, generated by **gen-9**. Collect **32 MORE
batches from the SAME generator** (`pv_gen9_s128.msgpack`,
`PolicyValueNetAttn().apply`, standing muzero recipe per Stage 1) →
homogeneous **64×2048 ≈ 128k**. ⚠ Do NOT generate the new batches from
gen-10 — a mixed-generator corpus confounds the width comparison, and
the whole point is DATA as the only variable vs the sweep. Restart-safe
per-shard as usual; you only pay for the delta 32 batches (~half a fresh
collect). The existing 32 batches become `batches[:32]` of the 128k.

**Stage 2 — two arms on the 128k corpus** (full 20k, `data_parallel`,
ES armed):

| GEN | model | file | role |
|:--|:--|:--|:--|
| `10c_ctrl` | `PolicyValueNetAttn()` (128/2) | `pv_gen10c_ctrl_..._s128...` | width baseline @128k + coverage test |
| `10c` | `PolicyValueNetAttn(hidden=256, num_heads=8)` (256/2) | `pv_gen10c_h256_..._s128...` | THE test — width well-fed |

- 64k control is the **existing 10-ctrl** (128/2 @ `batches[:32]`) — no
  retrain; it IS the coverage baseline for `10c_ctrl`.
- Watch the value U-curve on `10c` — if 128k still bends it, take the
  `_es` snapshot at the value floor (as 10a did at 64k). If it now
  trains clean through 20k, that alone is evidence the starvation is
  fixed.

**Stage 3 — decisive gate: `10c` (256/2) vs `10c_ctrl` (128/2), both
@128k**, raw-vs-raw, two seeds. Also gate each vs gen-10 (= 10-ctrl) for
the absolute step. **Fork:**
- **10c beats 10c_ctrl** → capacity was DATA-blocked. Scale data+width
  together; THEN do a real **gen-11** from the gen-10 generator with the
  wider arch. (And revisit the export `--hidden/--heads/--layers` flags.)
- **10c ties/loses well-fed** → capacity is genuinely dead. The
  ~+2.5/gen corpus refresh is the only lever → gen-9 self-play has
  PLATEAUED; next gain needs a NEW target source (operator saturated,
  JTR off-limits, capacity dead) — a strategic pivot, not another arch.
- Coverage side-read: `10c_ctrl` (128/2 @128k) vs 10-ctrl (128/2 @64k) —
  does more data move ctrl at all? (Expect little; ctrl wasn't starved.)

## Current strategic state (SUPERSEDED — see gen-10 sweep above; kept for the gen-8d_mz recipe detail)

**CHAMPION: gen-8d_mz (`pv_gen8d_mz.msgpack`) — `PolicyValueNetAttn`,
64k muzero corpus (K=16×64, pb_c=1.25), full 20k (NO early stop
needed).** Promoted 2026-07-06 on raw +13.7/+11.0 (both seeds ***).
The teacher searcher is the recipe change that did it: classical
PUCT (`search_variant="muzero"`) replaced Gumbel-read-by-visits at
collection — see the 2026-07-06 log arc for the searcher post-mortem
and the reinterpretation of the retired fuel gauge (it measured
Gumbel's readout starvation, not teacher exhaustion; the muzero
operator margin is a MEANINGFUL gauge again — +10.5 at gen-7).
⚠ Champion + generator are ATTN: every cell that loads them needs
`PolicyValueNetAttn().apply`; a `PolicyValueNet` template silently
mangles the params. Gate cells against pre-attn generations need TWO
model instances.

- **The attn recipe (updated 2026-07-06): full 20k; early stopping is
  CONTINGENT on the corpus.** Muzero corpora: the 64k gen-8d_mz run
  showed NO U-curve through 20k (holdout v ≈ 0.074, seen-vs-holdout
  gap closed) — train full 20k, keep full logs + `snapshot_every=500`
  as insurance, early-stop only if an upturn appears. The old
  U-minimum table (15 batches → 7k, 31 → 10k, floors 0.113–0.115)
  applies to GUMBEL-corpus runs only — the U-curve was the value head
  memorizing weak-play outcome noise. Weight decay stays SHELVED.
  Everything else per Stage 2, plus `model=PolicyValueNetAttn()` and
  `data_parallel=True`. Training-health check on muzero corpora:
  holdout v ≈ 0.074 band, policy CE ≈ 0.72 band (peaked targets —
  not comparable to the old 0.9585 gumbel-target floor).
**Queued work (order decided end-of-session 2026-07-04):**

0. **DONE (2026-07-06): the gen-8 RETAKE — gen-8d_mz PROMOTED
   (+13.7/+11.0 raw, both seeds ***; trained full 20k, NO U-curve,
   no `_es` needed). NEXT: gen-9, same recipe, SRC=8d_mz.** The
   muzero collection config is now the STANDING recipe (see Stage 1
   note below); re-probe the operator (muzero K=16×64 vs new raw)
   while the gen-9 corpus collects. Original arc: `search_variant="muzero"` (classical
   PUCT via `mctx.muzero_policy`, landed `sxznyotm`) beats gen-7 raw
   **+11.8*** at K=45×64** where Gumbel read −1.1 ns — same budget,
   same net, same deals; pb_c plateau 0.64–2.5. Pre-collection probe
   DONE: the margin holds cheap — **SRC=7b_es_mz collection config =
   `search_variant="muzero", pb_c_init=1.25, num_determinizations=16,
   num_simulations=64`** (+10.5 at the standing 1,024-exp/move cost;
   worlds-over-depth: 8×128 only +7.1), `dirichlet_fraction=0`,
   τ=1.0, 32×2048, per-chip B=8 pending a quick `profile_collect_fn`
   check. Dose-response is LIVE again. Full
   arc: log 2026-07-06 entries + plan NEXT block. ⚠ The Gumbel
   default is UNCHANGED in code — collection cells must pass
   `search_variant="muzero"` explicitly.
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
