# Jass experiment log

Chronological record of experiments, arena results, and diagnostics for the
Jass AlphaZero project. **Strategy, current state, and the step plan live in
`docs/jass_plan.md`; the per-generation procedure is `docs/jass_sop.md`.**
This file is the append-only history: entries are dated, oldest first, moved
here verbatim from the plan's old per-step **Results** slots (2026-07-02
restructure). Append new results here; the plan keeps only conclusions and
pointers.

All arena numbers are pts/game from the challenger's perspective unless noted;
gates use the swapped-deal paired t-test harness (see plan Working agreements).

---

## ~2026-05 — Step 0 preliminary: random-play V₀ loses decisively

(Preliminary, pre-pairing-fix arena, 100 unpaired games.) V-MCTS K=64 with
random-play V₀: **31 wins vs 69** for the random-rollout K=8 N=8 baseline,
t ≈ −6 (decisive, ≈20+ pts/game deficit). Confirms the thesis negative result
in this stack. V₀ was trained to plateau (colab: ~1k gradient steps at batch
8192, loss flat after ~500), so this is a clean measurement of the
random-play-value ceiling, not an undertraining artifact. Below-parity (not
just parity) is explained by: (a) V₀'s approximation error is *biased* and
highly correlated across the 64 determinizations (similar feature inputs), so
it doesn't average out the way rollout noise does, and argmax action selection
harvests the bias; (b) late-game random rollouts are near-exact (tiny
remaining tree) while V₀'s error is constant across stages; (c) V₀ also picks
trump, where random-play values are least informative.

## 2026-06-10 — Step 0 RECORDED BASELINE

(Swapped-deal-paired arena, 100 games / 50 pairs, colab 1×1 v5 TPU.)
Challenger V-MCTS K=64 with random-play V₀ vs random-rollout K=8 N=8:
**win 33%, mean −37.5 pts/game, sd(game)=66.8, sd(pair mean)=32.5, t=−8.08**
on pair means (p≈1e-10). Consistent with the preliminary run, deficit even
larger. This is the yardstick Step 1 generations must beat. Note: sd(pair
mean)≈half sd(game) → pairing gives ~2× effective sample size here (same-deal
games diverge via trump choices, so cancellation is partial).

## 2026-06-12 — Step 1 generation 1 (V-greedy): GATE FAILED; Step 1 closed

V₁ trained with canonical settings (1000 epochs × 8192, ~5.7 s/epoch on 1×1
v5 TPU; train/eval loss 0.124/0.127, fairly flat by 1000) on V₀-greedy data
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
   and replay-buffer mixing (Step 1 tasks 2–3).

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

## 2026-06-12 — Step 2 run 1: policy head learned nothing (target bug, fixed)

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

## 2026-06-12 — Step 2 run 2: policy learns; gate (b) FAILED; root cause found

Run 2 (argmax-target fix): corpus 24 × 4096 search games (K=8 V₁-leaf, play
τ=10, argmax targets), 600 epochs: **eval policy CE 1.35 → 0.90, plateau by
~epoch 200** (uniform-over-legal ≈ 1.3; the floor includes irreducible K=8
determinization noise in the argmax). Quota constraint lifted (2026-06-12),
so future runs can use bigger corpora / 1k-game arenas freely.

Gates: **(a) PASSED, (b) FAILED.**

- Gate (a), PV value head vs V₁ as K=64 leaf (1000 games): **+1.4,
  t=1.6, p=0.10** — at least V₁'s equal; trunk sharing cost nothing.
- Gate (b), policy-only vs uniform random (512 games): τ=1 (samples
  the raw policy, ~40% mass on the search argmax per CE 0.90):
  **−0.6, neutral**. Near-greedy τ=0.1: **43.5% wins, −11.2,
  t=−6.2 — significantly WORSE than random.** Sharpening hurts ⇒
  errors are confident and correlated (same pathology as the Step 0
  analysis), not uniform.
- **Diagnostics:** D1 — teacher (greedy K=8 V₁-search) vs
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

## 2026-06-13 — Step 2 run 3 (fixed architecture): BOTH GATES PASS; Step 2 closed

Same 24×4096 corpus. Training: total 1.60 → 0.59, **v 0.29 → 0.12, p 1.3 →
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
- **Yardstick re-measured (2026-06-13, 1000 games, ~1.6 s/game —
  rollout side dominates): PV-MCTS K=64 vs rollout K=8 N=8: −20.6,
  *** — first movement of this gap since Step 0** (V₀ −37.5, V₁
  −31.1, both at 100-game ±9 precision; this one ±2.5). Matches the
  gate (a) transfer prediction (−31.1 + 12.6 ≈ −18.5). **This is the
  number Step 3's first PUCT generation must beat.**

**STEP 2 CLOSED 2026-06-13.** Success criteria met: value head beats
(not just matches) Step 1's V; policy-only clearly beats random
(+9.9), though below the teacher's +33.7 — CE was still falling at
epoch 600, so an extension run (`num_epochs=1200`, same checkpoint,
resume) may close some of that gap; worth doing before or alongside
early Step 3, but PUCT retrains the policy on visit distributions
regardless.

## 2026-06-13/14 — run 3 extended (same checkpoint, no architecture change)

Resumed past epoch 600 — policy CE still falling, approaching a plateau.
Eval losses (total / value / policy CE):

- epoch 600 (run 3 gate point): 0.59 / 0.12 / 0.48
- epoch 2000: 0.47 / 0.08 / 0.39
- epoch 10000: 0.39 / 0.080 / 0.32
- epoch 20000: 0.36 / 0.076 / 0.29

Policy CE dropped from 0.48 (the value already gated at this point: +12.6
over V₁) to 0.29 — well below the old (pre-identity-fix) architecture's
0.90 floor and getting closer to the teacher's implied target (teacher
greedy-K8-V₁-search was +33.7 vs policy-only +9.9 at CE 0.48). Value loss
also kept improving (0.12 → 0.076).

## 2026-06-14 — re-gate of the 20k-epoch checkpoint (= gen-0)

(1000 games each, same arena setup as the run3@600 gates, p=0.0000 to 4dp
for both.)

- Gate (a) vs V₁ (K=64 both): **60.2% win, mean +13.2, p<0.0001** —
  essentially matches run3@600's +12.6, i.e. no regression and a small
  further gain. Value head is at least holding its Step-2 win over V₁.
- Gate (b) policy-only (τ=1) vs uniform random: **72.9% win, mean +33,
  p<0.0001** — a large jump from run3@600 (τ=0.1, +9.9), and now
  essentially matches the search-teacher's own strength (greedy
  K8-V₁-search vs random: 72.3% win, +33.7, 512 games). The policy head
  has nearly closed the gap to the teacher it was distilled from.

Net effect: the 20k-epoch checkpoint is at least as good as the
run3@600 checkpoint on value and a clear, significant improvement on
policy.

## 2026-06-14 — rollout yardstick on the 20k checkpoint

PV-MCTS K=64 vs rollout K=8 N=8, 1000 games via `run_batched_arena` with the
PV value head as leaf: **mean −19.8, t=−14.8 vs zero, SE ≈ 1.3.**
Indistinguishable from run3@600's −20.6 (Δ0.8 ≈ 0.3σ on a combined SE
≈ 2.8). This is the *correct* result, not a stall: the yardstick is a
V-MCTS arena, so only the value head is exercised — and the value head
barely moved (gate vs V₁ +12.6 → +13.2, also within noise). The 20k
extension's gains were on the policy head (CE 0.48 → 0.29, policy-only
+9.9 → +33), which does not enter a V-MCTS leaf at all; it is only
load-bearing as PUCT priors. So gen-0 is fully validated (no value
regression) and −20.6/−19.8 is the gen-0 baseline that **generation 1
(PUCT-trained, priors active) must beat.** (Loading note: the PV
checkpoint must be restored with a `PolicyValueNet` template — a
`ValueNet` template silently downcasts it to a 4-Dense tree that fails
only at apply with `ScopeParamNotFoundError: Dense_4`.)

## 2026-06-15 — GENERATION 1 — first PUCT generation, PROMOTED

Gen-1 (fresh `PolicyValueNet`) trained on a gen-0 PUCT corpus
(`make_puct_collect_fn`, K=8, **sims=16**, τ=1.0; 32k games = 8×4096) +
the Step-2 corpus, **50/50 replay mix** (`collect_fn=[puct_fn,
step2_fn]`), 20k epochs, host corpus + `split=2` (2048 games/step for
TPU memory). Eval: value loss 0.27 → 0.10 (≈ gen-0's 0.076);
**policy CE flat at 1.30 the entire run.**

- **The CE was NOT a stall — it was the entropy floor of the soft PUCT
  target.** CE against a soft target is floored at the target's entropy
  (here ≈ 1.2–1.3: the visit dist had max mass ~0.52 over ~3.5
  actions); uniform-over-legal is also ≈ 1.3, so "stuck at 1.30" and
  "matching the target" are indistinguishable in CE. Diagnose soft
  targets with top-1 agreement / target entropy, never raw CE.
- **Gate (c) — gen-1 PUCT vs gen-0 PUCT** (`make_puct_action_fn`,
  greedy, K=8/sims=64, `policy_match`, 1000 games): **seed 0 +4.9
  (t=3.2), seed 2 +4.6 (t=2.9), both p<0.005 — PROMOTE.** The
  project's first generation to climb via a search-improved policy
  (cf. Step 1 V-greedy's +2.7 that saturated — this is the right
  mechanism). Note cumulative-mean regression-to-mean: a hot first
  chunk read +16, the rest +4.6 — trust the full-sample t, not the
  running mean's path.
- **Gates (a)/(b) — value-head only — a WASH.** (a) gen-1 vs gen-0
  V-MCTS K=64: +2.8 (p=0.028). (b) gen-1 vs rollout K=8 N=8: −22.4
  (t=−16.4) vs gen-0's −19.8 → −2.6, p≈0.18, ns. **Opposite signs ⇒ no
  real value change** (consistent with gen-1's slightly-higher value
  loss; it trained on less data than gen-0). **Lesson: V-MCTS arenas
  cannot see policy gains; PUCT-vs-PUCT is the load-bearing Step-3
  gate.**
- Promoted artifact: `pv_gen1.msgpack`. Next: gen-2 with gen-1 as
  generator.

## 2026-06-16 — GENERATION 2 (3-way mix, sims=16) — GATE FAILED, not promoted

Gen-2 (fresh `PolicyValueNet`) trained on a **3-way ⅓ replay mix**
newest-first, `collect_fn=[gen1-PUCT, gen0-PUCT, step2]` — i.e. the
gen-1-generated PUCT corpus (`make_puct_collect_fn`, K=8, sims=16, τ=1.0,
32k games), the gen-0-generated PUCT corpus, and the Step-2 corpus, ⅓ epochs
each. Same everything else as gen-1 (20k epochs, batch 4096, `split=2`,
policy_weight=1.0, augment=True). Eval: value loss → **0.1023**
(≈ gen-1's 0.10), policy CE pinned at the soft-target floor ~1.33 as
expected (uninformative).

- **The 3-way round-robin makes train loss bounce period-3** — a
  *liveness* signal, not progress: the one-hot Step-2 batch can drive
  CE toward 0 (train ≈0.83) while the two soft PUCT batches are floored
  at their target entropy (train ≈1.33/1.42). All expected; only eval
  value loss is a real progress signal here, and it's not gateable.
- **Gate — gen-2 PUCT vs gen-1 PUCT** (`make_puct_action_fn`, greedy,
  K=8/sims=64, `policy_match`, 1000 games): **seed 0 −5.0 (t=−3.5),
  seed 2 −4.1 (t=−2.8) — both significantly NEGATIVE. REGRESSION.**
  Gen-2 *lost* to gen-1 by ≈ a full generation, landing back at
  **≈ gen-0 strength** (a near-mirror of gen-1's +4.7 gain over gen-0).
  Not promoted; **gen-1 remains champion and generator.**
- **Leading hypothesis: the 3-way mix.** The only structural change
  from gen-1's climbing recipe (2-way 50/50 `[gen0-PUCT, step2]`) was
  the mix: gen-2 cut the newest/best corpus from 50%→33% and added ⅓
  of an *older* PUCT generation (gen-0). Because each generation is
  retrained **from scratch** (not warm-started like canonical AZ), the
  mix composition is more load-bearing — a fresh net trained ⅓ on
  gen-0's behavior is dragged toward gen-0's policy. The landing-spot
  (≈ gen-0) is consistent with this. (Reducing Step-2 from 50→33%
  should only help — it's the weakest, one-hot V₁-leaf teacher — so it
  is unlikely to be the cause.)
- **Next: gen-2b** — controlled retry changing exactly one thing vs the
  *proven* gen-1 recipe: the generator. 2-way 50/50
  `[gen1-PUCT, step2]`, gen-0-PUCT dropped; checkpoint
  `pv_gen2b_ckpt.msgpack`. If gen-2b climbs → the 3-way window was the
  bug; adopt "newest-PUCT + step2 only" as the standing recipe. If
  gen-2b also fails to climb → hypothesis shifts to **PUCT iteration
  saturating at sims=16**; next lever is sharper targets (higher-sims
  corpus), not the mix.

## 2026-06-16 — GENERATION 2b — WASH; the operator diagnostic + sims sweep

Gen-2b = gen-1's exact recipe (fresh `PolicyValueNet`, 2-way 50/50
`[gen1-PUCT, step2]`, gen-0-PUCT dropped, 20k epochs, eval value loss →
0.1034 ≈ gen-1's 0.10). The controlled A/B vs gen-2: same generator, same
newest corpus, only the mix differs.

- **Gate — gen-2b PUCT vs gen-1 PUCT** (greedy, K=8/sims=64,
  `policy_match`, 1000 games): **seed 0 −2.0 (t=−1.4, p=0.15), seed 2
  +1.1 (t=0.7, p=0.47) — both ns, opposite signs ⇒ a WASH.** gen-2b is
  statistically indistinguishable from gen-1. Two findings: (1) **the
  3-way mix WAS harmful** — switching back to 2-way recovered the full
  −4.7 gen-2 regression to ~0; adopt 2-way `[newest-PUCT, step2]` as the
  standing recipe. (2) **gen-1's recipe no longer climbs** — one climb
  (gen-1) then saturation. Not promoted; **gen-1 remains champion.**
- **DIAGNOSTIC — teacher vs student (the "why").** Expert iteration only
  climbs by how much the teacher (PUCT search) beats the student (the raw
  policy it's distilled into). Measured directly: **gen-1 PUCT(sims=16,
  the corpus-gen config) vs gen-1 policy-only (raw, τ=0.05),
  `policy_match`, 300 games/seed: seed 0 −9.3 (t=−6.7), seed 2 −8.5
  (t=−6.1).** The teacher plays **~9 pts/game WORSE than the student** —
  the improvement operator hasn't just saturated, **at gen-1's strength
  it has gone NEGATIVE.** This fully explains gen-2b's wash: the corpus
  was generated by a process that plays *below* gen-1's own policy, so
  distilling its visit distributions cannot climb (it washed rather than
  regressed because soft targets still correlate with the policy and the
  step2 anchor + the net's inability to fit noise pull it back to the
  mean policy). The crossover: at gen-0 the policy was weak → sims=16
  PUCT > gen-0 policy → gen-1 climbed; gen-1's policy then **outgrew** the
  sims=16 search.
- **Two explanations, opposite prescriptions (resolved by the sweep
  below):** (A) **too few sims** — 16 sims over K=8 dets is ~1–2 ply, not
  enough to override a now-good prior; *predicts PUCT beats raw policy
  again at higher sims* → bump corpus sims. (B) **strategy fusion** (the
  determinization flaw) — each det searches a known-cards world and
  commits to moves only good under perfect info; summed-visit argmax
  fuses these into a move bad under real uncertainty, while the
  info-state raw policy handles uncertainty better; *predicts more sims
  makes it WORSE* → lever is Step 4/5 (value head, capacity,
  imperfect-info), and **the raw policy head may already be our strongest
  player** (meaning the PUCT-vs-PUCT gate is testing the wrong thing).
- **SWEEP RESULT — explanation A: too few sims.** gen-1 PUCT
  vs gen-1 raw policy (300 games/seed, A=PUCT): sims=16 **−9** → sims=64
  **+4.1** (t=1.67, p=0.10) → sims=128 **+10.2** (t=3.8, p=0.0002) →
  sims=256 **+12.0** (t=4.0, p=0.0001). Monotone increasing, crosses zero
  ~sims 40–50, healthy by 128, plateauing by 256. **Strategy fusion (B) is
  NOT dominant — more search helps, not hurts — and the raw policy is NOT
  our best player** (PUCT at sims≥64 beats it). The operator was simply
  starved: 16 sims can't outrun a good prior, 128 can.
- **NEXT — gen-2 at sims=128.** Regenerate the gen-1 PUCT corpus at
  **sims=128** (the knee: +10 fuel, clears the +8 bar; sims=256 buys only
  +2 more for ~2.5× the cost), keep the proven 2-way 50/50
  `[gen1-PUCT, step2]` mix, retrain fresh, re-gate gen-2 PUCT vs gen-1
  PUCT. **This supersedes the gen-1-era "sims=16 was enough" note** — that
  held only while the policy was weak (gen-0); a stronger prior needs a
  deeper search to improve on it.
- **COST CORRECTION (supersedes the super-linear claim in the plan's
  Operational facts):** measured arena cost (`policy_match`, K=8, vmapped
  over 10 pairs) was **~250 / 600 / 1500 ms/game at sims=64 / 128 / 256**
  — only ~×2.4–2.5 per doubling, roughly *linear* in sims, NOT
  "multiplier ≈ num_simulations". sims=256 over 300 games took just
  432 s. Re-time the *collector* at sims=128 before assuming a 128-sim
  corpus is expensive. (Update: the *collector* at sims=128 is much
  heavier than the arena — ~3 s/game/chip; see the gen-2(s128) entry.)

## 2026-06-17 — GENERATION 2 (sims=128 corpus) — CLIMBED +12, PROMOTED

The sweep fix applied: regenerated the gen-1 PUCT corpus at **sims=128**
(32k games = 16×2048, K=8, τ=1.0, generated on a 2×4 via `pmap` over 8
chips, ~3.3 h), retrained a fresh `PolicyValueNet` on the **proven 2-way
50/50** `[gen1-PUCT-s128, step2]` mix (20k epochs; corpus batches are 2048
so `split=1` keeps 2048 games/step, step2 stays `split=2`). Eval value loss
settled at **0.14** — higher than gen-1's 0.10, but on a different
(sims=128) holdout and irrelevant per the value-is-a-wash pattern.

- **Gate — gen-2(s128) PUCT vs gen-1 PUCT** (greedy, K=8/sims=64,
  `policy_match`, 1000 games): **seed 0 +11.8 (t=8.0), seed 2 +13.1
  (t=9.0), p≈0 — PROMOTE.** ~3× gen-1's +4.7 climb over gen-0, and the
  decisive confirmation of the whole diagnostic chain: the sims=16
  operator was *starved* (negative), sims=128 restored strong fuel, the
  loop climbed harder than ever. The +12 is entirely priors (value loss
  rose); value remains a wash.
- **Promoted artifact: `pv_gen2_s128.msgpack` (CHAMPION).** Confirmed
  recipe going forward: 2-way 50/50 `[newest-PUCT-s128, step2]`,
  **sims=128 corpus** (sims=16 is dead — it produces anti-signal once the
  policy is strong). **Next: gen-3** with gen-2(s128) as generator, same
  sims=128 / 2-way recipe; re-gate gen-3 vs gen-2. A second consecutive
  sims=128 climb confirms the loop is self-sustaining.
- **COLLECTOR COST (corrects the earlier "re-time the collector" TODO):**
  sims=128 data-gen measured **~3 s/game/chip** — ~5× the arena's
  ~600 ms/game (same arena-vs-collector gap the plan saw at sims=64:
  250 vs 504 ms), and far steeper than the arena's near-linear scaling.
  32k games ≈ 27 h on 1 chip → ~3.3 h on a 2×4 (8 chips, `pmap`-sharded
  collector; near-perfect 8×). Budget high-sims corpora accordingly.
  (Later superseded by the 2026-06-21 B=64/chip profiling — see the
  plan's "HOW TO SCALE STAGE 1".)

## 2026-06-17 — GENERATION 3 (sims=128) — CLIMBED +14, PROMOTED; loop self-sustaining

Straight repeat of the locked recipe: regenerated the PUCT corpus at
sims=128 with **gen-2(s128)** as generator (32k games, 2×4 pmap), retrained
fresh on 2-way 50/50 `[gen2-PUCT-s128, step2]` (`split=1`/`split=2`, 20k
epochs).

- **Gate — gen-3 PUCT vs gen-2(s128) PUCT** (greedy, K=8/sims=64,
  `policy_match`): **seed 0 +13.9 (t=9.5), p≈0 — PROMOTE.** Seed 2 was
  preempted; a t=9.5 over 500 pairs is not reversible and both prior
  climbs had seeds agree closely, so promoted on seed 0 (seed-2 re-run
  pending only for record completeness, not as a gate).
- **This is the second consecutive sims=128 climb → the loop is
  self-sustaining** (the criterion set in the gen-2 entry). Gains are not
  diminishing: gen-1 +4.7 → gen-2 +12 → gen-3 +14. Value loss again
  irrelevant; the climb is entirely priors.
- Promoted artifact: **`pv_gen3_s128.msgpack` (CHAMPION).** Next:
  **gen-4** (same recipe, gen-3 generator) to extend the ladder, **or
  pivot to Step 4** — the loop is proven, so an external calibration
  (JassTheRipper arena) and/or net scaling is now the higher-value move.

## 2026-06-20 — DECISION: keep cranking; semi-automate; recalibrate; queue the value-head bet

Made after the JTR calibration (next entry) showed the model is the limiter:

1. **Keep cranking the loop (gen-4, gen-5, …)** — it's the proven climber
   and gains are NOT diminishing; the marginal generation is the cheapest
   strength we can buy and there's no diminishing-returns signal yet.
2. **Semi-automate the *serial* single-generation pipeline** (not a
   hands-off chain): one resumable Colab flow `collect → train → gate`
   that **checkpoints each stage to Drive** so a generation = one launch
   that survives preemption. Do NOT fully automate a multi-gen chain — a
   silently-bad generator poisons everything downstream, and colab
   preemptions/quota make long unattended chains fragile.
3. **Re-calibrate vs JTR POWERFUL every ~2 generations** so we track
   ABSOLUTE strength, not just self-relative pts. Open question that
   decides how long to crank: does +12 self-relative buy ~12 absolute pts
   vs POWERFUL, or far less? (backfill gen-2-vs-POWERFUL + measure
   gen-4-vs-POWERFUL to get the conversion rate.)
4. **Queue the higher-upside bet for when loop gains flatten: net
   scaling + the stuck value head.** The value head has been a WASH every
   generation since Step 2, yet it IS the leaf evaluator in JTR's search —
   so it likely caps absolute strength. Attention over the 36 card rows
   (vs mean pool) is the natural Step-4 unlock. Cheap diagnostic first
   (is the wash data or architecture?) before a full new architecture line
   that restarts the generation ladder.

(Item 1's "not diminishing" was superseded by gen-4/gen-5; item 4 became
the active track.)

## 2026-06-20 — gen-4 RECIPE (planned): straight repeat, generator = gen-3

Nothing changes but the generator; the only NEW work is wrapping the three
stages as one resumable, per-stage-checkpointed flow (DECISION item 2).
Stages (champion = `pv_gen3_s128.msgpack`):

1. **COLLECT (TPU 2×4, ~3.3 h).** Regenerate the PUCT corpus with gen-3 as
   generator: `make_puct_collect_fn(pv_model.apply, gen3_params,
   num_determinizations=8, num_simulations=128, temperature=1.0)`, **32k
   games = 16×2048**, `pmap`-sharded over the 8 chips (~3 s/game/chip).
   ⚠ This is the mandatory fresh half of the corpus — it's gen-3's
   visit-distribution targets, the actual improvement signal. **Checkpoint
   the corpus to Drive** (`corpus_puct_gen3_16x2048_s128k8.pickle`, host
   numpy, list-of-per-batch-tuples — match the existing collector format,
   not a concatenated 6-tuple). The **step2 corpus is the fixed 50% anchor,
   reused as-is** (`corpus_k8_v1_24x4096.pkl`) — never regenerated.
2. **TRAIN (TPU, ~1 h).** Fresh `PolicyValueNet`, 2-way 50/50
   `collect_fn=[gen3-PUCT-s128, step2]`, 20k epochs, `policy_weight=1.0`,
   `augment=True`. Memory: corpus on HOST (numpy, never `jnp.asarray` the
   whole thing); gen-3-PUCT batches are 2048 → `split=1`, step2 stays
   `split=2` (keeps ≤2048 games/step under the ~12.2 G TPU). Checkpoint
   `pv_gen4_ckpt.msgpack` to Drive (resume = rerun the same `train_pv_model`
   call). Expect value loss ~0.14 and policy CE pinned at the soft-target
   entropy floor — **both uninformative; do NOT gate on them** (diagnose
   soft targets by top-1 agreement, never raw CE).
3. **GATE (CPU, `JAX_PLATFORMS=cpu`).** gen-4 PUCT vs gen-3 PUCT,
   `make_puct_action_fn` greedy, **K=8 / sims=64**, via `policy_match`,
   **1000 games, two seeds (0 and 2)**. PROMOTE on a significant positive
   (p<0.05, expect t large if the loop holds). The V-MCTS gates (value
   head, rollout yardstick) are SECONDARY — expect them flat (value is a
   wash). Save the gate output to Drive alongside the checkpoint.
4. **CALIBRATE (every ~2 gens; do it at gen-4).** Export gen-4 to TF
   SavedModel, run JTR `gen-4 vs POWERFUL` at 64 runs/det (≥300 paired
   games for a ~3 pt/game effect) to get the absolute-strength conversion
   rate vs the gen-3 −22/game baseline.

**Decision after gen-4:** if it climbs again AND the JTR gap closes
materially → keep cranking (gen-5…). If it climbs self-relative but the
JTR gap barely moves → the conversion rate is poor; pivot sooner to net
scaling / the value head (DECISION item 4). If it WASHES → suspect the
sims=128 operator is now starved at gen-3's strength (re-run the
PUCT-vs-raw-policy sims sweep; the crossover rises as the policy
strengthens) → bump corpus sims before blaming the architecture.

## 2026-06-20 — Step 4: gen-3 exported to JassTheRipper, externally calibrated

The export pipeline (Flax → TF SavedModel, repo commits 7f217a3 / d9498fd)
and a JTR-side "real PUCT" integration are done: JTR now exposes the full
softmax prior `P(s,a)` (`PuctPriorPolicy`), `MCTS.findChildrenPuct` uses it,
and cross-determinization aggregation is **summed root visit counts** (not
Q-sum) when a prior is active — the two changes needed to make the pgx
policy gain visible. (Full detail in `~/Documents/src/JassTheRipper/IDEAS.md`.)

- **Integration validated — gen-3 > gen-2 reproduces externally.** Both
  sides real PUCT, RUNS mode, swapped-deal paired t-test, at 64 runs/det:
  seed 42 −14.6 (p=0.003, 100 pairs), seed 43 −6.4 (p=0.020, 300 pairs)
  — gen-3 ahead, **replicated** (negative = gen-3 ahead; per-pair = 2
  games). The old argmax-tip + Q-sum setup had WASHED this (p=0.65). No
  clean depth crossover: 64/det is the sweet spot, 128 oddly weak, 256
  moderate (not pgx's tidy monotone sims curve). Magnitude is modest
  (~3 pts/game) — JTR's determinized search dilutes the policy gain vs
  pgx's own +14/game self-relative.
- **Absolute calibration triangle (100 games each, paired; negative =
  name1 behind, per-game ≈ ½ the per-pair figure):**

  | matchup | per-game | p |
  |:--|:--|:--|
  | gen-3 (value+policy) vs **POWERFUL** (classical) | **−22** | <0.0001 |
  | gen-3 vs **FAST_TEST** (weakest MCTS) | −5 | 0.14 (ns) |
  | **POWERFUL** vs **FAST_TEST** | +20 | <0.0001 |
  | gen-3 @ 256 runs/det vs **POWERFUL** | −9 | 0.006 |

- **Strength ordering: POWERFUL ≫ FAST_TEST ≳ gen-3.** The pgx agent is
  **weak in absolute terms** — loses decisively to the classical bot and
  is even nominally behind the weakest MCTS baseline. Giving gen-3 ~13×
  more search (20 → 256 runs/det, *exceeding* POWERFUL's own 200) only
  halves the deficit (−22 → −9) and does NOT close it.
- **CONCLUSION: the model is the limiter, not the JTR integration or
  search depth.** The integration is correct (gen-3 > gen-2 is visible);
  gen-3 is just an early, pgx-simplified-self-play net. The lever is
  **stronger pgx models** (more generations / net scaling), per the
  2026-06-20 DECISION. This is the calibrated external opponent Step 4
  asked for — answer: not yet competitive.

## 2026-06-21 — GENERATION 4 — PROMOTED but decelerated; then the REFRAME

Straight repeat of the locked recipe (gen-3 generator, sims=128, 2-way
50/50, 20k epochs), and the first corpus collected at the profiled
**64 games/chip** optimum (~48 min vs 3.3 h; see the plan's "HOW TO SCALE
STAGE 1"). Operational notes: training died at 5k epochs and was **resumed**
from the `checkpoint_every=500` checkpoint (RNG fast-forward); an early gate
read was a false alarm off **stale scrollback** (it matched gen-3's
+13.9/t=9.5 exactly — the tell). A separate near-miss: a 9 s "training" run
that had silently resumed a *completed* gen-3 checkpoint because
`checkpoint_path` wasn't bumped to the gen-4 file — caught before gating
(would have been gen-3-vs-gen-3). Lesson reinforced: derive every path from
a single `GEN` anchor (`SRC = GEN-1`), and print net fingerprints at gate
time.

- **Gate — gen-4 PUCT vs gen-3 PUCT** (greedy K=8/sims=64, `policy_match`,
  500 pairs/seed): **seed 0 +2.6 (t=1.9, p=0.06, win 50.9%, sign p=0.16,
  ns), seed 2 +4.4 (t=2.9, p=0.0035, win 53.4%, sign p=0.0078).** Both
  seeds positive, seed 2 significant → **PROMOTE (`pv_gen4_s128.msgpack`)**,
  but ~+3.5 combined is a hard drop from gen-3's +14 — back to gen-1-size
  gains. ⚠ Training-health confirm still pending (did eval value loss
  settle ~0.14 / top-1 agreement healthy?) to rule out the death/resume
  understating the climb. (Later confirmed: eval value loss 0.1393 ≈ 0.14.)
- **Initial read: the deceleration the gen-4 plan flagged.** ~+3.5 on
  sims=128 ≈ gen-1's +4.7 on sims=16 — one generation before the sims=16
  stall. Leading hypothesis at the time: the operator starved again
  (crossover rose with policy strength); planned diagnostic: gen-3 PUCT vs
  gen-3 RAW, swept sims 128/256/384. Strategic overlay: sims-bumping is
  consumable (treadmill — the crossover keeps rising), net scaling is
  structural.
- **DIAGNOSTICS IN (evening) — REFRAME: gen-4 did NOT stall; the PUCT@64
  GATE went blind.** Both diagnostics on CPU, existing nets:
  - **Operator (gen-3 PUCT@128 vs gen-3 raw, 300 pairs): +26.3, t=13.9**
    — *bigger* than gen-1's +10. Operator NOT starved; sims-bumping is OFF
    (more sims only widens the +26 gap, doesn't help the loop convert it).
    So s256/s384 were not needed (s128 already settled it) — the
    interrupted sweep cost nothing.
  - **Policy (gen-4 raw vs gen-3 raw, policy-only τ=0.05, 300 pairs):
    +15.0, t=8.9** — the policy is still climbing *hard*. The +3.5
    PUCT-vs-PUCT gate was a MEASUREMENT ARTIFACT: at sims=64 the search
    compensates for prior quality, compressing a +15 raw gain to +3.5
    searched. PUCT@64 has lost sensitivity to policy gains (early on weak
    priors were load-bearing in PUCT@64; no longer).
  - **Verdict:** loop is alive (policy ~+15/gen, not saturated), but (a)
    **gate gen-5 raw-vs-raw**, not PUCT@64 (low-sims PUCT, 8–16, is an
    alternative sensitive gate; PUCT@64 stays as a deployed-strength
    check); and (b) the *searched* agent's +3.5 fingers the **VALUE
    head** (stuck since Step 2) as the cap on deployed strength → **net
    scaling on the value head** (attention vs mean pool, validated on the
    existing corpus). Supersedes the sims-vs-arch fork (sims branch is
    dead). Training health fine (gen-4 eval value loss 0.1393 ≈ 0.14).
  - **Efficiency read (pocket):** a +15-better prior buys the same
    searched strength at FEWER sims (priors load-bearing at low sims) —
    JTR's "quality at fewer playouts" lever.

## 2026-07-01 — GENERATION 5 — policy expert-iteration SATURATED (not promoted)

Trained on `corpus_puct_gen4_16x2048_s128k8.pickle` (gen-4 @ sims=128, K=8),
2-way 50/50 with step2, 20k epochs, `policy_weight=1.0`, `augment=True`;
eval value loss 0.1382. **Champion stays gen-4.** The verdict: **policy
expert-iteration has SATURATED at sims=128 — the loop's policy climb
plateaued. NOT a policy-net capacity ceiling, NOT a pipeline bug.** The full
diagnostic chain (all artifacts ruled out, then the mechanism found):

- **Three flat gates, gen-5 vs gen-4** (τ=0.05 raw unless noted): raw seed 0
  **+1.5 (t=0.9)**, raw seed 2 **+2.4 (t=1.57)**, **PUCT@16 +0.5 (t=0.21)** —
  all ns. The low-sims PUCT gate (meant to be *more* sensitive) is the
  flattest: search can't amplify a policy edge that isn't there. Contrast
  gen-4-over-gen-3 raw was **+15 (t=8.9)**.
- **Not an artifact:** fingerprints differ (gen-4 param-sum 871.3 → gen-5
  908.7, so the net *did* change — mostly the value head); trained to 20k
  (eval value loss **0.1382** ≈ 0.14); provenance clean (collect used gen-4 @
  sims=128, train read `corpus_puct_gen4_16x2048_s128k8.pickle`; `GEN=5/SRC=4`
  anchor makes a gen-3 pickup impossible).
- **Operator NOT exhausted:** gen-4 PUCT@128 vs gen-4 raw = **+11.1 (t=2.94**,
  80 pairs, seed-looped — see the SOP OOM note**)** — down from gen-3's +26
  but still real teaching signal.
- **Distillation gap (held-out batch, 76 341 alive positions) — the smoking
  gun:** gen-4 raw AND gen-5 raw both agree with the gen-4-teacher `pi` at
  **0.634 (Δ=+0.001)** — training on gen-4's targets moved the policy argmax
  essentially *zero*. On the **36.6%** of positions where the search overrode
  gen-4 raw, gen-5 **adopted the teacher only 16.5%** and **kept gen-4's old
  move 73.2%**. And the teacher's visit target is **diffuse exactly there**:
  peak visit mass **0.367 on corrections** vs **0.531 overall**.
- **Mechanism:** the teacher's residual +11 edge now lives in *soft* visit
  distributions (peak ~0.37 on the moves that matter), which carry no argmax
  training gradient — the student already sits at a CE optimum w.r.t. them.
  So the *distillable-into-argmax* signal collapsed far faster than the raw
  operator margin: conversion of operator→raw-gain went **~58% (gen-3→4,
  +15/+26) → ~0% at the argmax level (gen-4→5, +2/+11)**.

**What this REDIRECTS (supersedes the gen-4 "keep cranking gen-5" step):**

1. **Do NOT scale the policy net** — Δ agreement +0.001 refutes underfit; a
   bigger policy head has no gradient to exploit.
2. **Cheap probe: sharpen the target via sims 128→256.** Testable prediction:
   peak-visit-mass on correction positions should rise. Validate on a *small*
   corpus (does peak sharpen? does the operator widen?) before committing a
   full generation.
3. **Higher upside: value-head scaling** (attention over the 36 card rows vs
   mean pool). A better leaf evaluator makes the search itself more decisive —
   lifting both the +11 operator AND future target sharpness. This is the
   pre-registered Step-4 pivot, now with evidence it's the right head.

## 2026-07-02 — step2-mix ablation, 20% arm — GATE FLAT; adoption UP but drift cancels it

First arm of the pre-registered probe (plan Status snapshot, 2026-07-02):
retrain gen-5 on the existing gen-4 corpus with the step2 anchor cut
50%→20% (`collect_fn=[puct_fn]*4 + [step2_fn]`, epoch round-robin; 20k
epochs, fresh net, otherwise the locked recipe). Training healthy: eval
value loss **0.1358** (≈ the 0.14 band; slightly *better* than the 50/50
gen-5's 0.1382 — no value-head-coverage degradation at 20%).

- **Gate — mix80 raw vs gen-4 raw** (τ=0.05, `policy_match`, 300
  pairs/seed): **seed 0 +1.0 (t=0.58), seed 2 +1.4 (t=0.80) — both ns**,
  numerically the same as the 50/50 gen-5 gates (+1.5/+2.4). Raising the
  PUCT share of gradient steps 50%→80% bought zero strength.
- **Distillation-gap readout (same held-out batch as the gen-5
  diagnostic; gen-4 agreement 0.634 and peaks 0.367/0.531 reproduce):**
  mix80 agrees with the gen-4 teacher at **0.621** (DOWN from gen-5's
  0.634); on corrections it **adopted the teacher 24.7%** (up from 16.5%)
  and **kept gen-4's move 58.8%** (down from 73.2%; a *third* move 16.5%,
  up from 10.3%).
- **The accounting (corrections = 36.6% of positions; non-corrections =
  63.4%, where gen-4 already matches the teacher):** on the agreed
  positions mix80 matches only **~83.7%** vs gen-5's ~90.5%. In totals it
  gained ~3.0% of positions as adopted corrections and lost ~4.3% off
  agreed-good moves → net agreement −1.3%, net strength ~0. The freed
  policy moved *symmetrically* — toward the teacher where the target is
  diffuse, away from it where it was already right.
- **MECHANISM REVISION (sharpens the gen-5 verdict):** the diffuse targets
  do carry an argmax gradient — the 50% anchor was absorbing it (adoption
  rose the moment it was cut), and its real function was *stabilizing the
  agreed moves*. The binding constraint is **target signal-to-noise**: at
  peak mass 0.37 the per-position margin is too weak for movement to
  convert into strength. So "student at a CE optimum / no gradient" →
  "gradient exists, but at current target sharpness it moves the policy
  without strengthening it."
- **Consequences:** (1) target **sharpening (sims 128→256 and/or K 8→16)
  is now the front-runner** — it directly raises the per-position margin;
  (2) the **0% arm** (pending) becomes the third point on the anchor
  dose-response curve (prediction: adoption up further, agreement down
  further, gate flat-to-negative); (3) **warm-start from gen-4** is the
  discriminating follow-up — initialization-as-anchor should retain agreed
  moves while accumulating corrections; a gain would prove the corrections
  carry signal, a flat result at high retention would confirm they are
  noise and only sharpening can help. (Needs an `init_params=` path in
  `train_pv_model`.)

## 2026-07-02 — step2-mix ablation, 0% arm — CLIMBED +11.8/+16.2; the ANCHOR WAS THE PLATEAU

Second arm of the pre-registered probe: same gen-4 corpus, **100% PUCT
epochs** (`collect_fn=[puct_fn]`, no step2 at all), otherwise the locked
recipe. Eval value loss **0.1332** — the best of the three arms (50/50
0.1382, 20% 0.1358, 0% 0.1332): no value-coverage degradation visible on
the PUCT-batch holdout.

- **Gate — mix100 raw vs gen-4 raw** (τ=0.05, `policy_match`, 300
  pairs/seed): **seed 0 +11.8 (t=6.67), seed 2 +16.2 (t=8.0)** — both p≈0,
  a gen-4-sized climb (gen-4 over gen-3 raw was +15) from a net trained on
  the SAME corpus whose 50/50 mix gated flat (+1.5/+2.4). **Two-seed
  promotion criterion MET → gen-5b.** Save the final mix100 net as
  `pv_gen5b_s128.msgpack` (gen-2b naming precedent; do NOT overwrite the
  50/50 `pv_gen5_s128.msgpack`).
- **Distillation-gap readout** (same held-out batch; gen-4 agreement 0.634
  and peaks 0.367/0.531 reproduce): mix100 agrees with the teacher at
  **0.66** (up from gen-5's 0.634); on corrections **adopted 33.5%**
  (16.5% → 24.7% → 33.5% across 50/20/0%), **kept gen-4's move 52.3%**
  (73.2 → 58.8 → 52.3). Non-correction agreement ~84.8% — the same plateau
  as the 20% arm's ~83.7%.
- **The dose-response resolves the mechanism — and CORRECTS the 20%-arm
  entry above.** Adoption rises steadily as the anchor shrinks while the
  drift off agreed moves is a roughly FIXED cost (~15–16% at both 20% and
  0%): at 50% the anchor absorbed the gradient entirely, at 20% adoption
  gains ≈ drift cost (the flat gate was the unlucky middle of the curve),
  at 0% adoption doubles again and wins outright. "Target SNR is the
  binding constraint" (20%-arm entry) was WRONG: **the diffuse sims=128
  corrections carry real, convertible signal — the anchor was both
  suppressing adoption and eating the conversion.**
- **The 2026-07-01 gen-5 VERDICT ("policy expert-iteration saturated") is
  OVERTURNED.** No saturation, no target-sharpness wall (yet) — a recipe
  bug of the same family as the gen-2 3-way regression, one notch subtler:
  stale sharp targets don't just drag a from-scratch net toward old
  behavior, at 50% they null out a subtle fresh signal completely.
  **Standing recipe → 100% newest-PUCT** (plan + SOP updated). Lesson:
  ablate the mix before declaring saturation — the anchor was the only
  never-ablated recipe component.
- **Open before the champion/generator switch: the PUCT@64 deployed check
  vs gen-4** (gen-5b's value head trained on champion-self-play positions
  only, and it is the search leaf; the holdout and the raw gate can't see
  off-distribution value error, the searched arena can). Then: gen-6 on
  the new recipe, operator re-measure (gen-5b PUCT@128 vs raw), JTR
  re-calibration (owed since gen-4).

## 2026-07-02 — gen-5b PUCT@64 deployed check PASSED — PROMOTED (CHAMPION)

gen-5b PUCT@64 vs gen-4 PUCT@64 (greedy K=8/sims=64, `policy_match`, 600
games / 300 swapped-deal pairs, seed 0): **+2.2/game, win 50.3%, t=+1.076
(p=0.28, ns), sign 148W/145L (ns).** Positive-and-flat = the pre-registered
pass: this is normal PUCT@64 compression (gen-4's +15 raw read as +3.5 at
this gate, its seed 0 just +2.6 ns — +2.2 is the same band), and **no sign
of the value-coverage risk** (a damaged leaf evaluator would read clearly
negative). Combined with the two-seed raw gate (+11.8/+16.2) → **gen-5b
PROMOTED: champion and gen-6 generator (`pv_gen5b_s128.msgpack`).**

Strategic note: the *searched* agent has barely moved since gen-3 (+3.5 at
gen-4, +2.2 ns here) while the raw policy gained ~+27–31 — the value head
remains the deployed-strength cap. Step-4 value-head scaling stays queued;
the JTR re-calibration will quantify how much raw-policy gain converts to
absolute strength.

## 2026-07-02 — JTR re-calibration at gen-5b (owed since gen-4): gap to POWERFUL roughly HALVED

gen-4 and gen-5b exported (`scripts/extract_pv_weights.py` →
`scripts/export_pv_savedmodel.py`) into JTR's
`src/main/resources/models/{pv_gen4_s128,pv_gen5b_s128}/export` and run
through JTR's real-PUCT integration (swapped-deal paired t-test).

- **gen-5b vs gen-4, real PUCT** (SWEEP_64 = 64 runs/det, both sides,
  500 pairs / 1000 games, seed 42): **+16.7/pair (+8.35/game), t=7.53,
  p=0.0000, sign 297W-174L-29T** — external confirmation of the internal
  climb (gen-5b was +11.8/+16.2 raw vs gen-4 in pgx itself, see the 0% arm
  entry above). Same pattern as the gen-3>gen-2 result: real PUCT surfaces
  a generation gain that the old argmax-tip + Q-sum setup would have washed.
- **gen-5b vs POWERFUL (classical), 250 pairs across 2 seeds:** seed 42
  (50 pairs) −12.6/pair (−6.3/game, t=−1.74, p=0.089 ns); seed 43
  (200 pairs) **−20.7/pair (−10.35/game, t=−5.6, p=0.0000)**; pair-weighted
  combined ≈ **−19.1/pair (≈−9.5/game)**.
- **Absolute-strength gap to POWERFUL has roughly HALVED since gen-3**
  (−22/game → ≈−9.5/game), while remaining a clear, significant loss. This
  answers the open question from the 2026-06-20 DECISION: the self-relative
  policy gains since gen-3 (gen-3→gen-4 raw +15, gen-4→gen-5b raw +27–31 via
  the step2-mix-ablation fix) DO convert substantially into absolute
  strength — not just self-play-relative noise that dies against a
  classical opponent, as gen-3's own weak showing against POWERFUL had left
  open.
- **gen-4's own POWERFUL calibration was never run** (deferred straight to
  gen-5b per the 2026-06-21 gen-4 REFRAME) — the −22 → −9.5 trendline has
  only two points (gen-3, gen-5b), not a full per-generation curve.
- Runtime note: the gen-5b-vs-gen-4 self-play arena (both sides doing pgx NN
  inference at every node) ran ~4 s/game (1000 games ≈ 72 min wall-clock);
  the POWERFUL-calibration runs were ~3× faster per game (~1.2 s/game) since
  classical random rollouts are cheaper per node than repeated NN forward
  passes on both sides.

## 2026-07-02 — GENERATION 6 — GATE FLAT; operator fuel gauge ≈ 0 — the sims=128 teacher is EXHAUSTED

First generation collected AND trained entirely on the new recipe (gen-5b
generator, `corpus_puct_gen5b_16x2048_s128k8.pickle` @ sims=128/K=8, 100%
newest-PUCT, 20k epochs). **Champion stays gen-5b.**

- Training healthy: reached 20k, eval value loss **0.1331** (the 0.13–0.14
  band); fingerprints differ (`new_params` ≠ `src_params`) — not an artifact.
- **Gate — gen-6 raw vs gen-5b raw** (τ=0.05, 300 pairs/seed): **seed 0
  +2.1 (p=0.18), seed 2 +1.3 (p=0.44) — both ns.** This is the first flat
  gate with no artifact left to blame: the anchor is retired, the standing
  gate is raw-vs-raw (no gate masking), provenance/health clean. Per the
  gen-5b promotion entry's pre-registration, a flat raw gate now means
  saturation for real.
- **Operator fuel gauge — gen-5b PUCT@128 (greedy, K=8) vs gen-5b raw
  (τ=0.05)**, 80 pairs × 3 seeds = 240 pairs (same method as the gen-5
  diagnostic): **+3.0/game, p=0.18 ns.** Trendline: **+26 (gen-3) → +11.1
  (gen-4) → +3.0 ns (gen-5b)** — search at sims=128 is now statistically
  indistinguishable from the raw policy it wraps.
- **Verdict: the sims=128/K=8 teacher is EXHAUSTED — gen-6's flat gate is
  fuel exhaustion, not a training failure.** The corpus targets contained
  essentially nothing beyond gen-5b raw, so there was nothing to distill.
  This completes the mechanism story across generations: at gen-5 the
  operator still held +11 but ~0% of it distilled into argmax (target S/N,
  peak visit mass 0.37 on corrections); at gen-5b the operator margin
  itself is gone. Policy expert-iteration at sims=128 has genuinely
  converged — the anchor explained the *previous* plateau, not this one.
- **Next (both pre-registered in the gen-5 REDIRECT):**
  1. **Cheap CPU probe before any collection:** does the operator re-open
     at sharper search? gen-5b PUCT@256 (and/or K=16) vs gen-5b raw,
     seed-looped — drop to ≤40 pairs/chunk, the @256 tree roughly doubles
     the @128 working set that already OOMs above ~80 pairs. A real margin
     at 256 = the crank restarts with sharper targets (→ gen-7 at
     sims=256/K=16, after re-profiling the per-chip collect optimum;
     B×K≈512 says expect per-chip batch ~32 and ~2× Stage-1 wall-clock).
     Still ~0 = more search on the current net teaches nothing.
  2. **Step-4 value-head scaling** (attention over the 36 card rows vs mean
     pool): a better leaf evaluator is what would make deeper search
     decisive again, and the JTR re-calibration (above) shows model gains
     convert to absolute strength. If the @256 probe comes back ~0, this
     becomes the only live lever.

## 2026-07-03 — Sharpening probe at gen-5b: NO search axis re-opens the operator — DECISION: pivot to Step-4 value-head scaling

The probe pre-registered yesterday: gen-5b PUCT (greedy) vs gen-5b raw
(τ=0.05), `policy_match`, 240 pairs per arm, seed-looped with chunk size
scaled to the tree working set (80×3 / 40×6 / 20×12 for K·sims of
8·128 / 8·256 / 16·128 — the B×K rule applied to the diagnostic).

| arm                  | mean/game | p      |
|----------------------|-----------|--------|
| @128, K=8 (baseline) | +3.0      | 0.18   |
| @256, K=8            | +2.4      | 0.26   |
| @128, K=16           | +3.8      | 0.0698 |

- **Doubling sims bought nothing** (+2.4 vs the +3.0 baseline) — the search
  is not depth-limited; the leaf evaluations feeding it are the constraint.
- **K=16 is the best arm and near-significant** — determinization noise is
  plausibly a real, small effect. But the magnitude is decision-irrelevant:
  +4 of fuel sits in the same band as the exhausted baseline, nothing like
  the +11–26 margins that drove actual climbs. And gen-6 just calibrated
  fuel→gate conversion on this recipe: +3.0 of operator margin produced a
  +2 ns gate. A K=16 corpus (+~4 fuel, 2× collection working set) projects
  to another flat generation. Not run to more seeds on purpose — a firmer
  +3.8 changes nothing.
- **DECISION: the Step-3 crank is CLOSED on this net at every reachable
  sharpness.** The value head is the cap — as predicted since the gen-4
  REFRAME, and now with external evidence (JTR: gap to POWERFUL halved)
  that model gains convert to absolute strength. **Next work: Step-4
  value-head scaling — attention over the 36 card rows replacing mean
  pooling.**
- **What re-opens Step 3:** a materially better net. The operator margin is
  a function of the leaf evaluator, so after the value-head upgrade
  trains, re-run this exact probe (K=16 arm first — it was the live axis)
  before deciding whether to crank generations again.

## 2026-07-03 — gen-6b (PolicyValueNetAttn, first Step-4 net): NEGATIVE — not promoted

First architecture arm: `PolicyValueNetAttn` (header broadcast into each
card row, 2 pre-LN self-attention blocks over the 36 rows, 4 heads,
hidden 128, learned-query attention pool replacing the mean; 393k params
vs 111k), trained from scratch on gen-6's exact corpus
(`corpus_puct_gen5b_16x2048_s128k8.pickle`, 100% PUCT, 20k epochs,
`policy_weight=1.0`, `augment=True`, flat Adam 3e-4) → 
`pv_gen6b_s128.msgpack`. Clean A/B vs gen-6, which trained
PolicyValueNet on this corpus and gated flat.

- **Training:** eval value loss **0.1476** — WORSE than the old
  architecture's 0.1331/0.1332 on the same corpus/holdout construct;
  policy CE 0.9506. Plateau confirmed by a 20k→21.4k extension probe
  (v oscillates 0.1468–0.1507, no trend) — fully trained, not
  undertrained at this lr.
- **Raw gate vs gen-5b raw** (τ=0.05, 300 pairs, seed 0): **+2.8,
  t=1.5, p=0.13 ns** — same band as gen-6's flat gate (+2.1/+1.3),
  as expected: the corpus's policy targets are exhausted (gen-6 entry).
- **PUCT@64 deployed check vs gen-5b** (greedy K=8/sims=64, 300 pairs,
  seed 0): **+1.0, t=0.49, p=0.62 ns** — the decisive test for the
  value-head hypothesis, and it's flat. Policy-only gains compress to
  +2–3.5 at this gate (gen-4/gen-5b), so a real leaf-evaluator win had
  to read clearly above that; it didn't.
- **VERDICT: NOT promoted; champion stays gen-5b.** But the architecture
  is not yet condemned: attention is a capacity superset (it can
  ~emulate mean-pool + the context head), yet it converged to a *worse*
  loss — an **optimization smell, not a capacity verdict**. Flat Adam
  3e-4 without warmup on transformer blocks is the standard way to get
  exactly this. **Next, in cost order: (1) lr warmup + small lr sweep
  (~1 h/arm on the 2×4 now); (2) value-only attention variant** (keep
  the old policy path, spend the new capacity on the value head alone —
  the original Step-4 framing). Keep `pv_gen6b_s128.msgpack`.
- Operational (details in SOP): the attn net OOM'd the standard
  2048-game train step (needs ~33 G vs 15.75 G/chip) → added
  `accum_steps` (gradient accumulation) and `data_parallel=True` (pmap
  over the 2×4; 20 s/100 epochs vs 90 s on 1×1+accum; a mid-run
  1×1→2×4 checkpoint handoff worked). Standing preference: one 2×4
  instance for everything while Step 3 is closed.

## 2026-07-03 — gen-6b post-mortem: it's OVERFITTING, not optimization — the "optimization smell" read above is WRONG

Seen-vs-holdout loss decomposition on the saved nets (sgd(0) step as a
pure loss readout; train[0] = first corpus batch, seen ~1.3k× modulo suit
augmentation; holdout = the eval batch):

| net | train[0] v | holdout v | train[0] p | holdout p |
|:--|:--|:--|:--|:--|
| gen-6b (attn) | **0.0731** | 0.1472 | 0.9536 | 0.9508 |
| gen-5b | 0.1341 | 0.1337 | 0.9650 | 0.9617 |

- **gen-6b's value head memorizes**: it fits seen data 2× better than the
  old net fits *anything* (0.073 — far past gen-5b's 0.134) and
  generalizes worse (0.147). **gen-5b's gap is ZERO** — the old
  architecture is capacity-saturated, not regularized.
- So the warmup/lr theory is falsified before being run: optimization is
  fine (it reached a much deeper train loss), capacity is ample —
  **regularization is what's missing.** Policy heads are clean on both
  nets (CE pinned at the soft targets' entropy floor, no overfit).
- Encouraging reframe of the gen-6b negative: capacity was the right
  thing to add; it currently converts to memorization instead of
  generalization. Also anecdotal (record lost to preemptions): eval v
  may have dipped lower mid-training — a U-curve minimum possibly below
  the old net's 0.133. The retrain will keep full logs and show it.
- **Next arm: weight decay** — `train_pv_model(..., 
  optimizer=optax.adamw(3e-4, weight_decay=1e-2))` (passthrough added
  2026-07-03), same corpus, full logs kept. Success signal before any
  arena: holdout v clearly below 0.133 WITHOUT the train[0]-vs-holdout
  gap re-opening. Escalation if insufficient: dropout in the attention
  blocks, then num_layers=1 (smaller attn net). Note plain adamw decays
  LayerNorm/bias params too (usually masked out in transformer recipes)
  — acceptable for the first arm, revisit if it underperforms.
  (Superseded same day by the early-stopping arm below, which promoted;
  the decay arm remains queued as the principled replacement for
  hand-picked stopping epochs.)

## 2026-07-03 — gen-6b_es (attn, early-stopped @ 7k): PROMOTED — first significant deployed-strength gain since gen-3

The full-log retrain of the exact gen-6b recipe traced the predicted
U-curve: **eval v bottoms at ~0.1146–0.1150, flat from ~6.0k to ~7.6k
epochs** (14% below the old architecture's 0.133 floor), overfit onset
~8k (v climbs monotonically after — 0.118 @ 9.5k, 0.126 @ 12k, headed
back to run 1's 0.147; train v falls throughout). Policy CE converged by
7k (0.9509 vs run 1's 20k 0.9508). Run killed at 12k; **early-stopped
retrain to 7,000 epochs** (same recipe, fresh anchor) →
`pv_gen6b_es_s128.msgpack`, gated vs champion gen-5b:

- **Raw gate (promotion): seed 0 +10.3, t=6.0; seed 2 +7.4, t=4.38** —
  both p«0.05, a generation-class climb (gen-2/gen-5b band), from the
  SAME corpus that gated flat twice (gen-6 old-arch +2.1/+1.3 ns;
  gen-6b full-20k +2.8 ns).
- **PUCT@64 deployed check: +5.2, t=2.59, p=0.01** — significantly
  positive AND above the +2–3.5 band where pure-policy gains compress
  (gen-4 +3.5, gen-5b +2.2 ns). First significant movement at this gate
  since gen-3: **the value-head upgrade converts to searched strength**,
  and no value-coverage damage from the off-distribution risk (new
  architecture, leaves trained on gen-5b self-play, early-stopped).
- **PROMOTED: gen-6b_es (`pv_gen6b_es_s128.msgpack`) is CHAMPION** and
  the generator for any gen-7 collection.

**Two interpretation corrections this result forces:**

1. **The gen-6 "fuel exhaustion" verdict was architecture-relative.**
   The corpus was not exhausted — it held a +10-class policy gain and a
   14% value-loss gain that the old architecture could not convert
   (and the full-20k attn run destroyed by memorization). "Nothing left
   to distill" should have been "nothing this net can extract." The
   operator fuel gauge (+3.0 ns at gen-5b) measures the *teacher's*
   margin over the raw policy, not the corpus's distillable content for
   a better student.
2. **Policy CE is blind even to generation-class play differences**:
   the 20k and 7k nets differ by eval CE 0.0001 (0.9508 vs 0.9509) and
   by +10/game in play. The standing "diagnose by top-1 agreement, not
   CE" lesson, strongest form yet.

**Next (in order):**
1. **Operator re-probe on gen-6b_es** (its PUCT vs its own raw, K=16 arm
   first, then @128 K=8 baseline; seed-looped chunks per the SOP memory
   rule) — the 2026-07-03 sharpening-probe DECISION pre-registered this:
   a materially better net is what re-opens Step 3. If the fuel gauge
   jumps off ~0, gen-7 collection restarts with gen-6b_es as generator.
2. **Weight-decay arm** (queued above): hold ≤0.115 at full convergence
   so future generations don't need hand-picked stopping epochs.
3. **Before gen-7 Stage 1:** re-profile the per-chip collect optimum —
   the attn net's forward is ~6.5× the old net's (CPU, B=512), so the
   B×K≈512 VMEM rule was profiled on the wrong net. Also: the JTR export
   scripts (`extract_pv_weights.py` / `export_pv_savedmodel.py`)
   hardcode `PolicyValueNet()` — they need attn support before the next
   external calibration.

## 2026-07-04 — Operator re-probe on gen-6b_es: gauge reads ZERO — search adds nothing over the new raw policy

The probe pre-registered by the sharpening-probe DECISION: gen-6b_es
PUCT@128 (greedy) vs its own raw (τ=0.05, both `attn_model.apply`),
`policy_match`, 240 pairs/arm, seed-looped (20×12 / 80×3 for K=16/K=8;
~70 s/chunk — the "minutes on CPU" SOP framing is stale for the attn
net, see the chip-parallelization TODO).

| arm                    | mean/game | p      |
|------------------------|-----------|--------|
| @128, K=16 (live axis) | −0.2      | 0.92   |
| @128, K=8 (baseline)   | −3.4      | 0.11   |

- **The operator did NOT re-open.** Fuel trendline at K=8·128: +26
  (gen-3) → +11.1 (gen-4) → +3.0 ns (gen-5b) → **−3.4 ns (gen-6b_es)**.
  The "a materially better net re-opens Step 3" hypothesis is refuted at
  every reachable sharpness: the value-head upgrade lifted RAW play past
  what its own search can improve on — leaf quality and policy quality
  rose together, so the margin between them stayed ~0.
- **K=8 leans negative** (−3.4, p=0.11): with no leaf-evaluation margin
  left to spend, K=8's determinization noise plausibly *costs* points;
  K=16 (−0.2) merely cancels it back to zero. Same K-is-the-live-axis
  reading as the gen-5b sweep, one generation later and 4 points lower.
- Same picture from the promotion gates, in hindsight: gen-6b_es beat
  gen-5b by MORE raw-vs-raw (+10.3/+7.4) than PUCT@64-vs-PUCT@64
  (+5.2) — search compresses this net's edge rather than amplifying it.

**What it means for gen-7 (the 64k DECISION stands, but re-priced):**

1. **Do NOT pay for sharpness.** Collect at the cheap baseline
   (K=8/sims=128) — K=16 and sims=256 buy zero play-strength margin at
   2× the working set. The pre-registered "collect sharp if the gauge
   jumps" branch is dead.
2. **The corpus case is now value-half + capacity only.** Value targets
   are game outcomes from play 10 pts/game stronger than the gen-6
   corpus's generator; the policy half's expected quality is "gen-6b_es
   raw + search noise". Per the standing fuel-gauge demotion (promotion
   entry: the gauge measures the teacher's margin over its own raw, NOT
   distillable content for a better student — gen-6b_es itself came out
   of a +3.0 ns corpus), a flat gauge doesn't kill the corpus; but any
   gen-7 policy gain must come from volume feeding the attn capacity,
   not from better targets.
3. **Cheap open question — does deployment still want search at all?**
   If PUCT@128 ≤ raw, PUCT@64 (the deployed/JTR config) is presumably
   also ≤ raw. A 240-pair gen-6b_es PUCT@64-vs-raw probe would settle
   whether the JTR calibration should submit the RAW policy (~65×
   cheaper per move) and whether the "deployed check" gate config needs
   rethinking.

## 2026-07-04 — Collect re-profile with the attn generator: knee at B=8, but ms/game barely moved — Stage 1 stays cheap

`profile_collect_fn`, gen-6b_es generator (`PolicyValueNetAttn`,
K=8/sims=128, temperature=1.0), one chip:

| B  | ms/game |
|----|---------|
| 4  | 684     |
| 8  | **615** |
| 16 | 753     |

- **The VMEM knee moved 8×: B×K≈512 → B×K≈64** (old net: min at B=64).
  Consistent with the attn forward's activations eating the scratchpad
  that used to hold tree state.
- **But the optimum ms/game is only ~9% worse than the old net's**
  (615 vs 563): Stage-1 collection was tree/VMEM-bound, not model-bound,
  so the ~6.5× (CPU) forward barely shows up at the per-chip optimum.
  The feared "attn generator re-prices Stage 1" scenario did not happen.
- **gen-7 64k pricing: ~160 s per 2048-game saved batch on the 2×4**
  (256 games/chip × 0.615 s) **× 32 batches ≈ 1.4 h** — about the same
  wall-clock as the old 32k corpus (~180 s/batch × 16). At B=8/chip a
  2048-game batch is 32 pmap calls of 64 games (vs the old 4×512), so
  expect a little extra host-dispatch overhead on top of the ~160 s.

## 2026-07-04 — gen-7 full-log train (64k corpus): U-curve bottom ~0.113 @ 7.5–11k, onset ~12k — bigger corpus helps, early stopping STAYS

First train on the 64k corpus (gen-6b_es generator, K=8/sims=128,
τ=1.0; 31 train batches × 2048, holdout = last batch, 76,334 positions).
Fresh `PolicyValueNetAttn`, locked recipe, `data_parallel=True` (8
devices), full 20k epochs with full logs; 4,463 s (~22 s/100).

- **Eval v: floor 0.1129–0.1135, flat ~7.5k–11k** (min 0.1129 @ ~9.1k),
  **overfit onset ~11.5–12k**, then a gentle climb: 0.115 @ 13k, 0.118 @
  16k, **0.121 @ 20k**. Train loss falls throughout. Policy CE plateaus
  at 0.9584 (NOT comparable to gen-6b's 0.9509 — different holdout and
  generator entropy).
- **vs gen-6b (15 batches, same architecture/recipe):** min 0.1146 →
  0.1129 (~1.5% deeper); onset ~8k → ~12k; 20k damage 0.147 → 0.121.
  Doubling the corpus bought a modest value-loss gain and much milder
  memorization.
- **Passes-per-batch hypothesis: REFUTED as a constant.** Predicted
  onset at 530 passes × 31 ≈ 16.5k; observed ~12k ≈ 390 passes/batch.
  More data delays overfit sublinearly, not at constant passes.
- **DECISION criterion resolved: the U-minimum did NOT move past 20k →
  early stopping does NOT retire.** It moves to ~10k for the 64k corpus
  (was 7k at 32k). The weight-decay arm (hold the floor at convergence)
  remains the principled fix, still queued.
- **Next:** early-stopped retrain to 10k (mid-flat, fresh ckpt path) →
  `pv_gen7_s128.msgpack` = the gen-7 candidate; gate vs champion
  gen-6b_es (raw-vs-raw ×2 seeds, then PUCT@64 deployed check — both
  nets attn, single template). Optional dose-response arm: retrain on
  `batches[:15]` + same holdout (gen-6b's corpus size, gen-7's
  generator) to separate generator-strength from corpus-size effects.

## 2026-07-04 — gen-7 (attn, 64k corpus, ES @10k): PROMOTED — the crank turned at fuel-gauge ZERO

Early-stopped retrain to 10k (mid-flat per the full-log run, fresh
anchor) → `pv_gen7_s128.msgpack`, gated vs champion gen-6b_es (both
attn, raw τ=0.05, 300 pairs/seed):

- **Raw gate (promotion): seed 0 +5.2, t=3.08, p=0.0022; seed 2 +10.2,
  t=5.74, p<1e-4** — both decisively significant, the second
  generation-class climb in two days.
- **PUCT@64 deployed check: +1.1, t=0.51, p=0.61 — FLAT.** The deployed
  config fully compresses the raw gain (conversion trend: gen-4 +3.5 →
  gen-5b +2.2 ns → gen-6b_es +5.2 sig → gen-7 +1.1 ns).
- **PROMOTED: gen-7 (`pv_gen7_s128.msgpack`) is CHAMPION** and the
  generator for gen-8. Plain numeric SRC/GEN anchors work again.

**The theory headline: the operator fuel gauge is RETIRED as a crank
gate.** Yesterday's probe measured gen-6b_es search at ZERO margin over
its own raw policy (−0.2/−3.4 ns) — and a corpus generated by exactly
that search still trained a +5/+10 student. The gauge-demotion caveat is
now fact: play-strength margin ≠ target information. Candidate
improvement channels (not yet separated): (a) value labels from
stronger self-play games feeding the shared trunk; (b) PUCT visit
distributions as a variance-reduced, determinization-ensembled
rendering of the raw policy — self-distillation that trains a better
student even when the ensemble doesn't PLAY better; (c) 64k volume
feeding attn capacity. The 15-batch dose-response arm would split (c)
from (a)+(b).

**The deployment question is now live:** if search can't amplify gen-7
(and PUCT@64 washed out a +5/+10 raw edge between generations), the
deployed/JTR config should plausibly be RAW (~65× cheaper per move).
Direct test queued: gen-7 PUCT@64 vs gen-7 raw, 240 pairs seed-looped.

**Next:** (1) gen-8 crank — same recipe end-to-end (64k @ K=8/sims=128,
ES @10k; ~2.4 h/generation now that the stopping epoch is known);
(2) gen-7 PUCT@64-vs-raw deployment probe; (3) weight-decay arm
(pipeline economics: retire the per-corpus-size full-log calibration
run); (4) optional 15-batch dose-response arm; (5) attn support in the
JTR export scripts, then a gen-7 POWERFUL calibration.

## 2026-07-04 — Deployment probe: gen-7 PUCT@64 vs its own raw = −6.3 (p=0.0033) — search HURTS at the deployed config; DEPLOY RAW

240 pairs (80×3 seed-looped), gen-7 PUCT@64 (K=8, greedy) vs gen-7 raw
(τ=0.05): **mean −6.3/game, t=−2.97, p=0.0033 — significantly
negative.** Search at the deployed config is now a handicap, not a
wash. Own-search-margin trendline: +26 (gen-3 @128) → +11.1 (gen-4) →
+3.0 ns (gen-5b) → −3.4 ns (gen-6b_es @128) → **−6.3 sig (gen-7 @64)**.

- **DECISION: the deployed config is RAW** (policy head, τ=0.05/greedy —
  ~65× cheaper per move). The JTR calibration should submit raw once the
  export scripts get attn support. **The "PUCT@64 deployed check" is
  RETIRED from the gate procedure** — deployed strength = raw strength,
  so the raw progress gate now covers both roles.
- **This explains the flat promotion check**: gen-7 PUCT@64 vs
  gen-6b_es PUCT@64 (+1.1 ns) was two nets each dragged down by their
  own search — the +5/+10 raw gap was real and search masked it. The
  deployed-conversion "trend" (+3.5 → +2.2 → +5.2 → +1.1) was measuring
  progressively worse search handicaps, not gain compression.
- **Mechanism (consistent with the whole arc):** the net's own move
  choice is now better than 64-sim/K=8 determinized search statistics —
  the search injects determinization noise and shallow-rollout value
  noise that the policy head has already averaged over in training.
- **Open question this raises for the CRANK: gen-8's teacher has
  NEGATIVE play margin.** gen-7 proved a zero-margin teacher still
  distills (+5/+10); whether a negative-margin teacher's visit
  distributions still carry sharpening signal is untested. Cheap
  decisive arm: **collect a RAW-generated corpus** (value labels from
  raw τ=1.0 self-play — ~65× cheaper, 64k in minutes; policy targets
  one-hot self-imitation) and gate its student against the PUCT-corpus
  student. If they tie, the improvement channel is value-labels +
  volume, Stage 1 drops from ~1.8 h to ~2 min, and search exits the
  loop entirely. If the PUCT student wins, the visit-distribution
  self-distillation channel is real and search stays as the
  target-generator only.

## 2026-07-05 — JTR re-calibration at gen-6b_es / gen-7: gap to POWERFUL closes to ZERO

Both attn nets exported through the updated scripts (commit b942b68a)
into JTR's `src/main/resources/models/{pv_gen6b_es_s128,pv_gen7b_es_s128}/export`
(JTR dir `pv_gen7b_es_s128` = pgx champion `pv_gen7_s128.msgpack`; the
`b_es` suffix is JTR-side naming only) and run through the same
real-PUCT harness as the 2026-07-02 calibration (SWEEP_64 = 64
runs/det, FLAT, RUNS mode, swapped-deal paired t-test), except **250
pairs / 500 games** per match (halved from 500 pairs — the prior
gen-5b-vs-gen-4 effect was significant with large margin), seed 42.
Results recorded in JTR's IDEAS.md (jj change `uvslltoswwys`):

| matchup | per-game | t | p | sign test | verdict |
|:--|:--|:--|:--|:--|:--|
| gen-6b_es vs gen-5b (SWEEP_64) | +6.75 | 4.89 | 0.0000 | 141W-93L-16T, p=0.0021 | gen-6b_es stronger |
| gen-7 vs gen-6b_es (SWEEP_64) | +1.0 | 0.70 | 0.48 | 124W-108L-18T ns | wash |
| gen-6b_es vs **POWERFUL** (classical) | +0.35 | 0.24 | 0.81 | 118W-122L-10T ns | tied |
| gen-7 vs **POWERFUL** (classical) | −0.25 | −0.18 | 0.85 | 112W-120L-18T ns | tied |

- **Headline: the absolute-strength gap to POWERFUL has closed to
  ZERO.** Trendline: gen-3 ≈−22/game → gen-5b ≈−9.5/game → gen-6b_es /
  gen-7 both flat ns. First time the pgx lineage matches the classical
  baseline under JTR's harness rather than just narrowing the gap.
- **gen-7 vs gen-6b_es washes under external PUCT too** — independent
  reproduction of the 2026-07-04 mechanism (search at this model
  strength masks raw gains; pgx's own PUCT@64 promotion check read
  +1.1 ns over a real +5.2/+10.2 raw gap). Not a contradiction of
  gen-7's raw gain.
- **Parity caveat:** the SOP-scoped numeric parity test (SavedModel vs
  `attn_model.apply` on random inputs) has not been run; the strongly
  positive gen-6b_es-vs-gen-5b external result is itself evidence the
  attn Keras port is sound (a broken forward pass would not beat the
  verified gen-5b export by +6.75/game).
- **Next (per the deploy-raw DECISION):** raw-policy arena in JTR —
  gen-7 raw vs gen-6b_es raw, and raw vs POWERFUL. Requires a
  `--pgx-raw` mode in JTR (the current `--pgx-policy` flag only wires
  the policy head in as a PUCT prior); if raw ≥ POWERFUL, the deployed
  config beats the classical baseline at ~65× less compute per move.

## 2026-07-05 — JTR raw arena, match 1: gen-7 RAW loses to POWERFUL −8.5/game — deploy-raw does NOT transfer to the external harness

First run of JTR's new `--pgx-raw` mode (argmax of the policy head
averaged over the round's determinizations — no tree, no rollouts;
JTR commit `qzzrmuqy`). gen-7 raw (SWEEP_64 determinization budget,
45 dets at trick 0) vs classical POWERFUL, 250 pairs / 500 games,
seed 42:

**−17.0/pair (−8.5/game), t=−6.20, p<0.0001, sign 85W-154L-11T
(p<0.0001) — decisively negative.** Contrast with yesterday's gen-7
PUCT@SWEEP_64 vs POWERFUL: −0.25/game, tied. Dropping search costs
gen-7 ~8 pts/game against the classical baseline — the OPPOSITE sign
of the internal deployment probe (gen-7 PUCT@64 vs own raw = −6.3,
p=0.0033, "search hurts").

Working interpretation (pending match 2, gen-7 PUCT vs gen-7 raw
external): **not a contradiction — the internal probe was
perfect-information.** pgx's env deals open hands and the net trains
on them; internal "raw" plays argmax on the TRUE state. JTR raw must
marginalize over hidden hands, and averaging policy probs across 45
sampled worlds is the crudest information-set aggregation, while
JTR's PUCT searches within each world and aggregates visit
statistics (real information-set reasoning + value lookahead that
can veto policy blunders against the out-of-distribution classical
opponent). So externally the comparison is "rich determinization
aggregation vs naive policy averaging" under hidden information, not
"search vs no search."

- Transitivity prediction for match 2: gen-7 PUCT beats gen-7 raw
  externally by ≈ +8/game (sign flipped from internal −6.3).
- If confirmed: **the deploy-raw DECISION is harness-scoped** — raw
  stays the deployed config for pgx-internal (perfect-info) gates,
  but JTR/real deployment keeps PUCT.
- Queued diagnostic to split the −8.5: `--cheating` raw (single
  forward pass on true hands = exactly the pgx-internal raw config;
  supported in the raw code path, CLI flag not yet exposed) vs
  POWERFUL. Ties POWERFUL → the whole gap is imperfect-info
  marginalization; still loses → internal raw gains are partly
  self-play-relative and don't transfer to classical opponents.

## 2026-07-05 — JTR raw arena, match 2: gen-7 PUCT beats gen-7 raw +10.15/game — external sign FLIP confirmed; deploy-raw DECISION re-scoped

gen-7 PUCT@SWEEP_64 vs gen-7 raw (same net both sides), 250 pairs /
500 games, seed 42: **+20.3/pair (+10.15/game), t=7.34, p<0.0001,
sign 171W-60L-19T (p<0.0001).** Decisive at 250 pairs.

The three external numbers are internally consistent: raw ≈
POWERFUL − 8.5; PUCT ≈ raw + 10.15 ≈ POWERFUL + 1.65 ≈ the observed
near-tie. And the match-1 transitivity prediction (≈ +8, sign
flipped from the internal −6.3) is confirmed at +10.15 — supporting
the perfect-info-vs-imperfect-info mechanism over "the internal
probe was wrong":

- **Internal (pgx, perfect info):** the net plays argmax on the TRUE
  state; 64-sim determinized search adds only noise → −6.3, search
  hurts.
- **External (JTR, hidden hands):** raw must marginalize by
  averaging policy probs over 45 sampled worlds (crude); PUCT
  searches within each world and aggregates visit statistics —
  information-set reasoning + value lookahead vs an
  out-of-distribution classical opponent → +10.15, search helps.

**DECISION (2026-07-05): deploy-raw is HARNESS-SCOPED.**
- pgx-internal (perfect-info gates, self-play): raw remains the
  deployed config; the raw gate stays the standing progress gate;
  the PUCT@64 deployed check stays retired.
- JTR / any real imperfect-info deployment: the submitted config is
  PUCT (SWEEP_64) — and that calibration is already done: gen-7
  PUCT ties POWERFUL (2026-07-05 entry above).
- The gen-8 raw-corpus A/B is UNAFFECTED: corpus collection happens
  in pgx's perfect-info env, where raw τ=1.0 self-play is a valid
  (and ~65× cheaper) generator; the open question there is target
  quality, not deployment strength.

Optional diagnostic still queued: `--cheating` raw vs POWERFUL
(true-hands single forward pass = exactly the pgx-internal raw
config; needs a CLI flag) to split the −8.5 into
imperfect-info-marginalization cost vs genuine raw-policy weakness
against classical opponents.

## 2026-07-05 — Cheating-raw diagnostic: perfect-info raw STILL loses to POWERFUL −7.5 — the marginalization hypothesis is REFUTED; the raw policy itself is the gap

gen-7 raw with `--cheating` (single forward pass on the TRUE hands —
exactly the pgx-internal raw config; JTR change `yovnospt`) vs
classical POWERFUL, 250 pairs / 500 games, seed 42:

**−15.0/pair (−7.5/game), t=−4.99, p<0.0001, sign 86W-156L-8T** —
nearly identical to fair (determinized) raw's −8.5/game.

- **The marginalization hypothesis from the match-1 entry is
  REFUTED**: giving raw the true state recovers only ~1 pt of the
  ~8.5-pt gap. Determinization-averaging is nearly free (also
  evidence the 45-world averaging implementation is sound — it
  tracks the true-state policy closely). The gen-7 raw policy is
  simply ~7.5/game weaker than POWERFUL's search, even with perfect
  information.
- **Where PUCT's +10.15 comes from, revised: the search process
  itself — and note the BUDGET asymmetry.** The internal "search
  hurts" probe (−6.3) ran 64 TOTAL sims; JTR's SWEEP_64 runs 64
  sims/det × 45 dets ≈ 2,880 expansions/move with root aggregation
  across worlds. The entire internal own-search-margin trendline
  (+26 → +11 → +3 → −6.3) was measured at 64–128 total sims. So the
  internal and external findings likely reconcile as a
  search-BUDGET effect, not an information effect: tiny searches
  now hurt (net's argmax is better), large determinized searches
  still help a lot.
- **Reading for internal gates:** raw-vs-raw progress gates remain
  valid as RELATIVE progress measures, but this shows self-play
  raw strength does not translate 1:1 into absolute strength vs an
  out-of-distribution classical opponent — the same caution the
  gen-3-era results raised. External absolute strength currently
  REQUIRES the search wrapper; the value head at 2,880-sim scale is
  doing real work (the "search exits the loop" narrative is
  internal-only, and specifically small-budget-internal only).
- Not re-opened here (standing DECISION): the internal search axis
  at collection. But if a cheap internal probe is ever wanted, the
  discriminating experiment is internal PUCT at a JTR-scale budget
  (~2–3k sims) vs raw — it would separate "budget" from
  "harness/opponent" cleanly.

## 2026-07-05 — gen-8 Arm A full-log train (gen-7 corpus, 64k, no WD): U-curve floor ~0.111 @ 8–11.5k, onset ~12k — ES @10k; whole curve shifted DOWN ~0.002 vs gen-7's run

Stage 2, gen-8 crank (scoped to the weight-decay comparison — SOP item
3). Standard recipe on the fresh gen-7-generated corpus (32×2048 PUCT,
K=8/sims=128; holdout 76,326 labeled positions), PolicyValueNetAttn,
data_parallel over the 2×4, full 20k with full logs, NO decay.

- **Eval-v U-curve: floor 0.1112–0.1116, flat ~8.2k–11.5k** (min
  ≈0.1112 at ~9.8k and ~10.9k — noise-level distinction), **overfit
  onset ~12k** (0.113 @13.5k, 0.115 @16k, ~0.120 @20k; train keeps
  falling). Same shape and same ES epoch as gen-7's 64k run
  (0.113 @ 7.5–11k, onset ~12k) — **ES @10k confirmed for Arm A-ES.**
- **The curve sits ~0.002 below gen-7's run everywhere** (floor 0.111
  vs 0.113): the gen-7-generated corpus trains a slightly better value
  head than the gen-6b_es corpus did — teacher quality still shows up
  in the labels even at fuel-gauge ZERO.
- Policy CE clean as always: converges ~0.9585–0.9590 by ~10k, no
  upturn through 20k. Value remains the only overfitting head.
- **Arm B success bar is now concrete: holdout v ≤ ~0.111 at 20k with
  the seen-vs-holdout gap closed** (this run's 20k lands 0.120).
- Operational: the run's 10k checkpoint was overwritten by the resumed
  10k→20k leg (slot files alternate) — Arm A-ES is a fresh
  `num_epochs=10_000` retrain per the standing procedure; Arm B =
  same call + `weight_decay=1e-2`, `num_epochs=20_000`, fresh GEN=
  checkpoint name (adamw opt_state ≠ adam's).

## 2026-07-05 — gen-8b_es (Arm A-ES, no WD, ES @10k) raw gate vs gen-7: WASH — NOT promoted; gen-7 stays champion

Stage 3 progress gate, raw-vs-raw (temperature=0.05), A=gen-8b_es /
B=gen-7, 300 pairs/seed. Nets fingerprinted distinct first.

- **Seed 0: mean +2.3, t=1.49, p=0.1362** (not significant).
- **Seed 2: mean −0.6, t=−0.39, p=0.6963** (null, wrong sign).
- **Verdict: FAIL.** Promote requires both seeds significant p<0.05,
  same sign; here the seeds don't even agree in direction. Early-
  stopping-only on the fresh gen-7 corpus does not clear gen-7.
- Consistent with the train log read: the eval-v floor shifted DOWN
  ~0.002 (0.111 vs 0.113), i.e. a marginally better value head, but
  policy CE converged to the same ~0.958 soft floor as gen-7 — and the
  raw gate reads the **policy**, so a flat policy floor → flat gate.
  The value gain is real but not gateable (value head is a wash every
  gen, standing result).
- **Next: gen-8c_wd** (`weight_decay=1e-2`, `num_epochs=20_000`, fresh
  checkpoint name) is now the live gen-8 candidate — the ES-only recipe
  is exhausted. Same raw-vs-raw gate vs gen-7. If 8c_wd also washes, the
  recipe/corpus isn't producing a gen-7→8 climb and the question moves
  to fuel (bigger corpus) over regularization.

## 2026-07-05 — gen-8c_wd (weight decay 1e-2, full 20k) raw gate vs gen-7: WASH — NOT promoted; BOTH gen-8 arms fail, gen-7 stays champion

Stage 3 gate, same setup (raw-vs-raw temp 0.05, A=gen-8c_wd / B=gen-7,
300 pairs/seed, fingerprints distinct).

- **Seed 0: mean +0.1, t=0.031, p=0.9752** (dead null).
- **Seed 2: mean −1.3, t=−0.738, p=0.4612** (null, wrong sign).
- **Verdict: FAIL — more decisively null than 8b_es.** Weight decay
  does not clear gen-7 either.

**gen-8 is CLOSED as a regularization play. Both arms wash.** ES-only
(8b_es) and WD (8c_wd), each trained on the fresh gen-7-generated 64k
corpus, are indistinguishable from gen-7 on the raw gate. Regularization
only governs the value-head U-curve (a gate wash every generation); it
was never going to move the policy, and the policy is what the gate
reads.

**The real finding: a same-size self-distillation round is a FIXED
POINT at gen-7.** New teacher, same corpus size (64k), fresh net →
zero raw-gate gain. This is the operator-gauge-ZERO situation made
concrete: perfect-info PUCT@128 no longer beats gen-7's raw policy
(gauge ZERO since gen-6b_es; gen-7 PUCT@64 measured −6.3 vs its own
raw), so the distillation targets ≈ the current raw policy, and
iterating reproduces the champion. The AlphaZero improvement operator
(search stronger than raw) has stalled internally — no gradient pulls
the net forward from self-play.

**What the levers look like now:**
- **More fuel (>64k corpus):** the fuel that worked gen-6→7 was
  CONFOUNDED with the architecture jump; on a pure same-arch round it
  just wash-tested. More of the same fixed-point targets won't move the
  policy — it only feeds the value head (gate wash). Low expected value
  for the raw gate.
- **Re-open the operator:** the 2026-07-03 sharpening probe found NO
  search axis (sims/determinizations/PUCT-c) re-opens the perfect-info
  gauge. If that holds, self-play distillation is saturated.
- **Net capacity / architecture:** the last real climb (gen-6b_es,
  +10/+7) WAS an architecture jump (attn). Step-4 value-head/net scaling
  is the standing candidate to add extractable capacity.
- **External headroom is REAL:** JTR arena has gen-7 raw −8.5 and even
  cheating-raw −7.5 vs POWERFUL (2026-07-05) — the raw policy genuinely
  is the gap. The improvement signal exists; the internal self-play loop
  just can't see it because its own teacher is exhausted. A teacher that
  produces targets stronger than the current raw policy (stronger/
  different search, or POWERFUL-style targets) is what would re-open the
  loop.

**DECISION PENDING (2026-07-05): gen-9 direction.** Not "bigger corpus"
by default — that was scoped when the fixed-point wasn't yet
established. Candidates: (a) net-capacity scaling; (b) a stronger
perfect-info teacher to re-open the operator; (c) the deferred
dose-response arm is now mainly an economics probe (can we SHRINK
corpus for free?), not a strength lever. Think before cranking.

## 2026-07-05 — CORRECTION: `num_simulations` is PER DETERMINIZATION — the "budget asymmetry" arithmetic was wrong; gen-9 option space framed + discriminating probe pre-registered

Code check (`jass_puct.py::puct_search`: the K dets are the mctx batch,
`num_simulations` runs per batch element — "Tree simulations per
determinization" in the docstring): every internal budget quoted as
"total sims" in the 2026-07-05 cheating-raw entry was off by a factor
of K.

- gen-7 deployed check PUCT@64 K=8 = **512 expansions/move** (not 64).
- The sharpening probe's biggest arms (gen-5b: @256 K=8, @128 K=16) and
  the gen-6b_es re-probe's live arm (K=16 @128) = **2,048
  expansions/move**.
- JTR SWEEP_64 = 64/det × 45 dets = **2,880 expansions/move**.

So the gap between "internal gauge reads ZERO" (gen-6b_es K=16@128,
−0.2) and "external search wins big" (gen-7 JTR, +10.15) is **1.4× in
budget, not 45×** — the pure-budget reconciliation is under-determined.
The genuinely un-probed axes at the attn generation:
(a) **worlds** — K=45 vs 16 (K was the live axis in both internal
    probes);
(b) **the searcher itself** — JTR runs classical PUCT over all legal
    actions with its own c/backup; internal is mctx Gumbel with
    `max_num_considered_actions=16` acting on summed visit counts;
(c) harness/opponent context.

**gen-9 option space (resolves the DECISION PENDING above):**

- **A. Re-open the operator with a JTR-mirror teacher.** Cheapest
  discriminating step — probe pre-registered below. If internal search
  at the JTR config beats gen-7 raw ≈ +8–10, gen-9 = collect at that
  config (2.8× the K=8@128 collection budget → Stage 1 ~4 h at 64k,
  less if the corpus shrinks — stronger targets should need fewer
  games). If it reads ~0, budget/worlds do NOT re-open the operator
  inside mctx-Gumbel; the JTR searcher/harness is the difference and
  the options move to B/C (or importing a JTR-style searcher as
  teacher).
- **B. Net capacity scaling.** Bigger attn trunk/heads trained on the
  EXISTING 64k gen-7 corpus — zero collection cost, pure train + gate.
  Precedent: gen-6b_es (+10/+7) extracted signal from a corpus that had
  gated flat twice for the old net. Risk: unlike gen-6b, today's
  targets ≈ the champion's raw policy (fixed point), so capacity may
  just reproduce gen-7 with more parameters; the ~0.9585 policy-CE
  floor may be target entropy, not a capacity limit. Cheap to falsify,
  and per the sharpening-probe DECISION a materially better net is
  itself what re-opens the search axes.
- **C. Better small-budget targets (Gumbel knobs).** The searcher is
  already `mctx.gumbel_muzero_policy` (improvement-guarantee machinery
  in place); unexplored: `max_num_considered_actions` 16 → wider, and
  training on the completed-Q `action_weights` output instead of raw
  summed visit counts (the current pi). Fallback if A washes or its
  economics hurt.
- **Ruled out:** more same-recipe fuel (fixed-point targets in greater
  volume), more regularization (gen-8 closed it), JTR/POWERFUL games as
  targets (standing DECISION, 2026-07-05), Step-5 imperfect-info work
  (addresses external raw strength, not the internal climb).

**Pre-registered probe (the Option-A discriminator):** gen-7 PUCT
(greedy) vs gen-7 raw (τ=0.05), `policy_match`, on the 2×4 with one
chunk per chip (`jax.pmap`, SOP probe-parallelization TODO now done),
8 pairs/chip × 8 chips × 5 seed rounds = 320 pairs/arm:

| arm | expansions/move | question |
|:--|:--|:--|
| K=45 × 64 | 2,880 | JTR mirror — does the external config win internally? |
| K=16 × 128 | 2,048 | anchor — replicate gen-6b_es's −0.2 on gen-7 |
| K=8 × 360 | 2,880 | worlds-vs-depth control (decisive only if arm 1 re-opens) |

Predictions: budget/worlds hypothesis → arm 1 ≈ +8–10 with arm 3
lagging it; searcher/harness hypothesis → all arms ≈ 0.

## 2026-07-06 — gen-9 direction probe: NO mctx config beats gen-7 raw at the JTR budget — Option A (budget/worlds) is CLOSED; the external +10.15 lives in the SEARCHER/harness

The pre-registered three-arm probe (2026-07-05 correction entry), run
on the 2×4 via the new pmap recipe — the chip-parallel `policy_match`
worked first try, ~8 min wall clock for all three arms at 320
pairs/arm (the old single-device estimate was ~14 min per *arm*).
gen-7 PUCT (greedy summed-visit argmax; champion alias gen-7b_es =
`pv_gen7_s128.msgpack`, the JTR export name — see 2026-07-05) vs
gen-7 raw (τ=0.05), same deals across arms:

| arm | exp/move | mean/game | t | p | sign (pairs) |
|:--|:--|:--|:--|:--|:--|
| K=45×64 (JTR mirror) | 2,880 | −1.1 | −0.66 | 0.51 ns | 150W/164L ns |
| K=16×128 (anchor) | 2,048 | −3.4 | −1.99 | 0.048 * | 137W/175L * |
| K=8×360 (depth ctl) | 2,880 | −9.8 | −5.52 | <0.0001 *** | 114W/200L *** |

- **Pre-registered prediction resolved: the searcher/harness
  hypothesis wins.** The JTR-mirror config reaches zero (−1.1 ns) —
  the predicted +8–10 budget effect is decisively excluded. No
  reachable mctx configuration produces a positive margin over raw.
- **K is a noise knob, not a strength knob.** At ~fixed budget the
  worlds axis is monotone but converges to zero FROM BELOW: −9.8
  (K=8) → −3.4 (K=16) → −1.1 (K=45). More determinizations only
  cancel determinization noise; they never add margin.
- **Depth actively HURTS at fixed budget**: 360 sims in 8 worlds is
  the worst config ever measured on an attn net (−9.8). Reading: a
  deep search inside one sampled world commits to that world's
  private information, so the per-world recommendation drifts from
  the information-set optimum, and 8 worlds can't average the drift
  out — while the raw baseline plays the TRUE state, making every
  determinization artifact pure cost.
- **Reconciliation with JTR (+10.15 PUCT-vs-raw, same net):** the
  cheating-raw diagnostic showed true-state raw ≈ marginalized raw
  (~1 pt), so the baselines are comparable — in JTR's harness the
  same net's search extracts ~+10 over raw where mctx-Gumbel at the
  same budget extracts ~0. **A teacher stronger than gen-7 raw
  demonstrably EXISTS; it lives in JTR's searcher (or harness), not
  in mctx sims/worlds.**
- **The question that now matters (Option A′, zero compute): what
  does JTR's `--pgx-policy` search evaluate leaves with?** If it
  blends its classical ROLLOUT machinery with the net, the +10 is
  rollout ground truth correcting value-head blind spots — an
  importable teacher signal (rollout-backed leaf evals at
  collection). If it is pure net priors+value, the delta is
  selection/backup mechanics (full-width classical PUCT vs Gumbel
  sequential halving read out via summed visits) — importable by
  mirroring JTR's search in mctx (`muzero_policy`, its c/backup) and
  re-running this probe.

**Option table after the probe:** A (mctx budget/worlds) CLOSED by
measurement. Live: **A′** — read JTR's search code, identify the
delta, import it, re-probe (code reading is free; this is the only
lever with a MEASURED +10 behind it); **B** — capacity scaling on the
existing 64k corpus (unchanged, still zero collection cost); **C** —
Gumbel `action_weights` targets (unchanged but expectations capped:
no internal config shows play-strength margin to distill).

## 2026-07-06 — A′ scoped by reading JTR's search code: leaf evals are the pgx VALUE HEAD (rollout hypothesis refuted for card play); the deltas are CLASSICAL FULL-WIDTH PUCT and heuristic-playout TRUMP search

Read the `--pgx-policy` path in JTR (MCTS.java, JassBoard.java,
MCTSHelper.kt, PgxPlayoutSelectionPolicy.java, StrengthLevel.kt). What
JTR's search actually does with the exported net:

- **Card-play leaf evaluation = the pgx value head, NOT rollouts.**
  `playout()` short-circuits on `board.hasScoreEstimator()` →
  `JassBoard.estimateScore()` → `pgxEstimator.predictValue(game)`
  (signed differential mapped to per-player points). MCTSHelper logs
  "Using pgx PolicyValueNet value head to determine the score". So the
  external +10.15 is extracted from the SAME two signals (policy prior
  + value leaf) the internal search uses — the rollout-ground-truth
  hypothesis is REFUTED for card play.
- **EXCEPT trump selection: `hasScoreEstimator()` is false there** —
  trump-phase search runs real playouts via
  `PgxPlayoutSelectionPolicy.runPlayout`, which delegates to the HEAVY
  RULE-BASED heuristic. So the PUCT arm's trump decisions are searched
  with JTR's expert rules while the raw arm's trump comes from the
  policy head. A candidate channel for part of the +10 — and one that
  imports JTR domain knowledge (adjacent to the no-JTR-games-in-
  training DECISION if ever used as a teacher).
- **Selection = classical full-width PUCT, no Gumbel:**
  `findChildrenPuct`: Q = mean backed-up score in POINTS (0–157,
  unvisited = 0), U = `puctC`(=100, "scaled for 0–157 reward range") ×
  P(s,a) × √(N_parent+1)/(1+n), full softmax prior from the policy
  head cached once per node, every legal child considered every visit.
  No sequential halving, no completed-Q, no 16-action root cap.
- **Budget confirmed:** SWEEP_64 = factor 5 → (9−round)×5
  determinizations (45 at trick 0, declining to 5) × 640/10 = 64
  runs/det. Matches the 2,880-expansions-at-trick-0 arithmetic; note
  the internal probe held K=45 constant across all tricks, JTR
  doesn't. Historical note in StrengthLevel.kt: the sweep was built to
  find where soft-prior PUCT beats the raw prior ("pgx crossover was
  ~sims 40-50, healthy by ~128").

**Discriminating next step (the new Option A′ probe): swap
`mctx.gumbel_muzero_policy` → `mctx.muzero_policy` in `jass_puct.py`**
(classical PUCT selection, full width; set
`dirichlet_fraction=0` for a deterministic teacher probe, tune
`pb_c_init` toward JTR's effective c — values are in points internally
too, v_scale=100) and re-run the 2026-07-06 K=45×64 probe unchanged.
- Reproduces ≈ +8–10 → the improvement operator is re-opened INSIDE
  pgx with net-only signals; gen-9 = collect with muzero_policy
  targets (and dose-response goes live).
- Still ≈ 0 → the +10 lives in the trump-phase heuristic playouts or
  the harness; next discriminator is a JTR arena arm with trump forced
  to the policy head on both sides.

## 2026-07-06 — A′ probe: classical PUCT (`muzero_policy`) beats gen-7 raw +11.8*** at the JTR budget — THE OPERATOR IS RE-OPENED; the bottleneck was the GUMBEL SEARCHER all along

Same budget (K=45×64 ≈ 2,880 expansions/move), same net (gen-7
champion), same deals (same seeds) as the 2026-07-06 Gumbel probe —
only the searcher swapped (`search_variant="muzero"`,
`dirichlet_fraction=0`; knob landed `sxznyotm`). 320 pairs/arm,
~2.5 min/arm on the 2×4:

| arm (pb_c_init) | mean/game | t | p | sign (pairs) |
|:--|:--|:--|:--|:--|
| 0.64 (JTR-equiv) | **+11.8** | +7.41 | <0.0001 *** | 204W/98L *** |
| 1.25 (AZ default) | +11.5 | +7.05 | <0.0001 *** | 210W/97L *** |
| 2.5 (high-expl) | +11.1 | +6.69 | <0.0001 *** | 205W/97L *** |

- **Reproduces JTR's external margin (+10.15) internally with
  NET-ONLY signals.** The trump-phase heuristic-playout channel is
  NOT needed to explain the external result.
- **pb_c is a plateau across 0.64–2.5** — no tuning cliff; the mctx
  default (1.25) is fine.
- **Direct searcher A/B at identical budget: Gumbel −1.1 ns vs
  classical PUCT +11.8***.** The searcher choice alone is worth ~13
  points of operator margin on this net.
- **REINTERPRETATION cascade — every "operator" number in this log
  was measured through Gumbel:** the fuel trendline (+26 → +11 → +3
  → −6.3), the sharpening probe ("no search axis re-opens"), the
  gauge-ZERO retirement, the internal deploy-raw finding (−6.3), and
  the gen-8 fixed point. All of it now reads as "the net outgrew
  SMALL-SIM GUMBEL-AS-READ-OUT-BY-SUMMED-VISITS", not "search (or
  self-play) is exhausted". The gen-8 wash follows by construction:
  its corpus targets were Gumbel visit counts ≈ the raw policy.
- Working note on WHY: Gumbel sequential halving spreads root visits
  across the considered set in phases, so a summed-visit readout is
  flat/noisy at 64 sims/det — and Gumbel's actual recommendation is
  completed-Q-based, which our visit-argmax aggregation never reads.
  Classical PUCT concentrates visits on its preferred action: summed
  visits ARE its native signal (and JTR's aggregation).

**DECISION (2026-07-06): gen-9 = the muzero-teacher crank.**
- **Pre-collection probe first (cheap):** muzero at the STANDING
  collection config K=8×128 (1,024 exp/move) vs gen-7 raw. Margin
  holds → Stage 1 cost unchanged (~1.4 h at 64k). Collapses → pay
  for K=45×64 (~2.8×) or find the knee (K=16×64, K=45×16, …).
- Collection call: `make_puct_collect_fn(..., search_variant=
  "muzero", pb_c_init=1.25)`; keep `dirichlet_fraction=0` (τ=1.0
  visit sampling already provides self-play diversity; noise-free
  targets match the probed teacher) and τ=1.0 per the standing
  recipe.
- **The 15-batch dose-response arm goes LIVE again** (stronger
  targets may need fewer games — it carries the corpus-size DECISION
  for the new recipe).
- The raw gate stays the progress gate; external deployment stays
  JTR-PUCT (unchanged).

## 2026-07-06 — Pre-collection probe: the muzero margin HOLDS at cheap configs — gen-9 collects at K=16×64 (standing cost class, +10.5)

Same setup as the pb_c sweep (muzero, pb_c=1.25, 320 pairs/arm, same
seeds/deals), cheap configs:

| arm | exp/move | mean/game | t | p | sign (pairs) |
|:--|:--|:--|:--|:--|:--|
| K=8×128 (standing collect cfg) | 1,024 | +7.1 | +3.97 | 0.0001 *** | 190W/117L *** |
| K=16×64 (worlds swap) | 1,024 | **+10.5** | +6.40 | <0.0001 *** | 200W/103L *** |
| K=8×64 (old deployed cfg) | 512 | +9.4 | +5.69 | <0.0001 *** | 192W/112L *** |

- **The margin holds at every cheap config** — no knee between 512
  and 2,880. Even 512 exp/move reads +9.4 where Gumbel read −6.3: a
  ~15.7-point searcher swing at the old deployed budget. (The
  internal "search hurts → deploy raw" finding is therefore
  searcher-scoped too; the raw gate stays the progress gate
  regardless — it measures the POLICY, which is what training moves.)
- **Worlds-over-depth, yet again, now for classical PUCT:** at equal
  budget 16×64 (+10.5) beats 8×128 (+7.1), and 8×64 (+9.4) ≥ 8×128
  (+7.1) at HALF the cost — extra per-world depth is
  worthless-to-harmful. The three probes agree: spend budget on
  worlds, keep per-world sims at 64.
- **K=16×64 delivers ~91% of the full JTR-budget margin (+11.5) at
  36% of the cost.**

**DECISION (2026-07-06): gen-9 Stage 1 = standard 32×2048 with
`make_puct_collect_fn(..., search_variant="muzero", pb_c_init=1.25,
num_determinizations=16, num_simulations=64)`, τ=1.0,
`dirichlet_fraction=0`.** Tree working set is unchanged (16×65 ≈
8×129 nodes/game), so per-chip B=8 should stand — quick
`profile_collect_fn` sanity check per SOP before the crank; wall
clock may even improve (half the sequential sims at 2× the leaf-eval
batch). Teacher margin +10.5 is gen-4-band fuel (+11.1 → +15 raw
gate) — the first real teacher signal since gen-6b.

## 2026-07-06 — NAMING: the muzero round is GEN-8 retaken (student gen-8d_mz_es), anchored SRC="7b_es_mz"

The generator is still the gen-7 champion, so this is gen-8 — **a
generation isn't abandoned when attempts fail; attempts get
suffixes** ("gen-9" in the 2026-07-06 entries above refers to this
round; a generation may take many attempts). Student net:
**gen-8d_mz_es** (d = next free attempt letter after the washed
8b_es/8c_wd; mz = muzero teacher, es = early-stopped per the
standing recipe). Anchor: the round's
`SRC` token is **`7b_es_mz`** — the champion's own label (`7b_es`)
plus the teacher-recipe tag (`mz`) — so every SOP-derived f-string
filename (corpus shards, checkpoint, final net) is self-describing
about generator AND recipe; the champion PARAMS file stays `7b_es`
(no `_mz` — load it via a separate CHAMP token). Corpus kept BIG on
purpose — 32×2048 like the washed arms — so the searcher swap is the
ONLY change.

## 2026-07-06 — gen-8d_mz full 20k train on the muzero corpus: NO U-curve — the first attn run that does not overfit; gates as the 20k net (no _es earned)

Stage 2 on `corpus_puct_gen7b_es_mz` (32×2048, muzero K=16×64
teacher), standard recipe, full 20k:

- **No overfit onset AT ALL through 20k** — eval total flat-to-falling
  to the end (0.7967 @19.9k, still drifting down), train≈eval gap
  ~0.005, value seen-vs-holdout gap essentially CLOSED. Every previous
  64k attn run turned up at ~12k. **Early stopping is unnecessary for
  this corpus: the gate candidate is the 20k net, named gen-8d_mz
  (the _es suffix is not earned).**
- **Holdout value loss ≈ 0.074** vs the 0.111–0.113 floor of every
  gumbel-corpus run — a ~33% drop. Comparability caveat: v is measured
  against THIS generator's outcome labels, and stronger/more
  consistent play is intrinsically more predictable, so part of the
  drop is label quality, not head quality. Same caveat for policy CE
  (~0.723 vs the old ~0.9585 soft floor — muzero visit targets are
  much more peaked, so the entropy floor itself is lower).
- The no-overfit result FITS the label-noise story: the U-curve was
  the value head memorizing outcome noise in weak self-play; muzero
  games have less of it. (It also retro-explains why bigger corpora
  only *softened* the U-curve before — more noise to average, same
  noise level per game.)
- Ops: resumed leg 15.5k→20k ran ~21 s/100 epochs, `training done
  [1006 s]`.

**Next: Stage 3 raw gate, gen-8d_mz (20k net) vs gen-7 champion
(`7b_es`), two seeds, 300 pairs, fingerprints first.** Teacher margin
was +10.5; the loss table says the net absorbed the sharper targets —
the gate says whether it converts.

## 2026-07-06 — gen-8d_mz raw gate vs gen-7: +13.7/+11.0 BOTH SEEDS *** — PROMOTED; the muzero-teacher crank converts at ~1:1

Stage 3 raw gate (temp 0.05, 300 pairs/seed, fingerprints distinct:
new=29546.98 / src=28867.91; baseline label printed the SRC corpus
token "7b_es_mz" but the loaded net is the gen-7 champion):

- **Seed 0: mean +13.7, t=+7.833, p<0.0001, sign 203W/79L***.**
- **Seed 2: mean +11.0, t=+6.493, p<0.0001, sign 189W/96L***.**
- **Verdict: PROMOTED. gen-8d_mz is CHAMPION** and the next-round
  generator. First promotion since gen-7 (2026-07-04), third attempt
  at gen-8, and the single change vs the two washed attempts was the
  teacher searcher.

Conversion: teacher margin +10.5 (muzero K=16×64 over gen-7 raw) →
student raw +11.0/+13.7 over the same opponent. The student fully
absorbed its teacher's edge — same 1:1-or-better band as gen-4
(+11.1 fuel → +15 gate). Together with the no-U-curve train and the
0.074 value floor, every stage of the round says the same thing: the
Gumbel-visits teacher was the binding constraint on the whole loop.

**The crank is TURNING again. Queue for the next rounds:**
1. **gen-9 (SRC=8d_mz): same recipe, new generator** — muzero
   K=16×64, 32×2048, full 20k (ES only if a U-curve reappears —
   watch the full logs). The standing question each round: does the
   operator margin (muzero vs new raw) persist at the new champion?
   Probe it while the corpus collects, per the old cadence.
2. **JTR re-calibration is OWED** — export gen-8d_mz. The headline
   external question: gen-7 raw lost to POWERFUL by −7.5/−8.5 and
   gen-8d_mz raw is ~+12 over gen-7 raw internally — raw may now
   MATCH the classical baseline with no search at all (and PUCT
   should clear it).
3. **Dose-response** (15-batch student on this corpus) — the
   corpus-size DECISION for the muzero recipe.
4. Parked: C′ gumbel-native-readout efficiency probe; B capacity
   scaling.

## 2026-07-06 — gen-9 full 20k train on the gen-8d_mz muzero corpus: NO U-curve AGAIN, loss floors drop further (v 0.0655, p 0.604)

Stage 2 on the gen-9 corpus (SRC=8d_mz, 32×2048, muzero K=16×64
teacher — the SAME recipe as gen-8d_mz, only the generator advanced
to the new champion), standard recipe, full 20k, 76,543 holdout
positions:

- **No overfit onset again through 20k** — eval falls monotonically
  to the end (0.6694 @18.8k, 0.6701 @19.9k, still drifting down),
  train≈eval gap ~0.005. Second consecutive muzero-corpus run with no
  U-curve; the label-noise story holds — the muzero generator makes
  clean outcome labels, so early stopping stays unnecessary and the
  gate candidate is the 20k net (no `_es`).
- **Holdout value loss v ≈ 0.0655** — even below gen-8d_mz's 0.074.
  Same caveat: a stronger, more consistent generator produces
  more-predictable outcomes, so part of the drop is label quality.
- **Policy CE p ≈ 0.604** vs gen-8d_mz's 0.723 — targets even more
  peaked (the gen-8d_mz teacher net now drives sharper muzero visit
  distributions than the gen-7 teacher did).
- Ops: full 20k from scratch, `training done [4418 s]` (~74 min; the
  gen-8d_mz 1006 s was only the resumed 15.5k→20k leg).

## 2026-07-06 — gen-9 raw gate vs gen-8d_mz: +2.8/+4.2 both seeds significant — PROMOTED, but the step collapses ~4–5× (crank decelerates)

Stage 3 raw gate (temp 0.05, 300 pairs/seed, fingerprints distinct
new=29639.75 / src=29546.98 = the gen-8d_mz champion; both labels
printed "gen-8d_mz raw" — the cosmetic SRC-token bug, but src fp
29546.98 = gen-8d_mz's own "new" fp from its promotion entry, so the
baseline IS the new champion):

| seed | W/L | win% | mean/game | t | sign (pairs) |
|:--|:--|:--|:--|:--|:--|
| 1 | 305/295 | 50.8% | **+2.8** | +2.319 p=0.0211 * | 151W/88L p=0.0001 *** |
| 2 | 309/291 | 51.5% | **+4.2** | +3.501 p=0.0005 *** | 145W/104L p=0.0111 * |

- **Verdict: PROMOTED.** Both seeds positive and significant on the
  pair-mean t-test → gen-9 is CHAMPION and the gen-10 generator. Two
  consecutive promotions since the teacher swap.
- **But the step collapsed ~4–5×:** gen-8d_mz beat gen-7 by
  +13.7/+11.0; gen-9 beats gen-8d_mz by only +2.8/+4.2. This is the
  first SAME-recipe iteration (new generator, teacher unchanged) — so
  the +11–14 was the one-time teacher-swap unlock, and iterating the
  loop on the fixed muzero recipe yields a much thinner gen-over-gen
  gain. Loss floors kept dropping (v 0.074→0.0655, p 0.723→0.604) but
  the raw gate barely moved — the net absorbed sharper targets that no
  longer translate into much extra raw strength vs the (already
  stronger) previous champion.
- Sign/mean tension, seed 1: 151W/88L pairs (strongly more wins,
  p=0.0001) yet mean only +2.8 — gen-9 wins many pairs by small
  margins and loses fewer-but-larger. Consistent with a small but real
  edge, not a fluke.

**DECISION (2026-07-06): promote gen-9, but treat +2.8/+4.2 as a
deceleration flag — re-run the OPERATOR PROBE before committing gen-10
to the same recipe.** The standing question each round is whether the
muzero-vs-raw operator margin persists at the new champion; a thin raw
gate suggests that margin may be compressing too. Probe muzero K=16×64
vs gen-9 raw:
- Margin holds (~+10) → the recipe still has fuel; run gen-10 as-is
  and expect another small step.
- Margin has compressed → the muzero teacher is nearing saturation
  against its own student; time to change a lever (B capacity scaling
  or corpus size) rather than iterate the fixed recipe.

**Queue (unchanged priorities, gen-9 now the generator):**
1. Operator probe (muzero vs gen-9 raw) — carries the gen-10 recipe
   decision, per above.
2. **JTR re-calibration still OWED** — export gen-8d_mz/gen-9; the
   headline external question (can raw now match POWERFUL with no
   search?) only got more pressing with two promotions banked.
3. Dose-response (15-batch student) — corpus-size DECISION.
4. Parked: C′ gumbel-native-readout probe; B capacity scaling (now a
   live candidate if the operator probe reads saturation).

## 2026-07-06 — Operator probe at the new champion: muzero K=16×64 vs gen-9 raw = −1.0 ns — the operator margin is GONE (recipe saturated)

Same probe config as the pre-collection sweep (muzero, pb_c=1.25,
K=16×64, greedy teacher vs raw temp 0.05), teacher and baseline BOTH
gen-9 (fp 693.58), 320 pairs on the 2×4:

- **mean −1.0/game, t=−0.765 p=0.4450 ns, sign 130W/140L p=0.5840 ns**
  (wins 309 / losses 331). The margin didn't just shrink — it's at
  zero (marginally negative).
- **The operator arc, top to bottom:** muzero K=16×64 read **+10.5
  over gen-7 raw** (pre-collection probe) → gen-8d_mz raw beat gen-7
  by +13.7/+11.0 → gen-9 raw beat gen-8d_mz by only +2.8/+4.2 → and
  now the SAME teacher **can't beat gen-9's raw at all.** The student
  has caught up to its teacher: at the K=16×64 collection budget,
  search no longer improves on the raw policy, so there is no target
  signal left for a gen-10 on the identical recipe to climb. This is
  the fixed point the plan anticipated ("with the anchor gone, a flat
  raw gate will then mean saturation for real").

**DECISION (2026-07-06): do NOT run gen-10 on the fixed muzero K=16×64
recipe — it would gate flat. Two candidate levers; one cheap
diagnostic decides between them BEFORE any collection/training:**
- **Sweep the teacher budget UP first (cheap, no training): muzero
  K=32×64 and K=45×64 vs gen-9 raw, same probe.** This is the SOP
  "starved operator" test — a small/negative margin at the current
  budget may just mean the teacher is under-resourced now that the
  student is strong.
  - **Margin reopens with more worlds** → the net still has headroom;
    gen-10 = crank the teacher harder (more worlds, keep sims=64 per
    worlds-over-depth) and pay the higher collection cost.
  - **Still flat at K=45×64** → search is not the constraint, NET
    CAPACITY is. Pull the **B capacity lever** (hidden=128 → larger
    `PolicyValueNetAttn`) — the parked item, now the live path.
- JTR re-calibration stays owed regardless (two promotions banked).

## 2026-07-06 — Teacher-budget sweep: muzero flat vs gen-9 raw at EVERY budget (K=16/32/45×64 = −1.0/−1.6/−0.1) — search SATURATED, not starved

Extends the operator probe up the worlds axis, same setup (muzero
pb_c=1.25, sims=64, greedy teacher vs raw temp 0.05, gen-9 both,
fp 693.58, 320 pairs/arm):

| K×64 | exp/move | mean/game | t | sign (pairs) |
|:--|:--|:--|:--|:--|
| 16×64 | 1,024 | −1.0 | −0.765 p=0.4450 ns | 130W/140L ns |
| 32×64 | 2,048 | −1.6 | −1.346 p=0.1793 ns | 117W/143L ns |
| 45×64 | 2,880 | −0.1 | −0.112 p=0.9107 ns | 130W/133L ns |

- **Flat everywhere — no reopening with more worlds.** Even at 2,880
  exp/move (2.8× the collection budget, = JTR/POWERFUL's ~2,880),
  search cannot beat gen-9's raw. The teacher isn't starved for
  worlds; the determinization-PUCT improvement operator has SATURATED
  against this policy. (The depth axis is already ruled out by the
  three worlds-over-depth probes — more sims/world was
  worthless-to-harmful even when a +10 margin existed.)
- **The net fully captures its own search:** search(gen-9) ≈
  raw(gen-9) means the policy is NOT a lossy compression of better
  targets — at this budget there ARE no better targets. So a bigger
  net trained on the same corpus has nothing extra to learn (training
  was already clean: no overfit, train≈eval, losses plateaued).
  **Pure B-capacity scaling on the current recipe is therefore likely
  a WASH — the earlier "flat at K=45 → pull the B lever" inference was
  too quick.** Capacity only pays if paired with a stronger teacher.
- **The binding fact is EXTERNAL and points elsewhere:** at the same
  2,880 exp/move, POWERFUL beats our raw by ~7.5 (2026-07-05, and
  perfect-info raw too), yet our own net-guided search at that budget
  adds nothing over raw. Better play EXISTS and is findable by a
  classical searcher at equal compute — the self-play loop just can't
  generate it. This is a policy-iteration fixed point: search(π) ≈ π,
  so iteration stalls regardless of net size.

**DECISION (2026-07-06): the muzero self-distillation crank is at its
fixed point. The lever is NOT more search and NOT (alone) more net —
it is a stronger TEACHER than the loop can self-generate. POWERFUL is
the existence proof, NOT the teacher: NO JTR games in the training mix
(standing DECISION 2026-07-05, plan Step 4b) — JTR is the preserved
external benchmark, and distilling from it would convert "ties/beats
POWERFUL" into eval-on-train. (An earlier version of this entry listed
POWERFUL-distillation as candidate #1; corrected same day.) Live
candidates, strongest first:**
1. **Stronger IN-HOUSE search operator.** (a) Decouple the teacher
   from the net's priors — flattened/uniform priors, stronger root
   exploration — so search stops being self-confirming; cheap operator
   probe vs gen-9 raw first, a reopened margin = the gen-10 teacher.
   (b) Port algorithmic differences from JTR's classical PUCT into
   `jass_puct.py` — its code/ideas are fair game, its games are not.
2. **B capacity scaling — only PAIRED with a stronger teacher** (1);
   worthless on the already-absorbed current corpus.
3. Low/parked: pure B-scaling on the current corpus (expected wash;
   keep only as a cheap null-check if it's ~free to run).

## 2026-07-06 — Grounded-teacher probes: all four arms FAIL — flat-prior arms confounded by the argmax-visits readout; gen-9 JTR re-calibration now DECISION-CRITICAL

The four probe arms from the plan's NEXT block, on the standing
operator-probe harness (320 pairs on the 2×4, greedy teacher vs gen-9
raw τ=0.05, muzero K=16×64, pb_c=1.25; knob-free baseline = −1.0 ns;
new `jass_puct` knobs `prior_mix_uniform` / `rollout_value_weight`):

| arm | mean/game | t | sign (pairs) |
|:--|:--|:--|:--|
| dirichlet 0.25 | −0.8 | −0.675 p=0.4999 ns | 131W/139L ns |
| flat prior λ=1 | **−13.7** | −7.844 p=0.0000 *** | 101W/210L *** |
| rollout value w=1 | −1.9 | −1.560 p=0.1196 ns | 131W/136L ns |
| classical λ=1 w=1 | **−24.3** | −12.794 p=0.0000 *** | 67W/246L *** |

- **Root noise: nothing** (−0.8 ≈ the −1.0 baseline) — expected for a
  greedy probe; still the standard ingredient for any future
  collection run.
- **Rollout value w=1 is FLAT, not worse** (−1.9 ns, and only ~2×
  slower: 121 s vs 58 s for 5 rounds). A single uniform-random
  playout is worth as much as the trained value head as leaf guidance
  at this budget. So the value head is NOT the binding half of the
  self-confirmation — **the priors are**: with prior-guided
  exploration at 64 sims over ≤9 actions, the tree follows π
  regardless of the evaluator.
- **The flat-prior collapses are NOT clean evidence that classical
  search is weak — they are the readout failing.** Our aggregation is
  argmax of SUMMED ROOT VISITS; JTR's ideas.md predicted this exact
  failure ("argmax-visits with UCB/uniform tree policy: visits still
  ~uniform within tree → argmax picks a near-random move. Worse than
  baseline"). With uniform priors, 64 sims/world doesn't concentrate
  visits, so the summed-visit argmax reads noise. A clean in-house
  classical replica needs JTR's **Q-sum-over-determinizations
  readout** — a readout knob we haven't built (deliberately: Q-sum
  was rejected for the *net-prior* search because it neutralizes the
  tree policy; for a uniform-prior search the trade flips).
- **STALENESS FLAG on the existence proof.** "POWERFUL beats our raw
  ~+8.5 at 2,880 exp/move" was measured 2026-07-05 against
  gen-6b_es/gen-7-era raw. gen-9 raw is ≈ +16 internal points above
  gen-7 (+13.7 then +2.8). If that transfers through the JTR harness,
  POWERFUL's edge over gen-9 may already be ≈0 — **the target the
  grounded-teacher hunt is chasing may no longer exist.**

**DECISION (2026-07-06): the gen-9 JTR re-calibration (owed since
gen-8d_mz) is now DECISION-CRITICAL, not hygiene — it determines
whether external headroom exists at all. Export gen-9 → JTR real-PUCT
harness vs POWERFUL before building anything else:**
- **gen-9 still clearly below POWERFUL** → headroom is real: build
  the Q-sum readout knob, re-probe classical λ=1 w=1 at K=45×64
  (POWERFUL-parity, in-house) with it; a win there = the gen-10
  teacher.
- **gen-9 ≈ or above POWERFUL** → the equal-compute existence proof
  is gone; the fixed point means we've *caught up with classical
  search at this budget class*. The remaining levers are then
  above-parity teachers (budget + capacity scaling) or stronger
  external engines (JTR++, KUS) as calibration targets — reframe the
  plan, and B-capacity moves back up on its own merits.

## 2026-07-06 — qsum probe first read: classical 16×64 = −61 vs gen-9 raw — a READOUT SIGN BUG, not a measurement (score-sum inverts when Q<0); fixed to visit-weighted mean Q

First run of the qsum classical arms: 16×64 read **−61** mean/game vs
gen-9 raw — *worse* than the −24.3 visits readout — and the 45×64 arm
crashed the runtime before completing (likely the working set: 8
pairs/chip × K=45 plus the rollout state is far past the profiled
B×K≈64 knee).

−61 was the bug's signature, not the search's strength. The first
qsum implementation copied JTR's aggregation literally: score-sum
Σ_k N_k(a)·Q_k(a). But JTR sums per-leaf *scores*, which are
non-negative (0..157); our Q is a **±157 points differential**. With
mixed-sign Q the sum inverts preferences in every from-behind
position: a lightly-visited terrible action (−40×2 = −80) outscores a
heavily-visited slightly-losing one (−1×40 = −40). Against random
(all positions winning, Q>0 throughout) the readout looked great
(+30.4 with a random-init net vs +10.8 by visits) — against the
stronger gen-9 raw, the classical side is behind half the time and
systematically picked its least-examined move from every losing
position. Textbook eval-asymmetry trap: validated on wins only.

**Fix (same day): `readout="qsum"` = visit-weighted MEAN Q across the
K trees** — sign-safe, and slightly better even vs random (+32.2).
Negative-Q regression unit added (`_qsum_scores`); 14 puct tests
pass. The −61 number is VOID; both qsum arms rerun with the fix:
- 16×64 as before;
- 45×64 with **PAIRS_PER_CHIP=4, ROUNDS=10** (same 320 pairs, half
  the per-chip working set) to stay under the memory knee.

## 2026-07-07 — Fixed qsum re-read: classical 16×64 = −42.3 vs gen-9 raw — mean-Q is noise-seeking with single-rollout evals; and the arm itself has a pb_c CONFOUND (exploration tuned for net priors)

The sign-safe mean-Q readout on the classical λ=1/w=1 arm (16×64,
standing harness): **−42.3 mean/game vs gen-9 raw** — still worse
than the −24.3 visits readout. (45×64 arm not rerun.)

Two lessons, one live confound:

- **Mean-Q argmax = max of noisy means.** Leaf evals are single
  uniform-random playouts (±157-scale variance); at 64 sims over ~9
  actions an action carries ~7 visits/tree. Argmax over per-action
  means systematically selects the luckiest-sampled action, not the
  best one. Robust-child (max visits) is the standard UCT readout for
  exactly this reason.
- **JTR's Σ N·score is ≈ robust child, not max-Q.** With non-negative
  per-leaf scores the visit mass dominates the sum, so JTR's readout
  is robust-child with a quality tilt. The 07-06 "readout confound"
  story was half right (visits underread the classical arm vs random:
  +10.8 vs +32.2) but readout choice does NOT explain the gap to
  gen-9 — no readout rescues this arm as configured.
- **The live confound: pb_c=1.25 is net-prior-tuned.** PUCT explores
  by c·P(a)·√N/(1+n); flattening P from a concentrated net prior to
  1/9 mechanically shrinks the exploration term ~5–9×. The classical
  arm searched at an effective c ≈ 1.25/9 ≈ 0.14, while JTR's
  classical operator runs at ≈ 0.64 normalized (its c=100 on the raw
  point scale) — an over-greedy tree that commits to lucky rollout
  lines. Uniform-prior PUCT tying JTR's UCB baseline (their
  heuristic-prior experiment) was measured at *their* c, not ours/9.

**DECISION (2026-07-07): one last cheap classical probe — rescale
exploration for flat priors: pb_c ∈ {5.6, 11.2} (≈ 0.64×9 and 2×
that), 16×64, BOTH readouts, standing harness. This is the fair
version of the classical arm, not knob-fishing: the flat prior
rescaled c by 1/9, the sweep undoes it. If the best arm is still
double-digit negative vs gen-9 raw, the in-house classical-teacher
branch is DEAD at this budget class regardless of readout — then the
gen-9 JTR re-calibration (still owed, still decision-critical) is the
only open question, and the remaining in-house levers on a headroom
verdict are (a) determinization QUALITY (learned who-has-card
sampling trained on OUR self-play games — idea class from JTR's
ideas.md, zero JTR game data) and (b) porting JTR PUCT algorithmic
diffs.**

## 2026-07-07 — pb_c-rescaled classical probe: best arm −11.2 — the exploration fix recovers ~13–30 pts but the branch CLOSES per the pre-registered rule; the numbers now PREDICT the recal

Classical λ=1/w=1 at 16×64, standing harness vs gen-9 raw (means
only; arm order = the ARMS dict insertion order of the probe cell):

| pb_c | readout | mean/game |
|:--|:--|:--|
| 5.6 | visits | −18.7 |
| 5.6 | qsum (mean-Q) | −14.8 |
| 11.2 | visits | −18.2 |
| 11.2 | **qsum (mean-Q)** | **−11.2** |

- **The pb_c confound was real:** rescaling exploration recovered the
  arm from −24.3 (visits @1.25) / −42.3 (qsum @1.25) to −11.2. And
  with a properly exploring tree the readouts flip: qsum beats visits
  at both c values (the tree now earns its Q estimates; mean-Q stops
  being pure noise-seeking).
- **Branch CLOSED per the pre-registered bar** ("still double-digit
  negative → dead"): best arm −11.2. A teacher must clearly *beat*
  raw; +11 more from budget (16→45 worlds ≈ 3×) is not plausible —
  that 3× bought the net-prior operator ~0, and the classical slope,
  while real, is shallow. No more search-operator probes.
- **Convergent prediction for the recal:** our in-house classical at
  1,024 exp/move reads ≈ −11 vs gen-9; POWERFUL at 2,880 read +8.5 vs
  gen-7-era raw, ≈ −8 vs gen-9 if the +16 internal climb transfers.
  Two independent estimates now say **gen-9 ≈ classical-search parity
  at these budget classes.** If the recal confirms (POWERFUL ≈ 0 or
  negative vs gen-9), the gen-9 fixed point isn't a failure — it's
  *having caught up with classical search*, and Step 4's external
  goal vs JTR is effectively met without ever training on it.

**DECISION (2026-07-07): search-operator probing is DONE. The gen-9
JTR re-calibration is the sole next measurement and the branch point
for the whole plan.**

## 2026-07-07 — EXTERNAL MILESTONE: gen-8d_mz and gen-9 cross ABOVE POWERFUL in JTR's harness — Step-4 external goal MET without ever training on JTR

The re-calibration (JTR repo commit `caa33f9`; 250 pairs / 500 games
each, seed 42, SWEEP_64, real PUCT on every model side, classical
POWERFUL as the external opponent):

| matchup | per-game | p | verdict |
|:--|:--|:--|:--|
| gen-8d_mz vs gen-7b_es | −0.05 | 0.9549 | wash |
| gen-9 vs gen-8d_mz | +4.45 | 0.0007 | gen-9 stronger |
| gen-8d_mz vs POWERFUL | **+4.2** | 0.0028 | ABOVE |
| gen-9 vs POWERFUL | **+5.05** | 0.0003 | ABOVE |

- **The trendline completes: −22 (gen-3) → −9.5 (gen-5b) → ≈0
  (gen-6b_es/7b_es) → +4..+5 (gen-8d_mz/gen-9).** The lineage caught
  classical POWERFUL at gen-6/7 and has now pulled clearly ahead —
  with ZERO JTR games in training. The generality bet (Step 4b
  DECISION 2026-07-05) PAID.
- Our parity prediction was right in direction, conservative in
  level: we projected POWERFUL ≈ −8 vs gen-9 raw; PUCT-wrapped gen-9
  measures +5.05 over POWERFUL.
- **The internal crank's small gen-9 step is REAL strength:** the
  +2.8/+4.2 internal raw gate showed up externally as +4.45/game
  (p=0.0007) through a completely different harness. (gen-8d_mz vs
  gen-7b_es was an external wash — the PUCT-flattens-policy-gains
  pattern seen at gen-7-vs-6 — so the big internal +13.7 partially
  masked externally, yet the next step transferred cleanly.)
- **LIVE SIGNAL — JTR's operator may still be a teacher.** Every
  model side in this run was JTR-PUCT-wrapped. Combining +5.05 (PUCT
  gen-9 vs POWERFUL) with our ≈−5..−8 projection for RAW gen-9 vs
  POWERFUL implies JTR's determinized-MCTS wrapper still extracts +10
  over raw from the same gen-9 net — where OUR operator extracts −1
  (saturated, 2026-07-06). Cross-run and indirect, so it needs one
  direct measurement: **gen-9 real-PUCT vs gen-9 raw (`--pgx-raw`),
  250 pairs, in JTR's harness.**

**DECISION (2026-07-07): milestone recorded; Step 4's external goal
vs JTR is MET. One measurement decides the next branch — the JTR-side
gen-9 PUCT-vs-raw operator margin:**
- **~+10 confirmed** → JTR's search operator improves on a net ours
  can't improve; port its algorithmic diffs into `jass_puct.py`
  (code/ideas fair game, games not) as the gen-10 teacher candidate.
- **≈0** → gen-9 is saturated against the whole JTR-class operator
  family; reframe to stronger external targets (JTR++, KUS) and
  B-capacity on its own merits.

## 2026-07-07 — RESOLVED: JTR's operator margin on gen-9 = +1.5/game ns — the ≈0 branch. Search saturation confirmed across TWO independent operators; my +10 projection was wrong

The direct measurement (JTR experiment_log.md § "gen-9 raw vs PUCT",
250 pairs / 500 games, seed 42, SWEEP_64, same gen-9 net both sides):

| matchup | per-game | p | verdict |
|:--|:--|:--|:--|
| gen-9 PUCT vs gen-9 raw | **+1.5** | 0.1596 | ns — operator margin gone |
| gen-9 raw vs POWERFUL | +2.5 | 0.0760 | raw itself trends ABOVE POWERFUL |

- **The ≈0 branch, decisively. My "+10, JTR's operator is a teacher"
  projection (previous entry) was WRONG** — I anchored raw gen-9 at
  ≈−5..−8 vs POWERFUL by reusing the STALE gen-7 raw number (−8.5).
  In reality raw gen-9 has climbed ~+11 over two generations to +2.5
  ABOVE POWERFUL, so the PUCT-over-raw gap it left is +1.5, not +10.
  Lesson: never carry a raw-vs-external baseline across generations —
  the raw policy is exactly what's improving.
- **Saturation is now confirmed across TWO independent operators:**
  our muzero PUCT extracts −1.0 over gen-9 raw (2026-07-06); JTR's
  mature classical determinized MCTS extracts +1.5 ns (this run). Two
  unrelated determinized-search implementations agree gen-9 policy has
  absorbed ~all the improvement this operator *class* can add — a far
  stronger fixed-point result than either operator alone. The gen-7→9
  collapse is monotone: JTR's operator margin went +10.15*** (gen-7)
  → +1.5 ns (gen-9) as the net strengthened.
- **Porting JTR's PUCT diffs is OFF the table as a gen-10 teacher** —
  there is no operator margin left to harvest, even from a mature
  classical implementation. The escape from the fixed point is NOT a
  better search operator of this class from any source.

**DECISION (2026-07-07): the gen-10-teacher hunt via better search is
CLOSED (own ops, budget/depth sweeps, flat-prior classical, JTR-op
port — all saturated). Step 4's external goal is MET and then some
(raw gen-9 alone ~ties/beats POWERFUL). POWERFUL stays the yardstick
— it's a fixed, well-understood reference, and the margin over it is
the progress metric now; the goal is simply to beat it by more each
generation (no need for a harder external benchmark yet — the +2.5
edge has ample measurement room before saturating high). With the
operator saturated, the primary lever is:**
1. **B-capacity scaling on a fresh gen-10 corpus** — now on its own
   merits, not gated on a stronger teacher (that gate WAS the
   saturated operator; with no operator headroom, capacity is the
   direct lever for a stronger *raw* policy, which is what carries
   strength now). Collect gen-10 with the standing muzero recipe,
   train a wider net; success = a bigger margin over POWERFUL.

NOT a lever: learned determinization sampling (who-has-card model). It
is a documented NOT-retry negative (JTR CardsEstimator + AR variants:
no signal vs uniform, ≥1000 games; plan "Negative results"), and
JTR's own cheating probe RULED OUT determinization noise as the raw
gap's cause (perfect info moved −8.5 → only −7.5). Both facts kill it.

## 2026-07-08 — LEARNED TRUMP SELECTION beats the heuristic AND ~doubles the margin over POWERFUL — the net's first validated win outside card play; symmetric-blind-spot worry resolved FAVORABLY

The trump-head probe (my 2026-07-07 side-probe suggestion) landed
decisively. Source: JTR repo commit `398db0e` (`--pgx-trump`: pick
trump by argmax of the pgx policy head's trump logits, indices 36–42,
averaged over the trumpf-phase determinization count — forward passes,
no search, no value head; Schiebe/logit-42 masked once already
shifted). Card play held IDENTICAL on both teams — the same isolation
trick as `--pgx-raw` — so every delta is the trump switch alone.

**A/B, net trump vs rule trump, raw cards both sides (2000 pairs pooled):**

| seed | per-game | p |
|:--|:--|:--|
| 42 | +2.1 | 0.0091 |
| 43 | +1.3 | 0.1263 |
| **pooled 2000** | **≈+1.7** | **≈0.004** |

Net trump beats the hand-tuned heuristic by ≈+1.7/game. Both seeds
positive; the split is the known un-seeded determinization RNG (a true
≈+3.4/pair effect vs SE≈1.6/pair sits on the significance boundary, so
single 1000-pair seeds scatter — pooled it's decisive).

**vs POWERFUL — the gain MORE THAN ADDS onto the card-play edge:**

| matchup | per-game | p |
|:--|:--|:--|
| raw + **net** trump vs POWERFUL (500 pairs pooled) | **≈+6.8** | ≈1e-6 |
| raw + rule trump vs POWERFUL (baseline, seed 42) | +2.5 | 0.076 |
| PUCT + **net** trump vs POWERFUL (50 pairs, seed 42) | **+11.6** | 0.0044 |
| PUCT + rule trump vs POWERFUL (baseline) | +5.05 | 0.0003 |

Enabling learned trump roughly DOUBLES the margin over POWERFUL at
both card-play strengths (2.5→6.8 raw, 5.05→11.6 PUCT). And the trump
switch amplifies monotonically with card-play strength:

| card backdrop | trump switch (rule→net) |
|:--|:--|
| raw vs raw | +2.1/game |
| raw vs POWERFUL | +4.3/game |
| PUCT vs POWERFUL | +6.55/game |

Reading: a stronger card player converts the positional edge a good
trump creates instead of squandering it — POWERFUL also punishes a bad
trump harder — so getting trump right compounds.

**Why this matters for us:**
- **First validated win for the net OUTSIDE card play**, and it
  RESOLVES the symmetric-blind-spot worry (2026-07-07) in the good
  direction: trump was the least-validated head (longest-horizon
  value, invisible to self-play if symmetric), and it turns out the
  learned trump is genuinely *better* than a decades-tuned heuristic,
  not a hidden weakness. The +5.05 and +2.5 vs POWERFUL had used
  heuristic trump on both sides, so this is a net-new, previously
  untested component paying off.
- **End-to-end all-net play:** trump was the last rule-based component
  on an otherwise all-net team; the net now drives both phases.
  "Beat POWERFUL by more" — achieved, ~2×, by enabling a head we
  already had.
- Unlike rule/MCTS trump ("never shifts by itself"), the net DOES
  choose Schiebe when the hand warrants it.

**Caveats (from source):** un-seeded determinization RNG (single seeds
scatter, hence pooling); the PUCT-vs-POWERFUL row is a single 50-pair
seed (clears at p=0.0044) and its baseline is cross-run, so the "~2×"
multipliers are approximate though direction-confirmed on all three
backdrops; ~5% invalid-card→random fallback on the raw path (tiny,
mildly asymmetric net vs rule; PUCT path is ~1%).

**DECISION (2026-07-08): learned trump PROMOTED as the deployed trump
selector (JTR side). For the pgx research line it is a strength
milestone and a generality datapoint, not a redirection — gen-10 stays
B-capacity scaling. The self-play recipe ALREADY trains the trump head
(it's the same policy over DECLARE actions), so no collection change is
needed; the gen-10 gate should just keep measuring the growing margin
over POWERFUL with net trump now enabled on the pgx side.**

## 2026-07-10 — gen-10 CONTROL (10-ctrl, 128/2/4 on the fresh gen-10 corpus) gates +2.5/+2.4 vs gen-9 — sets the "corpus-refresh alone" bar

First arm of the gen-10 capacity sweep (design 2026-07-09, jass_sop.md
"gen-10 — capacity sweep"). 10-ctrl is the SAME gen-9 architecture
(`PolicyValueNetAttn`, hidden=128, num_layers=2, num_heads=4) retrained
on the fresh gen-9-generated corpus — it isolates the corpus-refresh
gain from any capacity change (arms 10a/10b add width/depth).

**Gate — raw-vs-raw vs gen-9** (fingerprints differ: new 10-ctrl
29781.71 ≠ src gen-9 29639.75, so a real A/B, not a self-gate):

| seed | per-game | t | p | win% |
|:--|:--|:--|:--|:--|
| 0 | +2.5 | +2.083 | 0.0381 * | 50.8% |
| 2 | +2.4 | +2.190 | 0.0293 * | 52.2% |

Both seeds positive, both p<0.05 → 10-ctrl beats gen-9. ⚠ The gate
print MISLABELED both sides "gen-9 raw" (challenger label not bumped to
10-ctrl) — the SOP's documented GEN-for-CHAMP mixup; numbers valid
(fps differ), fix the label string for the a/b arms.

**Training health:** full run, flat eval (~0.603 total), NO U-curve
through ~22k; holdout v 0.0636 (at gen-9's 0.0655 floor), train≈eval
(no overfit). Policy CE dropped to ~0.54 (gen-9: 0.604) — the fresh
corpus's targets are sharper/more learnable, a good sign the corpus
carries real signal.

**Reading:** the corpus-refresh gain alone is **~+2.5**, in the same
band as the gen-9 same-recipe step (+2.8/+4.2) — self-play iteration is
still yielding ~+2.5/gen at FIXED capacity. This is now the BAR the
capacity arms must clear: 10a (256/2/8, width) and 10b (128/4/4, depth)
have to beat ~+2.5 for capacity to be earning its keep beyond the
fresher corpus. If an arm only matches it, capacity added nothing.

## 2026-07-10 — gen-10a (256/2/8, WIDTH arm): value-head U-curve returns; ES net does NOT clear the corpus-refresh bar — width is data-starved at 128k

Second arm of the gen-10 sweep: `PolicyValueNetAttn(hidden=256,
num_heads=8)` (~4× params), same shared corpus as 10-ctrl.

**Training — the value head overfit (U-curve back).** Policy CE stayed
flat/healthy (~0.538 the whole run, no overfit), but eval VALUE loss
bottomed ~0.063 at epoch 8–10k then climbed to 0.080 by 22k while train
kept dropping (0.60→0.56) — classic memorization, in the value head
only. The SAME corpus left 128-wide (10-ctrl) flat through 22k, so the
256 width is what re-opened the U-curve — as the sweep design warned.
Took the `_es` snapshot at **10k** (eval-total basin 0.6013–0.602;
policy CE flat after 10k, so a later snapshot buys no raw-policy
strength and only worsens the value head).

**Gate — 10a_es raw vs gen-9** (fp new 80430.02 ≠ src gen-9 29639.75;
gen-9 fp matches the ctrl baseline — clean A/B, labels fixed):

| seed | per-game | t | p |
|:--|:--|:--|:--|
| A | +0.7 | +0.513 | 0.6084 ns |
| B | +2.6 | +2.089 | 0.0375 * |

Pooled ≈ +1.65, p≈0.07 — one seed a flat wash.

**Verdict: WIDTH FAILS at 128k.** 10-ctrl set +2.5/+2.4 (both seeds
***) vs the same gen-9; 10a_es lands +0.7/+2.6 (one ns) — BELOW the
control. Doubling width produced a net no better than the plain 128-wide
retrain on the same corpus. Reading: the 256-wide net is DATA-STARVED
on 128k — extra capacity went into value memorization, ES caps it, and
the capped policy ≈ the control.

**Direct head-to-head CONFIRMS (10a_es raw vs 10-ctrl raw; fp new
80430.02 ≠ src ctrl 29781.71):** seed A −0.2 (p=0.8432 ns, 300W/300L
dead even), seed B −0.9 (p=0.4342 ns), pooled ≈ −0.55 ns. Width is a
dead wash vs the control, a hair negative — doubling width bought
NOTHING at 128k.

**The deeper tell: the POLICY head never overfit at either width** —
eval policy CE plateaued at ~0.538 for BOTH ctrl (128) and 10a (256);
only the value head overfit on 10a. So width didn't fail merely from
value memorization — the policy head extracted the SAME signal at both
widths. The binding constraint is the policy-target INFORMATION in the
corpus, not model capacity: 128-wide already saturates what gen-9+muzero
puts in each position. This is operator-saturation surfacing on the
training side — more params can't extract signal that isn't in the
targets.

**Fork (pre-committed):** run 10b (depth) to finish the sweep, but if it
also washes vs ctrl, promote 10-ctrl as gen-10 (+2.5 corpus-refresh win)
and pivot the next lever from CAPACITY to DATA — collect a bigger corpus
(more games = more value coverage + state diversity). The policy-CE
plateau says we're target-information-limited; open question whether more
games move the policy at all, or gen-9 self-play has plateaued and the
next gain needs a different target source (JTR games stay off-limits).

## 2026-07-10 — gen-10b (128/4/4, DEPTH arm): trains CLEAN (no U-curve, best holdout CE) but ties the control in the arena — capacity sweep CLOSED, gen-10 = 10-ctrl

Third/final arm of the gen-10 sweep: `PolicyValueNetAttn(num_layers=4)`
(~2× params vs gen-9, width unchanged), same shared corpus.

**Training — no overfit, unlike width.** Eval value loss stayed flat
~0.062–0.064 through 22k (no U-curve), policy CE drifted to 0.5358 —
the BEST holdout of the three (ctrl 0.5396, 10a 0.538). Gated the final
net, no ES needed. Depth was the safe capacity add; width (10a) was not.

**Gate vs gen-9** (fp new 48513.89 ≠ src gen-9 29639.75):

| seed | per-game | p |
|:--|:--|:--|
| A | +2.7 | 0.0324 * |
| B | +4.1 | 0.0015 ** |

Both significant, pooled ≈ +3.4 — clears the bar, nominally above ctrl.

**Gate vs 10-ctrl — direct depth test** (fp new 48513.89 ≠ src ctrl
29781.71): seed A +0.0 (p=0.9776 ns, 102/108), seed B −0.0 (p=0.9742 ns,
119/110). **Dead wash.** Depth ties the control head-to-head.

**SWEEP VERDICT (2026-07-10): capacity is NOT the lever at 128k.** All
three arms — ctrl 128/2, 10a 256/2, 10b 128/4 — are arena-equal:

| arm | vs gen-9 | vs ctrl (direct) | train |
|:--|:--|:--|:--|
| 10-ctrl (128/2) | +2.5/+2.4 ** | — (bar) | clean |
| 10a (256/2, width) | +0.7/+2.6 | −0.55 ns (wash) | value U-curve, ES@10k |
| 10b (128/4, depth) | +2.7/+4.1 ** | +0.0 ns (wash) | clean, best CE |

The single most telling fact: **10b got the best holdout policy CE yet
still ties ctrl in the arena** — a better fit to the targets converts to
ZERO extra strength. We're at the CORPUS's information ceiling, not a
capacity ceiling; the ~+2.5–3.4 over gen-9 is entirely the corpus
refresh (matches gen-9's own +2.8/+4.2 same-recipe step). This is
operator-saturation (search(π)≈π) surfacing on the training side: more
params can't extract signal the saturated teacher didn't put in.

**gen-10 = 10-ctrl (128/2/4), PROMOTED** — all three arena-equal, so
take the smallest (no reason to carry 2× params for zero gain). Footnote:
depth (128/4) is the capacity that did NOT overfit at 128k, so it's the
arch to revisit IF a bigger corpus is collected — not width.

**Next lever = DATA/coverage, not capacity (operator + capacity both
saturated).** Open question, to test: does a bigger corpus (more games =
more state coverage + value outcomes) move the raw POLICY at all, or only
the value head? If per-position policy info is truly capped by the
saturated operator, more games mainly buys coverage/value — a diminishing
bet. If even more data washes on policy, gen-9 self-play has plateaued and
the next real gain needs a NEW target source (operator saturated, JTR
games off-limits) — the project's hard wall.

## 2026-07-11 — gen10c (128k width re-feed): capacity is DEAD — well-fed 256/2 STILL washes vs 128/2 despite the best holdout CE of the sweep; gen-9 self-play has PLATEAUED

Resolves the sweep's one open confound (width's 64k wash was
data-starvation). Extended the existing 32×2048 gen-9 corpus to
**64×2048 (~128k)** by collecting 32 more batches from the SAME
generator (gen-9) — data is the only variable vs the sweep. Still the
gen-10 family (gen-9-generated); NOT gen-11. Two arms trained on 128k.

**Arm 1 — 10c_ctrl (128/2 @128k), the coverage control.** Trained
clean (no U-curve). Gates:
- vs gen-9: +1.7 ns (p=0.1402) / +2.8 * (p=0.0296) — ~the usual +2.5
  corpus step.
- vs 10-ctrl (128/2 @**64k**): +0.1 / −1.1, both ns → **more data does
  NOTHING for the small net.** ctrl wasn't starved; coverage alone can't
  move it. (holdout v floors ~0.0686 on the 128k batch — don't compare
  across corpus sizes.)

**Arm 2 — 10c (256/2 @128k), THE test.** Value U-curve MILDER than at
64k but not gone: U-min pushed 8k→~13k, floor 0.0665, overfit climb only
+0.005 by 24k (vs 10a's +0.017 at 64k) — starvation partially relieved,
256-wide still somewhat data-hungry. Took `_es` at ~13k. **Best holdout
policy CE of the entire sweep: ~0.5352** (128/2 @128k 0.5396; width @64k
0.538) — the first arm where width fits the policy targets measurably
better than the small net. Gates:
- vs gen-9: +2.2 ns (p=0.1103) / +2.8 * (p=0.0155).
- **vs 10c_ctrl (128/2 @128k) — DECISIVE:** +0.7 (p=0.5228) / −0.0
  (p=0.9914, 300/300), pooled ≈ +0.35 **dead wash.**

**VERDICT: capacity is dead; the plateau is real.** Width well-fed at
128k, with the best CE in the sweep, converts to ZERO arena strength —
the THIRD straight "better holdout CE fails to convert" (10b depth; 10c
width-fed; + coverage washed). Both live escape hypotheses — "width was
just starved" and "more data helps" — are killed by this round. We are
at the CORPUS's information ceiling: gen-9+muzero puts a fixed amount of
playing strength into each target, 128/2 already extracts all of it, and
neither more params nor more games adds anything. gen-9 self-play has
PLATEAUED; the ~+2.5/gen corpus refresh just reshuffles the same info.

Honest caveat: width-vs-narrow drifted −0.55 (@64k) → +0.35 (@128k) as
data doubled — a faint nudge for width, but both ns, inside the noise;
after three CE-doesn't-convert results, not worth a 256k collect to
chase. **No champion change: gen-10 = 10-ctrl** (all three 128k nets
arena-equal to it). Next gain requires a NEW TARGET SOURCE, not more
self-play scale — see plan NEXT.

## 2026-07-12 — Cheating-raw RE-RUN at gen-9, head-to-head: perfect information is worth ZERO to the raw policy — 101/250 paired deals end in identical scores

Re-run of the 2026-07-05 cheating diagnostic (that one was gen-7,
common-opponent design, and bounded the info value at ~1 pt). Motivated
by the target-source pivot: before investing in better card inference
(belief-weighted determinization etc.), re-measure what perfect hand
knowledge is even worth to the current net. Sharper design this time:
**same net both sides, head-to-head, paired stats** — no differencing
of two noisy vs-POWERFUL margins.

gen-9 cheating-raw (single forward pass on the TRUE hands) vs gen-9
fair-raw (policy averaged over the round's sampled determinizations),
JTR in-process arena (`ApplicationArena`, JTR commit 01ce4f1), both
sides `SWEEP_64` / `--pgx-raw`, `--cheating1`, 250 pairs / 500 games,
seed 42, rule-based trump both sides (symmetric; the trump phase is
NOT probed here):

**mean diff −1.0/pair (−0.5/game), t=−0.499, p=0.6182; sign
75W–74L–101T.** Dead wash — and 40% of paired deals produced EXACTLY
the same score, i.e. the true-hands policy mostly plays the same moves
as the 40-world average.

Reading: determinization-averaging at the input is completely free at
gen-9 (stronger than gen-7's "~1 pt"). Either hidden-hand information
is intrinsically near-worthless to a one-shot policy, or the net is
not USING the hands input at all. Those have very different
implications — discriminated by the sensitivity probe (next entry).
Companion arm still running: cheating-PUCT vs fair-PUCT at SWEEP_64
(never run before — bounds what perfect information is worth to the
SEARCH, which is what belief-weighted determinization would improve).

## 2026-07-12 — Hidden-hand sensitivity probe: the POLICY head is hands-BLIND (KL 0.003), the VALUE head is hands-AWARE (±28 pts) — and the aggregated-visits target TRAINS the blindness in

Probe script: `scripts/jass_hidden_hand_probe.py` (in-repo, exact
experiment). Method: 128 on-policy games with gen-9 (fingerprint
29639.75, sampled τ=1); at each of 3,407 card-play decisions with >1
legal move, hold all public info fixed and resample the hidden hands 8
times via `sample_determinization` (void-aware, the searchers' own
sampler); compare net outputs true-world vs resampled-worlds.

| head | sensitivity to WHERE the hidden cards sit |
|:--|:--|
| policy | KL(true‖world) mean **0.0030** (median 0.0013) vs entropy 0.80; argmax flips **4.1%** |
| value | std across worlds **28.5 pts** mean, p90 48.9 (vs mean \|v\| 60.9 pts) |

Per-trick: value std runs 41.7 pts (trick 0) → 11.8 (trick 7) —
uncertainty resolving as cards fall; policy KL is flat ~0.003
everywhere. **The policy head almost completely ignores card-matrix
columns 1–3 (partner/left/right holds); the value head conditions on
them heavily.** This fully explains the cheating-raw wash: cheating
feeds the policy head information it never uses.

**The blindness is STRUCTURAL, not a learning failure.** Collection
pairs true-state features (hands columns included) with a policy
target that is the visit sum across K=16 determinized trees — a target
that by construction depends only on the INFORMATION SET, not on the
true hidden hands. Given that label, the CE-optimal policy is exactly
a hands-blind one; training actively teaches the policy head that the
hands columns are noise. The value head's label (actual game outcome)
DOES depend on the true hands — so it learned to use them. One
pipeline, two heads, opposite lessons.

**Why this reframes the plateau:** the 16 per-world searches produce
16 DIFFERENT visit distributions (they must — the value head's 28-pt
world-sensitivity says the Q landscape differs per world: finesses
work in one world, fail in another). Summing them throws that variance
away — the aggregation step discards per-position information the
pipeline already computes, and it is precisely the hands-conditional
dimension the input features have room to encode. Capacity washed
because nothing hands-conditional was left in the labels for extra
params to learn. It also feeds the operator fixed point: each
determinized tree is guided by a hands-blind prior, so per-world
search re-derives hand-specific tactics from scratch through the value
head every time — the self-confirmation the grounded-teacher probes
kept hitting.

**NEW CANDIDATE — hands-conditional policy targets** (per-world visit
distributions paired with per-world features, instead of the aggregate
paired with true-state features). Same collection compute; the
per-world visits exist right before the sum. Deployment is unaffected:
JTR fair-raw already averages the policy over sampled worlds, which
becomes the CORRECT inference-time marginalization of a
hands-conditional policy (exactly how the value head is already used).
Gate design (pre-registered, decision order):

0. **Arm B informs (in flight):** cheating-PUCT vs fair-PUCT bounds
   the value-head-only oracle gain at SWEEP_64.
1. **Teacher-signal pre-probe (no training, cheap — the kill switch):**
   at the standing collection config (muzero K=16×64), measure
   across-world disagreement of the 16 root visit distributions per
   move (pairwise KL + argmax disagreement). If per-world teachers
   agree everywhere, the hands-conditional label adds no information
   at this budget → arm dies before any collection. Prediction from
   the value head's 28-pt sensitivity: they disagree substantially.
2. **Collection change:** per move, emit per-world policy rows
   ((cm_k, hd_k) of the determinized root, π_k = that tree's root
   visits) alongside the existing true-state row. Value label y stays
   on the true-state row ONLY (world-k features + true-world outcome
   is a mismatched value pair) → per-row value-loss mask in
   train_step, or first arm = per-world rows carry policy loss only.
   16 policy rows/move at the same game count; subsample worlds
   (e.g. 4) if corpus memory bites. Standard 32×2048 collect from the
   gen-10 generator.
3. **Train:** gen-11hc, standing recipe (128/2/4, ES armed).
   **Control: gen-11ctrl on the SAME collect's standard rows**
   (aggregated target) — same generator, so target construction is
   the only delta; ctrl also banks the routine ~+2.5 corpus refresh.
4. **Gates, in order:**
   a. **Mechanism check:** re-run `jass_hidden_hand_probe.py` on the
      student — policy KL must move well off 0.003. Still blind →
      the targets didn't bind; debug before any arena time.
   b. **Raw gate:** gen-11hc vs gen-11ctrl head-to-head, two seeds,
      DECISIVE (must clear the ctrl, not just gen-10 — capacity-sweep
      discipline); vs gen-10 for the record.
   c. **Operator re-probe:** muzero K=16×64 vs gen-11hc raw. The
      hypothesis predicts the margin RE-OPENS (hands-conditional
      priors make each world's search start from world-appropriate
      play). A re-opened operator restarts the crank — the big prize
      even if (b) is thin.
   d. **External:** JTR raw + PUCT margins as usual (fair mode).
   Fork: (a) fails → fix pairing. (a) passes but (b)+(c) wash →
   hands-conditional info doesn't convert; direction dead, exact
   endgame targets next. (c) re-opens → iterate the crank.

## 2026-07-12 — Cheating-PUCT vs fair-PUCT: +12.6/game*** — hidden-hand info is worth ZERO to the raw policy but +12.6 to the SEARCH; the imperfect-info pool is WIDE OPEN and the belief-model NOT-retry is re-scoped

The Arm-B companion to today's cheating-raw re-run — the arm never
run before (2026-07-05 probed cheating-RAW only). gen-9 cheating-PUCT
(all 45 "determinizations" = the true deal) vs gen-9 fair-PUCT
(uniform void-aware worlds), JTR in-process arena (commit 01ce4f1),
both sides `SWEEP_64` real PUCT (`--pgx-policy`), `--cheating1`,
250 pairs / 500 games, seed 42, rule-based trump both sides:

**+25.2/pair (+12.6/game), sd 35.2, t=11.302, p<0.0001; sign
161W–38L–51T (cheat scored 117.4% of fair's points).**

**The oracle dissociation, complete (same net, same day, same
harness):**

| path | Δ(perfect info − fair) | verdict |
|:--|:--|:--|
| raw (one forward pass) | −0.5/game, p=0.62 | worth ZERO |
| PUCT @ SWEEP_64 | **+12.6/game, p<0.0001** | worth more than the whole POWERFUL margin (+5.05) |

Mechanism (matches the sensitivity probe): hidden-hand information is
cashed through PLANNING — the tree simulates the opponents' actual
holdings and the hands-AWARE value head (±28 pts world-sensitivity)
evaluates true continuations. The hands-blind policy head cannot use
the same information in one shot, and was trained not to.

**Honest decomposition — +12.6 is the ORACLE bound, not the belief
payoff.** The gap bundles: (a) world QUALITY (searching the right
world); (b) aggregation losses of fair play (visits summed across 45
worlds — strategy-fusion-type averaging); (c) compute concentration
(45 trees on ONE world ≈ a deeper effective search on the relevant
world). A realistic belief model harvests only part of (a). The bound
being ~2.5× the entire POWERFUL margin still makes this the largest
measured headroom anywhere in the project right now.

**Saturation narrative REVISED:** the determinized-PUCT operator is
saturated on INFO-SET-DERIVABLE information (teacher-budget sweep
flat, JTR margin +1.5 ns, capacity dead — all still true). But a
+12.6/game pool sits in hidden-information quality that neither more
sims nor more params can reach, because fair search spreads its budget
over mostly-wrong worlds. Everything washes EXCEPT knowing where the
cards are.

**The NOT-retry is RE-SCOPED, not violated.** The 2026-07-07 "both
facts kill it" verdict on learned determinization sampling rested on:
(1) "JTR's own cheating probe ruled out determinization noise" — that
probe was cheating-RAW; it binds the raw-input path only, and today's
Arm B directly overturns it for the search path. (2) The thesis-era
CardsEstimator/AR no-signal (≥1000 games) — measured on a
rollout-leaf searcher with NO hands-aware evaluator; the converter
that turns world quality into points (a value head with ±28-pt hand
sensitivity) did not exist then. Plan rule: "don't retry without new
evidence" — +12.6*** is the new evidence. Scope going forward: dead
for raw-path marginalization; OPEN for search-world quality with the
current evaluator.

**Cheap next probe (pre-registered): oracle-mixture dose-response.**
Replace q of the 45 fair worlds with the true deal, q ∈ {0, 25, 50,
100}% (q=0 fair, q=100 cheating ≈ today's arms), same config. The
curve says how belief-model quality maps to points — i.e. how good an
inference model must be before it pays, and how much of +12.6 is world
quality (a) vs concentration (c). Runs in the same harness with a
small `ApplicationArena` flag.

Levers now on the table are COMPLEMENTARY, all feeding the same
mechanism: hands-conditional targets fix the prior INSIDE each world
(plan NEXT #1); belief-weighted determinization fixes WHICH worlds are
searched; exact endgame targets strengthen the value head that cashes
both. gen-10's JTR export + external re-calibration remain pending.

## 2026-07-12 — Teacher-signal pre-probe: across-world JSD 0.24 vs a 0.0000 same-world floor — the KILL SWITCH DOES NOT FIRE; the aggregation step discards ~35% of the target's entropy

Gate step 1 of hands-conditional policy targets (pre-registered
above), run same-day. Script: `scripts/jass_teacher_signal_probe.py`;
enabled by two new diagnostic knobs on `puct_search` (`cheat=True` —
all K worlds = the true state, the internal analogue of JTR's
`--cheating`; `return_visits=True` — per-tree root visits before
aggregation). 24 on-policy games played BY the standing teacher
itself (gen-9, muzero K=16×64, pb_c=1.25, dirichlet 0, τ=1 visit
sampling — collection-faithful), 645 card-play decisions with >1
legal move:

| metric | value |
|:--|:--|
| across-world JSD of the 16 root visit distributions | **0.241** mean (median 0.259, p90 0.438) |
| same-world floor (cheat=True, search-noise control) | **0.0000** — teacher is deterministic; ALL disagreement is world-driven |
| JSD / H(aggregate): share of the target's entropy that is across-world variation | **35%** |
| worlds whose visit argmax ≠ the aggregate argmax | **25.4%** |

Flat ~0.23–0.29 JSD across tricks 0–6 (dips to 0.16 at trick 7 as
worlds collapse). **In a quarter of sampled worlds, the per-world
search's top move is NOT the move the aggregated target teaches** —
and the marginal target actively averages those world-specific
recommendations away.

The full quantitative chain for hands-conditional targets now reads:
teacher knows hands-conditional play at JSD ≈ 0.24/position →
aggregation discards it → trained net retains KL ≈ 0.003 across
worlds (hidden-hand probe, ~80× less) → cheating-raw worth zero,
while the oracle pool sits at +12.6/game on the search path.
**Gate step 1 PASSED; step 2 (per-world collection rows) is GO
whenever we choose to spend it.** Runtime note: 24 games ≈ 5 min on
the local CPU — the probe re-runs cheaply on any future student
(mechanism check = its policy KL moving toward this JSD, not to it —
some across-world variance is value-driven, not prior-learnable).
