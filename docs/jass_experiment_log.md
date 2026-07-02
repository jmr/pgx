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
