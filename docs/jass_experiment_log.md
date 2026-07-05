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
