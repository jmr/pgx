# Jass Value Network: State Representation Options

## Background

We want to train V(state) → expected score to replace random rollouts in
determinized MCTS. The core design question is: what does V take as input?

### The imperfect information problem

Jass is an imperfect information game. Each player sees only their own hand.
A flat observation of 120 bits (my hand + collected cards + trump + trick)
is **not a sufficient Markov state**: two identical observations can arise from
very different game histories with very different expected values.

Example: ♥A has been played; ♥K has not. The value of ♥Q in my hand is much
higher if my partner holds ♥K (team controls the suit) than if an opponent
does (they can overplay any time hearts are led). A partial observation can't
distinguish these cases.

### How determinization resolves this

In our MCTS we call V at leaf nodes of a **determinized world**: a complete
assignment of all 36 cards to players, consistent with known constraints. Within
a determinization every hand is known, so ♥K's location is unambiguous. The
per-card relational value (♥Q given partner holds ♥K) is fully computable.

**Within a determinization, the full game state is Markov-complete.** History
only matters for *belief updating* under hidden information — which the
determinization has already resolved.

This means V should be trained on **full game states** (all hands known), not
on the partial observation a player actually sees at runtime.

Training data: run self-play games; at each step record `(full_state, terminal
reward)`. The full state includes all 4 hands, so it's the same kind of state
V will see during MCTS.

### Markov-completeness caveat

Markov-completeness within a determinization holds given fixed opponent policies.
V learns expected value against the **self-play policy distribution**, not the
game-theoretic optimum. This is fine for our use case, but when we iterate
self-play the policy drifts, so the value target shifts too. Standard practice
is to periodically retrain V on data from the current policy.

---

## Option A — Flat partial observation (current, 120 bits)

```
[0:36]    my hand
[36:72]   cards_collected.any(axis=0)  — any player's collected cards
[72:108]  cards currently in the trick
[108:112] who led (one-hot)
[112:119] trump mode (7 bits)
[119]     my declare turn
```

**Pros**
- Already implemented; used for `pgx.State.observation`
- Small input; fast network forward pass

**Cons**
- **Fatal for V: no score attribution.** `cards_collected.any(axis=0)` records
  which cards have been taken but not by which team. V literally cannot tell
  who is winning — a team leading 130–27 looks identical to one trailing 27–130.
  This is the primary reason Option A is unusable for a value function, separate
  from any Markov argument.
- Not Markov: even if score were fixed, doesn't encode who holds remaining cards,
  so can't distinguish "partner has ♥K" from "opponent has ♥K"
- Wrong setting for our use case: we call V on determinized states, not partial obs

---

## Option B — Flat determinized state (264 bits)

Extend the observation to include all 4 hands and full score attribution:

```
[0:36]    my hand (player 0's perspective)
[36:72]   partner's hand
[72:108]  left opponent's hand
[108:144] right opponent's hand
[144:180] my team's collected cards   (player | partner)
[180:216] opponents' collected cards
[216:252] cards currently in the trick (36 bits)
[252:256] who led (one-hot)
[256:262] trump mode (6 bits: ♦ ♥ ♠ ♣ Obenabe Undeufe; omit "not-declared"
          since V is called during card play after trump is set — add it back
          if V is also used for trump selection evaluation)
[262]     forehand_passed
[263:272] trick_num (one-hot over 0–8, 9 bits; derivable from cards_collected
          but explicit is friendlier to the network. Use 4 bits if binary-encoded.)
```

Total: 272 bits (one-hot trick_num) or 267 bits (binary). Fully Markov within a
determinization.

**Pros**
- Complete information: V can learn inter-hand relationships (♥Q + partner ♥K)
- Simple architecture: MLP or small dense net
- Easy to implement; straightforward training pipeline

**Cons**
- Flat layout obscures structure: the network must discover that card c in
  hand[partner] interacts with card c in hand[self], buried 36 bits apart
- No natural inductive bias for suit/rank locality or trump vs. plain suits
- May need a large network to learn cross-hand patterns from scratch

---

## Option C — Card × player matrix (36 × 12, conv/attention-friendly)

Represent the state as a matrix with one row per card, columns encoding each
card's status across players and the current trick:

```
For each of 36 cards:
  col  0: I hold it
  col  1: partner holds it
  col  2: left opponent holds it
  col  3: right opponent holds it
  col  4: my team has collected it
  col  5: opponent team has collected it
  col  6: it is currently in the trick
  col  7: I played it in the trick (one-hot, 4 bits total for cols 7–10;
          ordered leader-first so the column index equals play position
          within the current trick)
  col  8: partner played it in the trick
  col  9: left opponent played it in the trick
  col 10: right opponent played it in the trick
  col 11: it is trump (derived from trump mode + card suit; all-zero in
          Obenabe/Undeufe)
```

Shape: 36 × 12. Apply a small CNN or self-attention over the 36 card rows.

The column layout is **redundant by design**: col 6 is the OR of cols 7–10,
and a card's location is mutually exclusive across cols 0–6. Splitting
mutually-exclusive states across columns lets a single conv/attention layer
attend to each fact directly instead of having to disentangle a packed encoding.

**Play position within the current trick** is encoded by ordering the
player columns leader-first (cols 7–10 = leader, leader+1, leader+2, leader+3).
This way the network reads "who played in position k" off column 7+k without
having to combine `who_led` from the header with a separate identity column.

Header vector (concatenated after pooling):
```
  trump mode    (6 bits: ♦ ♥ ♠ ♣ Obenabe Undeufe; keep "not-declared" bit
                 if V is called for trump selection evaluation)
  forehand_passed (1 bit)
  trick_num       (scalar or one-hot 0–8)
  who_led         (one-hot, 4 bits)
```

**Pros**
- **The load-bearing advantage**: all information about card c lives in row c.
  A conv/attention layer immediately sees that partner holds ♥K (col 1, row ♥K)
  and I hold ♥Q (col 0, row ♥Q) without having to bridge a 36-position gap in
  a flat vector. Cross-hand relationships like ♥Q + partner ♥K are a single
  layer away.
- Suit/rank locality: cards for each suit are contiguous rows; a small conv
  kernel (size 9 = one suit) learns suit-level patterns naturally
- Efficient: attention over 36 tokens is tiny; a 2–3 layer network should suffice
- JassTheRipper's ScoreEstimator uses a similar structured matrix (73 × 18)
- Suit-permutation augmentation: in Obenabe and Undeufe all 4! = 24 suit
  permutations are valid (no suit is special). In trump modes, trump breaks
  the symmetry — you can freely permute the **three non-trump suits** (3! = 6×)
  but cannot swap trump with a non-trump suit. Total: 24× for non-trump modes,
  6× for each of the four trump modes.

**Cons**
- Slightly more complex architecture than a flat MLP
- Trump bit (col 11) changes semantics per mode: in trump modes it marks one
  suit; in Obenabe/Undeufe all or none are "trump" — handle this carefully
- Header vector needed for scalar context that doesn't fit the card matrix
- Rank-in-mode is implicit: in Undeufe the 6 is highest, in Obenabe the A is
  highest, in trump the J/9 jump above the A. The matrix encodes card identity
  via row index but doesn't directly tell the network its rank in the current
  mode — the net has to learn the lookup. Probably fine given 24× suit-permutation
  augmentation; add an explicit "rank-in-mode" column later if learning is slow.

**Recommended architecture**
- 36 × 12 card matrix → 2–3 conv layers (kernel size 1–9) or a single
  multi-head self-attention layer over 36 tokens
- If using attention, add a positional embedding (learned or fixed) per row
  so the network knows "this row is ♥K," not just some card. Conv gets this
  for free via the row index.
- Concatenate flattened output with header vector → dense → **raw scalar output
  (no tanh/sigmoid/clipping)**, MSE loss
- Training target: team score differential in [−157, 157] (not raw points in
  [0, 157]). Differential is what MCTS actually compares across actions, so it's
  the cleanest target; the network doesn't have to learn that 80 points is
  "good" or "bad" in isolation.

---

## Option D — Full play sequence (recurrent / Transformer)

Feed the complete sequence of `(player, card)` pairs played so far, plus the
current state, to an LSTM or Transformer.

```
Tokens: [INFO_TOKEN, card_1_played_by_p, card_2_played_by_p, ..., CURRENT_HAND]
```

**Pros**
- Captures full behavioral history: "player 2 played ♠7 rather than ♠J in
  trick 3 — they might be saving trump"
- Naturally handles variable-length game prefixes
- Most expressive; could learn patterns a flat state misses

**Cons**
- Overkill for V used inside a determinization: the current full state is already
  Markov-complete — the play sequence adds no information beyond who currently
  holds what
- Much harder to train and integrate with JAX/pgx's `vmap` self-play loop
- Sequence history matters if V is trained on partial observations (imperfect-info
  setting), but we've argued against that above

**When this makes sense**: if we later train a card-estimator network (predicting
what cards other players likely hold, to improve determinization quality), the
play sequence is the right input. V is a simpler problem.

---

## Recommendation

**Use Option C directly. Don't build B first.**

The data pipeline — self-play → (full_state, terminal_reward) records → MSE
training → plug into `_random_rollout` — is identical for B and C. The only
difference is a reshape and swapping a flat MLP for a 2-layer conv/attention
block. Building B first validates almost nothing that C wouldn't also validate,
and the risk is that B ships, kinda works, and the upgrade gets deprioritized.

**Code structure**

Separate `state → features` from `features → value` so the feature builder is
swappable. Keep a flat-MLP variant (Option B layout) available as a one-flag
sanity baseline — not a separate milestone, just a debugging tool when C's loss
looks weird.

**Success criterion**

First milestone: **V beats random rollout in MCTS at K=8, N=8**, not "V's MSE
is low." Tie the criterion to the downstream task from day one.

**Training setup**
- Generate self-play games with current K=8, N=8 MCTS
- At every game step, record the full determinized state (all 4 hands known
  during self-play) + terminal score differential for the current player's team
- Train V to minimize MSE against terminal differential
- Plug V into `jass_mcts.py`: call V once at the leaf instead of simulating
  38 random steps

**Trump selection with V**

During trump selection, for each legal action under the current mask
(at most 7: ♦ ♥ ♠ ♣ Obenabe Undeufe Schiebe; Schiebe is only legal for forehand
when `forehand_passed` is False, so partner sees at most 6), hypothetically apply
the action to get `state_after_trump`, construct the card matrix from that state
(keeping the "not-declared" trump bit in the header), call V, pick the argmax
over `legal_actions`. No separate trump heuristic needed.
