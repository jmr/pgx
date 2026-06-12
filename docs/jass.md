# Jass

## Description

Jass is the national card game of Switzerland, played with a 36-card deck by four players in two fixed partnerships. One player declares a game mode (trump suit, Obenabe, or Undeufe), and the two teams then compete over nine tricks to score points. A match is typically played over 8 or 12 hands, and the team with the most cumulative points wins.

This pgx environment models a single hand.

This implementation uses **tournament rules**: all game modes carry a 1× score multiplier. Melds and marriages are not supported.

## Rules

### Setup

Four players sit in fixed partnerships: team A is players 0 and 2; team B is players 1 and 3. Each player is dealt 9 cards from a 36-card deck (four suits × nine ranks: 6, 7, 8, 9, 10, J, Q, K, A).

### Trump Selection

Player 0 (forehand) acts first and must declare one of six game modes, or pass (*Schiebe*) to their partner (player 2). If player 0 passes, player 2 must declare a mode and cannot pass again.

**Game modes:**

| Action | Mode | Description |
|:---|:---|:---|
| 36 | ♦ Trump | Diamonds are trump |
| 37 | ♥ Trump | Hearts are trump |
| 38 | ♠ Trump | Spades are trump |
| 39 | ♣ Trump | Clubs are trump |
| 40 | Obenabe | No trump; Ace is highest |
| 41 | Undeufe | No trump; Six is highest |
| 42 | Schiebe | Pass to partner (forehand only) |

### Card Play

Nine tricks are played. Player 0 leads the first trick; thereafter the winner of each trick leads the next.

**Legal plays:**
- Any card is legal when leading.
- Otherwise you must follow the led suit if you hold any.
- In trump modes, if a non-trump suit is led and you cannot follow suit, you may play trump only if it beats the highest trump already in the trick — unless you hold only trump cards (in which case any trump is legal, including undertrumping).
- If trump is led and the Buur (trump Jack) is your only remaining trump, you are not required to play it and may discard off-suit.

### Scoring

The trick winner (the player who played the highest card of the led suit, or the highest trump if any trump was played) collects all four cards. 

The winner of the **last trick** receives a **5-point bonus**.

**Card point values:**

| Rank | Trump mode | Obenabe | Undeufe |
|:---:|---:|---:|---:|
| 6 | 0 | 0 | 11 |
| 7 | 0 | 0 | 0 |
| 8 | 0 | 8 | 8 |
| 9 | 0 (14 if trump) | 0 | 0 |
| 10 | 10 | 10 | 10 |
| J | 2 (20 if trump) | 2 | 2 |
| Q | 3 | 3 | 3 |
| K | 4 | 4 | 4 |
| A | 11 | 11 | 0 |

Total card points in every mode: 152 + 5 (last trick) = **157**.

**Card rank (highest wins trick):**

| Mode | Order (low → high) |
|:---|:---|
| Trump suit | 6 < 7 < 8 < 10 < Q < K < A < 9 (Nell) < J (Buur) |
| Non-trump (in trump mode) | 6 < 7 < 8 < 9 < 10 < J < Q < K < A |
| Obenabe | 6 < 7 < 8 < 9 < 10 < J < Q < K < A |
| Undeufe | A < K < Q < J < 10 < 9 < 8 < 7 < 6 |

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `4` |
| Number of actions | `43` |
| Observation shape | `(120,)` |
| Observation type | `bool` |
| Rewards | `{-157, ..., 0, ..., 157}` (integers, zero-sum across teams) |

## Observation

Each player observes 120 boolean features (partial information — other players' hands are hidden):

| Index | Description |
|:---:|:---|
| `[0:36]` | Cards currently in my hand (card encoding below) |
| `[36:72]` | Cards won in any completed trick (no team attribution) |
| `[72:108]` | Cards played in the current trick so far |
| `[108:112]` | Which player led the current trick (one-hot, players 0–3) |
| `[112:119]` | Declared game mode (one-hot: ♦, ♥, ♠, ♣, Obenabe, Undeufe, not-yet-declared) |
| `[119]` | 1 if it is this player's turn to declare trump |

Note: `[36:72]` records *which* cards have been collected but not *by which team*. This is intentional for the partial observation — score state is not directly observable. See the Value Network Features section below for a full-information representation.

**Card encoding** (indices 0–35):

| Index | Suit | Rank |
|:---:|:---:|:---:|
| 0–8 | ♦ | 6, 7, 8, 9, 10, J, Q, K, A |
| 9–17 | ♥ | 6, 7, 8, 9, 10, J, Q, K, A |
| 18–26 | ♠ | 6, 7, 8, 9, 10, J, Q, K, A |
| 27–35 | ♣ | 6, 7, 8, 9, 10, J, Q, K, A |

## Action

| Index | Meaning |
|:---:|:---|
| 0–35 | Play the card at that index (card play phase) |
| 36–41 | Declare trump/mode (trump selection phase; see table above) |
| 42 | Schiebe — pass trump selection to partner (forehand only) |

The `legal_action_mask` is `False` for all actions not currently valid.

## Rewards

Rewards are assigned at the end of the game. Each player receives their team's total score minus the opposing team's total score (zero-sum):

- Players 0 and 2 receive `score_A - score_B`
- Players 1 and 3 receive `score_B - score_A`

Rewards range from −157 to +157. A perfect win (taking all 157 points) yields +157 for the winning team.

## Termination

The game terminates after all nine tricks have been played.

## Value Network Features

`value_features(state, player)` in `pgx/_src/games/jass.py` builds a full-information
feature representation for a learned value function. It is called on a **determinized**
game state (all four hands known), as used inside MCTS leaf evaluation.

It returns two arrays:

### Card matrix — `(36, 12)` bool

One row per card; cards of the same suit are contiguous (♦: 0–8, ♥: 9–17, ♠: 18–26, ♣: 27–35).
Columns are player-relative (me / partner / left-opp / right-opp), so columns 0&7, 1&8, 2&9, 3&10
always refer to the same player.

| Col | Description |
|:---:|:---|
| 0 | I hold the card |
| 1 | Partner holds it |
| 2 | Left opponent holds it |
| 3 | Right opponent holds it |
| 4 | My team has collected it (I or partner won the trick) |
| 5 | Opponent team has collected it |
| 6 | Card is currently on the table in the trick |
| 7 | I played it in the current trick |
| 8 | Partner played it |
| 9 | Left opponent played it |
| 10 | Right opponent played it |
| 11 | Card is trump in the current mode |

Columns 0–5 and 7–10 are mutually exclusive per card (a card is held by exactly one player, or has been collected, or is in the trick). Column 6 is the OR of columns 7–10. Splitting mutually exclusive states across columns lets a single conv/attention layer attend to each fact directly.

Column 11 (`is_trump`) is zero in Obenabe and Undeufe modes (no suit is trump in those modes).

**Card identity is positional — and invisible to row-shared architectures.**
The 12 columns describe a card's *state*, not which card it is; identity (suit,
rank) is encoded only by row index. Any net that processes rows with shared
weights and pools symmetrically (the per-card-MLP + mean-pool trunk) therefore
cannot tell a Jack from a six. `PolicyValueNet` compensates by concatenating a
constant suit-one-hot ⊕ rank-one-hot (13 bits) onto each row *inside the
module* (`_CARD_IDENTITY` in `jass_value_net.py`) — the stored/collected
features stay (36, 12). The legacy `ValueNet` (V₀/V₁ artifacts) predates this
and is identity-blind; see the Step 2 run 2 post-mortem in `jass_plan.md`.

### Header — `(20,)` bool

Scalar context that does not fit neatly into the card matrix:

| Bits | Description |
|:---:|:---|
| `[0:6]` | Trump mode one-hot (♦, ♥, ♠, ♣, Obenabe, Undeufe) |
| `[6]` | Forehand passed (Schiebe was played) |
| `[7:16]` | Current trick number one-hot (0–8) |
| `[16:20]` | Trick leader one-hot (players 0–3) |

### Training target

The value network predicts the **score differential** for the current player's team: `score_my_team − score_opponent_team` ∈ [−157, 157]. This is the quantity MCTS compares across actions and is a cleaner target than raw points in [0, 157].

### Data augmentation

Suit permutations are a free augmentation, but trump breaks the symmetry. In Obenabe and Undeufe all 4! = 24 suit permutations are valid. In trump modes, the three non-trump suits can be permuted freely (3! = 6×) while the trump suit keeps its label.

**Implemented** in `jass_selfplay.py`: `sample_suit_permutation` / `apply_suit_permutation` / `augment_suits` (and `augment=` in `train_model` / `train_pv_model`). The transform relabels the 9-row suit blocks of the card matrix, the card actions 0–35 **and the trump-declare actions 36–39** of policy targets and legal masks, and the trump-suit one-hot in the header. Verified against the engine: features/masks of a directly suit-relabeled `GameState` match the transformed arrays exactly.

Note: in trump modes a full 4! permutation *with the trump one-hot remapped* would also be exactly equivalent (suit scores follow the trump label); the sampler conservatively keeps the trump suit fixed, so the header is unchanged in trump modes. The 4! extension is available for free if more diversity is ever wanted.

### Trump selection with V

During trump selection, for each legal action apply it hypothetically to get `state_after_trump`,
call `value_features`, run V, and pick the argmax over legal actions. No separate trump heuristic needed.

## Version History

- `v0` : Initial release
