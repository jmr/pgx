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

Each player observes 120 boolean features:

| Index | Description |
|:---:|:---|
| `[0:36]` | Cards currently in my hand (card encoding below) |
| `[36:72]` | Cards won in completed tricks (publicly known) |
| `[72:108]` | Cards played in the current trick so far |
| `[108:112]` | Which player led the current trick (one-hot, players 0–3) |
| `[112:119]` | Declared game mode (one-hot: ♦, ♥, ♠, ♣, Obenabe, Undeufe, not-yet-declared) |
| `[119]` | 1 if it is this player's turn to declare trump |

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

## Alternative Observation: 2D Plane Layout

For CNN or attention-based models, a `(15, 36)` boolean plane layout may be preferable. Cards of the same suit occupy contiguous indices (♦: 0–8, ♥: 9–17, ♠: 18–26, ♣: 27–35), so 1D convolutions across the card axis naturally capture suit-level structure.

| Plane | Description |
|:---:|:---|
| 0 | Cards in my hand |
| 1 | Cards won by my team in completed tricks |
| 2 | Cards won by opponents in completed tricks |
| 3–6 | Cards played in the current trick, one plane per seat relative to trick leader |
| 7–12 | Declared trump mode (one-hot: ♦, ♥, ♠, ♣, Obenabe, Undeufe) — constant across all 36 positions |
| 13 | Trump not yet declared — constant |
| 14 | It is my turn to declare trump — constant |

Compared to the flat layout, this splits "cards played in previous tricks" into two planes (my team vs. opponents), preserving information about who won each trick that the flat encoding discards.

## Version History

- `v0` : Initial release
