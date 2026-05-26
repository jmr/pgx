from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ──────────────────────────────────────────────────────────────────────────────
# Card encoding
#
#   card c = suit * 9 + rank_idx
#   suit:     0=♦  1=♥  2=♠  3=♣
#   rank_idx: 0=6  1=7  2=8  3=9  4=10  5=J  6=Q  7=K  8=A

CARD_SUIT = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 9)   # (36,)
CARD_RANK = jnp.tile(jnp.arange(9, dtype=jnp.int32), 4)      # (36,)

# Trick-winning rank for trump cards.
# Order: 6 < 7 < 8 < 10 < Q < K < A < 9(Nell) < J(Buur)
#        0   1   2   3    4   5   6   7          8
_TRUMP_RANK_TABLE = jnp.int32([0, 1, 2, 7, 3, 8, 4, 5, 6])  # indexed by rank_idx
CARD_TRUMP_RANK = _TRUMP_RANK_TABLE[CARD_RANK]  # (36,)

# Trick-winning rank for Undeufe: reversed normal order.
CARD_UNDEUFE_RANK = jnp.int32(8) - CARD_RANK  # (36,)

# ──────────────────────────────────────────────────────────────────────────────
# Scoring
#
# All modes sum to 152 card points + 5 last-trick bonus = 157 total.
#
# In trump mode the eight is always worth 0 (regardless of suit), so the trump
# suit's Buur (J=20) + Nell (9=14) replace the eight's 8 points exactly.

#                               6   7   8    9   10   J   Q   K   A
_NORMAL_SCORE     = jnp.int32([ 0,  0,  0,   0,  10,  2,  3,  4, 11])
_TRUMP_CARD_SCORE = jnp.int32([ 0,  0,  0,  14,  10, 20,  3,  4, 11])
_OBENABE_SCORE    = jnp.int32([ 0,  0,  8,   0,  10,  2,  3,  4, 11])
_UNDEUFE_SCORE    = jnp.int32([11,  0,  8,   0,  10,  2,  3,  4,  0])

# MODE_SCORES[mode, card] — point value of card in that mode.
# Modes 0–3: trump suit (♦/♥/♠/♣); 4: Obenabe; 5: Undeufe.
_suit_is_mode = jnp.arange(4, dtype=jnp.int32)[:, None] == CARD_SUIT[None, :]  # (4, 36)
MODE_SCORES = jnp.concatenate([
    jnp.where(_suit_is_mode, _TRUMP_CARD_SCORE[CARD_RANK], _NORMAL_SCORE[CARD_RANK]),  # (4, 36)
    _OBENABE_SCORE[CARD_RANK][None],                                                    # (1, 36)
    _UNDEUFE_SCORE[CARD_RANK][None],                                                    # (1, 36)
], axis=0)  # (6, 36)

# ──────────────────────────────────────────────────────────────────────────────
# Action space constants
#   0–35 : play card
#   36–41: declare mode (mode = action - DECLARE_OFFSET)
#   42   : Schiebe (forehand passes to partner)

DECLARE_OFFSET = 36
ACTION_SCHIEBE = 42
NUM_ACTIONS    = 43

TRUMP_OBENABE = 4
TRUMP_UNDEUFE = 5
RANK_JACK     = 5   # rank_idx of J (used for Buur detection)

LAST_TRICK_BONUS = 5


# ──────────────────────────────────────────────────────────────────────────────

class GameState(NamedTuple):
    current_player:  Array = jnp.int32(0)
    hands:           Array = jnp.zeros((4, 36), dtype=jnp.bool_)
    trump:           Array = jnp.int32(-1)   # -1 = not yet declared
    phase:           Array = jnp.int32(0)    # 0 = trump selection, 1 = card play
    forehand_passed: Array = jnp.bool_(False)
    trick_cards:     Array = -jnp.ones(4, dtype=jnp.int32)  # card per player, -1 = not played
    trick_leader:    Array = jnp.int32(0)
    led_suit:        Array = jnp.int32(-1)   # -1 before first card of trick
    cards_collected: Array = jnp.zeros((4, 36), dtype=jnp.bool_)
    trick_num:       Array = jnp.int32(0)    # 0–8; 9 after last trick resolved
    # void_in_suit[p, s]: player p is known to hold no cards of suit s.
    # Inferred when a follower plays off the led suit (excluding the Buur-exempt case).
    # Used by the MCTS determinization sampler to avoid impossible card assignments.
    void_in_suit:    Array = jnp.zeros((4, 4), dtype=jnp.bool_)


class Game:
    def init(self, key: Array) -> GameState:
        perm = jax.random.permutation(key, 36)
        player_cards = perm.reshape(4, 9)  # (4, 9)
        hands = jax.vmap(lambda cs: jnp.zeros(36, dtype=jnp.bool_).at[cs].set(True))(player_cards)
        return GameState(hands=hands)

    def legal_action_mask(self, state: GameState) -> Array:
        return jax.lax.cond(
            state.phase == 0,
            lambda: _trump_selection_mask(state),
            lambda: _card_play_mask(state),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        return jax.lax.cond(
            state.phase == 0,
            lambda: _trump_selection_step(state, action),
            lambda: _card_play_step(state, action),
        )

    def is_terminal(self, state: GameState) -> Array:
        return state.trick_num >= 9

    def rewards(self, state: GameState) -> Array:
        safe_trump = jnp.clip(state.trump, 0, 5)
        card_scores = MODE_SCORES[safe_trump]  # (36,)
        per_player = (state.cards_collected * card_scores[None, :]).sum(axis=1)  # (4,)
        # Last-trick bonus: winner of trick 8 is current trick_leader after resolution.
        last_trick_winner = state.trick_leader
        per_player = per_player.at[last_trick_winner].add(LAST_TRICK_BONUS)
        team_a = per_player[0] + per_player[2]  # players 0 & 2
        team_b = per_player[1] + per_player[3]
        diff = jnp.float32(team_a - team_b)
        # Players 0,2 get +diff; players 1,3 get -diff.
        signs = jnp.float32([1, -1, 1, -1])
        return jnp.where(self.is_terminal(state), diff * signs, jnp.zeros(4, jnp.float32))

    def observe(self, state: GameState, player: Array) -> Array:
        hand = state.hands[player]  # (36,)
        # Cards taken in previous complete tricks (by anyone).
        prev_taken = state.cards_collected.any(axis=0)  # (36,)
        # Cards currently on the table in this trick.
        valid = state.trick_cards >= 0  # (4,)
        safe  = jnp.where(valid, state.trick_cards, 0)
        in_trick = jnp.zeros(36, dtype=jnp.bool_).at[safe].max(valid)  # (36,)
        # Who led the current trick (one-hot, 4 bits).
        who_led = jnp.arange(4) == state.trick_leader  # (4,) bool
        # Trump mode one-hot (7 bits: ♦ ♥ ♠ ♣ Obenabe Undeufe not-declared).
        safe_trump = jnp.clip(state.trump, 0, 5)
        trump_oh = jnp.arange(6) == safe_trump          # (6,)
        trump_oh = jnp.where(state.trump >= 0, trump_oh, jnp.zeros(6, dtype=jnp.bool_))
        not_declared = jnp.bool_(state.trump < 0)        # (1,)
        trump_bits = jnp.append(trump_oh, not_declared)  # (7,)
        # My turn to declare.
        my_declare_turn = jnp.bool_(
            (state.phase == 0) & (state.current_player == player)
        )
        return jnp.concatenate([
            hand,                                         # [0:36]
            prev_taken,                                   # [36:72]
            in_trick,                                     # [72:108]
            who_led,                                      # [108:112]
            trump_bits,                                   # [112:119]
            jnp.bool_([my_declare_turn]),                 # [119]
        ])  # (120,)


# ──────────────────────────────────────────────────────────────────────────────
# Trump selection

def value_features(state: GameState, player: Array) -> tuple[Array, Array]:
    """Build value-network features from a fully-observed (determinized) game state.

    Returns:
        card_matrix : (36, 12) bool — one row per card, columns below.
        header      : (20,)  bool — scalar context.

    Card matrix columns (player-relative: me / partner / left-opp / right-opp):
        0  : I hold the card
        1  : partner holds it
        2  : left opponent holds it
        3  : right opponent holds it
        4  : my team has collected it
        5  : opponent team has collected it
        6  : card is currently in the trick
        7  : I played it in the current trick
        8  : partner played it
        9  : left opponent played it
        10 : right opponent played it
        11 : card is trump in the current mode

    Player-relative seat offsets: me=0, left-opp=1, partner=2, right-opp=3.
    Columns 0&7 / 1&8 / 2&9 / 3&10 always refer to the same player.

    Header:
        [0:6]   trump mode one-hot (♦ ♥ ♠ ♣ Obenabe Undeufe)
        [6]     forehand_passed
        [7:16]  trick_num one-hot (0–8)
        [16:20] trick_leader one-hot (players 0–3)
    """
    partner   = (player + 2) % 4
    left_opp  = (player + 1) % 4
    right_opp = (player + 3) % 4

    # ── Columns 0–3: who holds each card ──
    col0 = state.hands[player]     # (36,)
    col1 = state.hands[partner]
    col2 = state.hands[left_opp]
    col3 = state.hands[right_opp]

    # ── Columns 4–5: team collection ──
    col4 = state.cards_collected[player]   | state.cards_collected[partner]    # (36,)
    col5 = state.cards_collected[left_opp] | state.cards_collected[right_opp]

    # ── Columns 6–10: current trick ──
    valid = state.trick_cards >= 0                                    # (4,)
    safe  = jnp.where(valid, state.trick_cards, 0)                    # (4,)
    col6  = jnp.zeros(36, dtype=jnp.bool_).at[safe].max(valid)       # (36,)

    def _played_by(seat_offset: int) -> Array:
        """(36,) mask: True at the card played by the player at relative seat_offset."""
        p      = (player + seat_offset) % 4
        c      = state.trick_cards[p]
        played = (c >= 0) & valid[p]
        safe_c = jnp.where(played, c, 0)
        return jnp.zeros(36, dtype=jnp.bool_).at[safe_c].max(played)

    col7  = _played_by(0)   # me
    col8  = _played_by(2)   # partner
    col9  = _played_by(1)   # left opponent
    col10 = _played_by(3)   # right opponent

    # ── Column 11: is trump ──
    is_trump_mode = (state.trump >= 0) & (state.trump < 4)
    trump_suit    = jnp.clip(state.trump, 0, 3)
    col11 = is_trump_mode & (CARD_SUIT == trump_suit)                 # (36,)

    card_matrix = jnp.stack(
        [col0, col1, col2, col3, col4, col5,
         col6, col7, col8, col9, col10, col11],
        axis=1,
    )  # (36, 12)

    # ── Header ──
    safe_trump   = jnp.clip(state.trump, 0, 5)
    trump_oh     = (jnp.arange(6) == safe_trump) & (state.trump >= 0)  # (6,)
    trick_num_oh = jnp.arange(9) == state.trick_num                     # (9,)
    leader_oh    = jnp.arange(4) == state.trick_leader                  # (4,)

    header = jnp.concatenate([
        trump_oh,
        jnp.bool_([state.forehand_passed]),
        trick_num_oh,
        leader_oh,
    ])  # (20,)

    return card_matrix, header


# ──────────────────────────────────────────────────────────────────────────────
# Trump selection

def _trump_selection_mask(state: GameState) -> Array:
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    # Declare actions 36–41 always available.
    mask = mask.at[DECLARE_OFFSET : DECLARE_OFFSET + 6].set(True)
    # Schiebe (42) only available to forehand (player 0) who hasn't passed yet.
    can_schiebe = (state.current_player == 0) & ~state.forehand_passed
    return mask.at[ACTION_SCHIEBE].set(can_schiebe)


def _trump_selection_step(state: GameState, action: Array) -> GameState:
    is_schiebe = action == ACTION_SCHIEBE
    # If Schiebe: pass to partner (player 2), mark passed.
    # If declare: set trump, move to card-play phase.
    trump = jnp.where(is_schiebe, state.trump, action - DECLARE_OFFSET)
    phase = jnp.where(is_schiebe, jnp.int32(0), jnp.int32(1))
    forehand_passed = is_schiebe
    next_player = jnp.where(is_schiebe, jnp.int32(2), jnp.int32(0))
    return state._replace(
        trump=trump,
        phase=phase,
        forehand_passed=forehand_passed,
        current_player=next_player,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Card-play legal action mask

def _card_play_mask(state: GameState) -> Array:
    player = state.current_player
    hand = state.hands[player]  # (36,) bool

    is_trick_start = state.led_suit < 0

    is_trump_mode = (state.trump >= 0) & (state.trump < 4)
    trump_suit = jnp.clip(state.trump, 0, 3)  # safe index even in non-trump modes

    is_trump_card = CARD_SUIT == trump_suit  # (36,) — meaningless when not trump mode

    # Highest trump rank currently in the trick.
    valid = state.trick_cards >= 0                           # (4,)
    safe  = jnp.where(valid, state.trick_cards, 0)           # (4,)
    trick_is_trump = valid & (CARD_SUIT[safe] == trump_suit)  # (4,)
    trick_trump_ranks = jnp.where(trick_is_trump, CARD_TRUMP_RANK[safe], jnp.int32(-1))
    highest_trick_trump = trick_trump_ranks.max()            # -1 if none

    # Per-card: beats current highest trump in trick?
    card_beats_trick_trump = is_trump_card & (CARD_TRUMP_RANK > highest_trick_trump)

    # Follow-suit helpers.
    led = state.led_suit
    card_is_led = CARD_SUIT == led                            # (36,)
    has_led_suit = (hand & card_is_led).any()

    # Only-trump-in-hand: any card is always legal.
    has_only_trump = ~(hand & ~is_trump_card).any()

    # Buur exemption: Jack of trump is the only remaining trump card.
    is_buur = is_trump_card & (CARD_RANK == RANK_JACK)
    trump_in_hand = hand & is_trump_card
    has_only_buur = trump_in_hand.any() & ~(trump_in_hand & ~is_buur).any()

    led_is_trump = (led == trump_suit) & is_trump_mode

    # ── Case: trump led ──
    # Must follow trump; Buur exemption lets you play off-suit.
    follow_trump = hand & is_trump_card
    led_trump_mask = jnp.where(has_only_buur | ~trump_in_hand.any(), hand, follow_trump)

    # ── Case: non-trump led ──
    # Trump cards: legal only if they beat the current highest trick trump.
    # Non-trump cards: legal if void in led suit, or card matches led suit.
    non_trump_card_legal = ~is_trump_card & (~has_led_suit | card_is_led)
    trump_card_legal     = card_beats_trick_trump
    non_led_trump_mask   = (non_trump_card_legal | trump_card_legal) & hand

    # ── Combine trump-mode mask ──
    trump_mask = jnp.where(led_is_trump, led_trump_mask, non_led_trump_mask)
    trump_mask = jnp.where(has_only_trump, hand, trump_mask)

    # ── Non-trump mode (Obenabe / Undeufe): simple follow-suit ──
    plain_mask = jnp.where(has_led_suit, hand & card_is_led, hand)

    play_mask = jnp.where(is_trump_mode, trump_mask, plain_mask)
    play_mask = jnp.where(is_trick_start, hand, play_mask)

    # Embed into full action space (actions 0–35 only).
    full = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    return full.at[:36].set(play_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Card-play step

def _card_play_step(state: GameState, action: Array) -> GameState:
    player = state.current_player
    card   = action  # 0–35

    # Place card.
    hands       = state.hands.at[player].set(state.hands[player].at[card].set(False))
    trick_cards = state.trick_cards.at[player].set(card)
    trick_count = (trick_cards >= 0).sum()

    # Set led suit on first card of trick.
    led_suit = jnp.where(state.led_suit < 0, CARD_SUIT[card], state.led_suit)

    # Infer void-in-suit.
    # If a follower (not the trick leader) plays off the led suit, they are void
    # in that suit — unless trump is led in a trump mode (Buur exemption makes
    # the inference ambiguous: they may hold only the Buur, not be void in trump).
    is_trump_mode  = (state.trump >= 0) & (state.trump < 4)
    trump_suit     = jnp.clip(state.trump, 0, 3)
    is_follower    = state.led_suit >= 0                       # player is not the leader
    played_off_led = CARD_SUIT[card] != state.led_suit
    led_is_trump   = is_trump_mode & (state.led_suit == trump_suit)
    safe_led       = jnp.clip(state.led_suit, 0, 3)           # safe index for -1 case
    can_infer_void = is_follower & played_off_led & ~led_is_trump

    # Update void_in_suit[player, led_suit] ← True (vectorised outer product).
    player_onehot    = jnp.arange(4) == player                # (4,)
    suit_onehot      = jnp.arange(4) == safe_led              # (4,)
    void_update      = can_infer_void & player_onehot[:, None] & suit_onehot[None, :]  # (4, 4)
    new_void_in_suit = state.void_in_suit | void_update

    # After all 4 cards played: resolve trick.
    return jax.lax.cond(
        trick_count == 4,
        lambda: _resolve_trick(state._replace(
            hands=hands, trick_cards=trick_cards, led_suit=led_suit,
            void_in_suit=new_void_in_suit,
        )),
        lambda: state._replace(
            hands=hands,
            trick_cards=trick_cards,
            led_suit=led_suit,
            current_player=(player + 1) % 4,
            void_in_suit=new_void_in_suit,
        ),
    )


def _resolve_trick(state: GameState) -> GameState:
    winner = _trick_winner(state.trick_cards, state.led_suit, state.trump)

    # All 4 trick cards go to the winner's row in cards_collected.
    safe = jnp.where(state.trick_cards >= 0, state.trick_cards, 0)  # (4,)
    new_row = state.cards_collected[winner].at[safe].set(True)
    new_collected = state.cards_collected.at[winner].set(new_row)

    return state._replace(
        trick_cards=jnp.full(4, -1, dtype=jnp.int32),
        led_suit=jnp.int32(-1),
        trick_leader=winner,
        current_player=winner,
        cards_collected=new_collected,
        trick_num=state.trick_num + 1,
    )


def _trick_winner(trick_cards: Array, led_suit: Array, trump: Array) -> Array:
    """Return the player index (0–3) who wins the trick."""
    is_trump_mode = (trump >= 0) & (trump < 4)
    trump_suit = jnp.clip(trump, 0, 3)

    valid = trick_cards >= 0                              # (4,)
    safe  = jnp.where(valid, trick_cards, 0)             # (4,)

    suit = CARD_SUIT[safe]                               # (4,)
    rank = CARD_RANK[safe]                               # (4,)

    is_trump = (suit == trump_suit) & is_trump_mode      # (4,)
    is_led   = suit == led_suit                          # (4,)

    # Trump cards beat led-suit cards; offset their rank by 9.
    trump_eff = CARD_TRUMP_RANK[safe] + 9               # (4,)
    obenabe_eff = rank                                   # (4,)
    undeufe_eff = jnp.int32(8) - rank                   # (4,)

    eff_trump = jnp.where(is_trump, trump_eff, jnp.where(is_led, obenabe_eff, jnp.int32(-1)))
    eff_plain = jnp.where(is_led, obenabe_eff, jnp.int32(-1))
    eff_undeufe = jnp.where(is_led, undeufe_eff, jnp.int32(-1))

    eff = jnp.where(
        is_trump_mode, eff_trump,
        jnp.where(trump == TRUMP_OBENABE, eff_plain, eff_undeufe)
    )
    eff = jnp.where(valid, eff, jnp.int32(-2))

    return jnp.argmax(eff).astype(jnp.int32)
