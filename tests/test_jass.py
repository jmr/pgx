import jax
import jax.numpy as jnp

import pgx
from pgx._src.games.jass import (
    CARD_RANK, CARD_SUIT, CARD_TRUMP_RANK, CARD_UNDEUFE_RANK,
    MODE_SCORES, TRUMP_OBENABE, TRUMP_UNDEUFE,
    LAST_TRICK_BONUS, NUM_ACTIONS, ACTION_SCHIEBE, DECLARE_OFFSET,
    RANK_JACK,
    Game, GameState,
    _trump_selection_mask, _card_play_mask, _trick_winner,
)

game = Game()


# ──────────────────────────────────────────────────
# Constants

def test_card_encoding():
    # 36 cards, 4 suits × 9 ranks
    assert CARD_SUIT.shape == (36,)
    assert CARD_RANK.shape == (36,)
    assert (CARD_SUIT[:9] == 0).all()   # ♦
    assert (CARD_SUIT[9:18] == 1).all() # ♥
    assert (CARD_SUIT[18:27] == 2).all()# ♠
    assert (CARD_SUIT[27:] == 3).all()  # ♣
    assert (CARD_RANK[:9] == jnp.arange(9)).all()
    assert (CARD_RANK[27:] == jnp.arange(9)).all()


def test_trump_rank_order():
    # Buur (J) must be highest, Nell (9) second-highest
    jack_rank = CARD_TRUMP_RANK[5]   # ♦J = card 5, rank_idx 5
    nell_rank = CARD_TRUMP_RANK[3]   # ♦9 = card 3, rank_idx 3
    ace_rank  = CARD_TRUMP_RANK[8]   # ♦A = card 8
    six_rank  = CARD_TRUMP_RANK[0]   # ♦6 = card 0
    assert jack_rank > nell_rank > ace_rank > six_rank


def test_undeufe_rank_order():
    # Six is highest, Ace is lowest
    assert CARD_UNDEUFE_RANK[0] > CARD_UNDEUFE_RANK[8]  # ♦6 > ♦A
    assert (CARD_UNDEUFE_RANK == 8 - CARD_RANK).all()


def test_mode_scores_totals():
    # All 6 modes must sum to 152 card points (+ 5 last-trick bonus handled separately)
    for mode in range(6):
        assert int(MODE_SCORES[mode].sum()) == 152, f"mode {mode} total != 152"


def test_mode_scores_trump_buur_nell():
    # In ♦ trump (mode 0): ♦J=20, ♦9=14
    assert MODE_SCORES[0, 5] == 20   # ♦J
    assert MODE_SCORES[0, 3] == 14   # ♦9


def test_mode_scores_eight():
    # In trump modes: all 8s are 0
    eights = jnp.arange(4) * 9 + 2   # cards 2, 11, 20, 29
    for mode in range(4):
        for c in eights:
            assert MODE_SCORES[mode, c] == 0, f"mode {mode} card {c} (eight) should be 0"
    # In Obenabe and Undeufe: 8 is worth 8
    for c in eights:
        assert MODE_SCORES[TRUMP_OBENABE, c] == 8
        assert MODE_SCORES[TRUMP_UNDEUFE, c] == 8


def test_mode_scores_undeufe_six_ace():
    # Undeufe: 6=11, A=0
    sixes = jnp.arange(4) * 9 + 0
    aces  = jnp.arange(4) * 9 + 8
    for c in sixes:
        assert MODE_SCORES[TRUMP_UNDEUFE, c] == 11
    for c in aces:
        assert MODE_SCORES[TRUMP_UNDEUFE, c] == 0


# ──────────────────────────────────────────────────
# Init

def test_init_deal():
    key = jax.random.PRNGKey(0)
    state = game.init(key)
    # Each player gets exactly 9 cards
    for p in range(4):
        assert int(state.hands[p].sum()) == 9
    # No card dealt to two players
    total = state.hands.sum(axis=0)
    assert (total == 1).all()


def test_init_phase():
    state = game.init(jax.random.PRNGKey(0))
    assert state.phase == 0        # trump selection
    assert state.current_player == 0
    assert state.trump == -1


def test_init_trick_state():
    state = game.init(jax.random.PRNGKey(0))
    assert (state.trick_cards == -1).all()
    assert state.led_suit == -1
    assert state.trick_num == 0
    assert not state.cards_collected.any()


# ──────────────────────────────────────────────────
# Trump selection

def test_trump_selection_mask_forehand():
    state = game.init(jax.random.PRNGKey(0))
    mask = _trump_selection_mask(state)
    # Actions 36–41 legal, 42 (Schiebe) legal for player 0
    assert mask[DECLARE_OFFSET:DECLARE_OFFSET + 6].all()
    assert mask[ACTION_SCHIEBE]
    # No card actions
    assert not mask[:36].any()


def test_trump_selection_mask_after_schiebe():
    state = game.init(jax.random.PRNGKey(0))
    # Player 0 Schiebt
    state = game.step(state, jnp.int32(ACTION_SCHIEBE))
    assert state.forehand_passed
    assert state.current_player == 2
    mask = _trump_selection_mask(state)
    # Declare still legal
    assert mask[DECLARE_OFFSET:DECLARE_OFFSET + 6].all()
    # Schiebe not legal (player 2 cannot pass)
    assert not mask[ACTION_SCHIEBE]


def test_trump_selection_declare():
    state = game.init(jax.random.PRNGKey(0))
    state = game.step(state, jnp.int32(DECLARE_OFFSET + TRUMP_OBENABE))
    assert state.trump == TRUMP_OBENABE
    assert state.phase == 1          # moved to card play
    assert state.current_player == 0


def test_trump_selection_schiebe_then_declare():
    state = game.init(jax.random.PRNGKey(0))
    state = game.step(state, jnp.int32(ACTION_SCHIEBE))
    state = game.step(state, jnp.int32(DECLARE_OFFSET + 2))  # ♠ trump
    assert state.trump == 2
    assert state.phase == 1
    assert state.current_player == 0


# ──────────────────────────────────────────────────
# Helpers for building test states

def _hand(*cards):
    """Boolean hand array with the given card indices set."""
    h = jnp.zeros(36, dtype=jnp.bool_)
    return h.at[jnp.array(cards)].set(True)


def _card(suit, rank_idx):
    """Card index from suit (0-3) and rank_idx (0-8: 6,7,8,9,10,J,Q,K,A)."""
    return suit * 9 + rank_idx


# Handy named rank indices
_6, _7, _8, _9, _10, _J, _Q, _K, _A = range(9)

D, H, S, C = 0, 1, 2, 3   # suit constants


def _play_state(trump, led_suit, player, hand_cards, trick=None):
    """Minimal card-play GameState for testing legal_action_mask."""
    hands = jnp.zeros((4, 36), dtype=jnp.bool_)
    hands = hands.at[player].set(_hand(*hand_cards))
    tc = -jnp.ones(4, dtype=jnp.int32)
    if trick:
        for p, c in trick.items():
            tc = tc.at[p].set(c)
    return GameState(
        current_player=jnp.int32(player),
        hands=hands,
        trump=jnp.int32(trump),
        phase=jnp.int32(1),
        led_suit=jnp.int32(led_suit),
        trick_cards=tc,
    )


def _legal_cards(state):
    """Return set of legal card indices (actions 0-35) for state."""
    mask = _card_play_mask(state)
    return set(int(i) for i in jnp.where(mask[:36])[0])


# ──────────────────────────────────────────────────
# Legal action mask: follow-suit

def test_legal_mask_follow_suit_trump_mode():
    # Trump=♦, led=♥, hand has ♥K plus off-suit cards → only ♥K legal.
    hk = _card(H, _K)   # ♥K
    s6 = _card(S, _6)   # ♠6
    ca = _card(C, _A)   # ♣A
    state = _play_state(trump=D, led_suit=H, player=0, hand_cards=[hk, s6, ca])
    assert _legal_cards(state) == {hk}


def test_legal_mask_follow_suit_obenabe():
    # Obenabe: same follow-suit rule, no trump concept.
    hk = _card(H, _K)
    s6 = _card(S, _6)
    state = _play_state(trump=4, led_suit=H, player=0, hand_cards=[hk, s6])
    assert _legal_cards(state) == {hk}


def test_legal_mask_void_non_trump_mode():
    # Obenabe, void in led suit → all hand cards legal.
    s6 = _card(S, _6)
    ca = _card(C, _A)
    state = _play_state(trump=4, led_suit=H, player=0, hand_cards=[s6, ca])
    assert _legal_cards(state) == {s6, ca}


def test_legal_mask_trick_start():
    # No cards played yet → any hand card legal.
    hk = _card(H, _K)
    d6 = _card(D, _6)
    state = _play_state(trump=D, led_suit=-1, player=0, hand_cards=[hk, d6])
    assert _legal_cards(state) == {hk, d6}


# ──────────────────────────────────────────────────
# Legal action mask: trump restrictions

def test_legal_mask_no_undertrump():
    # Trump=♥, led=♦, trick has ♥8 (trump rank 2).
    # Hand: ♥6 (trump rank 0, can't beat ♥8), ♠K (non-trump off-suit).
    # ♥6 is illegal (undertrump); ♠K is legal (non-trump, void in led suit).
    h8 = _card(H, _8)
    h6 = _card(H, _6)
    sk = _card(S, _K)
    # Player 1 led ♦6, player 2 played ♥8 (trump)
    state = _play_state(
        trump=H, led_suit=D, player=0,
        hand_cards=[h6, sk],
        trick={1: _card(D, _6), 2: h8},
    )
    assert _legal_cards(state) == {sk}


def test_legal_mask_can_overtrump():
    # Same setup but hand has ♥J (Buur, trump rank 8) instead of ♥6.
    # Buur beats ♥8 → legal. ♠K also legal (off-suit, void in led ♦).
    h8 = _card(H, _8)
    hj = _card(H, _J)   # Buur
    sk = _card(S, _K)
    state = _play_state(
        trump=H, led_suit=D, player=0,
        hand_cards=[hj, sk],
        trick={1: _card(D, _6), 2: h8},
    )
    assert _legal_cards(state) == {hj, sk}


def test_legal_mask_only_trump_in_hand_allows_undertrump():
    # Trump=♥, led=♦, trick has ♥8. Hand: only ♥6 (trump, can't beat ♥8).
    # Because hand is all-trump, undertrump is allowed.
    h8 = _card(H, _8)
    h6 = _card(H, _6)
    state = _play_state(
        trump=H, led_suit=D, player=0,
        hand_cards=[h6],
        trick={1: _card(D, _6), 2: h8},
    )
    assert _legal_cards(state) == {h6}


def test_legal_mask_no_trump_in_trick_any_trump_ok():
    # Trump=♥, led=♦, no trump in trick yet, hand has ♥6 (low trump) + ♠K.
    # ♥6 is legal (any trump ok when no trump yet); ♠K legal (off-suit, void in ♦).
    h6 = _card(H, _6)
    sk = _card(S, _K)
    state = _play_state(
        trump=H, led_suit=D, player=0,
        hand_cards=[h6, sk],
        trick={1: _card(D, _6)},
    )
    assert _legal_cards(state) == {h6, sk}


def test_legal_mask_voluntary_trump_over_led_suit():
    # Trump=♥, led=♦, no trump in trick. Hand has ♦K (led suit) and ♥J (Buur).
    # ♦K: must follow → legal. ♥J: voluntary trump (highest since no trump yet) → also legal.
    dk = _card(D, _K)
    hj = _card(H, _J)
    state = _play_state(
        trump=H, led_suit=D, player=0,
        hand_cards=[dk, hj],
        trick={1: _card(D, _6)},
    )
    assert _legal_cards(state) == {dk, hj}


def test_legal_mask_voluntary_trump_must_beat():
    # Trump=♥, led=♦, trick has ♥9 (Nell, trump rank 7). Hand has ♦K and ♥6 (rank 0).
    # ♦K: follows led suit → legal. ♥6: can't beat ♥9 → illegal.
    dk = _card(D, _K)
    h9 = _card(H, _9)   # Nell
    h6 = _card(H, _6)
    state = _play_state(
        trump=H, led_suit=D, player=0,
        hand_cards=[dk, h6],
        trick={1: _card(D, _6), 2: h9},
    )
    assert _legal_cards(state) == {dk}


def test_legal_mask_trump_led_must_follow():
    # Trump=♠, led=♠. Hand has ♠10 (trump) and ♥K (off-suit). Must follow trump.
    s10 = _card(S, _10)
    hk  = _card(H, _K)
    state = _play_state(
        trump=S, led_suit=S, player=0,
        hand_cards=[s10, hk],
        trick={1: _card(S, _6)},
    )
    assert _legal_cards(state) == {s10}


def test_legal_mask_buur_exemption():
    # Trump=♣, led=♣. Hand has ♣J (Buur, only trump) and ♥A.
    # Buur exemption: don't have to play Buur → ♥A is also legal.
    cj = _card(C, _J)   # Buur
    ha = _card(H, _A)
    state = _play_state(
        trump=C, led_suit=C, player=0,
        hand_cards=[cj, ha],
        trick={1: _card(C, _6)},
    )
    assert _legal_cards(state) == {cj, ha}


def test_legal_mask_buur_exemption_only_when_sole_trump():
    # Trump=♣, led=♣. Hand has ♣J (Buur) AND ♣9 (Nell). Not sole trump → must follow.
    cj = _card(C, _J)
    c9 = _card(C, _9)
    ha = _card(H, _A)
    state = _play_state(
        trump=C, led_suit=C, player=0,
        hand_cards=[cj, c9, ha],
        trick={1: _card(C, _6)},
    )
    assert _legal_cards(state) == {cj, c9}


def test_legal_mask_trump_led_void_in_trump():
    # Trump=♦, led=♦, player has no ♦ cards → can play anything.
    hk = _card(H, _K)
    s6 = _card(S, _6)
    state = _play_state(
        trump=D, led_suit=D, player=0,
        hand_cards=[hk, s6],
        trick={1: _card(D, _J)},  # ♦J (Buur) led
    )
    assert _legal_cards(state) == {hk, s6}


def test_legal_mask_undeufe():
    # Undeufe: no trump. Led=♦. Hand has ♦K, ♠6. Only ♦K legal.
    dk = _card(D, _K)
    s6 = _card(S, _6)
    state = _play_state(trump=5, led_suit=D, player=0, hand_cards=[dk, s6])
    assert _legal_cards(state) == {dk}


# ──────────────────────────────────────────────────
# Trick winner

def _winner(trump, led_suit, cards_by_player):
    """cards_by_player: dict {player: card_index}."""
    tc = -jnp.ones(4, dtype=jnp.int32)
    for p, c in cards_by_player.items():
        tc = tc.at[p].set(c)
    return int(_trick_winner(tc, jnp.int32(led_suit), jnp.int32(trump)))


def test_trick_winner_normal_suit():
    # No trump, led=♦. ♦A beats ♦K.
    assert _winner(4, D, {0: _card(D, _6), 1: _card(D, _K), 2: _card(D, _A), 3: _card(D, _7)}) == 2


def test_trick_winner_off_suit_cannot_win():
    # Led=♦, player 3 plays ♠A (off-suit, not trump). ♦K wins.
    assert _winner(4, D, {0: _card(D, _6), 1: _card(D, _K), 2: _card(D, _7), 3: _card(S, _A)}) == 1


def test_trick_winner_trump_beats_led():
    # Trump=♥, led=♦. Any ♥ beats any ♦.
    assert _winner(H, D, {0: _card(D, _A), 1: _card(H, _6), 2: _card(D, _K), 3: _card(D, _Q)}) == 1


def test_trick_winner_buur_beats_nell():
    # Trump=♥. ♥J (Buur) beats ♥9 (Nell).
    assert _winner(H, D, {0: _card(D, _A), 1: _card(H, _9), 2: _card(H, _J), 3: _card(D, _K)}) == 2


def test_trick_winner_nell_beats_ace():
    # Trump=♥. ♥9 (Nell) beats ♥A.
    assert _winner(H, D, {0: _card(D, _A), 1: _card(H, _9), 2: _card(H, _A), 3: _card(D, _K)}) == 1


def test_trick_winner_obenabe():
    # Obenabe, led=♦. ♦A wins; ♠A is off-suit, doesn't count.
    assert _winner(4, D, {0: _card(D, _A), 1: _card(S, _A), 2: _card(D, _K), 3: _card(D, _6)}) == 0


def test_trick_winner_undeufe():
    # Undeufe, led=♦. ♦6 is highest (rank 8 in undeufe).
    assert _winner(5, D, {0: _card(D, _A), 1: _card(D, _6), 2: _card(D, _K), 3: _card(D, _7)}) == 1


def test_trick_winner_first_played_wins_tie_impossible():
    # With real cards there are no ties, but verify leader wins same-rank scenario
    # is irrelevant — just verify argmax picks correctly among led-suit cards.
    assert _winner(4, D, {0: _card(D, _K), 1: _card(D, _Q), 2: _card(D, _7), 3: _card(D, _6)}) == 0


# ──────────────────────────────────────────────────
# Scoring

def _make_terminal(trump, winner_player, all_cards_to_player, last_trick_winner=None):
    """Terminal GameState where one player collected all cards."""
    collected = jnp.zeros((4, 36), dtype=jnp.bool_)
    collected = collected.at[all_cards_to_player].set(True)
    return GameState(
        trump=jnp.int32(trump),
        trick_num=jnp.int32(9),
        trick_leader=jnp.int32(last_trick_winner if last_trick_winner is not None else all_cards_to_player),
        cards_collected=collected,
    )


def test_scoring_team_a_sweeps_trump():
    # Player 0 (team A) collects all cards in ♦ trump mode and won last trick.
    state = _make_terminal(trump=D, winner_player=0, all_cards_to_player=0)
    r = game.rewards(state)
    # team_a = 152 + 5 = 157, team_b = 0 → diff = 157
    assert r[0] == 157.0
    assert r[2] == 157.0
    assert r[1] == -157.0
    assert r[3] == -157.0


def test_scoring_team_b_sweeps():
    state = _make_terminal(trump=D, winner_player=1, all_cards_to_player=1)
    r = game.rewards(state)
    assert r[1] == 157.0
    assert r[3] == 157.0
    assert r[0] == -157.0
    assert r[2] == -157.0


def test_scoring_buur_nell_trump():
    # In ♦ trump: Buur (♦J=card 5) = 20, Nell (♦9=card 3) = 14.
    collected = jnp.zeros((4, 36), dtype=jnp.bool_)
    collected = collected.at[0, 5].set(True)   # ♦J (Buur)
    collected = collected.at[0, 3].set(True)   # ♦9 (Nell)
    state = GameState(
        trump=jnp.int32(D),
        trick_num=jnp.int32(9),
        trick_leader=jnp.int32(0),
        cards_collected=collected,
    )
    r = game.rewards(state)
    # Player 0: 20 + 14 + 5 (last trick) = 39. Team B: 0.
    # diff = 39 - 0 = 39
    assert r[0] == 39.0
    assert r[1] == -39.0


def test_scoring_obenabe_eight_counts():
    # Obenabe: ♦8 (card 2) = 8 points. Player 2 (team A) collects it + wins last trick.
    collected = jnp.zeros((4, 36), dtype=jnp.bool_)
    collected = collected.at[2, 2].set(True)   # ♦8
    state = GameState(
        trump=jnp.int32(TRUMP_OBENABE),
        trick_num=jnp.int32(9),
        trick_leader=jnp.int32(2),
        cards_collected=collected,
    )
    r = game.rewards(state)
    # team_a = 8 + 5 = 13, team_b = 0 → diff = 13
    assert r[0] == 13.0
    assert r[2] == 13.0
    assert r[1] == -13.0


def test_scoring_zero_sum():
    state = _make_terminal(trump=H, winner_player=0, all_cards_to_player=0)
    r = game.rewards(state)
    assert float(r.sum()) == 0.0


# ──────────────────────────────────────────────────
# Full game

def test_full_game_terminates():
    key = jax.random.PRNGKey(42)
    state = game.init(key)
    state = game.step(state, jnp.int32(DECLARE_OFFSET + 0))  # ♦ trump

    for _ in range(36):
        assert not game.is_terminal(state)
        mask = game.legal_action_mask(state)
        assert mask.any(), "no legal actions mid-game"
        action = jnp.int32(jnp.argmax(mask))
        state = game.step(state, action)

    assert game.is_terminal(state)


def test_api():
    env = pgx.make("jass")
    pgx.api_test(env, num=3, use_key=False)


def test_full_game_invariants():
    key = jax.random.PRNGKey(7)
    state = game.init(key)
    state = game.step(state, jnp.int32(DECLARE_OFFSET + TRUMP_OBENABE))

    for _ in range(36):
        mask = game.legal_action_mask(state)
        action = jnp.int32(jnp.argmax(mask))
        state = game.step(state, action)

    r = game.rewards(state)
    # Same-team players share the same reward
    assert r[0] == r[2]
    assert r[1] == r[3]
    # Zero-sum
    assert abs(float(r.sum())) < 1e-3
    # All cards collected
    assert state.cards_collected.sum() == 36
    # Total card points = 152 (last trick bonus handled in rewards, not collected)
    safe_trump = jnp.clip(state.trump, 0, 5)
    card_pts = int((state.cards_collected * MODE_SCORES[safe_trump]).sum())
    assert card_pts == 152


# ──────────────────────────────────────────────────
# Observation

def test_observe_in_trick_card_zero():
    """Card 0 (♦6) played first in a trick must appear in the in-trick observation bits.

    Regression: .at[safe].set(valid) with invalid slots mapped to index 0 would
    overwrite the True at card 0 with False, silently dropping ♦6 from the observation.
    """
    # Build a state where player 0 has just played ♦6 (card 0).
    hands = jnp.zeros((4, 36), dtype=jnp.bool_)
    hands = hands.at[1, 9].set(True)   # player 1: ♥6 (needs a card so hand is non-empty)
    hands = hands.at[2, 18].set(True)
    hands = hands.at[3, 27].set(True)

    state = GameState(
        current_player=jnp.int32(1),
        hands=hands,
        trump=jnp.int32(1),
        phase=jnp.int32(1),
        trick_cards=jnp.int32([0, -1, -1, -1]),   # player 0 played ♦6 (card 0)
        trick_leader=jnp.int32(0),
        led_suit=jnp.int32(0),
        cards_collected=jnp.zeros((4, 36), dtype=jnp.bool_),
        trick_num=jnp.int32(0),
    )

    obs = game.observe(state, jnp.int32(1))
    # in-trick bits are [72:108]; card 0 = ♦6 → index 72+0 = 72
    assert bool(obs[72]), "♦6 (card 0) must appear in the in-trick observation"


# ──────────────────────────────────────────────────
# Void-in-suit tracking

def test_void_leader_does_not_set_void():
    """The trick leader playing any card must not set void."""
    state = game.init(jax.random.PRNGKey(0))
    state = game.step(state, jnp.int32(DECLARE_OFFSET))  # declare ♦ trump
    # Player 0 leads — playing any card should not mark them void.
    card = int(jnp.argmax(state.hands[0]))
    next_state = game.step(state, jnp.int32(card))
    assert not next_state.void_in_suit.any()


def test_void_set_when_follower_plays_off_suit():
    """Follower playing a card not matching the led suit → void in led suit."""
    # Build a state where player 0 has led ♦6 (card 0) and player 1 holds no ♦ cards.
    # player 1 must play something off-suit → should be marked void in ♦ (suit 0).
    base = game.init(jax.random.PRNGKey(3))

    # Give player 0 only ♦6; give player 1 only ♥ cards; others get the rest.
    # Simplest: construct hands directly.
    hands = jnp.zeros((4, 36), dtype=jnp.bool_)
    hands = hands.at[0, 0].set(True)                       # player 0: ♦6
    hands = hands.at[1, 9].set(True)                       # player 1: ♥6
    hands = hands.at[2, 18].set(True)                      # player 2: ♠6
    hands = hands.at[3, 27].set(True)                      # player 3: ♣6

    state = GameState(
        current_player=jnp.int32(1),
        hands=hands,
        trump=jnp.int32(2),            # ♠ trump — ♦ is a plain suit
        phase=jnp.int32(1),
        trick_cards=jnp.int32([0, -1, -1, -1]),  # player 0 played ♦6
        trick_leader=jnp.int32(0),
        led_suit=jnp.int32(0),         # ♦ led
        cards_collected=jnp.zeros((4, 36), dtype=jnp.bool_),
        trick_num=jnp.int32(0),
    )

    # Player 1 plays ♥6 (card 9) — off led suit ♦.
    next_state = game.step(state, jnp.int32(9))
    assert bool(next_state.void_in_suit[1, 0]), "player 1 should be void in ♦"
    # No other voids should be set.
    assert not next_state.void_in_suit[0].any()
    assert not next_state.void_in_suit[2].any()
    assert not next_state.void_in_suit[3].any()
    assert not next_state.void_in_suit[1, 1:].any()


def test_void_not_set_trump_led_off_suit():
    """When trump is led, playing off-suit (Buur exemption) must NOT mark void in trump."""
    hands = jnp.zeros((4, 36), dtype=jnp.bool_)
    hands = hands.at[0, 0].set(True)   # player 0: ♦6 (trump, leads)
    hands = hands.at[1, 9].set(True)   # player 1: ♥6 (off-suit)
    hands = hands.at[2, 18].set(True)
    hands = hands.at[3, 27].set(True)

    state = GameState(
        current_player=jnp.int32(1),
        hands=hands,
        trump=jnp.int32(0),            # ♦ trump — ♦ is led
        phase=jnp.int32(1),
        trick_cards=jnp.int32([0, -1, -1, -1]),
        trick_leader=jnp.int32(0),
        led_suit=jnp.int32(0),         # ♦ led (= trump)
        cards_collected=jnp.zeros((4, 36), dtype=jnp.bool_),
        trick_num=jnp.int32(0),
    )

    # Player 1 plays ♥6 — trump is led, this is the Buur-exempt ambiguous case.
    next_state = game.step(state, jnp.int32(9))
    # We must NOT infer void in trump suit (suit 0) for player 1.
    assert not bool(next_state.void_in_suit[1, 0]), \
        "must not infer void in trump when trump is led (Buur exemption ambiguity)"


def test_void_accumulates_across_tricks():
    """Void flags from multiple tricks accumulate and are never cleared."""
    state = game.init(jax.random.PRNGKey(7))
    state = game.step(state, jnp.int32(DECLARE_OFFSET + 1))  # ♥ trump
    # Play two full tricks greedily; collect any voids set along the way.
    for _ in range(8):
        mask = game.legal_action_mask(state)
        state = game.step(state, jnp.int32(jnp.argmax(mask)))
    # Void flags can only be True, never go back to False.
    # Run a few more steps and check monotonicity.
    prev_void = state.void_in_suit
    for _ in range(4):
        mask = game.legal_action_mask(state)
        state = game.step(state, jnp.int32(jnp.argmax(mask)))
        assert (state.void_in_suit | prev_void == state.void_in_suit).all(), \
            "void flags must be monotonically increasing"
        prev_void = state.void_in_suit
