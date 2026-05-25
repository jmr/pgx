import jax.numpy as jnp

from pgx.jass import State as JassState

SUITS = ["♦", "♥", "♠", "♣"]
RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
MODES = ["♦", "♥", "♠", "♣", "Obe", "Ufe", "?"]


def _make_jass_dwg(dwg, state: JassState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    W = BOARD_WIDTH * GRID_SIZE
    H = BOARD_HEIGHT * GRID_SIZE

    dwg.add(dwg.rect((0, 0), (W, H), fill=color_set.background_color))

    g = dwg.g()

    def txt(text, x, y, size=12):
        g.add(dwg.text(
            text,
            insert=(x, y),
            fill=color_set.text_color,
            font_size=f"{size}px",
            font_family="Courier",
        ))

    x = state._x
    trump = int(x.trump)
    phase = int(x.phase)
    cur = int(x.current_player)

    # Mode line
    mode_str = MODES[trump] if 0 <= trump <= 5 else "?"
    phase_str = "trump" if phase == 0 else "play"
    txt(f"Phase:{phase_str}  Mode:{mode_str}  Player:{cur}", GRID_SIZE * 0.5, GRID_SIZE * 1, size=14)

    # Hands: one row per player
    for p in range(4):
        y_base = GRID_SIZE * (2.5 + p * 2.5)
        txt(f"P{p}:", GRID_SIZE * 0.3, y_base, size=11)
        col = 0
        for c in range(36):
            if x.hands[p, c]:
                suit = c // 9
                rank = c % 9
                card_str = SUITS[suit] + RANKS[rank]
                txt(card_str, GRID_SIZE * (1.2 + col * 1.5), y_base, size=11)
                col += 1

    # Current trick
    trick_y = GRID_SIZE * 12.5
    txt("Trick:", GRID_SIZE * 0.3, trick_y, size=11)
    for i in range(4):
        c = int(x.trick_cards[i])
        if c >= 0:
            suit = c // 9
            rank = c % 9
            card_str = SUITS[suit] + RANKS[rank]
        else:
            card_str = "--"
        txt(f"P{i}:{card_str}", GRID_SIZE * (1.5 + i * 2.5), trick_y, size=11)

    return g
