"""
Value-network diagnostics.

Top-level entry points:
  - run_calibration:  generate holdout, print stats, save scatter plot.
  - perspective_probe: V from all 4 seats on one mid-game state.

See scripts/jass_v_diagnostic.py for the CLI driver.
"""

import jax
import jax.numpy as jnp
from jax import Array

from pgx._src.games.jass import Game, value_features
from pgx._src.games.jass_selfplay import collect_batch
from pgx._src.games.jass_value_net import ValueNet, TARGET_SCALE


# ── Calibration ───────────────────────────────────────────────────────────────

def run_calibration(
    params, model: ValueNet, key: Array,
    n_games: int = 2048,
) -> tuple[float, "matplotlib.figure.Figure"]:
    """Full calibration diagnostic: generate holdout, print stats, plot.

    Returns (R², matplotlib Figure).  The caller is responsible for
    saving or displaying the figure (in Colab it renders inline).
    """
    import matplotlib.pyplot as plt

    print("Generating holdout games for calibration ...")
    cm, hd, labels, alive = collect_batch(key, n_games)
    cm = cm.reshape(-1, 36, 12)
    hd = hd.reshape(-1, 20)
    y = labels.reshape(-1)
    mask = alive.reshape(-1)

    # Filter to alive positions only.
    n_alive = int(mask.sum())
    idx = jnp.where(mask, size=mask.shape[0])[0]
    cm_a = cm[idx[:n_alive]]
    hd_a = hd[idx[:n_alive]]
    y_a = y[idx[:n_alive]]

    # Predict in chunks to avoid OOM.
    chunk_size = 4096
    preds = []
    for i in range(0, n_alive, chunk_size):
        p = model.apply(params, cm_a[i:i + chunk_size], hd_a[i:i + chunk_size])
        preds.append(p * TARGET_SCALE)
    pred = jnp.concatenate(preds)

    # ── Stats ─────────────────────────────────────────────────────────────
    mse = float(((pred - y_a) ** 2).mean())
    rmse = mse ** 0.5
    var_y = float(y_a.var())
    r2 = 1.0 - mse / var_y if var_y > 0 else float("nan")

    print(f"\n{'─' * 50}")
    print(f"  Calibration Stats ({n_alive} positions)")
    print(f"{'─' * 50}")
    print(f"  Actual   : mean={float(y_a.mean()):7.2f}  std={float(y_a.std()):7.2f}")
    print(f"  Predicted: mean={float(pred.mean()):7.2f}  std={float(pred.std()):7.2f}")
    print(f"  Var(y)   : {var_y:.2f}")
    print(f"  MSE      : {mse:.2f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"  R²       : {r2:.4f}")
    print(f"{'─' * 50}\n")

    if r2 < 0.1:
        print("  ⚠ R² near zero: V has almost no predictive signal.")
    elif r2 < 0.5:
        print("  ⚠ R² is mediocre. V has some signal but high variance.")
    else:
        print("  ✓ R² is reasonable.")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(y_a, pred, s=1, alpha=0.1)
    lo = min(float(y_a.min()), float(pred.min())) - 10
    hi = max(float(y_a.max()), float(pred.max())) + 10
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x")
    ax.set_xlabel("Actual differential")
    ax.set_ylabel("Predicted differential")
    ax.set_title(f"Value Net Calibration  (R²={r2:.3f})")
    ax.legend()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    return r2, fig


# ── Perspective probe ─────────────────────────────────────────────────────────

def perspective_probe(
    params, model: ValueNet, key: Array, n_moves: int = 18,
) -> list[float]:
    """Evaluate V from all 4 seats at one mid-game state.

    In a correct implementation:
      V(p0) ≈ V(p2)   (partners)
      V(p0) ≈ -V(p1)  (opponents)
      sum of all 4 ≈ 0

    Returns the four values [V(p0), V(p1), V(p2), V(p3)].
    """
    game = Game()
    init_key, play_key = jax.random.split(key)
    state = game.init(init_key)

    # Play n_moves random moves to reach mid-game.
    for _ in range(n_moves):
        play_key, sk = jax.random.split(play_key)
        mask = game.legal_action_mask(state)
        logits = jnp.where(mask, 0.0, -1e9)
        action = jax.random.categorical(sk, logits).astype(jnp.int32)
        state = game.step(state, action)

    print(f"\n{'─' * 50}")
    print(f"  Perspective Probe (trick {int(state.trick_num)}, "
          f"trump={int(state.trump)}, current_player={int(state.current_player)})")
    print(f"{'─' * 50}")

    vals = []
    for p in range(4):
        cm, hd = value_features(state, jnp.int32(p))
        v = float(model.apply(params, cm[None], hd[None])[0] * TARGET_SCALE)
        team = "A" if p % 2 == 0 else "B"
        vals.append(v)
        print(f"  V(player {p}, team {team}) = {v:+.2f}")

    print()
    print(f"  Sum of all 4:    {sum(vals):+.2f}  (should be ≈ 0)")
    print(f"  V(p0) + V(p1):   {vals[0] + vals[1]:+.2f}  (should be ≈ 0)")
    print(f"  V(p0) - V(p2):   {vals[0] - vals[2]:+.2f}  (should be ≈ 0, partners)")
    print(f"  V(p1) - V(p3):   {vals[1] - vals[3]:+.2f}  (should be ≈ 0, partners)")
    print(f"{'─' * 50}\n")

    if abs(sum(vals)) > 20:
        print("  ⚠ Sum far from 0: possible perspective bug.")
    if abs(vals[0] - vals[2]) > 10:
        print("  ⚠ Partners disagree: possible perspective bug.")
    return vals
