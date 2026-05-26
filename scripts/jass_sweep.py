"""
Sweep (K, N) configurations against the K=8, N=8 baseline.

Each pair of consecutive games uses the same random sequence but swaps which
team is the challenger to cancel out the forehand (trump-selection) advantage.

Statistical tests (following JassTheRipper Arena.java):
  - Paired t-test on score differences (challenger − baseline per game)
  - Two-sided sign test (binomial) on wins vs losses, excluding ties

Note: ties are theoretically impossible in Jass (total = 157 points, odd).
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from pgx._src.games.jass import Game, DECLARE_OFFSET
from pgx._src.games.jass_mcts import best_action

game = Game()
BASELINE_K, BASELINE_N = 8, 8


# ── Game runner ────────────────────────────────────────────────────────────────

def play_game(K_a, N_a, K_b, N_b, key):
    """Play one game. Team A = players {0,2}, Team B = players {1,3}.

    Returns team A's reward (positive = team A won, range ±157).
    """
    key, init_key = jax.random.split(key)
    state = game.init(init_key)
    for _ in range(40):
        if bool(game.is_terminal(state)):
            break
        p = int(state.current_player)
        key, subkey = jax.random.split(key)
        if p in (0, 2):
            action = best_action(state, jnp.int32(p), subkey,
                                 num_determinizations=K_a, num_rollouts=N_a)
        else:
            action = best_action(state, jnp.int32(p), subkey,
                                 num_determinizations=K_b, num_rollouts=N_b)
        state = game.step(state, action)
    return float(game.rewards(state)[0])


# ── Matchup runner ─────────────────────────────────────────────────────────────

def run_matchup(K_c, N_c, max_games, time_budget_s, seed=0):
    """Run challenger (K_c, N_c) vs baseline, alternating sides.

    Odd-indexed games: challenger = team A (forehand / trump selection advantage).
    Even-indexed games: challenger = team B.
    Score is always recorded from the challenger's perspective.

    Returns: np.ndarray of challenger score differences (shape [n_games]).
    """
    scores = []
    key = jax.random.PRNGKey(seed)
    t_start = time.perf_counter()

    for i in range(max_games):
        if time.perf_counter() - t_start > time_budget_s:
            break
        key, subkey = jax.random.split(key)
        if i % 2 == 0:
            score = play_game(K_c, N_c, BASELINE_K, BASELINE_N, subkey)
        else:
            score = -play_game(BASELINE_K, BASELINE_N, K_c, N_c, subkey)
        scores.append(score)

        if (i + 1) % 20 == 0:
            arr = np.array(scores)
            w = (arr > 0).sum()
            l = (arr < 0).sum()
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1:4d}]  wins={w}  losses={l}  mean={arr.mean():+.1f}"
                  f"  ({elapsed:.0f}s elapsed)", flush=True)

    return np.array(scores)


# ── Stats ──────────────────────────────────────────────────────────────────────

def _wilson_ci(wins, n, alpha=0.05):
    """Two-sided Wilson confidence interval for a proportion."""
    if n == 0:
        return float('nan'), float('nan')
    z = stats.norm.ppf(1 - alpha / 2)
    p = wins / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def print_stats(K_c, N_c, scores):
    n = len(scores)
    wins   = int((scores > 0).sum())
    losses = int((scores < 0).sum())
    ties   = n - wins - losses
    n_eff  = wins + losses

    # Paired t-test: is mean score diff significantly != 0?
    t_stat, p_t = stats.ttest_1samp(scores, 0)

    # Two-sided sign test (binomial H0: p_win = 0.5, excluding ties)
    binom = stats.binomtest(wins, n_eff, p=0.5, alternative='two-sided')
    p_sign = binom.pvalue

    win_rate = wins / n_eff if n_eff > 0 else float('nan')
    lo, hi   = _wilson_ci(wins, n_eff)

    def sig(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    print(f"\nChallenger  K={K_c:2d}  N={N_c}  vs  Baseline K={BASELINE_K} N={BASELINE_N}")
    print(f"  Games : {n}  wins={wins}  losses={losses}  ties={ties}")
    print(f"  Win%  : {100*win_rate:.1f}%  95% CI [{100*lo:.1f}%, {100*hi:.1f}%]")
    print(f"  Score : mean={scores.mean():+.1f}  sd={scores.std():.1f}"
          f"  (range {scores.min():.0f}..{scores.max():.0f})")
    print(f"  t-test: t={t_stat:+.3f}  p={p_t:.4f}  {sig(p_t)}")
    print(f"  sign  : p={p_sign:.4f}  {sig(p_sign)}")
    sys.stdout.flush()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Compile all (K, N) combinations up front so timing isn't polluted.
    print("Compiling all (K, N) combinations ...", flush=True)
    state0 = game.init(jax.random.PRNGKey(0))
    state0 = game.step(state0, jnp.int32(DECLARE_OFFSET))
    t0 = time.perf_counter()
    for K in (4, 8, 16, 32):
        for N in (2, 4, 8):
            best_action(state0, state0.current_player, jax.random.PRNGKey(0),
                        num_determinizations=K, num_rollouts=N)
    jax.effects_barrier()
    print(f"Compiled in {time.perf_counter()-t0:.1f}s\n", flush=True)

    COMBOS = [
        (K, N)
        for K in (4, 8, 16, 32)
        for N in (2, 4, 8)
        if (K, N) != (BASELINE_K, BASELINE_N)
    ]

    TOTAL_BUDGET_S  = 4 * 3600        # 4 hours
    PER_COMBO_S     = TOTAL_BUDGET_S / len(COMBOS)
    MAX_GAMES       = 1000

    print(f"Baseline  : K={BASELINE_K}  N={BASELINE_N}")
    print(f"Combos    : {len(COMBOS)}  ({COMBOS})")
    print(f"Budget    : {PER_COMBO_S/60:.0f} min per combo  (max {MAX_GAMES} games)\n",
          flush=True)

    sweep_start = time.perf_counter()
    for idx, (K_c, N_c) in enumerate(COMBOS):
        remaining = TOTAL_BUDGET_S - (time.perf_counter() - sweep_start)
        budget    = min(PER_COMBO_S, remaining / max(1, len(COMBOS) - idx))

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(COMBOS)}]  K={K_c}  N={N_c}  (budget {budget/60:.1f} min)",
              flush=True)

        scores = run_matchup(K_c, N_c, max_games=MAX_GAMES, time_budget_s=budget)
        print_stats(K_c, N_c, scores)

    elapsed = time.perf_counter() - sweep_start
    print(f"\n\nTotal elapsed: {elapsed/60:.1f} minutes")
