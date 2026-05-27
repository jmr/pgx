"""
Arena: V-MCTS challenger vs random-rollout baseline.

Library usage (e.g. from Colab):
    from pgx._src.games.jass_v_arena import run_arena
    scores = run_arena(params, k_v=64, games=200, hours=1)

CLI usage:
    python scripts/jass_v_arena.py --weights jass_v_weights.msgpack
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from pgx._src.games.jass import DECLARE_OFFSET, Game
from pgx._src.games.jass_mcts import best_action
from pgx._src.games.jass_value_net import TARGET_SCALE, ValueNet

_game = Game()


# ── Agent factories ────────────────────────────────────────────────────────────

def make_random_agent(K: int, N: int):
    """Agent using random rollouts."""
    def agent(state, player_id, key):
        return best_action(state, player_id, key,
                           num_determinizations=K, num_rollouts=N)
    return agent


def make_v_agent(K: int, params, model: ValueNet):
    """Agent using a trained value network at the leaf."""
    def agent(state, player_id, key):
        return best_action(state, player_id, key,
                           num_determinizations=K, num_rollouts=1,
                           v_params=params, v_apply=model.apply,
                           v_scale=TARGET_SCALE)
    return agent


# ── Game runner ────────────────────────────────────────────────────────────────

def play_game(agent_a, agent_b, key):
    """Play one game. Team A = players {0,2}, Team B = players {1,3}.

    Returns team A's score differential (positive = team A won).
    """
    key, init_key = jax.random.split(key)
    state = _game.init(init_key)
    for _ in range(40):
        if bool(_game.is_terminal(state)):
            break
        p = int(state.current_player)
        key, subkey = jax.random.split(key)
        agent = agent_a if p in (0, 2) else agent_b
        state = _game.step(state, agent(state, jnp.int32(p), subkey))
    return float(_game.rewards(state)[0])


# ── Matchup runner ─────────────────────────────────────────────────────────────

def _run_matchup(challenger, baseline, max_games: int, time_budget_s: float, seed: int = 0):
    """Run challenger vs baseline, alternating which team each plays.

    Even-indexed games: challenger = team A.
    Odd-indexed games:  challenger = team B.
    Score always recorded from the challenger's perspective.
    """
    scores  = []
    key     = jax.random.PRNGKey(seed)
    t_start = time.perf_counter()

    for i in range(max_games):
        if time.perf_counter() - t_start > time_budget_s:
            break
        key, subkey = jax.random.split(key)
        if i % 2 == 0:
            score = play_game(challenger, baseline, subkey)
        else:
            score = -play_game(baseline, challenger, subkey)
        scores.append(score)

        if (i + 1) % 20 == 0:
            arr = np.array(scores)
            w   = (arr > 0).sum()
            l   = (arr < 0).sum()
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1:4d}]  wins={w}  losses={l}  mean={arr.mean():+.1f}"
                  f"  ({elapsed:.0f}s elapsed)", flush=True)

    return np.array(scores)


# ── Stats ──────────────────────────────────────────────────────────────────────

def _wilson_ci(wins, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    z      = stats.norm.ppf(1 - alpha / 2)
    p      = wins / n
    denom  = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def print_stats(label_c: str, label_b: str, scores: np.ndarray):
    """Print arena statistics: win rate, CI, t-test, sign test."""
    n      = len(scores)
    wins   = int((scores > 0).sum())
    losses = int((scores < 0).sum())
    ties   = n - wins - losses
    n_eff  = wins + losses

    t_stat, p_t = stats.ttest_1samp(scores, 0)
    binom        = stats.binomtest(wins, n_eff, p=0.5, alternative="two-sided")
    p_sign       = binom.pvalue
    win_rate     = wins / n_eff if n_eff > 0 else float("nan")
    lo, hi       = _wilson_ci(wins, n_eff)

    def sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"\nChallenger  {label_c}  vs  Baseline  {label_b}")
    print(f"  Games : {n}  wins={wins}  losses={losses}  ties={ties}")
    print(f"  Win%  : {100*win_rate:.1f}%  95% CI [{100*lo:.1f}%, {100*hi:.1f}%]")
    print(f"  Score : mean={scores.mean():+.1f}  sd={scores.std():.1f}"
          f"  (range {scores.min():.0f}..{scores.max():.0f})")
    print(f"  t-test: t={t_stat:+.3f}  p={p_t:.4f}  {sig(p_t)}")
    print(f"  sign  : p={p_sign:.4f}  {sig(p_sign)}")
    sys.stdout.flush()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_arena(params, *, k_v=64, k_base=8, n_base=8,
              games=1000, hours=4.0, seed=0):
    """Run V-MCTS (with value net) vs random-rollout MCTS arena.

    Args:
        params:  Flax parameter tree from a trained ValueNet.
        k_v:     Determinizations for V-MCTS challenger.
        k_base:  Determinizations for random-rollout baseline.
        n_base:  Rollouts per action for baseline.
        games:   Maximum number of games to play.
        hours:   Time budget in hours.
        seed:    PRNG seed.

    Returns:
        np.ndarray of per-game score differentials from the challenger's
        perspective (positive = challenger won).
    """
    model = ValueNet()

    # ── Warm-up compilation ───────────────────────────────────────────────
    print("Compiling ...", flush=True)
    state0 = _game.init(jax.random.PRNGKey(0))
    state0 = _game.step(state0, jnp.int32(DECLARE_OFFSET))
    t0     = time.perf_counter()
    best_action(state0, state0.current_player, jax.random.PRNGKey(0),
                num_determinizations=k_v, num_rollouts=1,
                v_params=params, v_apply=model.apply, v_scale=TARGET_SCALE)
    best_action(state0, state0.current_player, jax.random.PRNGKey(0),
                num_determinizations=k_base, num_rollouts=n_base)
    jax.effects_barrier()
    print(f"  Compiled in {time.perf_counter()-t0:.1f}s\n", flush=True)

    # ── Run arena ─────────────────────────────────────────────────────────
    challenger = make_v_agent(k_v, params, model)
    baseline   = make_random_agent(k_base, n_base)

    label_c = f"V-MCTS K={k_v}"
    label_b = f"random K={k_base} N={n_base}"

    print(f"Challenger : {label_c}")
    print(f"Baseline   : {label_b}")
    print(f"Budget     : {hours:.1f} h  (max {games} games)\n", flush=True)

    scores = _run_matchup(challenger, baseline,
                          max_games=games,
                          time_budget_s=hours * 3600,
                          seed=seed)
    print_stats(label_c, label_b, scores)
    print(f"\nTotal elapsed: {(time.perf_counter()-t0)/60:.1f} minutes")
    return scores
