"""
Arena: V-MCTS challenger vs random-rollout baseline.

Games run as swapped-deal pairs: each deal is played twice with the teams
exchanged, and the significance tests run on per-pair mean scores. This
cancels deal-strength variance (which dominates single-game scores) and
the forehand advantage.

Library usage (e.g. from Colab):
    from pgx._src.games.jass_v_arena import run_arena, run_batched_arena
    scores = run_arena(params, k_v=64, games=200, hours=1)   # CPU
    scores = run_batched_arena(params, k_v=64, games=200)    # TPU/GPU

run_arena plays one game at a time (two host syncs per move — fine on CPU,
dispatch-bound on accelerators); run_batched_arena plays whole chunks of
games in lockstep inside jit and is the right harness on TPU/GPU.

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
from pgx._src.games.jass_mcts import best_action, make_search_action_fn
from pgx._src.games.jass_selfplay import policy_match
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
    """Run challenger vs baseline in swapped-deal pairs.

    Each pair replays the same deal (same PRNG key): challenger as team A in
    the first game, team B in the second. Score is always recorded from the
    challenger's perspective. Returned array has even length; consecutive
    entries (2j, 2j+1) form pair j.
    """
    scores  = []
    key     = jax.random.PRNGKey(seed)
    t_start = time.perf_counter()

    for _ in range(max_games // 2):
        if time.perf_counter() - t_start > time_budget_s:
            break
        key, pair_key = jax.random.split(key)
        scores.append(play_game(challenger, baseline, pair_key))
        scores.append(-play_game(baseline, challenger, pair_key))

        if len(scores) % 20 == 0:
            arr = np.array(scores)
            w   = (arr > 0).sum()
            l   = (arr < 0).sum()
            elapsed = time.perf_counter() - t_start
            print(f"  [{len(scores):4d}]  wins={w}  losses={l}  mean={arr.mean():+.1f}"
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

    # Inference on per-pair means: the two same-deal games cancel deal
    # strength and forehand advantage, so pair means have far lower variance
    # than single games.
    pair_means = scores.reshape(-1, 2).mean(axis=1)
    t_stat, p_t = stats.ttest_1samp(pair_means, 0)
    p_wins   = int((pair_means > 0).sum())
    p_losses = int((pair_means < 0).sum())
    binom    = stats.binomtest(p_wins, max(1, p_wins + p_losses),
                               p=0.5, alternative="two-sided")
    p_sign   = binom.pvalue

    win_rate = wins / n_eff if n_eff > 0 else float("nan")
    lo, hi   = _wilson_ci(wins, n_eff)

    def sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"\nChallenger  {label_c}  vs  Baseline  {label_b}")
    print(f"  Games : {n} ({len(pair_means)} swapped-deal pairs)"
          f"  wins={wins}  losses={losses}  ties={ties}")
    print(f"  Win%  : {100*win_rate:.1f}%  95% CI [{100*lo:.1f}%, {100*hi:.1f}%]  (per game)")
    print(f"  Score : mean={scores.mean():+.1f}  sd(game)={scores.std():.1f}"
          f"  sd(pair mean)={pair_means.std():.1f}")
    print(f"  t-test: t={t_stat:+.3f}  p={p_t:.4f}  {sig(p_t)}  (on pair means)")
    print(f"  sign  : p={p_sign:.4f}  {sig(p_sign)}  (pairs {p_wins}W/{p_losses}L)")
    sys.stdout.flush()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_arena(params, *, baseline_params=None, k_v=64, k_base=8, n_base=8,
              games=1000, hours=4.0, seed=0):
    """Run a V-MCTS challenger against a baseline agent.

    Default baseline is random-rollout MCTS (K=k_base, N=n_base). Pass
    baseline_params to instead play V-MCTS vs V-MCTS — the gated-promotion
    arena from docs/jass_plan.md Step 1, e.g.:

        run_arena(v1_params, baseline_params=v0_params, k_v=64, k_base=64)

    Args:
        params:  Flax parameter tree for the challenger's ValueNet.
        baseline_params: Optional ValueNet params for a V-MCTS baseline
            (K=k_base, N=1). None = random-rollout baseline.
        k_v:     Determinizations for V-MCTS challenger.
        k_base:  Determinizations for the baseline.
        n_base:  Rollouts per action for the random-rollout baseline
            (ignored when baseline_params is given).
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
    if baseline_params is None:
        best_action(state0, state0.current_player, jax.random.PRNGKey(0),
                    num_determinizations=k_base, num_rollouts=n_base)
    elif k_base != k_v:
        best_action(state0, state0.current_player, jax.random.PRNGKey(0),
                    num_determinizations=k_base, num_rollouts=1,
                    v_params=baseline_params, v_apply=model.apply,
                    v_scale=TARGET_SCALE)
    jax.effects_barrier()
    print(f"  Compiled in {time.perf_counter()-t0:.1f}s\n", flush=True)

    # ── Run arena ─────────────────────────────────────────────────────────
    challenger = make_v_agent(k_v, params, model)
    if baseline_params is None:
        baseline = make_random_agent(k_base, n_base)
        label_b  = f"random K={k_base} N={n_base}"
    else:
        baseline = make_v_agent(k_base, baseline_params, model)
        label_b  = f"V-MCTS K={k_base} (baseline weights)"

    label_c = f"V-MCTS K={k_v}"

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


def run_batched_arena(params, *, baseline_params=None, k_v=64, k_base=8,
                      n_base=8, games=100, chunk_pairs=25, seed=0,
                      v_apply=None, baseline_v_apply=None):
    """Batched drop-in for run_arena: same matchups, vmapped execution.

    All games in a chunk run in lockstep inside one jitted call (vmap over
    games, lax.scan over plies). Both agents' searches are evaluated on every
    board each ply and the move is selected by seat parity — 2× the per-move
    compute of run_arena, but ~chunk_pairs×2 parallelism with zero per-move
    dispatch, so it is the right harness on TPU/GPU (run_arena is
    dispatch-bound there). Same swapped-deal pairing and statistics.

    Note: identical seeds do NOT reproduce run_arena's games (different key
    plumbing), only the same kind of experiment.

    Args:
        params:  Flax parameter tree for the challenger's ValueNet.
        baseline_params: Optional ValueNet params for a V-MCTS baseline
            (K=k_base, N=1). None = random-rollout baseline (K=k_base,
            N=n_base).
        k_v:     Determinizations for the V-MCTS challenger.
        k_base:  Determinizations for the baseline.
        n_base:  Rollouts per action for the random-rollout baseline.
        games:   Total games (= 2 × number of swapped-deal pairs).
        chunk_pairs: Pairs per jitted call. Larger = more parallelism and
            memory; progress prints once per chunk.
        seed:    PRNG seed.
        v_apply: Apply function for the challenger's value leaf,
            (params, cm, hd) → (B,) scaled value. Defaults to
            ValueNet().apply. For a PolicyValueNet value head pass e.g.
            ``lambda p, cm, hd: pv_model.apply(p, cm, hd)[1]`` (define it
            once and reuse — a new callable means a new jit trace).
        baseline_v_apply: Same for the baseline (with baseline_params);
            defaults to ValueNet().apply.

    Returns:
        np.ndarray of per-game score differentials from the challenger's
        perspective, pair-adjacent (even length).
    """
    if v_apply is None or baseline_v_apply is None:
        model = ValueNet()
        v_apply = v_apply or model.apply
        baseline_v_apply = baseline_v_apply or model.apply
    challenger_fn = make_search_action_fn(
        num_determinizations=k_v, num_rollouts=1,
        v_params=params, v_apply=v_apply, v_scale=TARGET_SCALE)
    label_c = f"V-MCTS K={k_v}"
    if baseline_params is None:
        baseline_fn = make_search_action_fn(
            num_determinizations=k_base, num_rollouts=n_base)
        label_b = f"random K={k_base} N={n_base}"
    else:
        baseline_fn = make_search_action_fn(
            num_determinizations=k_base, num_rollouts=1,
            v_params=baseline_params, v_apply=baseline_v_apply,
            v_scale=TARGET_SCALE)
        label_b = f"V-MCTS K={k_base} (baseline weights)"

    print(f"Challenger : {label_c}")
    print(f"Baseline   : {label_b}")
    print(f"Games      : {games} in chunks of {chunk_pairs} pairs\n", flush=True)

    num_pairs = games // 2
    key = jax.random.PRNGKey(seed)
    chunks = []
    played_pairs = 0
    t0 = time.perf_counter()
    while played_pairs < num_pairs:
        n = min(chunk_pairs, num_pairs - played_pairs)
        key, chunk_key = jax.random.split(key)
        s = np.asarray(policy_match(challenger_fn, baseline_fn, chunk_key, n))
        chunks.append(s)
        played_pairs += n
        arr = np.concatenate(chunks)
        w, l = (arr > 0).sum(), (arr < 0).sum()
        print(f"  [{len(arr):4d}]  wins={w}  losses={l}  mean={arr.mean():+.1f}"
              f"  ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)

    scores = np.concatenate(chunks)
    print_stats(label_c, label_b, scores)
    print(f"\nTotal elapsed: {(time.perf_counter()-t0)/60:.1f} minutes")
    return scores
