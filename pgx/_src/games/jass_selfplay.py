"""
Vmapped self-play for Jass network training.

Value-only generators (same contract):

collect_batch(key, batch_size)
    Uniform-random play (generation 0).

make_v_collect_fn(v_apply, v_params, ...)(key, batch_size)
    V-greedy softmax play (generation >= 1): each legal action is scored by
    V(state after action) from the acting player's perspective and sampled
    with probability softmax(score / temperature).

Both return (cm, hd, labels, alive):
    cm    : (B, T, 36, 12) bool  card matrix per timestep
    hd    : (B, T, 20)     bool  header per timestep
    labels: (B, T)         f32   acting player's terminal differential
    alive : (B, T)         bool  False once game is terminal (label mask)

Policy+value generators (for PolicyValueNet, docs/jass_plan.md Step 2+)
additionally return per-step policy targets and legal masks:

make_search_collect_fn(...)(key, batch_size)
    Determinized-search self-play; policy target = one-hot of the chosen
    action (upgraded to visit distributions by the PUCT generator, Step 3).

make_policy_collect_fn(pv_apply, pv_params, ...)(key, batch_size)
    Fast generator: moves sampled from the policy head directly (no search).

Both return (cm, hd, labels, pi, legal, alive):
    pi    : (B, T, 43)     f32   policy target (zero on illegal actions)
    legal : (B, T, 43)     bool  legal_action_mask at each step

Flatten and filter by alive before feeding the trainer, or pass
alive.reshape(-1).astype(jnp.float32) as a sample-weight mask to avoid
dynamic shapes inside jit.
"""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from pgx._src.games.jass import CARD_SUIT, Game, NUM_ACTIONS, value_features
from pgx._src.games.jass_mcts import _action_scores, make_search_action_fn

_game = Game()
_MAX_STEPS = 38   # 2 trump-selection + 9*4 card-play steps


def random_action_fn(s, key: Array):
    """Uniform-random legal action."""
    mask   = _game.legal_action_mask(s)
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def make_v_action_fn(v_apply, v_params, *, v_scale: float = 100.0,
                     temperature: float = 10.0):
    """Build an action_fn(state, key) that plays with a value network.

    Every action's successor state is scored by V from the acting player's
    perspective; a legal action is sampled with probability
    softmax(score / temperature) (temperature in points, > 0).
    """
    all_actions = jnp.arange(NUM_ACTIONS, dtype=jnp.int32)

    def action_fn(s, k):
        next_states = jax.vmap(_game.step, in_axes=(None, 0))(s, all_actions)
        cm, hd = jax.vmap(value_features, in_axes=(0, None))(
            next_states, s.current_player
        )
        vals   = v_apply(v_params, cm, hd) * v_scale       # (A,) points
        mask   = _game.legal_action_mask(s)
        logits = jnp.where(mask, vals / temperature, -1e9)
        return jax.random.categorical(k, logits).astype(jnp.int32)

    return action_fn


def _play_one(action_fn, key: Array):
    """Run one full game with action_fn(state, key) selecting moves.

    Returns per-step features and terminal rewards.
    """
    init_key, play_key = jax.random.split(key)
    s0 = _game.init(init_key)

    def step_fn(carry, _):
        s, k = carry
        done = s.trick_num >= 9

        k, sk = jax.random.split(k)
        action = action_fn(s, sk)

        cm, hd = value_features(s, s.current_player)
        out = (cm, hd, s.current_player, ~done)

        ns = _game.step(s, action)
        # Hold state fixed once terminal so the scan stays well-defined.
        ns = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), s, ns)
        return (ns, k), out

    (final, _), (cm, hd, actor, alive) = jax.lax.scan(
        step_fn, (s0, play_key), None, length=_MAX_STEPS
    )
    # cm:    (T, 36, 12)
    # hd:    (T, 20)
    # actor: (T,) int32
    # alive: (T,) bool
    # rew:   (4,) float32

    rew = _game.rewards(final)  # (4,)
    return cm, hd, actor, alive, rew


def _collect(action_fn, key: Array, batch_size: int):
    """Run batch_size games in parallel with action_fn; label every step."""
    keys = jax.random.split(key, batch_size)
    cm, hd, actor, alive, rew = jax.vmap(
        functools.partial(_play_one, action_fn)
    )(keys)
    # rew:   (B, 4); actor: (B, T)
    # For each (b, t): labels[b, t] = rew[b, actor[b, t]]
    labels = jnp.take_along_axis(
        rew[:, jnp.newaxis, :],   # (B, 1, 4)
        actor[..., jnp.newaxis],  # (B, T, 1)
        axis=-1,
    ).squeeze(-1)                 # (B, T)
    return cm, hd, labels, alive


@functools.partial(jax.jit, static_argnames=("batch_size",))
def collect_batch(key: Array, batch_size: int):
    """Run batch_size uniform-random games in parallel; label every step.

    Returns:
        cm    : (B, T, 36, 12) bool
        hd    : (B, T, 20)     bool
        labels: (B, T)         float32 — acting player's terminal differential
        alive : (B, T)         bool    — mask out post-terminal padding
    """
    return _collect(random_action_fn, key, batch_size)


def make_v_collect_fn(v_apply, v_params, *, v_scale: float = 100.0,
                      temperature: float = 10.0):
    """Build a collect_fn(key, batch_size) that plays with a value network.

    All four seats select moves the same way: every action's successor state
    is scored by V from the acting player's perspective (same evaluation as
    the V-MCTS leaf, but without determinization — self-play states are
    fully known), and a legal action is sampled with probability
    softmax(score / temperature).

    Args:
        v_apply: Network apply function, (params, cm, hd) → scaled value.
        v_params: Network parameters (passed as a traced argument, so one
            compilation serves all generations of weights).
        v_scale: Multiplier from network output to points (TARGET_SCALE).
        temperature: Softmax temperature in points. Must be > 0; lower is
            greedier. At 10.0, actions within ~10 points of the best keep
            meaningful probability.

    Returns:
        collect_fn(key, batch_size) with the same contract as collect_batch.
    """
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _v_collect(params, key: Array, batch_size: int):
        action_fn = make_v_action_fn(v_apply, params,
                                     v_scale=v_scale, temperature=temperature)
        return _collect(action_fn, key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _v_collect(v_params, key, batch_size)

    return collect_fn


# ── Policy+value collection (PolicyValueNet training data) ────────────────────
#
# A policy_fn(state, key) → (action, pi) generalizes action_fn: pi (43,) is
# the policy training target for the step (one-hot of the chosen action for
# greedy/sampled players; the aggregated root visit distribution for PUCT).


def as_policy_fn(action_fn):
    """Lift an action_fn(state, key) → action to a policy_fn with one-hot pi."""

    def policy_fn(s, k):
        action = action_fn(s, k)
        return action, jax.nn.one_hot(action, NUM_ACTIONS, dtype=jnp.float32)

    return policy_fn


def make_policy_action_fn(pv_apply, pv_params, *, temperature: float = 1.0):
    """Build an action_fn(state, key) that samples from the policy head.

    The policy head is evaluated on the full-information value features of
    the actual state (self-play states are fully known), illegal logits are
    masked out, and a legal action is sampled from
    softmax(logits / temperature).

    Suitable for policy_match diagnostics (Step 2 success criterion:
    policy-only vs random) and as a fast data generator via
    make_policy_collect_fn.
    """

    def action_fn(s, k):
        cm, hd = value_features(s, s.current_player)
        logits, _ = pv_apply(pv_params, cm[None], hd[None])
        mask = _game.legal_action_mask(s)
        masked = jnp.where(mask, logits[0] / temperature, jnp.float32(-1e9))
        return jax.random.categorical(k, masked).astype(jnp.int32)

    return action_fn


def _play_one_pv(policy_fn, key: Array):
    """Run one full game with policy_fn; record features, pi, legal, rewards."""
    init_key, play_key = jax.random.split(key)
    s0 = _game.init(init_key)

    def step_fn(carry, _):
        s, k = carry
        done = s.trick_num >= 9

        k, sk = jax.random.split(k)
        action, pi = policy_fn(s, sk)

        cm, hd = value_features(s, s.current_player)
        legal = _game.legal_action_mask(s)
        out = (cm, hd, pi, legal, s.current_player, ~done)

        ns = _game.step(s, action)
        ns = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), s, ns)
        return (ns, k), out

    (final, _), (cm, hd, pi, legal, actor, alive) = jax.lax.scan(
        step_fn, (s0, play_key), None, length=_MAX_STEPS
    )
    rew = _game.rewards(final)  # (4,)
    return cm, hd, pi, legal, actor, alive, rew


def _collect_pv(policy_fn, key: Array, batch_size: int):
    """Run batch_size games in parallel with policy_fn; label every step."""
    keys = jax.random.split(key, batch_size)
    cm, hd, pi, legal, actor, alive, rew = jax.vmap(
        functools.partial(_play_one_pv, policy_fn)
    )(keys)
    labels = jnp.take_along_axis(
        rew[:, jnp.newaxis, :],   # (B, 1, 4)
        actor[..., jnp.newaxis],  # (B, T, 1)
        axis=-1,
    ).squeeze(-1)                 # (B, T)
    return cm, hd, labels, pi, legal, alive


@functools.partial(jax.jit, static_argnames=("batch_size",))
def collect_pv_batch(key: Array, batch_size: int):
    """Uniform-random play with the PV contract (one-hot pi of random moves).

    The policy targets carry no signal (behavior cloning of random play);
    useful as a smoke-test generator and for value-only pretraining with
    policy_weight=0.

    Returns:
        (cm, hd, labels, pi, legal, alive) — see module docstring.
    """
    return _collect_pv(as_policy_fn(random_action_fn), key, batch_size)


def make_search_collect_fn(v_apply=None, v_params=None, *,
                           num_determinizations: int = 8,
                           num_rollouts: int = 8,
                           v_scale: float = 100.0,
                           temperature: float = None):
    """Build a collect_fn(key, batch_size) playing with determinized search.

    Every seat runs best_action's evaluation per move (random-rollout
    leaves by default, V leaves when v_apply is given). The policy target
    pi is the one-hot of the search's ARGMAX action — the Step 2 training
    target, upgraded to visit distributions in Step 3. The *played* action
    is the argmax too when temperature=None, or sampled with probability
    softmax(scores/temperature) when temperature > 0 (exploration).
    Played action and target are deliberately decoupled: training on the
    sampled action instead would teach the policy the exploration noise —
    at temperature 10 that target is near-uniform over decent moves, and
    a policy head trained on it stays at uniform-over-legal (measured:
    eval CE 1.32 → 1.31 over 500 epochs).

    Much more expensive per game than collect_batch / make_v_collect_fn:
    each move runs K determinizations × A actions of leaf evaluation.
    Use V leaves (num_rollouts=1) and an accelerator for bulk generation.

    Returns:
        collect_fn(key, batch_size) → (cm, hd, labels, pi, legal, alive).
    """

    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _search_collect(params, key: Array, batch_size: int):
        policy_fn = make_search_policy_fn(
            v_apply, params,
            num_determinizations=num_determinizations,
            num_rollouts=num_rollouts, v_scale=v_scale,
            temperature=temperature)
        return _collect_pv(policy_fn, key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _search_collect(v_params, key, batch_size)

    return collect_fn


def make_search_policy_fn(v_apply=None, v_params=None, *,
                          num_determinizations: int = 8,
                          num_rollouts: int = 8,
                          v_scale: float = 100.0,
                          temperature: float = None):
    """policy_fn(state, key) → (action, pi) for search self-play.

    pi is the one-hot of the search's argmax (the training target);
    the returned action is that argmax when temperature=None, or a
    softmax(scores/temperature) sample for exploration. The search
    consumes the first of split(key); the sample the second.
    """

    def policy_fn(s, k):
        k_search, k_sample = jax.random.split(k)
        scores, mask = _action_scores(
            s, s.current_player, k_search,
            num_determinizations, num_rollouts,
            v_params, v_apply, v_scale)
        best = jnp.argmax(jnp.where(mask, scores, jnp.float32(-jnp.inf)))
        pi = jax.nn.one_hot(best, NUM_ACTIONS, dtype=jnp.float32)
        if temperature is None:
            return best.astype(jnp.int32), pi
        logits = jnp.where(mask, scores / temperature, jnp.float32(-1e9))
        action = jax.random.categorical(k_sample, logits).astype(jnp.int32)
        return action, pi

    return policy_fn


def make_policy_collect_fn(pv_apply, pv_params, *, temperature: float = 1.0):
    """Build a collect_fn(key, batch_size) sampling from the policy head.

    The fast generator (no search): same speed class as make_v_collect_fn.
    pi is the one-hot of the sampled action, so training the policy head on
    this data is behavior cloning of the generator itself — useful for the
    value target mostly; prefer search-generated pi for policy improvement.

    Returns:
        collect_fn(key, batch_size) → (cm, hd, labels, pi, legal, alive).
    """
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _policy_collect(params, key: Array, batch_size: int):
        action_fn = make_policy_action_fn(pv_apply, params,
                                          temperature=temperature)
        return _collect_pv(as_policy_fn(action_fn), key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _policy_collect(pv_params, key, batch_size)

    return collect_fn


# ── Suit-permutation data augmentation (docs/jass.md) ─────────────────────────
#
# Relabeling suits maps a position to an exactly equivalent one: permute the
# 9-row suit blocks of the card matrix, the card actions 0–35 and the
# trump-declare actions 36–39 of pi/legal, and the trump-suit one-hot in the
# header. Valid permutations: all 4! when no trump suit is set (trump
# selection steps, Obenabe, Undeufe); the 3! fixing the trump suit in trump
# modes (the header then stays unchanged by construction).


def sample_suit_permutation(key: Array, hd: Array) -> Array:
    """Sample a valid suit relabeling sigma (old suit → new suit) for a step.

    Args:
        hd: (20,) header of the step; bits [0:4] are the trump-suit one-hot
            (all zero during trump selection and in Obenabe/Undeufe).

    Returns:
        (4,) int32 permutation; sigma[s] is the new label of old suit s.
        Fixes the trump suit when one is set.
    """
    k_full, k_three = jax.random.split(key)
    full = jax.random.permutation(k_full, 4).astype(jnp.int32)

    is_trump_mode = hd[:4].any()
    trump_suit = jnp.argmax(hd[:4])
    # Non-trump suits in ascending order (stable sort puts the True last).
    others = jnp.argsort(jnp.arange(4) == trump_suit)[:3].astype(jnp.int32)
    q = jax.random.permutation(k_three, 3)
    fixed = jnp.arange(4, dtype=jnp.int32).at[others].set(others[q])

    return jnp.where(is_trump_mode, fixed, full)


def apply_suit_permutation(sigma: Array, cm: Array, hd: Array,
                           pi: Array = None, legal: Array = None):
    """Apply a suit relabeling to one step's features (and policy targets).

    Args:
        sigma: (4,) suit permutation, old → new.
        cm: (36, 12), hd: (20,); optionally pi: (43,) and legal: (43,).

    Returns:
        (cm, hd) or (cm, hd, pi, legal), relabeled.
    """
    inv = jnp.argsort(sigma)                       # new suit → old suit
    ranks = jnp.arange(36, dtype=jnp.int32) % 9
    card_gather = inv[CARD_SUIT] * 9 + ranks       # new card → old card

    cm_new = cm[card_gather]
    hd_new = hd.at[:4].set(hd[:4][inv])

    if pi is None:
        return cm_new, hd_new

    # Card actions 0–35 follow the cards; declare actions 36–39 follow the
    # suits; Obenabe/Undeufe/Schiebe (40–42) are unchanged.
    action_gather = jnp.concatenate([
        card_gather,
        36 + inv,
        jnp.arange(40, NUM_ACTIONS, dtype=jnp.int32),
    ])
    return cm_new, hd_new, pi[action_gather], legal[action_gather]


def augment_suits(key: Array, cm: Array, hd: Array,
                  pi: Array = None, legal: Array = None):
    """Apply an independent random suit relabeling to each step of a batch.

    Args:
        cm: (N, 36, 12), hd: (N, 20); optionally pi/legal: (N, 43).

    Returns:
        Same-shape (cm, hd) or (cm, hd, pi, legal).
    """
    keys = jax.random.split(key, cm.shape[0])
    sigmas = jax.vmap(sample_suit_permutation)(keys, hd)
    if pi is None:
        return jax.vmap(apply_suit_permutation)(sigmas, cm, hd)
    return jax.vmap(apply_suit_permutation)(sigmas, cm, hd, pi, legal)


# ── Policy arena (no search) ───────────────────────────────────────────────────


def _play_score(action_fn, key: Array):
    """Run one game with action_fn; return only the rewards (4,)."""

    def step_fn(carry, _):
        s, k = carry
        done = s.trick_num >= 9
        k, sk = jax.random.split(k)
        action = action_fn(s, sk)
        ns = _game.step(s, action)
        ns = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), s, ns)
        return (ns, k), None

    init_key, play_key = jax.random.split(key)
    s0 = _game.init(init_key)
    (final, _), _ = jax.lax.scan(step_fn, (s0, play_key), None,
                                 length=_MAX_STEPS)
    return _game.rewards(final)


@functools.partial(jax.jit, static_argnames=("action_fn_a", "action_fn_b",
                                             "num_pairs"))
def policy_match(action_fn_a, action_fn_b, key: Array, num_pairs: int):
    """Head-to-head policy arena with no search, fully vmapped.

    Plays num_pairs swapped-deal pairs: in the first game of a pair, policy
    A holds seats {0, 2} and B holds {1, 3}; the second game replays the
    same deal (same PRNG key) with seats exchanged. Useful for cheap
    policy-strength diagnostics, e.g. V-greedy vs random.

    Args:
        action_fn_a / action_fn_b: action_fn(state, key) → action, e.g.
            random_action_fn or make_v_action_fn(...). Must be hashable
            (they are jit static args); module-level functions and
            make_v_action_fn results qualify.
        key: PRNG key.
        num_pairs: Number of deal pairs (2 × num_pairs games).

    Returns:
        (2 * num_pairs,) float32 — per-game score from A's perspective,
        pair-adjacent, ready for jass_v_arena.print_stats.
    """

    def seat_select(a_seats_even):
        def action_fn(s, k):
            ka, kb = jax.random.split(k)
            act_a = action_fn_a(s, ka)
            act_b = action_fn_b(s, kb)
            a_to_move = (s.current_player % 2 == 0) == a_seats_even
            return jnp.where(a_to_move, act_a, act_b)
        return action_fn

    keys = jax.random.split(key, num_pairs)
    # rewards[0] is seat 0's differential = team {0,2}'s.
    s_ab = jax.vmap(lambda k: _play_score(seat_select(True), k)[0])(keys)
    s_ba = jax.vmap(lambda k: _play_score(seat_select(False), k)[0])(keys)
    # First game: A is team {0,2} → +rewards[0]. Swapped game: A is team
    # {1,3} → −rewards[0].
    return jnp.stack([s_ab, -s_ba], axis=1).reshape(-1)


def profile_collect_fn(collect_fn, *, start=16, max_batch=None, reps=2,
                       seed=0, stop_when_worse=True, patience=1, verbose=True):
    """Sweep games-per-chip (batch_size) to find the ms/game OPTIMUM.

    Stage-1 (data-collection) is the per-generation bottleneck and the one
    embarrassingly-parallel stage (docs/jass_plan.md, "HOW TO SCALE STAGE 1").
    Before adding chips, find the per-chip batch that minimises ms/game. Run on
    a **1x1** TPU: the per-chip properties it measures transfer directly to each
    pmap shard of the production collector.

    The signal is ms/game vs batch_size, and it is NOT monotonic: the PUCT
    collector is bound by the on-chip-memory (VMEM) working set of the mctx
    tree ([batch, K, sims] of GameState embeddings), not HBM. So ms/game falls
    as the batch fills the chip, hits a sharp minimum when the tree still fits
    fast scratchpad, then jumps *up* (superlinearly) once it spills to HBM —
    while HBM usage stays tiny, so OOM never comes. Measured 2026-06-21 at
    K=8/sims=128: min at B=64 (563 ms/game), then 128→2048, 256→2885 (a peak,
    not a plateau). The optimum couples B,K,sims (≈ a fixed tree working set,
    B*K≈512) — re-profile when num_determinizations or num_simulations change.

    Therefore stop at the minimum (stop_when_worse), do NOT "double until OOM":
    OOM is unreachable here and the post-minimum points are exponentially
    expensive (the 2026-06-21 unbounded run wasted ~6 h past a knee visible at
    the second point).

    Generic over any collector with the (key, batch_size) contract
    (make_puct_collect_fn, make_search_collect_fn, make_policy_collect_fn, ...).

    Args:
        collect_fn: collector returned by one of the make_*_collect_fn builders.
        start: smallest batch_size to try (doubles each step).
        max_batch: hard cap on batch_size (None = no cap; rely on
            stop_when_worse / OOM to terminate).
        reps: measured runs per batch_size; the min is reported (least noise).
            The one-time compile (static batch_size recompiles per size) is
            timed separately and excluded.
        seed: PRNG seed.
        stop_when_worse: stop once ms/game has risen above the running minimum
            for `patience` consecutive steps (the optimum is behind us). True
            by default — this is what keeps the sweep from running away on a
            VMEM-bound workload. Set False to force a full sweep to max_batch.
        patience: consecutive worse-than-best steps tolerated before stopping.
            1 (default) stops right after the first regression — cheapest, but
            a single noisy high sample can trip it; raise to 2 with low `reps`
            if the curve is noisy (each extra step past the min is costly).
        verbose: print a table row per batch_size.

    Returns:
        list of per-batch dicts {batch_size, compile_s, run_s, ms_per_game,
        peak_gb, limit_gb}; an OOM row has run_s=None and an 'error' key.
    """
    import time

    dev = jax.local_devices()[0]

    def _hbm():  # peak_bytes_in_use is a session high-water mark (no reset API)
        s = dev.memory_stats() or {}
        return s.get("peak_bytes_in_use", 0), s.get("bytes_limit", 0)

    key = jax.random.PRNGKey(seed)
    rows = []
    if verbose:
        print(f"{'B':>7} {'compile_s':>10} {'run_s':>8} {'ms/game':>9} "
              f"{'peak_GB':>8} {'limit_GB':>9}", flush=True)
    t_all = time.time()
    B = start
    best_ms, best_b, worse = float("inf"), None, 0
    while max_batch is None or B <= max_batch:
        ks = jax.random.split(jax.random.fold_in(key, B), reps + 1)
        try:
            t0 = time.time()
            jax.block_until_ready(collect_fn(ks[0], B))  # compile + warm
            compile_s = time.time() - t0
            run_s = float("inf")
            for r in range(reps):
                t0 = time.time()
                jax.block_until_ready(collect_fn(ks[r + 1], B))
                run_s = min(run_s, time.time() - t0)
            peak, limit = _hbm()
            row = dict(batch_size=B, compile_s=compile_s, run_s=run_s,
                       ms_per_game=1000 * run_s / B,
                       peak_gb=peak / 1e9, limit_gb=limit / 1e9)
            rows.append(row)
            if verbose:
                el = time.time() - t_all
                print(f"{B:>7} {compile_s:>10.1f} {run_s:>8.2f} "
                      f"{row['ms_per_game']:>9.2f} {row['peak_gb']:>8.2f} "
                      f"{row['limit_gb']:>9.2f}   [{el:.0f}s elapsed]", flush=True)
        except Exception as e:  # noqa: BLE001 — RESOURCE_EXHAUSTED etc.
            rows.append(dict(batch_size=B, compile_s=None, run_s=None,
                             ms_per_game=None, peak_gb=None, limit_gb=None,
                             error=str(e)[:160]))
            if verbose:
                print(f"{B:>7}  OOM/err at B={B} (rare — this workload is "
                      f"VMEM-bound, not HBM): {str(e)[:80]}", flush=True)
            break
        if row["ms_per_game"] < best_ms:
            best_ms, best_b, worse = row["ms_per_game"], B, 0
        elif stop_when_worse:
            worse += 1
            if worse >= patience:
                if verbose:
                    print(f"        stop: ms/game past its min "
                          f"({best_ms:.0f} @ B={best_b}) for {worse} step(s) "
                          f"-> optimum is B={best_b}", flush=True)
                break
        B *= 2
    return rows
