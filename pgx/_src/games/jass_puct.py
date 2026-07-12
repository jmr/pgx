"""Determinized PUCT for Jass via mctx (Option B) — the AlphaZero search.

For each of K determinizations of the current information state, run a
batched tree search — classical full-width PUCT via `mctx.muzero_policy`
by default (`search_variant="muzero"`, the JTR-style operator and standing
collector since the 2026-07-06 pivot), or the retired
`mctx.gumbel_muzero_policy` (`search_variant="gumbel"`) — with the PolicyValueNet
supplying priors and leaf values, and the real game engine (`Game.step`)
as the dynamics. The K trees are aggregated by SUMMING ROOT
VISIT COUNTS and acting on the summed counts — the load-bearing choice from
docs/jass_plan.md (Q-sum aggregation neutralizes the tree policy). That
holds for NET-PRIOR search only: under flat priors visits stay ~uniform
and the visit readout picks near-noise, so `readout="qsum"` provides a
JTR-style Q aggregation instead — visit-weighted MEAN Q across the K
trees (2026-07-06 probe log; see `_qsum_scores` for why mean, not sum).

Grounded-teacher knobs (2026-07-06, the gen-9 fixed-point escape —
see the plan's NEXT block): `prior_mix_uniform` mixes the net's priors
toward uniform-over-legal at every node; `rollout_value_weight` blends
leaf values toward a uniform-random playout-to-terminal return (real
points). At 1.0/1.0 the search is classical, net-free determinized
MCTS (per-world trees over sampled determinizations, then aggregated
— PIMC family, like JTR's POWERFUL; not single-tree ISMCTS).

Sign conventions (mctx backs up q(parent, a) = reward + discount * v(child)):
- every node's value is from the perspective of the player to move there
  (the net's value is already acting-player-relative);
- reward on an edge is from the parent mover's perspective (nonzero only
  on the step that ends the game);
- discount is +1 when the child mover is on the parent mover's team,
  -1 otherwise (teams are seat parity: {0,2} vs {1,3}), and 0 once the
  game is over.

Everything is pure JAX: usable under jit and vmapped over games (the inner
mctx batch dimension is the K determinizations).
"""

import functools

import jax
import jax.numpy as jnp
import mctx
from jax import Array

from pgx._src.games.jass import Game, GameState, NUM_ACTIONS, value_features
from pgx._src.games.jass_mcts import sample_determinization

_game = Game()

_ILLEGAL = jnp.float32(-1e9)


def _qsum_scores(visits: Array, qvalues: Array, legal: Array) -> Array:
    """Visit-weighted MEAN Q per action across the K trees.

    (K, A) visits/qvalues → (A,) scores, −inf where illegal or unvisited
    in every tree (argmax-safe as-is). Mean, NOT JTR's raw score-sum:
    our Q is a ±157 points differential, and Σ N·Q inverts preferences
    whenever Q<0 — it scores a lightly-visited terrible action above a
    heavily-visited slightly-losing one (measured: −61 vs gen-9 raw,
    2026-07-06, vs −24.3 for the visits readout). JTR can sum because
    its per-leaf scores are non-negative (0..157).
    """
    n = visits.sum(axis=0)                               # (A,)
    q = (visits * qvalues).sum(axis=0) / n.clip(1.0)     # (A,)
    return jnp.where(legal & (n > 0), q, -jnp.inf)


def _hold_if(done: Array, old: GameState, new: GameState) -> GameState:
    """Per-leaf where(done, old, new) with done broadcast over leading dim."""
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
        old, new,
    )


def _rollout_value(states: GameState, key: Array) -> Array:
    """Uniform-random playout to terminal; points for each state's mover.

    The classical (grounded) leaf evaluation: play random legal moves to
    the end of the game and score the real outcome from the perspective
    of the player to move at `states` — no learned function involved.
    Already-terminal states are held in place and scored as they stand
    (callers zero terminal values, matching `_pv_eval`'s convention).
    """
    movers = states.current_player                       # (B,)

    def cond(carry):
        s, _ = carry
        return jnp.any(s.trick_num < 9)

    def body(carry):
        s, k = carry
        k, k_act = jax.random.split(k)
        legal = jax.vmap(_game.legal_action_mask)(s)     # (B, 43)
        action = jax.random.categorical(
            k_act, jnp.where(legal, 0.0, _ILLEGAL))      # (B,)
        done = s.trick_num >= 9
        next_s = jax.vmap(_game.step)(s, action.astype(jnp.int32))
        return _hold_if(done, s, next_s), k

    final, _ = jax.lax.while_loop(cond, body, (states, key))
    rewards = jax.vmap(_game.rewards)(final)             # (B, 4)
    return jnp.take_along_axis(
        rewards, movers[:, None], axis=1).squeeze(-1)


def _pv_eval(
    pv_apply,
    pv_params,
    states: GameState,
    v_scale: float,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    rollout_key: Array = None,
):
    """Evaluate the net on a batch of states from each mover's perspective.

    prior_mix_uniform / rollout_value_weight are the grounded-teacher
    knobs (2026-07-06 fixed-point escape; see docs/jass_plan.md). Both
    must be trace-time Python floats: they gate computation with `if`,
    so a value of 0.0 costs nothing.

    Returns:
        logits: (B, 43) masked to the legal actions of each state.
        value : (B,) in points, 0 for terminal states.
        legal : (B, 43) bool.
    """
    cm, hd = jax.vmap(value_features)(states, states.current_player)
    logits, value = pv_apply(pv_params, cm, hd)
    legal = jax.vmap(_game.legal_action_mask)(states)
    logits = jnp.where(legal, logits, _ILLEGAL)
    if prior_mix_uniform:
        lam = prior_mix_uniform
        probs = jax.nn.softmax(logits, axis=-1)
        n_legal = legal.sum(axis=-1, keepdims=True).clip(1)
        uniform = jnp.where(legal, 1.0 / n_legal, 0.0)
        logits = jnp.where(
            legal,
            jnp.log(((1.0 - lam) * probs + lam * uniform).clip(1e-9)),
            _ILLEGAL)
    value = value * v_scale
    if rollout_value_weight:
        w = rollout_value_weight
        value = (1.0 - w) * value + w * _rollout_value(states, rollout_key)
    done = states.trick_num >= 9
    value = jnp.where(done, 0.0, value)
    return logits, value, legal


def _make_recurrent_fn(pv_apply, v_scale: float,
                       prior_mix_uniform: float = 0.0,
                       rollout_value_weight: float = 0.0):
    """Build the mctx recurrent_fn over batched GameState embeddings."""

    def recurrent_fn(params, rng_key, action, states: GameState):
        prev_player = states.current_player           # (B,)
        prev_done = states.trick_num >= 9             # (B,)

        next_states = jax.vmap(_game.step)(states, action.astype(jnp.int32))
        next_states = _hold_if(prev_done, states, next_states)

        # Terminal reward from the parent mover's perspective; zero if the
        # game was already over before this edge (no double counting).
        rewards = jax.vmap(_game.rewards)(next_states)    # (B, 4)
        reward = jnp.take_along_axis(
            rewards, prev_player[:, None], axis=1).squeeze(-1)
        reward = jnp.where(prev_done, 0.0, reward)

        logits, value, _ = _pv_eval(
            pv_apply, params, next_states, v_scale,
            prior_mix_uniform=prior_mix_uniform,
            rollout_value_weight=rollout_value_weight,
            rollout_key=rng_key)

        done = next_states.trick_num >= 9
        same_team = (next_states.current_player % 2) == (prev_player % 2)
        discount = jnp.where(same_team, 1.0, -1.0)
        discount = jnp.where(done, 0.0, discount)

        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return output, next_states

    return recurrent_fn


def puct_search(
    state: GameState,
    player_id: Array,
    key: Array,
    pv_params,
    pv_apply,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "muzero",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    readout: str = "visits",
    *,
    cheat: bool = False,
    return_visits: bool = False,
) -> tuple[Array, ...]:
    """Run K determinized tree searches and aggregate the K roots.

    Args:
        state: Current (true or self-play) game state.
        player_id: Acting player; determinizations keep this hand fixed.
        pv_params / pv_apply: PolicyValueNet weights and apply function
            ((params, cm, hd) → (logits (B,43), value (B,))).
        num_determinizations: K sampled worlds (the mctx batch dimension).
        num_simulations: Tree simulations per determinization.
        v_scale: Net output → points (TARGET_SCALE of the training run).
        max_num_considered_actions: Gumbel sequential-halving width at the
            root. Gumbel variant only.
        search_variant: "muzero" (default, `mctx.muzero_policy` —
            classical full-width PUCT, the JTR-style operator and the
            standing collector since the 2026-07-06 searcher pivot) or
            "gumbel" (`mctx.gumbel_muzero_policy` — the retired
            summed-visit teacher; kept for the parked C′ efficiency
            probe only, DO NOT use as a collector: its visit readout is
            the impedance mismatch behind the gen-8 fixed point).
        pb_c_init: Classical-PUCT exploration constant on mctx's
            per-node-normalized Q. Muzero variant only. JTR's c=100 on
            the raw 0–157 point scale ≈ 0.64 here.
        dirichlet_fraction: Root Dirichlet-noise share. Muzero variant
            only; 0.0 (default) = deterministic teacher, AlphaZero
            self-play uses 0.25.
        prior_mix_uniform: λ share of uniform-over-legal mixed into the
            net's priors at the root and every tree node; 1.0 = flat
            (uniform-prior) PUCT. Grounded-teacher knob (2026-07-06):
            breaks the prior half of the search(π)≈π self-confirmation.
        rollout_value_weight: w share of a uniform-random playout-to-
            terminal return (real points) blended into every leaf value;
            1.0 replaces the value head entirely (classical evaluation).
            Breaks the value half of the self-confirmation. Costs up to
            38 extra env steps per node expansion. Both knobs must be
            Python floats at trace time (0.0 compiles to a no-op).
        readout: How the K root results become one score vector.
            "visits" (default): SUM ROOT VISIT COUNTS — correct when the
            priors concentrate visits (the net-prior search). "qsum":
            JTR-style Q aggregation, visit-weighted mean Q across the
            K trees — correct for flat-prior search, where visits stay
            ~uniform and the visit readout picks near-noise (measured:
            classical λ=1/w=1 read by visits = −24.3 vs gen-9 raw,
            2026-07-06; see `_qsum_scores` for why mean, not sum).
        cheat: All K "determinizations" are the TRUE state (no hand
            resampling) — the internal analogue of JTR's `--cheating`.
            Diagnostic only (oracle probes, search-noise floors); not a
            fair player and not a collector.
        return_visits: Also return the per-tree root visit counts
            (K, 43) BEFORE aggregation — the per-world teacher signal
            (hands-conditional-targets probe, log 2026-07-12).

    Returns:
        (scores, legal): (43,) float32 aggregated root scores and the
        (43,) bool legal mask of the information state. readout="visits":
        summed visit counts, zero on illegal actions. readout="qsum":
        visit-weighted mean Q in points, −inf on actions that are
        illegal or unvisited in every tree (argmax-safe as-is).
        With return_visits=True: (scores, legal, visits) with visits
        (K, 43) int32 per-tree root visit counts.
    """
    if readout not in ("visits", "qsum"):
        raise ValueError(f"unknown readout: {readout!r}")
    K = num_determinizations
    det_key, search_key, root_key = jax.random.split(key, 3)
    if cheat:
        det_states = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (K, *x.shape)), state
        )                                                # (K,) true state
    else:
        det_states = jax.vmap(
            lambda k: sample_determinization(state, player_id, k)
        )(jax.random.split(det_key, K))                  # (K,) GameState

    logits, value, _ = _pv_eval(
        pv_apply, pv_params, det_states, v_scale,
        prior_mix_uniform=prior_mix_uniform,
        rollout_value_weight=rollout_value_weight,
        rollout_key=root_key)
    root = mctx.RootFnOutput(
        prior_logits=logits, value=value, embedding=det_states)

    # The legal mask is an information-state property: identical across
    # determinizations (hands of others don't constrain the mover's moves).
    legal = _game.legal_action_mask(state)               # (43,)
    invalid = jnp.broadcast_to(~legal, (K, NUM_ACTIONS))

    recurrent_fn = _make_recurrent_fn(
        pv_apply, v_scale, prior_mix_uniform, rollout_value_weight)
    if search_variant == "gumbel":
        out = mctx.gumbel_muzero_policy(
            params=pv_params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid.astype(jnp.float32),
            max_num_considered_actions=max_num_considered_actions,
        )
    elif search_variant == "muzero":
        out = mctx.muzero_policy(
            params=pv_params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid.astype(jnp.float32),
            pb_c_init=pb_c_init,
            dirichlet_fraction=dirichlet_fraction,
        )
    else:
        raise ValueError(f"unknown search_variant: {search_variant!r}")

    summary = out.search_tree.summary()
    visits = summary.visit_counts                        # (K, 43)
    if readout == "visits":
        scores = jnp.where(legal, visits.sum(axis=0), 0.0)
    else:  # "qsum" — qvalues are root-mover perspective in all K trees;
        # unvisited children hold Q=0, so N·Q weighting ignores them.
        scores = _qsum_scores(visits, summary.qvalues, legal)
    if return_visits:
        return scores, legal, visits
    return scores, legal


@functools.partial(jax.jit, static_argnames=(
    "pv_apply", "num_determinizations", "num_simulations",
    "max_num_considered_actions", "search_variant",
    "prior_mix_uniform", "rollout_value_weight", "readout"))
def puct_action(
    state: GameState,
    player_id: Array,
    key: Array,
    pv_params,
    pv_apply,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "muzero",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    readout: str = "visits",
) -> Array:
    """Greedy PUCT move: argmax of the aggregated root scores."""
    scores, legal = puct_search(
        state, player_id, key, pv_params, pv_apply,
        num_determinizations, num_simulations, v_scale,
        max_num_considered_actions, search_variant, pb_c_init,
        dirichlet_fraction, prior_mix_uniform, rollout_value_weight,
        readout)
    scored = jnp.where(legal, scores, -jnp.inf)
    return jnp.argmax(scored).astype(jnp.int32)


def make_puct_policy_fn(
    pv_apply,
    pv_params,
    *,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "muzero",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    readout: str = "visits",
    temperature: float = None,
):
    """Build a policy_fn(state, key) → (action, pi) for jass_selfplay.

    readout="visits" (default): pi is the normalized summed visit
    distribution — the Step 3 policy training target. The executed
    action is the visit argmax when temperature is None, otherwise
    sampled ∝ visits^(1/temperature) (AlphaZero-style exploration;
    temperature=1 samples the visit distribution itself).

    readout="qsum": greedy argmax of the JTR-style Q mass; pi is the
    one-hot of that action (the classical teacher's recommendation —
    visit counts carry no policy signal under flat priors). temperature
    is visit-count semantics and would be silently meaningless here, so
    it must be None; use dirichlet_fraction for exploration instead.
    """
    if readout == "qsum" and temperature is not None:
        raise ValueError(
            "readout='qsum' requires temperature=None (greedy); "
            "temperature samples visit counts, which carry no policy "
            "signal under a Q readout — use dirichlet_fraction for "
            "exploration instead")

    def policy_fn(state: GameState, key: Array):
        k_search, k_sample = jax.random.split(key)
        scores, legal = puct_search(
            state, state.current_player, k_search, pv_params, pv_apply,
            num_determinizations, num_simulations, v_scale,
            max_num_considered_actions, search_variant, pb_c_init,
            dirichlet_fraction, prior_mix_uniform, rollout_value_weight,
            readout)
        if readout == "qsum":
            action = jnp.argmax(scores)          # already −inf masked
            pi = jax.nn.one_hot(action, NUM_ACTIONS)
        elif temperature is None:
            pi = scores / scores.sum().clip(1.0)
            action = jnp.argmax(jnp.where(legal, scores, -jnp.inf))
        else:
            pi = scores / scores.sum().clip(1.0)
            logits = jnp.where(
                legal, jnp.log(scores.clip(1e-9)) / temperature, _ILLEGAL)
            action = jax.random.categorical(k_sample, logits)
        return action.astype(jnp.int32), pi

    return policy_fn


def make_puct_action_fn(pv_apply, pv_params, **kwargs):
    """action_fn(state, key) → action wrapper (for policy_match / arenas)."""
    policy_fn = make_puct_policy_fn(pv_apply, pv_params, **kwargs)

    def action_fn(state: GameState, key: Array) -> Array:
        action, _ = policy_fn(state, key)
        return action

    return action_fn


def make_puct_collect_fn(
    pv_apply,
    pv_params,
    *,
    num_determinizations: int = 8,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    max_num_considered_actions: int = 16,
    search_variant: str = "muzero",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    readout: str = "visits",
    temperature: float = 1.0,
):
    """Build a collect_fn(key, batch_size) generating PUCT self-play data.

    readout="qsum" requires an explicit temperature=None (greedy) —
    pair it with dirichlet_fraction for corpus diversity.

    The Step 3 generator: every seat plays PUCT; policy targets are the
    aggregated root visit distributions; value labels are the acting
    player's terminal differential. Same contract as
    jass_selfplay.make_search_collect_fn:
    (cm, hd, labels, pi, legal, alive).
    """
    from pgx._src.games.jass_selfplay import _collect_pv

    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _puct_collect(params, key: Array, batch_size: int):
        policy_fn = make_puct_policy_fn(
            pv_apply, params,
            num_determinizations=num_determinizations,
            num_simulations=num_simulations,
            v_scale=v_scale,
            max_num_considered_actions=max_num_considered_actions,
            search_variant=search_variant,
            pb_c_init=pb_c_init,
            dirichlet_fraction=dirichlet_fraction,
            prior_mix_uniform=prior_mix_uniform,
            rollout_value_weight=rollout_value_weight,
            readout=readout,
            temperature=temperature)
        return _collect_pv(policy_fn, key, batch_size)

    def collect_fn(key: Array, batch_size: int):
        return _puct_collect(pv_params, key, batch_size)

    return collect_fn
