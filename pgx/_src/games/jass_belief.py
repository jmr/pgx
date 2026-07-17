"""Belief-weighted determinization for Jass.

The belief-quality probe priced the route (SOP "Belief-quality probe",
log 2026-07-17): Bayes-inverting the gen-11hc policy concentrates
q̄ ≈ 0.56 of the world mass on the true world — 3× the pre-registered
buy bar. This module is the buy branch (SOP "Belief-weighted
determinization — integration", 2026-07-17): run the probe's particle
filter at every decision and let the searchers act on the weighted
belief instead of uniform world samples.

Per decision:
  1. sample N void-consistent candidate worlds for the mover
     (`sample_determinization` — the uniform proposal, fresh each
     decision);
  2. weight each world w by the hc likelihood of the OTHER three
     players' observed moves so far,
     log L(w) = Σ_t log P_hc(move_t | state_t under w), the hc policy
     evaluated from the mover-at-t's seat (`world_log_likelihoods`, the
     scoring core shared with jass_probes.belief_quality_probe);
  3. act on the belief: belief PUCT draws its K root determinizations
     with replacement ∝ weights (mid-game ESS ~2.5 — duplicated trees
     ARE the q mass, do NOT deduplicate; and no true-world injection,
     that was the probe's measurement device only); belief fair raw
     plays the exact weighted mixture of per-world policies.

Plumbing: the arena/selfplay `action_fn(state, key)` interface cannot
express the filter — `GameState` does not record who played which card
(`cards_collected` is by trick-WINNER), so the observed-move history is
NOT recoverable from the current state. Belief agents therefore take
`(state, traj, key)` with `traj` a `PublicTrajectory`: the 38-slot
stacked states/actions buffer of the probe's `play_game` record,
threaded by the driver (`belief_policy_match` / `run_belief_arena`;
lift plain agents with `as_traj_action_fn`). The recorded states are
the driver's TRUE states, but the filter reads only the mover's own
hand and public fields/diffs (cards played in [t, now) = hand diffs),
so belief agents are fair by construction — the oracle-contamination
rule (raw TRUE-state arenas, log 2026-07-15) does not bite here.

Cost per decision ≈ N × 38 hc evals — same order as one K=16×64
search; the HBM knob is the driver's chunk size.
"""

import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pgx._src.games.jass import Game, GameState, value_features
from pgx._src.games.jass_mcts import sample_determinization

_game = Game()

_MAX_STEPS = 38  # 2 trump-selection + 9*4 card-play steps


class PublicTrajectory(NamedTuple):
    """The public record of a game so far, in fixed 38-slot buffers.

    states : (T,)-stacked GameState — the state BEFORE each move. Slots
        at and past the current step hold padding (masked by `valid`).
    actions: (T,) int32 — the move played at each step.
    valid  : (T,) bool — slot holds a recorded (pre-terminal) step.

    The mover at step t is states.current_player[t].
    """
    states: GameState
    actions: Array
    valid: Array


def empty_trajectory(state: GameState) -> PublicTrajectory:
    """All-padding trajectory buffer (slots prefilled with `state`)."""
    states = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None], (_MAX_STEPS, *x.shape)), state)
    return PublicTrajectory(
        states=states,
        actions=jnp.zeros(_MAX_STEPS, jnp.int32),
        valid=jnp.zeros(_MAX_STEPS, jnp.bool_))


def record_step(traj: PublicTrajectory, t: Array, state: GameState,
                action: Array, valid: Array = True) -> PublicTrajectory:
    """Write (state, action) into slot t; valid=False keeps it padding."""
    return PublicTrajectory(
        states=jax.tree_util.tree_map(
            lambda buf, x: buf.at[t].set(x), traj.states, state),
        actions=traj.actions.at[t].set(action),
        valid=traj.valid.at[t].set(valid))


def world_log_likelihoods(hc_apply, hc_params, states: GameState,
                          actions: Array, score_mask: Array,
                          current_hands: Array, worlds: Array) -> Array:
    """log L(w) = Σ_t log P_hc(actions[t] | states[t] under w) over the mask.

    The particle filter's scoring core (shared with
    jass_probes.belief_quality_probe). Past states under a candidate
    world need no replay: hands at step t are the world's hands NOW
    plus the cards each player publicly played in [t, now), read off
    the recorded hands as a diff — public information.

    Args:
        hc_apply / hc_params: likelihood net,
            ((params, cm, hd) → (logits (B,43), value (B,))).
        states: (T,)-stacked GameState trajectory.
        actions: (T,) int32 observed moves.
        score_mask: (T,) bool — steps entering the sum. The caller
            selects past valid steps whose mover is not the believer;
            masked slots may hold arbitrary padding.
        current_hands: (4, 36) bool — hands at the probed decision.
        worlds: (N, 4, 36) bool — candidate hands at the probed decision.

    Returns:
        (N,) float32 log-likelihoods (0 where the mask is empty).
    """
    replay = states.hands & ~current_hands[None]      # (T, 4, 36)
    recon = worlds[:, None] | replay[None]            # (N, T, 4, 36)
    steps = jnp.arange(actions.shape[0])

    def score_step(hands_t, t):
        st = jax.tree_util.tree_map(
            lambda x: x[t], states)._replace(hands=hands_t)
        cm, hd = value_features(st, st.current_player)
        logits, _ = hc_apply(hc_params, cm[None], hd[None])
        mask = _game.legal_action_mask(st)
        lp = jax.nn.log_softmax(
            jnp.where(mask, logits[0], jnp.float32(-1e9)))
        return jnp.where(score_mask[t], lp[actions[t]], 0.0)

    return jax.vmap(
        lambda rh: jax.vmap(score_step)(rh, steps).sum())(recon)


def make_belief_world_fn(hc_apply, hc_params, *, num_particles: int = 32,
                         mix_uniform: float = 0.0):
    """Build world_fn(state, traj, key) → (worlds (N, 4, 36), weights (N,)).

    One particle-filter pass for the mover: N fresh void-consistent
    candidate worlds, weighted by the softmax-normalized hc likelihood
    of the other players' recorded moves. At the first decision of a
    game (empty history) the weights are exactly uniform.

    Args:
        hc_apply / hc_params: likelihood net (gen-11hc).
        num_particles: N sampled worlds (probe used 32).
        mix_uniform: λ share of uniform blended into the weights — the
            guard against a degenerate/misspecified likelihood, analog
            of prior_mix_uniform (default 0; pre-register before the
            arena).
    """

    def world_fn(state: GameState, traj: PublicTrajectory, key: Array):
        player = state.current_player
        worlds = jax.vmap(
            lambda k: sample_determinization(state, player, k).hands
        )(jax.random.split(key, num_particles))       # (N, 4, 36)
        score_mask = traj.valid & (traj.states.current_player != player)
        logl = world_log_likelihoods(
            hc_apply, hc_params, traj.states, traj.actions,
            score_mask, state.hands, worlds)
        weights = jax.nn.softmax(logl)
        if mix_uniform:
            weights = ((1.0 - mix_uniform) * weights
                       + mix_uniform / num_particles)
        return worlds, weights

    return world_fn


def make_belief_puct_action_fn(
    pv_apply,
    pv_params,
    hc_apply,
    hc_params,
    *,
    num_particles: int = 32,
    mix_uniform: float = 0.0,
    num_determinizations: int = 16,
    num_simulations: int = 64,
    v_scale: float = 100.0,
    search_variant: str = "muzero",
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    prior_mix_uniform: float = 0.0,
    rollout_value_weight: float = 0.0,
    readout: str = "visits",
    temperature: float = None,
):
    """Belief-weighted PUCT: action_fn(state, traj, key) → action.

    The same searcher as jass_puct.make_puct_action_fn, but the K root
    determinizations are drawn with replacement ∝ belief weights from N
    candidate worlds instead of uniformly (duplicated roots carry the q
    mass — no dedup). Search net (pv_*) and likelihood net (hc_*) are
    separate arguments: the standing arena searches with the incumbent
    net and weighs worlds with gen-11hc.

    temperature=None (default) plays the greedy argmax of the
    aggregated scores; otherwise the action is sampled ∝
    scores^(1/temperature) (visits readout only, as in
    make_puct_policy_fn).
    """
    # In-function import: keeps this module (and jass_probes, which
    # imports it) free of jass_puct's mctx dependency.
    from pgx._src.games.jass_puct import puct_search

    if readout == "qsum" and temperature is not None:
        raise ValueError(
            "readout='qsum' requires temperature=None (greedy); "
            "temperature samples visit counts, which carry no policy "
            "signal under a Q readout")
    world_fn = make_belief_world_fn(hc_apply, hc_params,
                                    num_particles=num_particles,
                                    mix_uniform=mix_uniform)

    def action_fn(state: GameState, traj: PublicTrajectory,
                  key: Array) -> Array:
        k_world, k_draw, k_search, k_sample = jax.random.split(key, 4)
        worlds, weights = world_fn(state, traj, k_world)
        idx = jax.random.categorical(
            k_draw, jnp.log(weights.clip(1e-30)),
            shape=(num_determinizations,))
        det_states = jax.vmap(
            lambda h: state._replace(hands=h))(worlds[idx])
        scores, legal = puct_search(
            state, state.current_player, k_search, pv_params, pv_apply,
            num_determinizations, num_simulations, v_scale,
            search_variant=search_variant, pb_c_init=pb_c_init,
            dirichlet_fraction=dirichlet_fraction,
            prior_mix_uniform=prior_mix_uniform,
            rollout_value_weight=rollout_value_weight,
            readout=readout, det_states=det_states)
        if temperature is None:
            return jnp.argmax(
                jnp.where(legal, scores, -jnp.inf)).astype(jnp.int32)
        logits = jnp.where(
            legal, jnp.log(scores.clip(1e-9)) / temperature,
            jnp.float32(-1e9))
        return jax.random.categorical(k_sample, logits).astype(jnp.int32)

    return action_fn


def make_belief_fair_raw_action_fn(pv_apply, pv_params, hc_apply, hc_params,
                                   *, num_particles: int = 32,
                                   mix_uniform: float = 0.0,
                                   temperature: float = None):
    """Belief-weighted fair raw: action_fn(state, traj, key) → action.

    jass_probes.make_fair_raw_action_fn with the uniform world average
    replaced by the belief-weighted mixture
    Σ_w weight(w) · softmax(policy | w) — the exact belief marginal
    over the N particles (no resampling noise, so no K here). Plays the
    argmax when temperature is None (the standing fair-raw arena
    config), else samples ∝ p^(1/temperature).
    """
    world_fn = make_belief_world_fn(hc_apply, hc_params,
                                    num_particles=num_particles,
                                    mix_uniform=mix_uniform)

    def action_fn(state: GameState, traj: PublicTrajectory,
                  key: Array) -> Array:
        k_world, k_sample = jax.random.split(key)
        worlds, weights = world_fn(state, traj, k_world)
        mask = _game.legal_action_mask(state)

        def one_world(h):
            ws = state._replace(hands=h)
            cm, hd = value_features(ws, state.current_player)
            logits, _ = pv_apply(pv_params, cm[None], hd[None])
            return jax.nn.softmax(jnp.where(mask, logits[0],
                                            jnp.float32(-1e9)))

        p = (weights[:, None] * jax.vmap(one_world)(worlds)).sum(0)
        if temperature is None:
            return jnp.argmax(jnp.where(mask, p, -1.0)).astype(jnp.int32)
        logits = jnp.where(mask, jnp.log(p.clip(1e-9)) / temperature,
                           jnp.float32(-1e9))
        return jax.random.categorical(k_sample, logits).astype(jnp.int32)

    return action_fn


def as_traj_action_fn(action_fn):
    """Lift a plain action_fn(state, key) to the (state, traj, key) form."""

    def traj_action_fn(state, traj, key):
        del traj
        return action_fn(state, key)

    return traj_action_fn


def _play_score_traj(action_fn, key: Array) -> Array:
    """One game with a trajectory-threaded action_fn; returns rewards (4,)."""
    init_key, play_key = jax.random.split(key)
    s0 = _game.init(init_key)
    traj0 = empty_trajectory(s0)

    def step_fn(carry, t):
        s, traj, k = carry
        done = s.trick_num >= 9
        k, sk = jax.random.split(k)
        action = action_fn(s, traj, sk)
        traj = record_step(traj, t, s, action, valid=~done)
        ns = _game.step(s, action)
        ns = jax.tree_util.tree_map(
            lambda a, b: jnp.where(done, a, b), s, ns)
        return (ns, traj, k), None

    (final, _, _), _ = jax.lax.scan(step_fn, (s0, traj0, play_key),
                                    jnp.arange(_MAX_STEPS))
    return _game.rewards(final)


@functools.partial(jax.jit, static_argnames=("action_fn_a", "action_fn_b",
                                             "num_pairs"))
def belief_policy_match(action_fn_a, action_fn_b, key: Array,
                        num_pairs: int) -> Array:
    """Swapped-deal policy arena for trajectory-threaded action_fns.

    jass_selfplay.policy_match with the PublicTrajectory carry threaded
    to both agents. The trajectory is the game's shared public record;
    each belief agent reads it from its own seat's perspective at
    decision time. Lift plain agents with as_traj_action_fn. As in
    policy_match, both agents are evaluated on every board each ply and
    the move is selected by seat parity (~2× per-move compute, zero
    per-move dispatch), and identical keys replay the same deals.

    Returns:
        (2 * num_pairs,) float32 — per-game score from A's perspective,
        pair-adjacent, ready for jass_v_arena.print_stats.
    """

    def seat_select(a_seats_even):
        def action_fn(s, traj, k):
            ka, kb = jax.random.split(k)
            act_a = action_fn_a(s, traj, ka)
            act_b = action_fn_b(s, traj, kb)
            a_to_move = (s.current_player % 2 == 0) == a_seats_even
            return jnp.where(a_to_move, act_a, act_b)
        return action_fn

    keys = jax.random.split(key, num_pairs)
    s_ab = jax.vmap(lambda k: _play_score_traj(seat_select(True), k)[0])(keys)
    s_ba = jax.vmap(lambda k: _play_score_traj(seat_select(False), k)[0])(keys)
    return jnp.stack([s_ab, -s_ba], axis=1).reshape(-1)


def run_belief_arena(challenger_fn, baseline_fn, *, pairs: int = 300,
                     chunk_pairs: int = 10, seed: int = 0,
                     label_c: str = "belief", label_b: str = "baseline"):
    """Chunked belief arena with the standing statistics.

    The measurement harness of the SOP integration section: challenger
    (a belief agent) vs baseline (typically a lifted uniform-sampling
    agent), swapped-deal pairs, jass_v_arena.print_stats at the end.
    chunk_pairs is the memory/progress knob (each chunk is one jitted
    lockstep call over 2 × chunk_pairs games).

    Returns:
        np.ndarray of per-game score differentials from the
        challenger's perspective, pair-adjacent.
    """
    from pgx._src.games.jass_v_arena import print_stats

    key = jax.random.PRNGKey(seed)
    chunks = []
    played = 0
    t0 = time.perf_counter()
    while played < pairs:
        n = min(chunk_pairs, pairs - played)
        key, chunk_key = jax.random.split(key)
        chunks.append(np.asarray(
            belief_policy_match(challenger_fn, baseline_fn, chunk_key, n)))
        played += n
        arr = np.concatenate(chunks)
        w, l = (arr > 0).sum(), (arr < 0).sum()
        print(f"  [{len(arr):4d}]  wins={w}  losses={l}"
              f"  mean={arr.mean():+.1f}"
              f"  ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    scores = np.concatenate(chunks)
    print_stats(label_c, label_b, scores)
    return scores
