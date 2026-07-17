"""On-policy diagnostic probes for Jass policy+value nets.

Library core for the probe scripts (scripts/*.py are thin CLI wrappers;
they don't ship in the pip package, this module does — colab imports
these functions directly).

hidden_hand_probe — does the net USE the hidden-hand input columns?
(log 2026-07-12; the gen-11 mechanism check.) Play on-policy games with
the given net (policy-sampled, τ=1); at every card-play state with more
than one legal move, hold the public information fixed, resample the
hidden hands via sample_determinization (void-aware — the same sampler
the determinized searchers use), and compare the net's outputs on the
true world vs the resampled worlds:

    policy head : KL(p_true || p_world) over legal moves, argmax flips
    value head  : std of v across worlds, in points (× TARGET_SCALE)

A hands-blind head shows KL ~ 0 / flips ~ 0 / std ~ 0. Reference numbers
for gen-9 (2026-07-12): policy KL 0.003 vs entropy 0.80, flips 4.1% —
hands-blind; value std 28.5 pts vs mean |v| 61 pts — hands-aware. The
mechanism check for hands-conditional policy targets is this probe's
policy KL moving well off ~0.003 on the student.

belief_quality_probe — how much world mass does Bayes-inverting a
world-conditional policy buy? (SOP "Belief-quality probe" 2026-07-15;
the gate on belief-weighted determinization.) Particle filter alongside
self-play: at every decision, sample N void-consistent worlds for the
mover (plus the TRUE world as particle 0), weight each world w by
Π_t P_hc(observed move_t | w) over the full history of the OTHER three
players' moves, and report the normalized weight mass on the true world
(effective q) against the uniform 1/(N+1) baseline. Payoff estimate:
12.6·q̄ per game (dose-response, LINEAR, log 2026-07-13); pre-registered
bar q̄ ≥ ~0.2 → buy the search integration, q̄ ≲ 0.05 → route dead.

make_fair_raw_action_fn — the standing fair eval mode for hands-aware
nets (log 2026-07-15): raw play on TRUE states is oracle-contaminated
once a policy can read the opponent columns; fair raw marginalizes the
policy over sampled worlds (average of the legal-masked softmax).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass import Game, value_features
from pgx._src.games.jass_belief import world_log_likelihoods
from pgx._src.games.jass_mcts import sample_determinization
from pgx._src.games.jass_selfplay import make_policy_action_fn
from pgx._src.games.jass_value_net import TARGET_SCALE

_MAX_STEPS = 38  # 2 trump-selection + 9*4 card-play steps

_game = Game()


def hidden_hand_probe(pv_apply, pv_params, *, games: int = 128,
                      worlds: int = 8, seed: int = 0) -> dict:
    """Measure hidden-hand sensitivity of both heads, on-policy.

    Args:
        pv_apply / pv_params: net apply function
            ((params, cm, hd) → (logits (B,43), value (B,))) and weights.
        games: On-policy games to probe.
        worlds: Resampled determinizations per probed state.
        seed: PRNG seed.

    Returns:
        dict of (games, T) numpy arrays over the probed steps —
        kl (mean KL(p_true||p_world)), flip (share of worlds whose
        argmax ≠ the true-world argmax), ent (true-world policy entropy
        over legal), v0 (true-world value, net scale), vstd (value std
        across worlds, net scale), nlegal, trick, valid (card-play step
        with >1 legal move; mask for all statistics) — plus scalars
        games, worlds, runtime_s. Feed to print_hidden_hand_report.
    """

    def apply_net(state, player):
        cm, hd = value_features(state, player)
        logits, v = pv_apply(pv_params, cm[None], hd[None])
        return logits[0], v[0]

    def probe_state(state, key):
        """Metrics for one state: true-world output vs K resampled worlds."""
        player = state.current_player
        logits0, v0 = apply_net(state, player)
        mask = _game.legal_action_mask(state)

        def one_world(k):
            ws = sample_determinization(state, player, k)
            return apply_net(ws, player)

        wlogits, wv = jax.vmap(one_world)(
            jax.random.split(key, worlds))  # (K, 43), (K,)

        def logp(l):
            return jax.nn.log_softmax(jnp.where(mask, l, -1e9))

        lp0 = logp(logits0)
        lpw = jax.vmap(logp)(wlogits)
        p0 = jnp.exp(lp0)
        kl = jnp.sum(jnp.where(mask, p0[None] * (lp0[None] - lpw), 0.0),
                     axis=-1)
        flip = jnp.mean((jnp.argmax(lpw, axis=-1)
                         != jnp.argmax(lp0)).astype(jnp.float32))
        ent = -jnp.sum(jnp.where(mask, p0 * lp0, 0.0))
        return kl.mean(), flip, ent, v0, wv.std(), mask.sum()

    def play_and_probe(key):
        init_key, play_key = jax.random.split(key)
        s0 = _game.init(init_key)

        def step_fn(carry, _):
            s, k = carry
            done = s.trick_num >= 9
            k, ak, pk = jax.random.split(k, 3)

            logits0, _ = apply_net(s, s.current_player)
            mask = _game.legal_action_mask(s)
            action = jax.random.categorical(
                ak, jnp.where(mask, logits0, jnp.float32(-1e9))
            ).astype(jnp.int32)

            kl, flip, ent, v0, vstd, nlegal = probe_state(s, pk)
            valid = (~done) & (s.phase == 1) & (nlegal > 1)
            out = (kl, flip, ent, v0, vstd, nlegal, s.trick_num, valid)

            ns = _game.step(s, action)
            ns = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done, a, b), s, ns)
            return (ns, k), out

        _, outs = jax.lax.scan(step_fn, (s0, play_key), None,
                               length=_MAX_STEPS)
        return outs

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(seed), games)
    outs = jax.jit(jax.vmap(play_and_probe))(keys)
    kl, flip, ent, v0, vstd, nlegal, trick, valid = [
        np.asarray(x) for x in outs]
    return dict(kl=kl, flip=flip, ent=ent, v0=v0, vstd=vstd,
                nlegal=nlegal, trick=trick, valid=valid,
                games=games, worlds=worlds, runtime_s=time.time() - t0)


def print_hidden_hand_report(res: dict) -> None:
    """Print the standard report for a hidden_hand_probe result dict."""
    kl, flip, ent, v0, vstd, nlegal, trick = (
        res["kl"], res["flip"], res["ent"], res["v0"], res["vstd"],
        res["nlegal"], res["trick"])
    print(f"probe ran in {res['runtime_s']:.1f}s"
          f"  ({res['games']} games x {res['worlds']} worlds)")

    m = res["valid"].astype(bool)
    print(f"\n{m.sum()} card-play positions with >1 legal move")
    print(f"policy KL(true||world):  mean {kl[m].mean():.4f}"
          f"   median {np.median(kl[m]):.4f}"
          f"   p90 {np.percentile(kl[m], 90):.4f}")
    print(f"policy entropy (true):   mean {ent[m].mean():.4f}")
    print(f"argmax flip rate:        {flip[m].mean():.3%}")
    print(f"value std across worlds: mean {vstd[m].mean() * TARGET_SCALE:.2f} pts"
          f"   p90 {np.percentile(vstd[m], 90) * TARGET_SCALE:.2f} pts")
    print(f"(value scale: |v_true| mean"
          f" {np.abs(v0[m]).mean() * TARGET_SCALE:.1f} pts)")

    print("\nby trick:  n     KL      flip     v_std(pts)  entropy  n_legal")
    for t in range(9):
        tm = m & (trick == t)
        if tm.sum() == 0:
            continue
        print(f"  {t}:   {tm.sum():5d}  {kl[tm].mean():.4f}"
              f"  {flip[tm].mean():7.3%}"
              f"  {vstd[tm].mean() * TARGET_SCALE:8.2f}"
              f"  {ent[tm].mean():7.3f}  {nlegal[tm].mean():5.2f}")


def make_fair_raw_action_fn(pv_apply, pv_params, *, worlds: int = 16,
                            temperature: float = None):
    """Build the world-averaged (fair) raw action_fn(state, key) → action.

    The standing fair raw eval mode for hands-aware nets (log
    2026-07-15): the internal raw arena feeds `value_features` of the
    TRUE state, which is a cheating eval once a policy can read the
    opponent columns. Fair raw marginalizes instead: sample `worlds`
    void-consistent determinizations from the mover's perspective,
    average the legal-masked softmax across them, and play the argmax
    (temperature=None, the gen-11 arena config) or sample
    ∝ p^(1/temperature).
    """

    def action_fn(state, key):
        player = state.current_player
        k_worlds, k_sample = jax.random.split(key)
        mask = _game.legal_action_mask(state)

        def one_world(k):
            ws = sample_determinization(state, player, k)
            cm, hd = value_features(ws, player)
            logits, _ = pv_apply(pv_params, cm[None], hd[None])
            return jax.nn.softmax(jnp.where(mask, logits[0],
                                            jnp.float32(-1e9)))

        p = jax.vmap(one_world)(jax.random.split(k_worlds, worlds)).mean(0)
        if temperature is None:
            return jnp.argmax(jnp.where(mask, p, -1.0)).astype(jnp.int32)
        logits = jnp.where(mask, jnp.log(jnp.clip(p, 1e-9)) / temperature,
                           jnp.float32(-1e9))
        return jax.random.categorical(k_sample, logits).astype(jnp.int32)

    return action_fn


def uniform_pv_apply(params, cm, hd):
    """Constant-logits pv_apply: the legality-only baseline arm.

    Pass as `hc_apply` (params=None) to belief_quality_probe to measure
    the world mass bought by the LEGALITY channel alone — the legal-move
    set is world-dependent evidence (1/n_legal(w) normalization, and
    worlds under which an observed move was illegal are excluded) even
    with no policy shape at all. The hc run minus this arm isolates
    what the net's world-conditional policy adds.
    """
    b = cm.shape[0]
    return jnp.zeros((b, 43)), jnp.zeros((b,))


def belief_quality_probe(hc_apply, hc_params, *, games: int = 64,
                         particles: int = 16, seed: int = 0,
                         actor_action_fn=None, game_chunk: int = 16) -> dict:
    """Particle-filter belief probe: price the hc-likelihood route.

    At every decision of every game, sample `particles` void-consistent
    worlds for the mover (fresh each decision — the resample-from-
    scratch v0) and prepend the TRUE world as particle 0. Weight each
    world w by its likelihood of the OTHER three players' observed
    moves so far: log L(w) = Σ_t log P_hc(move_t | state_t under w),
    with the hc policy evaluated from the mover-at-t's seat. Past
    states under w need no replay: hands at t are w's hands plus the
    cards each player publicly played in [t, T). The scoring core is
    jass_belief.world_log_likelihoods, shared with the deployed
    belief-weighted searchers.

    The actor defaults to the hc net itself, raw sampled at τ=1 on the
    true state — the matched-likelihood setting, i.e. the CEILING of
    the route (a dead reading here kills it a fortiori). Pass
    `actor_action_fn(state, key) → action` (e.g. a gen-10 raw or
    make_fair_raw_action_fn agent) to measure under mismatched play.

    Args:
        hc_apply / hc_params: likelihood net (gen-11hc),
            ((params, cm, hd) → (logits (B,43), value (B,))).
        games: Self-play games to probe.
        particles: Sampled worlds per decision (true world is added on
            top, so the uniform baseline is 1/(particles+1)).
        seed: PRNG seed.
        actor_action_fn: Optional move generator for the games.
        game_chunk: Games per jitted call. The scoring batch holds
            game_chunk × (particles+1) × 38 net activations live at
            once, so this is the HBM knob (128×33 in one call OOMs a
            16G device). Per-game results don't depend on the chunking
            (same keys per game; only float reduction order differs).

    Returns:
        dict of numpy arrays over (games, T) probe steps — q (weight
        mass on exact-true worlds), n_scored (past moves scored),
        n_unknown (hidden cards), trick, phase, valid — and per-particle
        (games, T, particles+1) arrays weights (normalized, particle 0
        = true world), placement (fraction of hidden cards in the
        correct hand), misplaced (count) — plus scalars games,
        particles, runtime_s. Feed to print_belief_quality_report.
    """
    if actor_action_fn is None:
        actor_action_fn = make_policy_action_fn(hc_apply, hc_params,
                                                temperature=1.0)

    def play_game(key):
        """On-policy game; record the state/action/mover at every step."""
        init_key, play_key = jax.random.split(key)
        s0 = _game.init(init_key)

        def step_fn(carry, _):
            s, k = carry
            done = s.trick_num >= 9
            k, ak = jax.random.split(k)
            action = actor_action_fn(s, ak)
            ns = _game.step(s, action)
            ns = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done, a, b), s, ns)
            return (ns, k), (s, action, s.current_player, ~done)

        _, traj = jax.lax.scan(step_fn, (s0, play_key), None,
                               length=_MAX_STEPS)
        return traj

    def probe_game(traj, key):
        states, actions, movers, valid = traj

        def probe_step(_, xs):
            T, k = xs
            sT = jax.tree_util.tree_map(lambda x: x[T], states)
            probed = sT.current_player

            sampled = jax.vmap(
                lambda kk: sample_determinization(sT, probed, kk).hands
            )(jax.random.split(k, particles))             # (N, 4, 36)
            worlds = jnp.concatenate([sT.hands[None], sampled])

            steps = jnp.arange(_MAX_STEPS)
            inc = (steps < T) & (movers != probed) & valid
            logl = world_log_likelihoods(
                hc_apply, hc_params, states, actions, inc,
                sT.hands, worlds)
            weights = jax.nn.softmax(logl)                # (N+1,)

            valid_trick = sT.trick_cards >= 0
            safe_trick = jnp.where(valid_trick, sT.trick_cards, 0)
            in_trick = jnp.zeros(36, jnp.bool_).at[safe_trick].max(
                valid_trick)
            unknown = (~sT.hands[probed] & ~sT.cards_collected.any(axis=0)
                       & ~in_trick)                       # (36,)
            owner_true = jnp.argmax(sT.hands, axis=0)     # (36,)
            owner_w = jnp.argmax(worlds, axis=1)          # (N+1, 36)
            misplaced = (unknown[None]
                         & (owner_w != owner_true[None])).sum(-1)
            placement = 1.0 - misplaced / jnp.maximum(unknown.sum(), 1)
            q = (weights * (misplaced == 0)).sum()

            return None, (q, weights, placement.astype(jnp.float32),
                          misplaced, inc.sum(), unknown.sum(),
                          sT.trick_num, sT.phase, valid[T])

        _, outs = jax.lax.scan(
            probe_step, None,
            (jnp.arange(_MAX_STEPS), jax.random.split(key, _MAX_STEPS)))
        return outs

    def run_one(key):
        play_key, probe_key = jax.random.split(key)
        return probe_game(play_game(play_key), probe_key)

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(seed), games)
    # Pad to a whole number of chunks (one jit shape), trim after.
    n_chunks = -(-games // game_chunk)
    pad = n_chunks * game_chunk - games
    keys = jnp.concatenate([keys, keys[:pad]]) if pad else keys
    run = jax.jit(jax.vmap(run_one))
    chunks = []
    for i in range(n_chunks):
        chunks.append(jax.device_get(
            run(keys[i * game_chunk:(i + 1) * game_chunk])))
        if n_chunks > 1:
            print(f"  chunk {i + 1}/{n_chunks} [{time.time() - t0:.0f}s]")
    outs = [np.concatenate(x)[:games] for x in zip(*chunks)]
    q, weights, placement, misplaced, n_scored, n_unknown, trick, phase, \
        valid = outs
    return dict(q=q, weights=weights, placement=placement,
                misplaced=misplaced, n_scored=n_scored, n_unknown=n_unknown,
                trick=trick, phase=phase, valid=valid, games=games,
                particles=particles, runtime_s=time.time() - t0)


def print_belief_quality_report(res: dict) -> None:
    """Print the standard report for a belief_quality_probe result dict."""
    n1 = res["particles"] + 1
    uniform = 1.0 / n1
    m = res["valid"].astype(bool)
    q, w, mis = res["q"], res["weights"], res["misplaced"]
    placement, trick, phase = res["placement"], res["trick"], res["phase"]

    # Placement over the SAMPLED particles only (excluding the injected
    # true world), likelihood-reweighted vs the sampler's uniform average.
    ws = w[..., 1:]
    ws = ws / np.maximum(ws.sum(-1, keepdims=True), 1e-9)
    place_w = (ws * placement[..., 1:]).sum(-1)
    place_u = placement[..., 1:].mean(-1)
    ess = 1.0 / np.maximum((w ** 2).sum(-1), 1e-9)

    def mass_within(d):
        return (w * (mis <= d)).sum(-1), (mis <= d).mean(-1)

    print(f"probe ran in {res['runtime_s']:.1f}s"
          f"  ({res['games']} games x {res['particles']}+1 particles)")
    print(f"\n{m.sum()} probed decisions"
          f"  (uniform baseline 1/{n1} = {uniform:.4f})")

    cp = m & (phase == 1)
    qbar, qbar_cp = q[m].mean(), q[cp].mean()
    print(f"effective q̄ (mass on true world): {qbar:.4f} overall"
          f"   {qbar_cp:.4f} card-play only")
    print(f"predicted payoff 12.6·q̄:          "
          f"{12.6 * qbar:+.2f} /game overall"
          f"   {12.6 * qbar_cp:+.2f} card-play only")
    print("pre-registered bar: q̄ ≥ ~0.2 → BUY integration;"
          " q̄ ≲ 0.05 → route dead")

    for d in (1, 2, 4):
        mw, mu = mass_within(d)
        print(f"mass within {d} misplaced cards:   {mw[m].mean():.4f}"
              f"   (uniform {mu[m].mean():.4f})")
    print(f"placement (sampled worlds):       weighted {place_w[m].mean():.4f}"
          f"   uniform {place_u[m].mean():.4f}")
    print(f"effective sample size:            mean {ess[m].mean():.1f}"
          f" / {n1}")

    mass2, mass2_u = mass_within(2)
    print("\nby trick:     n    n_scored     q̄    mass(d≤2)  unif(d≤2)"
          "  place_w  place_u    ESS")
    rows = [("trump", m & (phase == 0))] + [
        (f"    {t}", m & (phase == 1) & (trick == t)) for t in range(9)]
    for label, tm in rows:
        if tm.sum() == 0:
            continue
        print(f"  {label}:  {tm.sum():5d}  {res['n_scored'][tm].mean():8.1f}"
              f"  {q[tm].mean():.4f}   {mass2[tm].mean():.4f}"
              f"     {mass2_u[tm].mean():.4f}"
              f"   {place_w[tm].mean():.4f}   {place_u[tm].mean():.4f}"
              f"  {ess[tm].mean():5.1f}")
