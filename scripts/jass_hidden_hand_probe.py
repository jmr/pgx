"""
Hidden-hand sensitivity probe: does the net USE the hidden-hand inputs?

Usage:
    python scripts/jass_hidden_hand_probe.py --weights pv_gen9_s128.msgpack
    python scripts/jass_hidden_hand_probe.py --weights ... --games 256 --worlds 16

Method (log entry 2026-07-12): play on-policy games with the given
PolicyValueNetAttn (sampled, tau=1). At every card-play state with more
than one legal move, hold the public information fixed and resample the
hidden hands via sample_determinization (void-aware — the same sampler
the determinized searchers use). Compare the net's outputs on the true
world vs the resampled worlds:

    policy head : KL(p_true || p_world) over legal moves, argmax flip rate
    value head  : std of v across worlds, in points (x TARGET_SCALE)

A hands-blind head shows KL ~ 0 / flips ~ 0 / std ~ 0. Reference numbers
for gen-9 (2026-07-12): policy KL 0.003 vs entropy 0.80, flips 4.1% —
hands-blind; value std 28.5 pts vs mean |v| 61 pts — hands-aware. The
mechanism-check for hands-conditional policy targets is this probe's
policy KL moving well off ~0.003 on the student.
"""

import argparse
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass import Game, value_features
from pgx._src.games.jass_mcts import sample_determinization
from pgx._src.games.jass_value_net import PolicyValueNetAttn, TARGET_SCALE

_MAX_STEPS = 38  # 2 trump-selection + 9*4 card-play steps

game = Game()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to PolicyValueNetAttn .msgpack checkpoint")
    parser.add_argument("--games", type=int, default=128,
                        help="On-policy games to probe (default: 128)")
    parser.add_argument("--worlds", type=int, default=8,
                        help="Resampled worlds per position (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PolicyValueNetAttn()
    with open(args.weights, "rb") as f:
        variables = flax.serialization.msgpack_restore(f.read())
    fp = float(sum(jnp.abs(x).sum()
                   for x in jax.tree_util.tree_leaves(variables)))
    print(f"loaded {args.weights}  fingerprint={fp:.2f}")

    def apply_net(state, player):
        cm, hd = value_features(state, player)
        logits, v = model.apply(variables, cm[None], hd[None])
        return logits[0], v[0]

    def probe_state(state, key):
        """Metrics for one state: true-world output vs K resampled worlds."""
        player = state.current_player
        logits0, v0 = apply_net(state, player)
        mask = game.legal_action_mask(state)

        def one_world(k):
            ws = sample_determinization(state, player, k)
            return apply_net(ws, player)

        wlogits, wv = jax.vmap(one_world)(
            jax.random.split(key, args.worlds))  # (K, 43), (K,)

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
        s0 = game.init(init_key)

        def step_fn(carry, _):
            s, k = carry
            done = s.trick_num >= 9
            k, ak, pk = jax.random.split(k, 3)

            logits0, _ = apply_net(s, s.current_player)
            mask = game.legal_action_mask(s)
            action = jax.random.categorical(
                ak, jnp.where(mask, logits0, jnp.float32(-1e9))
            ).astype(jnp.int32)

            kl, flip, ent, v0, vstd, nlegal = probe_state(s, pk)
            valid = (~done) & (s.phase == 1) & (nlegal > 1)
            out = (kl, flip, ent, v0, vstd, nlegal, s.trick_num, valid)

            ns = game.step(s, action)
            ns = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done, a, b), s, ns)
            return (ns, k), out

        _, outs = jax.lax.scan(step_fn, (s0, play_key), None,
                               length=_MAX_STEPS)
        return outs

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.games)
    outs = jax.jit(jax.vmap(play_and_probe))(keys)
    kl, flip, ent, v0, vstd, nlegal, trick, valid = [
        np.asarray(x) for x in outs]
    print(f"probe ran in {time.time() - t0:.1f}s"
          f"  ({args.games} games x {args.worlds} worlds)")

    m = valid.astype(bool)
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


if __name__ == "__main__":
    main()
