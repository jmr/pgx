"""
Teacher-signal pre-probe: do the K per-world searches DISAGREE?

Usage:
    python scripts/jass_teacher_signal_probe.py --weights pv_gen9_s128.msgpack
    python scripts/jass_teacher_signal_probe.py --weights ... --games 32 --no-null

The kill switch for hands-conditional policy targets (log 2026-07-12,
gate step 1): the standing collector aggregates K=16 determinized root
visit distributions into one info-set-marginal target. Per-world
targets only carry extra information if the K distributions actually
DISAGREE across worlds at the standing budget. This probe measures
that disagreement on-policy, before any collection or training.

Method: play games with the standing teacher itself (muzero K x sims,
aggregate visits, tau=1 sampling — collection-faithful). At every
card-play decision with >1 legal move, read the per-tree root visits
(puct_search(..., return_visits=True)) and report:

    JSD        generalized Jensen-Shannon divergence of the K
               normalized visit distributions = H(mean) - mean(H);
               0 = all worlds produce the same target
    JSD/H      share of the aggregate target's entropy that is
               across-world variation (the hands-conditional share)
    argmax !=  fraction of worlds whose visit argmax differs from the
               aggregate argmax (how often the marginal target's top
               move is not the per-world search's top move)

Null floor (--null, default on): the same search with cheat=True — all
K trees on the TRUE world. With dirichlet_fraction=0 the teacher is
deterministic, so the floor should be ~0; it validates that measured
JSD is world signal, not search noise.

Reference (gen-9, standing config, 2026-07-12): see log entry.
"""

import argparse
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass import Game
from pgx._src.games.jass_puct import puct_search
from pgx._src.games.jass_value_net import PolicyValueNetAttn

_MAX_STEPS = 38  # 2 trump-selection + 9*4 card-play steps

game = Game()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to PolicyValueNetAttn .msgpack checkpoint")
    parser.add_argument("--games", type=int, default=16,
                        help="On-policy games to probe (default: 16)")
    parser.add_argument("--worlds", type=int, default=16,
                        help="Determinizations K (default: 16, standing)")
    parser.add_argument("--sims", type=int, default=64,
                        help="Simulations per tree (default: 64, standing)")
    parser.add_argument("--pb-c", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Also run the cheat=True noise floor")
    args = parser.parse_args()

    model = PolicyValueNetAttn()
    with open(args.weights, "rb") as f:
        variables = flax.serialization.msgpack_restore(f.read())
    fp = float(sum(jnp.abs(x).sum()
                   for x in jax.tree_util.tree_leaves(variables)))
    print(f"loaded {args.weights}  fingerprint={fp:.2f}")
    print(f"teacher: muzero K={args.worlds}x{args.sims}, "
          f"pb_c={args.pb_c}, dirichlet=0 (standing collection config)")

    def pv_apply(params, cm, hd):
        return model.apply(params, cm, hd)

    def search(state, key, cheat):
        return puct_search(
            state, state.current_player, key, variables, pv_apply,
            num_determinizations=args.worlds,
            num_simulations=args.sims,
            pb_c_init=args.pb_c,
            search_variant="muzero",
            cheat=cheat,
            return_visits=True,
        )

    def disagreement(visits, legal):
        """(K,43) visits + (43,) legal -> JSD, H(mean), argmax stats."""
        p = visits / visits.sum(axis=-1, keepdims=True).clip(1.0)  # (K,43)
        pbar = p.mean(axis=0)                                      # (43,)

        def ent(q):
            return -jnp.sum(jnp.where(q > 0, q * jnp.log(q), 0.0), axis=-1)

        jsd = ent(pbar) - ent(p).mean()
        agg_amax = jnp.argmax(jnp.where(legal, visits.sum(axis=0), -1.0))
        mismatch = jnp.mean(
            (jnp.argmax(visits, axis=-1) != agg_amax).astype(jnp.float32))
        return jsd, ent(pbar), mismatch

    def play_and_probe(key):
        init_key, play_key = jax.random.split(key)
        s0 = game.init(init_key)

        def step_fn(carry, _):
            s, k = carry
            done = s.trick_num >= 9
            k, sk, nk, ak = jax.random.split(k, 4)

            scores, legal, visits = search(s, sk, cheat=False)
            jsd, hbar, mismatch = disagreement(visits, legal)

            if args.null:
                _, _, nvisits = search(s, nk, cheat=True)
                jsd0, _, _ = disagreement(nvisits, legal)
            else:
                jsd0 = jnp.float32(0.0)

            valid = (~done) & (s.phase == 1) & (legal.sum() > 1)
            out = (jsd, jsd0, hbar, mismatch, legal.sum(), s.trick_num,
                   valid)

            # collection-faithful action: sample ~ aggregate visits (tau=1)
            logits = jnp.where(legal, jnp.log(scores.clip(1e-9)),
                               jnp.float32(-1e9))
            action = jax.random.categorical(ak, logits).astype(jnp.int32)
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
    jsd, jsd0, hbar, mismatch, nlegal, trick, valid = [
        np.asarray(x) for x in outs]
    print(f"probe ran in {time.time() - t0:.1f}s"
          f"  ({args.games} games, K={args.worlds}x{args.sims})")

    m = valid.astype(bool)
    hm = hbar[m].clip(1e-9)
    print(f"\n{m.sum()} card-play positions with >1 legal move")
    print(f"across-world JSD:        mean {jsd[m].mean():.4f}"
          f"   median {np.median(jsd[m]):.4f}"
          f"   p90 {np.percentile(jsd[m], 90):.4f}")
    if args.null:
        print(f"same-world floor (cheat): mean {jsd0[m].mean():.4f}")
    print(f"aggregate entropy H(p̄):  mean {hbar[m].mean():.4f}")
    print(f"JSD / H(p̄):              mean {(jsd[m] / hm).mean():.3f}")
    print(f"world argmax ≠ aggregate: {mismatch[m].mean():.3%}")

    print("\nby trick:  n     JSD     floor   JSD/H    amax≠   n_legal")
    for t in range(9):
        tm = m & (trick == t)
        if tm.sum() == 0:
            continue
        print(f"  {t}:   {tm.sum():5d}  {jsd[tm].mean():.4f}"
              f"  {jsd0[tm].mean():.4f}"
              f"  {(jsd[tm] / hbar[tm].clip(1e-9)).mean():7.3f}"
              f"  {mismatch[tm].mean():7.3%}  {nlegal[tm].mean():5.2f}")


if __name__ == "__main__":
    main()
