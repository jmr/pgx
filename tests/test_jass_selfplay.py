import jax
import jax.numpy as jnp

from pgx._src.games.jass import CARD_SUIT, DECLARE_OFFSET, Game, value_features
from pgx._src.games.jass_selfplay import (
    apply_suit_permutation,
    augment_suits,
    collect_batch,
    collect_pv_batch,
    make_policy_action_fn,
    make_policy_collect_fn,
    make_search_collect_fn,
    make_v_action_fn,
    make_v_collect_fn,
    policy_match,
    random_action_fn,
    sample_suit_permutation,
)
from pgx._src.games.jass_value_net import TARGET_SCALE, PolicyValueNet, ValueNet

_game = Game()


B = 4
T = 38  # _MAX_STEPS


def _check_batch(cm, hd, labels, alive):
    assert cm.shape == (B, T, 36, 12)
    assert hd.shape == (B, T, 20)
    assert labels.shape == (B, T)
    assert alive.shape == (B, T)
    assert cm.dtype == jnp.bool_
    assert hd.dtype == jnp.bool_

    for b in range(B):
        n_alive = int(alive[b].sum())
        # 36 card plays + 1 or 2 trump-selection steps (Schiebe).
        assert n_alive in (37, 38)
        # alive is a prefix: no revival after the game ends.
        assert bool(alive[b, :n_alive].all())
        assert not bool(alive[b, n_alive:].any())

        y = labels[b][alive[b]]
        # Every step is labeled with the acting player's terminal
        # differential: same magnitude all game, range [-157, 157].
        assert jnp.all(jnp.abs(y) == jnp.abs(y[0]))
        assert jnp.abs(y[0]) <= 157


def test_collect_batch_shapes_and_labels():
    cm, hd, labels, alive = collect_batch(jax.random.PRNGKey(0), B)
    _check_batch(cm, hd, labels, alive)


def test_v_collect_fn_shapes_and_labels():
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_v_collect_fn(model.apply, params, v_scale=TARGET_SCALE)
    cm, hd, labels, alive = collect_fn(jax.random.PRNGKey(0), B)
    _check_batch(cm, hd, labels, alive)


def test_v_collect_fn_deterministic_and_param_sensitive():
    model = ValueNet()
    p1 = model.init(jax.random.PRNGKey(1),
                    jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    p2 = model.init(jax.random.PRNGKey(2),
                    jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    f1 = make_v_collect_fn(model.apply, p1, v_scale=TARGET_SCALE,
                           temperature=1.0)
    f2 = make_v_collect_fn(model.apply, p2, v_scale=TARGET_SCALE,
                           temperature=1.0)

    a = f1(jax.random.PRNGKey(0), B)
    b = f1(jax.random.PRNGKey(0), B)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))

    # Different weights play differently (low temperature, same key/deals).
    c = f2(jax.random.PRNGKey(0), B)
    assert not all(jnp.array_equal(x, y) for x, y in zip(a, c))


def test_policy_match_random_vs_random_is_balanced():
    scores = policy_match(random_action_fn, random_action_fn,
                          jax.random.PRNGKey(0), 128)
    assert scores.shape == (256,)
    # Same policy on both sides: no edge beyond pair-cancelled noise.
    pair_means = scores.reshape(-1, 2).mean(axis=1)
    assert abs(float(pair_means.mean())) < 15.0
    # Scores are valid differentials.
    assert jnp.all(jnp.abs(scores) <= 157)


def test_policy_match_v_policy_runs_and_is_deterministic():
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    v_fn = make_v_action_fn(model.apply, params, v_scale=TARGET_SCALE,
                            temperature=1.0)
    a = policy_match(v_fn, random_action_fn, jax.random.PRNGKey(0), 4)
    b = policy_match(v_fn, random_action_fn, jax.random.PRNGKey(0), 4)
    assert a.shape == (8,)
    assert jnp.array_equal(a, b)


def _check_pv_batch(cm, hd, labels, pi, legal, alive):
    _check_batch(cm, hd, labels, alive)
    assert pi.shape == (B, T, 43)
    assert legal.shape == (B, T, 43)
    assert pi.dtype == jnp.float32
    assert legal.dtype == jnp.bool_
    # pi is a distribution supported on legal actions (alive steps only).
    live = alive[..., None]
    assert jnp.allclose(jnp.where(alive, pi.sum(-1), 1.0), 1.0)
    assert not jnp.any((pi > 0) & ~legal & live)
    # Every alive step has at least one legal action.
    assert jnp.all(legal.any(-1) | ~alive)


def _pv_params(seed=1):
    model = PolicyValueNet()
    return model, model.init(jax.random.PRNGKey(seed),
                             jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))


def test_search_collect_fn_contract():
    collect_fn = make_search_collect_fn(num_determinizations=2,
                                        num_rollouts=1, temperature=10.0)
    out = collect_fn(jax.random.PRNGKey(0), B)
    _check_pv_batch(*out)
    # Greedy search variant too (one-hot pi by construction).
    greedy_fn = make_search_collect_fn(num_determinizations=2, num_rollouts=1)
    out = greedy_fn(jax.random.PRNGKey(0), B)
    _check_pv_batch(*out)


def test_search_policy_fn_targets_argmax_not_sampled_action():
    """pi must be the search argmax even when play is temperature-sampled.

    Training on the sampled action instead teaches the policy the
    exploration noise (at high temperature: uniform-over-legal).
    """
    from pgx._src.games.jass_mcts import best_action
    from pgx._src.games.jass_selfplay import make_search_policy_fn

    state = _game.init(jax.random.PRNGKey(3))
    state = _game.step(state, jnp.int32(DECLARE_OFFSET))
    legal = _game.legal_action_mask(state)

    hot = make_search_policy_fn(num_determinizations=2, num_rollouts=1,
                                temperature=1e6)
    sampled_differs = 0
    for i in range(20):
        key = jax.random.PRNGKey(i)
        action, pi = hot(state, key)
        # The target is the greedy search action for this step's search
        # key (first half of the split), regardless of what was played.
        k_search, _ = jax.random.split(key)
        expected = best_action(state, state.current_player, k_search,
                               num_determinizations=2, num_rollouts=1)
        assert int(jnp.argmax(pi)) == int(expected)
        assert float(pi.sum()) == 1.0 and float(pi.max()) == 1.0
        assert bool(legal[action])
        sampled_differs += int(action) != int(expected)
    # At temperature 1e6 play is ~uniform over legal: it must frequently
    # deviate from the argmax target (else targets track the sample).
    assert sampled_differs >= 5

    # Greedy variant: played action and target coincide by construction.
    greedy = make_search_policy_fn(num_determinizations=2, num_rollouts=1)
    action, pi = greedy(state, jax.random.PRNGKey(0))
    assert int(jnp.argmax(pi)) == int(action)


def test_search_collect_fn_with_v_leaf():
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_search_collect_fn(model.apply, params,
                                        num_determinizations=2,
                                        num_rollouts=1,
                                        v_scale=TARGET_SCALE)
    _check_pv_batch(*collect_fn(jax.random.PRNGKey(0), B))


def test_policy_collect_fn_contract_and_determinism():
    model, params = _pv_params()
    collect_fn = make_policy_collect_fn(model.apply, params)
    a = collect_fn(jax.random.PRNGKey(0), B)
    _check_pv_batch(*a)
    b = collect_fn(jax.random.PRNGKey(0), B)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))


def test_policy_action_fn_in_policy_match():
    model, params = _pv_params()
    p_fn = make_policy_action_fn(model.apply, params, temperature=1.0)
    scores = policy_match(p_fn, random_action_fn, jax.random.PRNGKey(0), 4)
    assert scores.shape == (8,)
    assert jnp.all(jnp.abs(scores) <= 157)


# ── Suit-permutation augmentation ──────────────────────────────────────────────


def _permute_state(state, sigma):
    """Relabel suits of a GameState directly (test oracle)."""
    inv = jnp.argsort(sigma)
    ranks = jnp.arange(36, dtype=jnp.int32) % 9
    old_of_new = inv[CARD_SUIT] * 9 + ranks       # gather map for card masks
    new_of_old = sigma[CARD_SUIT] * 9 + ranks     # for card indices

    tc = state.trick_cards
    tc_new = jnp.where(tc >= 0, new_of_old[jnp.clip(tc, 0)], -1)
    led_new = jnp.where(state.led_suit >= 0,
                        sigma[jnp.clip(state.led_suit, 0, 3)], -1)
    is_suit_trump = (state.trump >= 0) & (state.trump < 4)
    trump_new = jnp.where(is_suit_trump,
                          sigma[jnp.clip(state.trump, 0, 3)], state.trump)
    return state._replace(
        hands=state.hands[:, old_of_new],
        cards_collected=state.cards_collected[:, old_of_new],
        trick_cards=tc_new.astype(jnp.int32),
        led_suit=led_new.astype(jnp.int32),
        trump=trump_new.astype(jnp.int32),
        void_in_suit=state.void_in_suit[:, inv],
    )


def _midgame_state(seed, declare):
    state = _game.init(jax.random.PRNGKey(seed))
    state = _game.step(state, jnp.int32(declare))
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(6):  # into the second trick
        key, sk = jax.random.split(key)
        state = _game.step(state, random_action_fn(state, sk))
    return state


def _assert_permutation_consistent(state, sigma):
    """Permuted features/masks must equal features/masks of permuted state."""
    p = state.current_player
    cm, hd = value_features(state, p)
    legal = _game.legal_action_mask(state)
    pi = legal / legal.sum()  # any distribution supported on legal

    cm_a, hd_a, pi_a, legal_a = apply_suit_permutation(sigma, cm, hd, pi, legal)

    perm_state = _permute_state(state, sigma)
    cm_o, hd_o = value_features(perm_state, p)
    legal_o = _game.legal_action_mask(perm_state)

    assert jnp.array_equal(cm_a, cm_o)
    assert jnp.array_equal(hd_a, hd_o)
    assert jnp.array_equal(legal_a, legal_o)
    assert jnp.allclose(pi_a, legal_o / legal_o.sum())


def test_suit_permutation_matches_engine_trump_mode():
    state = _midgame_state(0, DECLARE_OFFSET + 1)  # ♥ trump
    sigma = jnp.int32([2, 1, 3, 0])                # fixes ♥
    _assert_permutation_consistent(state, sigma)


def test_suit_permutation_matches_engine_obenabe_full_4():
    state = _midgame_state(2, DECLARE_OFFSET + 4)  # Obenabe
    sigma = jnp.int32([3, 0, 1, 2])                # full 4-cycle
    _assert_permutation_consistent(state, sigma)


def test_suit_permutation_trump_selection_phase():
    state = _game.init(jax.random.PRNGKey(7))      # trump not yet declared
    sigma = jnp.int32([1, 2, 3, 0])
    _assert_permutation_consistent(state, sigma)


def test_sample_suit_permutation_fixes_trump():
    state = _midgame_state(4, DECLARE_OFFSET + 2)  # ♠ trump (suit 2)
    _, hd = value_features(state, state.current_player)
    seen = set()
    for i in range(40):
        sigma = sample_suit_permutation(jax.random.PRNGKey(i), hd)
        assert jnp.array_equal(jnp.sort(sigma), jnp.arange(4))
        assert int(sigma[2]) == 2  # trump suit fixed
        seen.add(tuple(int(x) for x in sigma))
    assert len(seen) == 6  # all 3! permutations of the non-trump suits


def test_sample_suit_permutation_full_when_no_trump_suit():
    state = _midgame_state(5, DECLARE_OFFSET + 4)  # Obenabe
    _, hd = value_features(state, state.current_player)
    seen = set()
    for i in range(200):
        sigma = sample_suit_permutation(jax.random.PRNGKey(i), hd)
        assert jnp.array_equal(jnp.sort(sigma), jnp.arange(4))
        seen.add(tuple(int(x) for x in sigma))
    assert len(seen) == 24  # all 4! permutations


def test_augment_suits_batch_contract():
    cm, hd, y, pi, legal, alive = collect_pv_batch(jax.random.PRNGKey(0), B)
    flat = (cm.reshape(-1, 36, 12), hd.reshape(-1, 20),
            pi.reshape(-1, 43), legal.reshape(-1, 43))
    cm_a, hd_a, pi_a, legal_a = augment_suits(jax.random.PRNGKey(1), *flat)
    assert cm_a.shape == flat[0].shape and cm_a.dtype == flat[0].dtype
    assert hd_a.shape == flat[1].shape
    # Relabeling preserves counts: cards held, legal moves, pi mass.
    assert jnp.array_equal(cm_a.sum((1, 2)), flat[0].sum((1, 2)))
    assert jnp.array_equal(legal_a.sum(-1), flat[3].sum(-1))
    assert jnp.allclose(pi_a.sum(-1), flat[2].sum(-1))
    # Value-only variant.
    cm_v, hd_v = augment_suits(jax.random.PRNGKey(1), flat[0], flat[1])
    assert jnp.array_equal(cm_v, cm_a)
    assert jnp.array_equal(hd_v, hd_a)


def test_v_collect_matches_random_play_distribution_contract():
    # The two generators must be drop-in interchangeable for train_model:
    # identical pytree structure, shapes, and dtypes.
    model = ValueNet()
    params = model.init(jax.random.PRNGKey(1),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    collect_fn = make_v_collect_fn(model.apply, params, v_scale=TARGET_SCALE)
    rand = collect_batch(jax.random.PRNGKey(0), B)
    vsel = collect_fn(jax.random.PRNGKey(0), B)
    for r, v in zip(rand, vsel):
        assert r.shape == v.shape
        assert r.dtype == v.dtype
