# Jass MCTS Design

> **Roadmap:** see `docs/jass_plan.md` for the incremental plan from the current
> determinized MCTS to a full AlphaZero-style agent (expert iteration, policy
> head, PUCT via mctx), including step status and arena results.

Jass is a game of imperfect information: each player knows only their own hand, the declared trump, and the cards that have been played publicly. Standard MCTS assumes a fully observable state, so it cannot be applied directly. This document describes the determinization approach and two implementation options.

## Background: Determinization

**Determinization** converts an imperfect-information game into a sequence of perfect-information games. For a given information state (what the current player knows), we sample a *consistent* assignment of unknown cards to opponents — a *determinization* — and run a perfect-information solver on it. Repeating this K times and aggregating the results gives a robust action choice.

This is sometimes called IS-MCTS (Information Set MCTS) in the literature. JassTheRipper (a competitive Java Jass agent) uses this approach: K independent MCTS trees, one per determinization, with the best action chosen by plurality vote across trees.

### What the current player knows

From `GameState`, player `p`'s information state at any point is:

| Known | Source |
|:---|:---|
| Own hand | `hands[p]` |
| All cards won in completed tricks (by whom) | `cards_collected` (4×36 bool) |
| Cards played so far in the current trick | `trick_cards` (−1 if not yet played) |
| Declared trump / game mode | `trump` |
| Who led the current trick | `trick_leader` |
| Current trick number | `trick_num` |

Unknown: the cards still held by the other three players.

### How many cards does each opponent hold?

At the start of trick `trick_num` (0-indexed), each player has played `trick_num` cards in completed tricks, plus 1 if they have already played in the current trick:

```
cards_in_hand[p] = 9 - trick_num - (1 if trick_cards[p] >= 0 else 0)
```

The three opponents' hand sizes always sum to the number of unknown cards, so the partition is always exact.

### Sampling a determinization (JAX)

1. Compute `unknown_mask`: the 36-bit mask of cards not in `hands[p]`, not in `cards_collected`, and not in `trick_cards`.
2. Gather the unknown card indices.
3. `jax.random.permutation(key, unknown_indices)` — uniformly random shuffle.
4. Slice: opponent 0 gets the first `n0` cards, opponent 1 the next `n1`, opponent 2 the last `n2`.
5. Build a new `GameState` identical to the original except `hands` is replaced with the sampled assignment (keeping `hands[p]` fixed).

This sampling is **unbiased** (uniform over all consistent assignments). It is vmappable: `jax.vmap(sample_determinization, in_axes=(None, None, 0))(state, player_id, keys)` produces K determinized states in one call.

### Suit-voiding refinement (optional, not yet implemented)

If player `q` failed to follow suit `s` at some earlier trick, they are known to hold no cards of suit `s`. This eliminates impossible determinizations and concentrates probability on plausible ones. Implementing this requires tracking a `(4, 4)` bool array `void_in_suit[player, suit]` in `GameState`. JassTheRipper uses this constraint and optionally refines further with a neural network card estimator.

---

## Option A: Determinized Rollouts (implemented)

The simplest approach: instead of building a UCT tree, evaluate each legal action by averaging the outcome of N random rollouts to game completion, across K determinizations.

```
score(action) = mean over k in 1..K, n in 1..N of:
    rollout(apply(determinization_k, action))
```

**Structure:**

```python
# Outer vmap: K determinizations
# Inner vmap: N rollouts per determinization
# All vmapped over actions simultaneously

det_states  = vmap(sample_determinization)(state, player, keys_K)        # (K,)
after_action = vmap(vmap(apply_action))(det_states, legal_actions)        # (K, A)
scores       = vmap(vmap(vmap(random_rollout)))(after_action, keys_K_A_N) # (K, A, N)
best_action  = argmax(mean(scores, axis=(0, 2)))                          # (A,) → scalar
```

**Advantages:**
- Trivially JAX-compatible — no dynamic data structures.
- Fully vectorized; runs on GPU without modification.
- Fast to implement and easy to reason about.

**Disadvantages:**
- No tree reuse: every rollout starts from the root. UCT concentrates search on promising branches; flat rollouts treat all actions equally until scores diverge.
- Weaker than UCT for the same compute budget, especially early in the game when branching is high.

**Complexity:** O(K × A × N) rollout steps, each of length up to ~37 steps (rest of game). With K=32, A=36, N=8: ~9,000 rollouts of ~20 steps average = ~180,000 environment steps per move decision. At 320k steps/sec on CPU, that is ~0.6 seconds per move. On GPU, sub-100ms is plausible.

---

## Option B: UCT with Pre-allocated Tree (future)

Proper UCT MCTS requires a tree with per-node visit counts and value estimates. In JAX this means pre-allocating fixed-size arrays and using `jax.lax.while_loop` for the MCTS loop.

**Recommended path: use [mctx](https://github.com/google-deepmind/mctx)**, DeepMind's JAX MCTS library. mctx provides:
- `mctx.gumbel_muzero_policy` and `mctx.muzero_policy` — batched UCT over vmapped root states
- Fixed-size tree representation internally
- Plugs in via a `recurrent_fn` callback: `(params, key, action, embedding) → (output, next_embedding)`

For Jass, the `embedding` would be the `GameState`, `recurrent_fn` would call `env.step`, and the root embeddings would be the K determinized states. mctx handles the tree expansion and backpropagation.

**Integration sketch:**

```python
import mctx

def recurrent_fn(params, key, action, state):
    next_state = env.step(state, action)
    # value estimate: 0 during play, actual reward at terminal
    value = jnp.where(next_state.terminated,
                      next_state.rewards[next_state.current_player], 0.0)
    output = mctx.RecurrentFnOutput(
        reward=next_state.rewards[next_state.current_player],
        discount=jnp.where(next_state.terminated, 0.0, 1.0),
        prior_logits=jnp.zeros(NUM_ACTIONS),  # uniform prior; replace with policy network
        value=value,
    )
    return output, next_state

# Vmap over K determinizations
root = mctx.RootFnOutput(
    prior_logits=jnp.zeros((K, NUM_ACTIONS)),
    value=jnp.zeros(K),
    embedding=det_states,  # (K,) batch of GameState
)
policy_output = mctx.gumbel_muzero_policy(
    params=None, rng_key=key, root=root,
    recurrent_fn=recurrent_fn, num_simulations=128,
)
# Aggregate: vote across K determinizations
action = jnp.argmax(jnp.sum(policy_output.action_weights, axis=0))
```

**Advantages over Option A:**
- UCT concentrates computation on promising branches — much stronger for the same budget.
- mctx is battle-tested and GPU-optimized.
- Supports adding a policy/value network later (AlphaZero-style).

**Challenges:**
- mctx dependency.
- Aggregating across determinizations: **sum visit counts across the K trees and argmax** (not Q-sum). JassTheRipper's instrumentation showed Q-sum aggregation neutralizes the tree policy entirely (UCB c was a wash across 7 orders of magnitude); visit counts must be load-bearing for policy priors to matter. See `docs/jass_plan.md`.
- The `recurrent_fn` interface assumes the state is the embedding — works cleanly since `GameState` is a JAX pytree.

---

## Comparison

| | Option A (rollouts) | Option B (mctx UCT) |
|:---|:---:|:---:|
| Implementation complexity | Low | Medium |
| JAX compatibility | Native | Via mctx |
| GPU-friendly | Yes | Yes |
| Search quality | Moderate | High |
| Policy network support | No | Yes |
| Dependency | None | mctx |

## References

- JassTheRipper source: `~/Documents/src/JassTheRipper`
  - Determinization sampling: `CardKnowledgeBase.java`
  - MCTS engine: `MCTS.java`, `Node.java`
  - Board interface: `Board.kt`, `JassBoard.java`
- mctx: https://github.com/google-deepmind/mctx
- Cowling et al. (2012): "Information Set Monte Carlo Tree Search" — the IS-MCTS paper
