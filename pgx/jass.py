import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.jass import NUM_ACTIONS, Game, GameState
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    current_player:   Array = jnp.int32(0)
    observation:      Array = jnp.zeros(120, dtype=jnp.bool_)
    rewards:          Array = jnp.float32([0.0, 0.0, 0.0, 0.0])
    terminated:       Array = jnp.bool_(False)
    truncated:        Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    _step_count:      Array = jnp.int32(0)
    _x:               GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "jass"


class Jass(core.Env):
    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        x = self._game.init(key)
        return State(
            current_player=x.current_player,
            legal_action_mask=self._game.legal_action_mask(x),
            _x=x,
        )  # type: ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self._game.step(state._x, action)
        terminated = self._game.is_terminal(x)
        return state.replace(  # type: ignore
            current_player=x.current_player,
            legal_action_mask=self._game.legal_action_mask(x),
            rewards=self._game.rewards(x),
            terminated=terminated,
            _x=x,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x, player_id)

    @property
    def id(self) -> core.EnvId:
        return "jass"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 4
