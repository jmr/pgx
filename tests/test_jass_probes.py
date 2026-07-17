import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass_probes import (
    belief_quality_probe,
    hidden_hand_probe,
    print_belief_quality_report,
    print_hidden_hand_report,
    uniform_pv_apply,
)
from pgx._src.games.jass_value_net import PolicyValueNet


def test_hidden_hand_probe_contract(capsys):
    model = PolicyValueNet()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, 36, 12)), jnp.zeros((1, 20)))
    G, T = 2, 38
    res = hidden_hand_probe(model.apply, params, games=G, worlds=2, seed=0)

    for name in ("kl", "flip", "ent", "v0", "vstd", "nlegal", "trick",
                 "valid"):
        assert res[name].shape == (G, T), name
    m = res["valid"].astype(bool)
    assert m.any() and not m.all()          # games end inside T
    # KL and entropy are nonnegative; flip is a share of worlds.
    assert np.all(res["kl"][m] >= -1e-5)
    assert np.all(res["ent"][m] >= -1e-5)
    assert np.all((res["flip"][m] >= 0) & (res["flip"][m] <= 1))
    assert np.all(res["vstd"][m] >= 0)

    print_hidden_hand_report(res)
    out = capsys.readouterr().out
    assert "policy KL(true||world)" in out
    assert "by trick:" in out


def test_belief_quality_probe_contract(capsys):
    G, T, N = 2, 38, 2
    res = belief_quality_probe(uniform_pv_apply, None,
                               games=G, particles=N, seed=0, game_chunk=G)
    for name in ("q", "n_scored", "n_unknown", "trick", "phase", "valid"):
        assert res[name].shape == (G, T), name
    for name in ("weights", "placement", "misplaced"):
        assert res[name].shape == (G, T, N + 1), name
    m = res["valid"].astype(bool)
    assert m.any() and not m.all()
    assert np.all((res["q"][m] >= 0) & (res["q"][m] <= 1 + 1e-5))
    assert np.allclose(res["weights"][m].sum(-1), 1.0, atol=1e-4)
    # Particle 0 is the injected true world: never misplaced.
    assert np.all(res["misplaced"][m][:, 0] == 0)
    assert np.allclose(res["placement"][m][:, 0], 1.0, atol=1e-5)

    print_belief_quality_report(res)
    out = capsys.readouterr().out
    assert "effective q̄" in out
    assert "by trick:" in out
