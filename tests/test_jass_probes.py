import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.jass_probes import (
    hidden_hand_probe,
    print_hidden_hand_report,
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
