"""Recovery machinery: known wS is recovered; wS=0 is not flagged positive."""

import numpy as np

from model_validation import recovery as rec


def _synthetic_design(seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for m in range(6):
        for _ in range(4):
            n = 7
            r = rng.normal(size=n)
            dV = rng.normal(size=n) * 0.3
            S = rng.normal(size=n)
            g = rng.choice([1.0, -1.0, 0.0], size=n)
            obs = rng.dirichlet(np.ones(n))   # replaced by simulate() in the harness
            blocks.append(rec.Block(r, dV, S, g, obs, f"m{m}"))
    return rec.Design(blocks=blocks, scales={"r_mean": 1.0, "dV_mean": 1.0, "S_mean": 1.0})


def test_parameter_recovery_covers_truth_and_no_false_positive():
    design = _synthetic_design()
    res = rec.parameter_recovery(design, wS_grid=(0.0, 1.5), n_sim=80, seed=1)
    grid = {row["wS_true"]: row for row in res["grid"]}
    # wS = 0 truth: CI must include 0 (no false positive)
    assert grid[0.0]["ci"][0] <= 0.0 <= grid[0.0]["ci"][1]
    # positive truth: recovered with correct (positive) sign and CI covering truth
    assert grid[1.5]["wS_recovered_mean"] > 0
    assert grid[1.5]["covers_truth"]


def test_fit_recovers_sign_on_clean_data():
    design = _synthetic_design(seed=3)
    rng = np.random.default_rng(7)
    sim = rec.simulate(design, {"w0": 0.0, "wr": 1.0, "wV": 0.5, "wS": 2.0},
                       kappa=200.0, rng=rng)
    w = rec.fit(sim.blocks, "full")
    assert w["wS"] > 0
