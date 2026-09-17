"""Reduction guards: K=1 B-D, and `full` collapses to `bd_baseline` at wS=0."""

import numpy as np

from model_validation import config as cfgmod
from model_validation import latent_regressors as lr
from model_validation import link_function as lk

cfg = cfgmod.validated_config()


def _kl_to_target(mu):
    sig2, mu_T, sig2_T = cfg.sigma2, cfg.p_T_mu, cfg.p_T_sigma2
    return (np.log(np.sqrt(sig2_T / sig2))
            + (sig2 + (mu - mu_T) ** 2) / (2.0 * sig2_T) - 0.5)


def test_bd_updates_reduce_to_single_state_at_K1():
    seq = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64)
    dummy_bpost = np.zeros((len(seq) - 1, 1))   # ignored when n_contexts == 1
    got = lr.bd_updates(seq, dummy_bpost, cfg, n_contexts=1)

    # independent plain B-D single-state reference
    logf = cfg.tone_logfreq
    mu = cfg.p_T_mu
    V_prev = -_kl_to_target(mu)
    ref = np.empty(len(seq) - 1)
    for i in range(len(seq) - 1):
        o = logf[seq[i + 1]]
        mu = mu + cfg.alpha * (o - mu)
        V_now = -_kl_to_target(mu)
        ref[i] = V_now - V_prev
        V_prev = V_now
    assert np.allclose(got, ref, atol=1e-12)


def test_full_collapses_to_bd_baseline_when_wS_zero():
    rng = np.random.default_rng(0)
    r, dV, S = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6)
    w0, wr, wV = 0.3, 1.2, 0.7
    A_full = lk.aesthetic_value(r, dV, S, lk.Weights(w0=w0, wr=wr, wV=wV, wS=0.0))
    A_bd = w0 + wr * r + wV * dV          # the Brielmann-Dayan value (no S term)
    assert np.allclose(A_full, A_bd)
