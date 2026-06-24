"""Belief-filter behaviour: diagnosticity drives S with the correct sign."""

import numpy as np

from model_validation import config as cfgmod
from model_validation import latent_regressors as lr

cfg = cfgmod.validated_config()
sym2idx = cfgmod.symbol_to_index(cfg)
# group 1: M_EE = Grammar A, M_SC = Grammar B
M_EE, M_SC = cfg.emission_matrices(1)


def _seq(indices):
    return np.array(indices, dtype=np.int64)


def test_belief_is_normalised_and_bounded():
    seq = _seq([0, 1, 2, 3, 4, 5, 0, 1])
    tr = lr.hmm_filter(seq, M_EE, M_SC, cfg)
    assert np.allclose(tr.b_post.sum(axis=1), 1.0)
    assert np.all(tr.b_post >= -1e-9) and np.all(tr.b_post <= 1 + 1e-9)
    assert np.allclose(tr.b_prior.sum(axis=1), 1.0)


def test_nondiagnostic_sequence_gives_zero_S():
    # descending steps i -> i-1 are a 'rare' (0.07) in BOTH grammars -> equal
    # likelihoods -> belief never moves -> S == 0 at every step.
    seq = _seq([0, 5, 4, 3, 2, 1, 0])
    tr = lr.hmm_filter(seq, M_EE, M_SC, cfg)
    assert np.allclose(tr.S, 0.0, atol=1e-9)


def test_grammarA_sequence_drives_S_positive():
    # ascending steps i -> i+1 are Grammar A's dominant (0.60) but Grammar B's
    # secondary (0.12): diagnostic toward EE (=A for group 1) -> S > 0, b_EE up.
    seq = _seq([0, 1, 2, 3, 4, 5, 0, 1, 2])
    tr = lr.hmm_filter(seq, M_EE, M_SC, cfg)
    assert tr.S.mean() > 0
    assert tr.b_post[-1, 0] > cfg.b0[0]   # belief in EE exceeds the prior


def test_grammarB_sequence_drives_S_nonpositive():
    # i -> i+3 is Grammar B's dominant (0.60) but Grammar A's secondary (0.12):
    # diagnostic toward SC -> S <= 0, b_EE down.
    seq = _seq([0, 3, 0, 3, 0, 3, 0, 3])
    tr = lr.hmm_filter(seq, M_EE, M_SC, cfg)
    assert tr.S.mean() <= 1e-9
    assert tr.b_post[-1, 0] < cfg.b0[0]


def test_complexity_ordering_in_r():
    # r (negative surprise) should rank dominant > secondary > rare under the
    # FULL grammar: dominant transitions are the most predictable.
    asc = _seq([0, 1, 2, 3, 4, 5])          # all-dominant under A
    rare = _seq([0, 5, 4, 3, 2, 1])         # all-rare under A
    r_dom = lr.hmm_filter(asc, M_EE, M_SC, cfg).r_hmm.mean()
    r_rare = lr.hmm_filter(rare, M_EE, M_SC, cfg).r_hmm.mean()
    assert r_dom > r_rare
