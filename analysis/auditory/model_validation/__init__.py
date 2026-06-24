"""model_validation — behavioural validation of the extended aesthetic-value model.

Tests whether a semantic value term S(t) (an HMM context-belief signal over the
EE/SC grammar association) explains maze preference behaviour over and above the
fluency-only Brielmann & Dayan baseline, out of sample, with the predicted
EE > SC sign.

This package is READ-ONLY with respect to the experiment code and the recorded
results: it reads logged sessions from a results tree and writes only into a
caller-supplied output directory. It never regenerates stimulus sequences and
never modifies src/auditory or the recordings.

See SPEC.md for the full design and the ordered pipeline. Phase 1 (data loading,
sanity, model-free design analysis, bigram latent regressors, collinearity and
parameter recovery) runs on the installed scientific stack. Phase 2 (PyMC nested
models, LOO, posterior predictive, individual differences) requires `pymc` and
`arviz` and is imported lazily.
"""

__version__ = "0.1.0"
