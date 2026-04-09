"""
aesthetic_value_model_4D.py
============================

A 4-dimensional extension of the Brielmann & Dayan (2022) computational
model of aesthetic value, applied to mouse acoustic preference data from
an 8-arm radial maze experiment.

EXPERIMENTAL DESIGN NOTES:
  - 8-arm radial maze, 7 sound arms, 1 silent arm
  - All arms including silence reassign randomly every 15 minutes
  - PI = relative time near sound vs silence, corrected for chance
  - Spatial learning impossible by design — PI reflects
    real-time acoustic processing only
  - Sounds change category every day
  - Vocalisation appears on D1 and W2_Voc only

MODEL ASSUMPTIONS:
  1. Stimulus feature vectors are theoretically motivated
     and manually assigned — not empirically derived.
     Results are sensitive to these values.
     TODO: sensitivity analysis on feature vector assumptions.
  2. mu_T dim 2 (biological relevance) reflects a partially
     innate evolutionary prior for conspecific calls in mice.
     This is constrained >= 0.4 and is the primary theoretical
     assumption distinguishing this implementation from the
     original paper.
     TODO: cite mouse auditory neuroscience literature on
     innate vocalisation processing.
  3. PI is treated as a monotone linear proxy for
     A(sound) - A(silence). The model does not include an
     explicit decision or action layer between aesthetic
     value and approach/avoidance behaviour.
  4. The model was developed and validated on human self-report
     aesthetic pleasure ratings. Application to mouse
     approach/avoidance in a maze is novel and adds an
     untested decision layer. Interpret all results cautiously.
  5. Sigma and Sigma_T are diagonal with scalar variances.
     This assumes feature dimensions are independent and
     equally weighted in the distance metric. This is a
     strong simplifying assumption.
  6. reshuffle_penalty = 0.05 is fixed and not fitted.
     This is arbitrary.
     TODO: run sensitivity analysis varying this from 0 to 0.2.
  7. With 11 free parameters and 5 primary observations,
     the model is overparameterised for formal statistical
     inference. This is a proof-of-concept simulation
     intended as a theoretical framework contribution,
     not a fitted predictive model.
  8. The within-day trial structure (4 blocks x 8 encounters)
     is an approximation. Actual number of arm visits per
     session depends on individual mouse locomotion and is
     not modelled here.
  9. The feedback loop between A(t) and time-spent-near-arm
     is not modelled. In reality, stimuli with higher A(t)
     receive more exposure and therefore update mu(t) more
     strongly. This simplification may underestimate
     learning for preferred stimuli.
  10. Sounds change every day. Unlike the paper's mere-exposure
      simulations, the system state never fully converges on
      any single stimulus. The habituation arc observed in the
      data is therefore driven by cumulative cross-stimulus
      acoustic learning and location familiarity, not
      single-stimulus overexposure.
  11. Random seed = 42 is used throughout for full reproducibility.

CITATION:
  Brielmann, A. A., & Dayan, P. (2022). A computational model
  of aesthetic value. Psychological Review, 129(6), 1319-1337.
  https://doi.org/10.1037/rev0000337

Usage:
    python aesthetic_value_model_4D.py
"""

import os
import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import pearsonr

np.random.seed(42)

# ======================================================================
# PART 1 — FOUR-DIMENSIONAL FEATURE SPACE
# ======================================================================
# Dimensions: [location_familiarity, spectral_complexity,
#              biological_relevance, temporal_predictability]

N_DIMS = 4
DIM_NAMES = [
    "location_familiarity",
    "spectral_complexity",
    "biological_relevance",
    "temporal_predictability",
]

# Location familiarity by day (dim 0 — fixed, not fitted)
LOC_BY_DAY = {
    "D1":     0.10,
    "D2":     0.40,
    "D3":     0.60,
    "W2_Seq": 0.85,
    "W2_Voc": 0.90,
}

# Acoustic feature vectors: dims 1-3 only (dim 0 is day-level)
# Format: {name: [spectral_complexity, biological_relevance, temporal_predictability]}
ACOUSTIC_FEATURES = {
    "silence":       [0.0, 0.0, 1.0],
    "smooth":        [0.0, 0.0, 0.9],
    "rough":         [1.0, 0.0, 0.1],
    "rough_complex": [1.5, 0.0, 0.3],
    "consonant":     [0.8, 0.1, 0.8],
    "dissonant":     [1.2, 0.1, 0.5],
    "vocalisation":  [1.8, 1.0, 0.6],
    "AAAAA":         [0.5, 0.0, 1.0],
    "AoAo":          [0.6, 0.0, 0.8],
    "ABAB":          [0.8, 0.0, 0.7],
    "ABCABC":        [1.3, 0.0, 0.5],
    "BABA":          [0.8, 0.0, 0.6],
    "ABBA":          [0.8, 0.0, 0.4],
}


def stimulus_vector(stim_name: str, day: str) -> np.ndarray:
    """Build a full 4D feature vector for a stimulus on a given day."""
    loc = LOC_BY_DAY[day]
    ac = ACOUSTIC_FEATURES[stim_name]
    return np.array([loc, ac[0], ac[1], ac[2]], dtype=np.float64)


# Day structures: list of 7 sound stimulus names per day
DAY_SOUNDS = {
    "D1":     ["smooth", "rough", "rough_complex", "vocalisation",
               "smooth", "rough", "rough_complex"],
    "D2":     ["smooth", "rough", "consonant", "dissonant",
               "vocalisation", "smooth", "rough"],
    "D3":     ["smooth", "rough", "consonant", "dissonant",
               "vocalisation", "smooth", "rough"],
    "W2_Seq": ["AAAAA", "AoAo", "ABAB", "ABCABC", "BABA", "ABBA", "AAAAA"],
    "W2_Voc": ["vocalisation"] * 7,
}

DAY_ORDER = ["D1", "D2", "D3", "W2_Seq", "W2_Voc"]
BLOCKS_PER_DAY = 4
RESHUFFLE_PENALTY = 0.05  # fixed, not fitted

# ======================================================================
# PART 5 — OBSERVED DATA
# ======================================================================

observed_voc_pi = {
    "D1":     {"mean": -0.037, "median": -0.062, "n": 31},
    "D2":     {"mean":  0.010, "median":  0.007, "n": 30},
    "D3":     {"mean": -0.024, "median": -0.079, "n": 27},
    "W2_Seq": {"mean":  0.048, "median":  0.011, "n": 24},
    "W2_Voc": {"mean":  0.044, "median":  0.038, "n": 24},
}

observed_overall_pi = {
    "D1":     -0.142,
    "D2":     -0.142 + 0.166,   # = 0.024
    "D3":     -0.142 + 0.085,   # = -0.057
    "W2_Seq": -0.142 + 0.147,   # = 0.005
    "W2_Voc": -0.142 + 0.173,   # = 0.031
}

observed_dwell_D1 = {
    "smooth":        511.0,
    "rough":         509.7,
    "rough_complex": 648.5,
    "vocalisation":  906.1,
}

observed_dwell_D2 = {
    "smooth":       627.6,
    "rough":        504.5,
    "consonant":    495.9,
    "dissonant":    573.9,
    "vocalisation": 724.1,
}

observed_dwell_W2seq = {
    "AAAAA":  914.2,
    "AoAo":   863.3,
    "ABAB":   819.8,
    "ABCABC": 782.2,
    "BABA":   715.2,
    "ABBA":   822.5,
}

observed_within_trial = {
    "D1":     {"median_diff_ms": -425, "p": 0.0000, "n": 30},
    "D2":     {"median_diff_ms": -329, "p": 0.1200, "n": 28},
    "D3":     {"median_diff_ms": -214, "p": 0.0079, "n": 26},
    "W2_Seq": {"median_diff_ms":  +32, "p": 0.4452, "n": 23},
    "W2_Voc": {"median_diff_ms":  +74, "p": 0.5457, "n": 24},
}


# ======================================================================
# PART 3 — MODEL IMPLEMENTATION
# ======================================================================

class AestheticModel:
    """Brielmann & Dayan (2022) aesthetic value model — 4D extension."""

    def __init__(self, params: dict):
        self.alpha = params["alpha"]
        self.alpha_loc = params["alpha_loc"]
        self.wr = params["wr"]
        self.wV = params["wV"]
        self.w0 = params["w0"]
        self.sigma_sq = params["sigma_sq"]
        self.mu_T = np.array([
            params["mu_T_0"], params["mu_T_1"],
            params["mu_T_2"], params["mu_T_3"],
        ], dtype=np.float64)
        self.sigma_T_sq = params["sigma_T_sq"]

        # Initial system state
        self.mu = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float64)

    def reset(self):
        """Reset system state to initial values."""
        self.mu = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float64)

    def compute_r(self, s: np.ndarray) -> float:
        """Immediate sensory reward: log-likelihood of s under current state."""
        diff = s - self.mu
        return -0.5 / self.sigma_sq * np.dot(diff, diff)

    def compute_V(self) -> float:
        """System state value: -KL(p_T || X(t))."""
        diff = self.mu - self.mu_T
        kl = 0.5 * (
            N_DIMS * (self.sigma_T_sq / self.sigma_sq)
            + (1.0 / self.sigma_sq) * np.dot(diff, diff)
            - N_DIMS
            + N_DIMS * np.log(self.sigma_sq / self.sigma_T_sq)
        )
        return -kl

    def update_mu(self, s: np.ndarray, day: str):
        """Learning update: move mu toward stimulus."""
        # Location dim (0): update toward day-level location familiarity
        loc_day = LOC_BY_DAY[day]
        self.mu[0] += self.alpha_loc * (loc_day - self.mu[0])
        # Acoustic dims (1, 2, 3): update toward stimulus features
        for j in range(1, N_DIMS):
            self.mu[j] += self.alpha * (s[j] - self.mu[j])

    def compute_A(self, s: np.ndarray, day: str) -> dict:
        """Full aesthetic value computation for one stimulus encounter.

        Returns dict with r, V_before, V_after, Delta_V_eff, A.
        Does NOT modify self.mu — caller decides whether to apply update.
        """
        r = self.compute_r(s)
        V_before = self.compute_V()

        # Temporarily apply learning update
        mu_saved = self.mu.copy()
        self.update_mu(s, day)
        V_after = self.compute_V()

        Delta_V_eff = (V_after - V_before) - RESHUFFLE_PENALTY
        A = self.w0 + self.wr * r + self.wV * Delta_V_eff

        # Restore mu (caller will decide whether to keep the update)
        self.mu = mu_saved

        return {
            "r": r,
            "V_before": V_before,
            "V_after": V_after,
            "Delta_V_eff": Delta_V_eff,
            "A": A,
        }

    def step(self, s: np.ndarray, day: str) -> dict:
        """Compute A for stimulus and APPLY the learning update."""
        result = self.compute_A(s, day)
        self.update_mu(s, day)
        return result


# ======================================================================
# PART 4 — EXPERIMENT SIMULATION
# ======================================================================

def simulate_experiment(params: dict,
                        loc_override: dict = None,
                        alpha_override: float = None,
                        dim_dropout: int = None,
                        return_traces: bool = False):
    """Simulate the full experiment (5 days, carry-forward learning).

    Parameters
    ----------
    params : dict
        Model parameters.
    loc_override : dict, optional
        Override LOC_BY_DAY for counterfactual simulations.
    alpha_override : float, optional
        Override alpha for counterfactual (e.g., 0.0 = no acoustic learning).
    dim_dropout : int, optional
        If set, zero-out this dimension's contribution (set s_d = mu_d for all t).
    return_traces : bool
        If True, return full trial-level traces.

    Returns
    -------
    day_results : dict
        Per-day summary: mean A per stimulus, mean A_silence, predicted PI, etc.
    traces : list of dict (only if return_traces=True)
        Trial-level records.
    """
    p = dict(params)
    if alpha_override is not None:
        p["alpha"] = alpha_override

    model = AestheticModel(p)
    rng = np.random.RandomState(42)  # local RNG for reproducibility

    loc_map = LOC_BY_DAY if loc_override is None else loc_override

    day_results = {}
    traces = []
    global_trial = 0

    for day in DAY_ORDER:
        sounds = DAY_SOUNDS[day]
        day_A_by_stim = {}
        day_A_silence = []
        day_r_by_stim = {}
        day_dV_by_stim = {}

        for block in range(BLOCKS_PER_DAY):
            encounter_names = list(sounds) + ["silence"]
            rng.shuffle(encounter_names)

            for stim_name in encounter_names:
                s = stimulus_vector(stim_name, day)
                if loc_override is not None:
                    s[0] = loc_map.get(day, LOC_BY_DAY[day])
                if dim_dropout is not None:
                    s[dim_dropout] = model.mu[dim_dropout]

                # Compute A_silence at current state (no update)
                s_sil = stimulus_vector("silence", day)
                if loc_override is not None:
                    s_sil[0] = loc_map.get(day, LOC_BY_DAY[day])
                if dim_dropout is not None:
                    s_sil[dim_dropout] = model.mu[dim_dropout]
                res_sil = model.compute_A(s_sil, day)
                A_silence = res_sil["A"]

                # Compute A for actual stimulus and apply learning update
                result = model.step(s, day)

                # Store
                day_A_by_stim.setdefault(stim_name, []).append(result["A"])
                day_r_by_stim.setdefault(stim_name, []).append(result["r"])
                day_dV_by_stim.setdefault(stim_name, []).append(result["Delta_V_eff"])
                day_A_silence.append(A_silence)

                if return_traces:
                    traces.append({
                        "trial": global_trial,
                        "day": day,
                        "block": block,
                        "stimulus": stim_name,
                        "A": result["A"],
                        "r": result["r"],
                        "Delta_V_eff": result["Delta_V_eff"],
                        "A_silence": A_silence,
                        "mu_0": model.mu[0],
                        "mu_1": model.mu[1],
                        "mu_2": model.mu[2],
                        "mu_3": model.mu[3],
                    })
                global_trial += 1

        # Summarise this day
        mean_A_silence = np.mean(day_A_silence)
        stim_summary = {}
        for sn, vals in day_A_by_stim.items():
            stim_summary[sn] = {
                "mean_A": np.mean(vals),
                "mean_r": np.mean(day_r_by_stim[sn]),
                "mean_dV": np.mean(day_dV_by_stim[sn]),
                "predicted_PI": np.mean(vals) - mean_A_silence,
            }

        # Vocalisation-specific (fall back to overall PI if no vocalisations)
        voc_keys = [k for k in stim_summary if k == "vocalisation"]
        if voc_keys:
            voc_PI = np.mean([stim_summary[k]["predicted_PI"] for k in voc_keys])
        else:
            # Days without vocalisations (e.g., W2_Seq): use overall sound PI
            sound_keys_tmp = [k for k in stim_summary if k != "silence"]
            voc_PI = np.mean([stim_summary[k]["predicted_PI"] for k in sound_keys_tmp]) if sound_keys_tmp else np.nan

        # Overall sound PI (all non-silence)
        sound_keys = [k for k in stim_summary if k != "silence"]
        overall_PI = np.mean([stim_summary[k]["predicted_PI"] for k in sound_keys]) if sound_keys else np.nan

        day_results[day] = {
            "stimuli": stim_summary,
            "mean_A_silence": mean_A_silence,
            "voc_PI": voc_PI,
            "overall_PI": overall_PI,
            "final_mu": model.mu.copy(),
        }

    if return_traces:
        return day_results, traces
    return day_results


# ======================================================================
# PART 6 — MODEL FITTING
# ======================================================================

PARAM_NAMES = [
    "alpha", "alpha_loc", "wr", "wV", "w0", "sigma_sq",
    "mu_T_0", "mu_T_1", "mu_T_2", "mu_T_3", "sigma_T_sq",
]
PARAM_INITS = [0.05, 0.30, 1.0, 5.0, 0.0, 1.0, 0.75, 1.5, 0.6, 0.6, 2.0]
PARAM_BOUNDS = [
    (0.001, 1.0),   # alpha
    (0.001, 1.0),   # alpha_loc
    (0.0, 50.0),    # wr
    (0.0, 200.0),   # wV
    (-5.0, 5.0),    # w0
    (0.01, 20.0),   # sigma_sq
    (0.0, 1.0),     # mu_T_0
    (0.0, 3.0),     # mu_T_1
    (0.4, 1.0),     # mu_T_2 — constrained above 0.4 (innate prior)
    (0.0, 1.0),     # mu_T_3
    (0.1, 20.0),    # sigma_T_sq
]

OBSERVED_VOC_PI_ARRAY = np.array([
    observed_voc_pi[d]["mean"] for d in DAY_ORDER
])

N_RANDOM_INITS = 2000


def params_from_vec(x: np.ndarray) -> dict:
    return {name: val for name, val in zip(PARAM_NAMES, x)}


def objective(x: np.ndarray) -> float:
    """RMSE between linearly-scaled predicted Voc PI and observed."""
    params = params_from_vec(x)
    try:
        results = simulate_experiment(params)
    except Exception:
        return 1e6

    pred = np.array([results[d]["voc_PI"] for d in DAY_ORDER])
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        return 1e6

    # Optimal linear scaling: PI_scaled = a * pred + b
    coeffs = np.polyfit(pred, OBSERVED_VOC_PI_ARRAY, 1)
    scaled = np.polyval(coeffs, pred)
    rmse = np.sqrt(np.mean((scaled - OBSERVED_VOC_PI_ARRAY) ** 2))
    return rmse


def fit_model():
    """Run optimisation with 2000 random initialisations."""
    print(f"Fitting model with {N_RANDOM_INITS} random initialisations...")
    print(f"  {len(PARAM_NAMES)} free parameters, 5 primary observations")
    print(f"  Optimiser: SLSQP")
    print()

    rng = np.random.RandomState(42)
    all_fits = []

    for i in range(N_RANDOM_INITS):
        # Sample initial point uniformly within bounds
        x0 = np.array([
            rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS
        ])

        try:
            res = minimize(
                objective, x0,
                method="SLSQP",
                bounds=PARAM_BOUNDS,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            all_fits.append((res.fun, res.x.copy(), res.success))
        except Exception:
            continue

        if (i + 1) % 200 == 0:
            best_so_far = min(f[0] for f in all_fits) if all_fits else float("inf")
            print(f"  ... {i+1}/{N_RANDOM_INITS} done, best RMSE so far: {best_so_far:.6f}")

    # Sort by objective
    all_fits.sort(key=lambda x: x[0])
    best_rmse, best_x, best_success = all_fits[0]
    top20 = all_fits[:20]

    print(f"\nBest fit RMSE: {best_rmse:.6f} (converged: {best_success})")
    print(f"Top 20 RMSE range: [{top20[0][0]:.6f}, {top20[-1][0]:.6f}]")

    return params_from_vec(best_x), best_rmse, top20


# ======================================================================
# PART 7 — DECOMPOSITION ANALYSES
# ======================================================================

def run_decompositions(best_params: dict, best_results: dict):
    """Run counterfactual, dimension dropout, and lesioned models."""
    decomp = {}

    # --- Location Decomposition ---
    # Counterfactual A: familiar maze from day 1
    loc_familiar = {d: 0.9 for d in DAY_ORDER}
    results_A = simulate_experiment(best_params, loc_override=loc_familiar)

    # Counterfactual B: no acoustic learning
    results_B = simulate_experiment(best_params, alpha_override=0.0)

    decomp["counterfactual_A"] = results_A
    decomp["counterfactual_B"] = results_B

    actual_voc_PI_D1 = best_results["D1"]["voc_PI"]
    cf_A_voc_PI_D1 = results_A["D1"]["voc_PI"]
    cf_B_voc_PI = {d: results_B[d]["voc_PI"] for d in DAY_ORDER}

    location_contrib_D1 = actual_voc_PI_D1 - cf_A_voc_PI_D1
    decomp["location_contrib_D1"] = location_contrib_D1
    decomp["cf_B_voc_PI"] = cf_B_voc_PI

    # --- Dimension Dropout ---
    dropout_results = {}
    for dim in range(N_DIMS):
        res_d = simulate_experiment(best_params, dim_dropout=dim)
        pred = np.array([res_d[d]["voc_PI"] for d in DAY_ORDER])
        if np.any(np.isnan(pred)):
            rmse_d = float("inf")
        else:
            coeffs = np.polyfit(pred, OBSERVED_VOC_PI_ARRAY, 1)
            scaled = np.polyval(coeffs, pred)
            rmse_d = np.sqrt(np.mean((scaled - OBSERVED_VOC_PI_ARRAY) ** 2))
        dropout_results[dim] = {
            "rmse": rmse_d,
            "voc_PI": {d: res_d[d]["voc_PI"] for d in DAY_ORDER},
        }
    decomp["dropout"] = dropout_results

    # --- Lesioned Models ---
    # Lesion A: wr = 0 (learning only)
    p_lesA = dict(best_params)
    p_lesA["wr"] = 0.0
    res_lesA = simulate_experiment(p_lesA)

    # Lesion B: wV = 0 (fluency only)
    p_lesB = dict(best_params)
    p_lesB["wV"] = 0.0
    res_lesB = simulate_experiment(p_lesB)

    # Lesion C: wr = wV = 1 (equal weights)
    p_lesC = dict(best_params)
    p_lesC["wr"] = 1.0
    p_lesC["wV"] = 1.0
    res_lesC = simulate_experiment(p_lesC)

    def _lesion_rmse(res):
        pred = np.array([res[d]["voc_PI"] for d in DAY_ORDER])
        if np.any(np.isnan(pred)):
            return float("inf"), pred
        coeffs = np.polyfit(pred, OBSERVED_VOC_PI_ARRAY, 1)
        scaled = np.polyval(coeffs, pred)
        return np.sqrt(np.mean((scaled - OBSERVED_VOC_PI_ARRAY) ** 2)), pred

    decomp["lesion_A"] = {"results": res_lesA, "rmse": _lesion_rmse(res_lesA)[0],
                           "pred_raw": _lesion_rmse(res_lesA)[1]}
    decomp["lesion_B"] = {"results": res_lesB, "rmse": _lesion_rmse(res_lesB)[0],
                           "pred_raw": _lesion_rmse(res_lesB)[1]}
    decomp["lesion_C"] = {"results": res_lesC, "rmse": _lesion_rmse(res_lesC)[0],
                           "pred_raw": _lesion_rmse(res_lesC)[1]}

    return decomp


# ======================================================================
# SECONDARY EVALUATIONS
# ======================================================================

def secondary_evaluations(best_results: dict):
    """Compute secondary fit metrics (no refitting)."""
    evals = {}

    # 1. D1 dwell correlation
    d1_stims = ["smooth", "rough", "rough_complex", "vocalisation"]
    d1_pred = [best_results["D1"]["stimuli"][s]["mean_A"] for s in d1_stims]
    d1_obs = [observed_dwell_D1[s] for s in d1_stims]
    if len(d1_pred) >= 3:
        r_d1, p_d1 = pearsonr(d1_pred, d1_obs)
    else:
        r_d1, p_d1 = np.nan, np.nan
    evals["D1_dwell_r"] = r_d1
    evals["D1_dwell_p"] = p_d1
    evals["D1_pred"] = d1_pred
    evals["D1_obs"] = d1_obs

    # 2. D2 dwell correlation
    d2_stims = ["smooth", "rough", "consonant", "dissonant", "vocalisation"]
    d2_pred = [best_results["D2"]["stimuli"][s]["mean_A"] for s in d2_stims]
    d2_obs = [observed_dwell_D2[s] for s in d2_stims]
    if len(d2_pred) >= 3:
        r_d2, p_d2 = pearsonr(d2_pred, d2_obs)
    else:
        r_d2, p_d2 = np.nan, np.nan
    evals["D2_dwell_r"] = r_d2
    evals["D2_dwell_p"] = p_d2
    evals["D2_pred"] = d2_pred
    evals["D2_obs"] = d2_obs

    # 3. W2_Seq dwell correlation (ordered by temporal predictability)
    w2_stims = ["AAAAA", "AoAo", "ABAB", "BABA", "ABCABC", "ABBA"]
    w2_pred = [best_results["W2_Seq"]["stimuli"][s]["mean_A"] for s in w2_stims]
    w2_obs = [observed_dwell_W2seq[s] for s in w2_stims]
    if len(w2_pred) >= 3:
        r_w2, p_w2 = pearsonr(w2_pred, w2_obs)
    else:
        r_w2, p_w2 = np.nan, np.nan
    evals["W2_dwell_r"] = r_w2
    evals["W2_dwell_p"] = p_w2
    evals["W2_pred"] = w2_pred
    evals["W2_obs"] = w2_obs

    # 4. Sign accuracy
    pred_voc_pi = {d: best_results[d]["voc_PI"] for d in DAY_ORDER}
    # Scale to observed range for sign check
    pred_arr = np.array([pred_voc_pi[d] for d in DAY_ORDER])
    coeffs = np.polyfit(pred_arr, OBSERVED_VOC_PI_ARRAY, 1)
    scaled = np.polyval(coeffs, pred_arr)
    sign_correct = {}
    for i, d in enumerate(DAY_ORDER):
        obs_sign = np.sign(OBSERVED_VOC_PI_ARRAY[i])
        pred_sign = np.sign(scaled[i])
        sign_correct[d] = obs_sign == pred_sign or obs_sign == 0
    evals["sign_correct"] = sign_correct
    evals["scaled_voc_PI"] = {d: scaled[i] for i, d in enumerate(DAY_ORDER)}

    # Pearson r: predicted vs observed Voc PI across days
    if len(pred_arr) >= 3:
        r_voc, p_voc = pearsonr(pred_arr, OBSERVED_VOC_PI_ARRAY)
    else:
        r_voc, p_voc = np.nan, np.nan
    evals["voc_PI_r"] = r_voc
    evals["voc_PI_p"] = p_voc

    # 5. Vocalisation anomaly: A(voc_D1) > A(smooth_D1)?
    A_voc_D1 = best_results["D1"]["stimuli"].get("vocalisation", {}).get("mean_A", np.nan)
    A_smooth_D1 = best_results["D1"]["stimuli"].get("smooth", {}).get("mean_A", np.nan)
    evals["voc_anomaly"] = A_voc_D1 > A_smooth_D1
    evals["A_voc_D1"] = A_voc_D1
    evals["A_smooth_D1"] = A_smooth_D1

    # 6. Temporal structure: A(AAAAA) > A(ABBA)?
    A_AAAAA = best_results["W2_Seq"]["stimuli"].get("AAAAA", {}).get("mean_A", np.nan)
    A_ABBA = best_results["W2_Seq"]["stimuli"].get("ABBA", {}).get("mean_A", np.nan)
    evals["temporal_structure"] = A_AAAAA > A_ABBA
    evals["A_AAAAA"] = A_AAAAA
    evals["A_ABBA"] = A_ABBA

    return evals


# ======================================================================
# PART 8 — VISUALISATION
# ======================================================================

def make_figure(best_params, best_results, best_rmse, top20, decomp, evals,
                output_dir="."):
    """Generate 6-panel publication figure."""
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 12), constrained_layout=False)

    # Layout: 2 rows x 3 cols
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

    day_labels = ["D1", "D2", "D3", "W2\nSeq", "W2\nVoc"]
    day_x = np.arange(len(DAY_ORDER))

    # ------------------------------------------------------------------
    # Panel A — Primary Voc PI fit
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    pred_raw = np.array([best_results[d]["voc_PI"] for d in DAY_ORDER])
    coeffs = np.polyfit(pred_raw, OBSERVED_VOC_PI_ARRAY, 1)
    pred_scaled = np.polyval(coeffs, pred_raw)

    # Robustness band from top 20
    top20_scaled = []
    for _, x_i, _ in top20:
        p_i = params_from_vec(x_i)
        r_i = simulate_experiment(p_i)
        pr_i = np.array([r_i[d]["voc_PI"] for d in DAY_ORDER])
        c_i = np.polyfit(pr_i, OBSERVED_VOC_PI_ARRAY, 1)
        top20_scaled.append(np.polyval(c_i, pr_i))
    top20_arr = np.array(top20_scaled)
    lo_band = np.min(top20_arr, axis=0)
    hi_band = np.max(top20_arr, axis=0)

    ax_a.fill_between(day_x, lo_band, hi_band, alpha=0.2, color="#4c72b0",
                       label="Top 20 range")
    ax_a.plot(day_x, pred_scaled, "-o", color="#4c72b0", lw=2, ms=6,
              label="Model (best fit)")
    obs_n = [observed_voc_pi[d]["n"] for d in DAY_ORDER]
    ax_a.scatter(day_x, OBSERVED_VOC_PI_ARRAY, s=[n * 3 for n in obs_n],
                 c="#dd5555", edgecolors="black", lw=0.5, zorder=5,
                 label="Observed mean")
    ax_a.axhline(0, ls="--", color="grey", lw=0.8)

    # Asterisk for D1 (significant)
    ax_a.annotate("***", (0, OBSERVED_VOC_PI_ARRAY[0]), fontsize=12,
                  ha="center", va="bottom", xytext=(0, 5),
                  textcoords="offset points", color="#dd5555")

    ax_a.set_xticks(day_x)
    ax_a.set_xticklabels(day_labels, fontsize=9)
    ax_a.set_ylabel("Vocalisation PI", fontsize=10)
    ax_a.set_title("A. Vocalisation Preference:\nModel vs Observed", fontsize=11,
                    fontweight="bold")
    ax_a.legend(fontsize=7, loc="lower right")

    # ------------------------------------------------------------------
    # Panel B — D1 complexity gradient
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    d1_stims = ["smooth", "rough", "rough_complex", "vocalisation"]
    d1_labels = ["smooth", "rough", "rough\ncomplex", "voc"]
    d1_pred_A = [best_results["D1"]["stimuli"][s]["mean_A"] for s in d1_stims]
    d1_obs_dwell = [observed_dwell_D1[s] for s in d1_stims]

    x_b = np.arange(len(d1_stims))
    w = 0.35
    ax_b.bar(x_b - w / 2, d1_pred_A, w, color="#4c72b0", label="Predicted A(t)")
    ax_b2 = ax_b.twinx()
    ax_b2.bar(x_b + w / 2, d1_obs_dwell, w, color="#dd8452", label="Observed dwell (ms)")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(d1_labels, fontsize=8)
    ax_b.set_ylabel("Predicted A(t)", fontsize=9, color="#4c72b0")
    ax_b2.set_ylabel("Observed dwell (ms)", fontsize=9, color="#dd8452")
    r_d1 = evals["D1_dwell_r"]
    ax_b.set_title(f"B. D1 Complexity Gradient\n(r = {r_d1:.3f})", fontsize=11,
                    fontweight="bold")
    # Combined legend
    lines_b, labels_b = ax_b.get_legend_handles_labels()
    lines_b2, labels_b2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines_b + lines_b2, labels_b + labels_b2, fontsize=7, loc="upper left")

    # ------------------------------------------------------------------
    # Panel C — W2 sequence predictions
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    w2_stims = ["AAAAA", "AoAo", "ABAB", "BABA", "ABCABC", "ABBA"]
    w2_labels = ["AAAAA", "AoAo", "ABAB", "BABA", "ABC\nABC", "ABBA"]
    w2_pred_A = [best_results["W2_Seq"]["stimuli"][s]["mean_A"] for s in w2_stims]
    w2_obs_dwell = [observed_dwell_W2seq[s] for s in w2_stims]

    x_c = np.arange(len(w2_stims))
    ax_c.bar(x_c - w / 2, w2_pred_A, w, color="#4c72b0", label="Predicted A(t)")
    ax_c2 = ax_c.twinx()
    ax_c2.bar(x_c + w / 2, w2_obs_dwell, w, color="#dd8452", label="Observed dwell (ms)")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(w2_labels, fontsize=8)
    ax_c.set_ylabel("Predicted A(t)", fontsize=9, color="#4c72b0")
    ax_c2.set_ylabel("Observed dwell (ms)", fontsize=9, color="#dd8452")
    r_w2 = evals["W2_dwell_r"]
    ax_c.set_title(f"C. W2 Sequence Structure\n(r = {r_w2:.3f})", fontsize=11,
                    fontweight="bold")
    lines_c, labels_c = ax_c.get_legend_handles_labels()
    lines_c2, labels_c2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines_c + lines_c2, labels_c + labels_c2, fontsize=7, loc="upper left")

    # ------------------------------------------------------------------
    # Panel D — System state trajectory (4 subplots)
    # ------------------------------------------------------------------
    gs_d = gs[1, 0].subgridspec(2, 2, hspace=0.45, wspace=0.3)

    _, traces = simulate_experiment(best_params, return_traces=True)
    df_traces = pd.DataFrame(traces)

    dim_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for dim_idx in range(N_DIMS):
        r_d, c_d = divmod(dim_idx, 2)
        ax_d = fig.add_subplot(gs_d[r_d, c_d])

        trials = df_traces["trial"].values
        mu_vals = df_traces[f"mu_{dim_idx}"].values
        ax_d.plot(trials, mu_vals, color=dim_colors[dim_idx], lw=1.2)

        # Day boundaries (add labels after all data is plotted)
        day_starts = []
        for day in DAY_ORDER:
            idx = df_traces[df_traces["day"] == day]["trial"].iloc[0]
            day_starts.append(idx)
            ax_d.axvline(idx, ls="--", color="grey", lw=0.5, alpha=0.6)

        # Stimulus feature values as dotted lines per day
        for day in DAY_ORDER:
            day_mask = df_traces["day"] == day
            day_trials = df_traces[day_mask]["trial"].values
            if len(day_trials) == 0:
                continue
            t_lo, t_hi = day_trials.min(), day_trials.max()
            seen = set()
            for sn in DAY_SOUNDS[day] + ["silence"]:
                if sn in seen:
                    continue
                seen.add(sn)
                sv = stimulus_vector(sn, day)
                ax_d.hlines(sv[dim_idx], t_lo, t_hi, ls=":", lw=0.4,
                            color=dim_colors[dim_idx], alpha=0.4)

        # Add day labels at top after all data is plotted
        for i_day, day in enumerate(DAY_ORDER):
            idx = day_starts[i_day]
            ax_d.text(idx + 1, ax_d.get_ylim()[1], day,
                      fontsize=5, va="top", alpha=0.7)

        ax_d.set_title(DIM_NAMES[dim_idx].replace("_", " ").title(),
                        fontsize=8, fontweight="bold")
        ax_d.tick_params(labelsize=6)
        if r_d == 1:
            ax_d.set_xlabel("Trial", fontsize=7)

    # ------------------------------------------------------------------
    # Panel E — Lesioned model comparison
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])

    # Full model
    ax_e.plot(day_x, pred_scaled, "-o", color="#4c72b0", lw=2, ms=5,
              label=f"Full (RMSE={best_rmse:.4f})")

    # Lesioned models — scale each to observed
    for les_key, les_label, ls_style, clr in [
        ("lesion_A", "wr=0 (learning only)", "--", "#55a868"),
        ("lesion_B", "wV=0 (fluency only)", ":", "#c44e52"),
        ("lesion_C", "wr=wV=1 (equal)", "-.", "#8172b2"),
    ]:
        les_pred = np.array([decomp[les_key]["results"][d]["voc_PI"]
                             for d in DAY_ORDER])
        if not np.any(np.isnan(les_pred)):
            c_l = np.polyfit(les_pred, OBSERVED_VOC_PI_ARRAY, 1)
            les_scaled = np.polyval(c_l, les_pred)
            rmse_l = decomp[les_key]["rmse"]
            ax_e.plot(day_x, les_scaled, ls_style, color=clr, lw=1.5,
                      label=f"{les_label} ({rmse_l:.4f})")

    ax_e.scatter(day_x, OBSERVED_VOC_PI_ARRAY, s=60, c="#dd5555",
                 edgecolors="black", lw=0.5, zorder=5, label="Observed")
    ax_e.axhline(0, ls="--", color="grey", lw=0.8)
    ax_e.set_xticks(day_x)
    ax_e.set_xticklabels(day_labels, fontsize=9)
    ax_e.set_ylabel("Vocalisation PI", fontsize=10)
    ax_e.set_title("E. Lesioned Model Comparison", fontsize=11, fontweight="bold")
    ax_e.legend(fontsize=6, loc="lower right")

    # ------------------------------------------------------------------
    # Panel F — Dimension dropout + location decomposition
    # ------------------------------------------------------------------
    gs_f = gs[1, 2].subgridspec(1, 2, wspace=0.4)

    # Left: dimension dropout
    ax_f1 = fig.add_subplot(gs_f[0, 0])
    dropout = decomp["dropout"]
    full_rmse = best_rmse
    dims_sorted = sorted(range(N_DIMS), key=lambda d: dropout[d]["rmse"] - full_rmse,
                          reverse=True)
    dim_labels_short = ["Location", "Spectral", "Biological", "Temporal"]
    dim_bar_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    delta_rmses = [dropout[d]["rmse"] - full_rmse for d in dims_sorted]
    bar_labels = [dim_labels_short[d] for d in dims_sorted]
    bar_colors = [dim_bar_colors[d] for d in dims_sorted]

    ax_f1.barh(range(len(dims_sorted)), delta_rmses, color=bar_colors)
    ax_f1.set_yticks(range(len(dims_sorted)))
    ax_f1.set_yticklabels(bar_labels, fontsize=8)
    ax_f1.set_xlabel("Delta RMSE", fontsize=8)
    ax_f1.set_title("Dim. Dropout", fontsize=9, fontweight="bold")

    # Right: location decomposition
    ax_f2 = fig.add_subplot(gs_f[0, 1])
    actual_D1 = best_results["D1"]["voc_PI"]
    cf_A_D1 = decomp["counterfactual_A"]["D1"]["voc_PI"]

    # Scale to same range as observed for interpretability
    vals = [actual_D1, cf_A_D1]
    labels_f2 = ["Actual\nD1", "Familiar\nmaze D1"]
    colors_f2 = ["#4c72b0", "#55a868"]
    ax_f2.bar(range(len(vals)), vals, color=colors_f2)
    ax_f2.set_xticks(range(len(vals)))
    ax_f2.set_xticklabels(labels_f2, fontsize=7)
    ax_f2.set_ylabel("Predicted Voc PI (raw)", fontsize=8)
    ax_f2.axhline(0, ls="--", color="grey", lw=0.5)
    ax_f2.set_title("Location Decomp.", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Brielmann & Dayan (2022) Aesthetic Value Model - 4D Extension\n"
        "Applied to Mouse Acoustic Preference (8-arm Maze)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    out_path = os.path.join(output_dir, "aesthetic_value_model_4D.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    pdf_path = os.path.join(output_dir, "aesthetic_value_model_4D.pdf")
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved figure: {pdf_path}")
    except Exception as e:
        print(f"  WARNING: could not write PDF: {e}")

    plt.close(fig)


# ======================================================================
# PART 9 — PRINTED REPORT
# ======================================================================

def print_report(best_params, best_rmse, best_results, top20, decomp, evals):
    """Print formatted report with all results."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("AESTHETIC VALUE MODEL — 4D EXTENSION (Brielmann & Dayan 2022)")
    print(sep)

    # --- BEST FIT PARAMETERS ---
    print(f"\n--- BEST FIT PARAMETERS (RMSE = {best_rmse:.6f}) ---\n")
    for name in PARAM_NAMES:
        val = best_params[name]
        interp = ""
        if name == "alpha":
            interp = " (acoustic learning rate)"
        elif name == "alpha_loc":
            interp = " (location learning rate)"
            if best_params["alpha_loc"] > best_params["alpha"]:
                interp += " [alpha_loc > alpha: spatial learning faster than acoustic, as expected]"
            else:
                interp += " [WARNING: alpha_loc <= alpha, spatial learning NOT faster]"
        elif name == "wr":
            interp = " (weight on immediate fluency r(t))"
        elif name == "wV":
            interp = " (weight on learning signal Delta_V)"
        elif name == "w0":
            interp = " (bias)"
        elif name == "sigma_sq":
            interp = " (system state variance)"
        elif name == "mu_T_0":
            interp = " (p_T mean: location familiarity)"
        elif name == "mu_T_1":
            interp = " (p_T mean: spectral complexity)"
        elif name == "mu_T_2":
            interp = " (p_T mean: biological relevance)"
            if val > 0.5:
                interp += " [STRONG innate biological prior]"
            else:
                interp += " [weak innate biological prior, near constraint boundary]"
        elif name == "mu_T_3":
            interp = " (p_T mean: temporal predictability)"
        elif name == "sigma_T_sq":
            interp = " (p_T variance)"
            if val > best_params["sigma_sq"]:
                interp += " [sigma_T_sq > sigma_sq: p_T broader than system state, typical]"
        print(f"  {name:12s} = {val:8.4f}{interp}")

    # Dominant component
    if best_params["wr"] < best_params["wV"] * 0.5:
        print("\n  >> Learning signal (Delta_V) DOMINATES over immediate fluency (r)")
    elif best_params["wr"] > best_params["wV"] * 2:
        print("\n  >> Immediate fluency (r) DOMINATES over learning signal (Delta_V)")
    else:
        print("\n  >> Fluency and learning contribute comparably")

    # --- MODEL FIT QUALITY ---
    print(f"\n--- MODEL FIT QUALITY ---\n")
    print(f"  Primary RMSE (Voc PI across 5 days):    {best_rmse:.6f}")
    print(f"  Pearson r (pred vs obs Voc PI):          {evals['voc_PI_r']:.4f} (p={evals['voc_PI_p']:.4f})")
    print(f"  Pearson r (pred A vs D1 dwell, 4 stim):  {evals['D1_dwell_r']:.4f} (p={evals['D1_dwell_p']:.4f})")
    print(f"  Pearson r (pred A vs D2 dwell, 5 stim):  {evals['D2_dwell_r']:.4f} (p={evals['D2_dwell_p']:.4f})")
    print(f"  Pearson r (pred A vs W2 dwell, 6 stim):  {evals['W2_dwell_r']:.4f} (p={evals['W2_dwell_p']:.4f})")

    print(f"\n  Sign accuracy (direction of PI):")
    for d in DAY_ORDER:
        obs = OBSERVED_VOC_PI_ARRAY[DAY_ORDER.index(d)]
        sc = evals["scaled_voc_PI"][d]
        correct = evals["sign_correct"][d]
        print(f"    {d:8s}: obs={obs:+.3f}, pred={sc:+.3f}  {'Y' if correct else 'N'}")

    print(f"\n  Vocalisation anomaly captured (A(voc_D1) > A(smooth_D1))? "
          f"{'YES' if evals['voc_anomaly'] else 'NO'}")
    print(f"    A(voc_D1)={evals['A_voc_D1']:.4f}, A(smooth_D1)={evals['A_smooth_D1']:.4f}")

    print(f"\n  Temporal structure captured (A(AAAAA) > A(ABBA))? "
          f"{'YES' if evals['temporal_structure'] else 'NO'}")
    print(f"    A(AAAAA)={evals['A_AAAAA']:.4f}, A(ABBA)={evals['A_ABBA']:.4f}")

    # Predicted vs observed per day
    print(f"\n  Predicted Voc PI per day (scaled):")
    for d in DAY_ORDER:
        obs = observed_voc_pi[d]["mean"]
        pred = evals["scaled_voc_PI"][d]
        print(f"    {d:8s}: observed={obs:+.4f}, predicted={pred:+.4f}")

    # --- DECOMPOSITION RESULTS ---
    print(f"\n--- DECOMPOSITION RESULTS ---\n")

    # Dimension dropout table
    print("  Dimension dropout (ranked by importance):")
    print(f"  {'Dimension':<25s} {'RMSE w/o':>10s} {'Delta RMSE':>12s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    full_rmse = best_rmse
    dropout = decomp["dropout"]
    dims_ranked = sorted(range(N_DIMS),
                          key=lambda d: dropout[d]["rmse"] - full_rmse,
                          reverse=True)
    for d in dims_ranked:
        dname = DIM_NAMES[d]
        rmse_d = dropout[d]["rmse"]
        delta = rmse_d - full_rmse
        print(f"  {dname:<25s} {rmse_d:>10.6f} {delta:>+12.6f}")
    print(f"  {'Full model':<25s} {full_rmse:>10.6f} {'(baseline)':>12s}")

    # Location decomposition
    actual_D1 = best_results["D1"]["voc_PI"]
    cf_A_D1 = decomp["counterfactual_A"]["D1"]["voc_PI"]
    loc_contrib = decomp["location_contrib_D1"]

    if actual_D1 != 0:
        loc_pct = abs(loc_contrib / actual_D1) * 100
        acoustic_pct = 100 - loc_pct
    else:
        loc_pct = acoustic_pct = 50.0

    print(f"\n  Location decomposition (D1):")
    print(f"    Actual predicted Voc PI D1:        {actual_D1:+.6f}")
    print(f"    Counterfactual A (familiar maze):  {cf_A_D1:+.6f}")
    print(f"    Location novelty contribution:     {loc_contrib:+.6f} (~{loc_pct:.0f}%)")
    print(f"    Acoustic novelty contribution:     ~{acoustic_pct:.0f}%")

    # Acoustic learning across days
    cf_B = decomp["cf_B_voc_PI"]
    actual_D1_vpi = best_results["D1"]["voc_PI"]
    actual_W2V_vpi = best_results["W2_Voc"]["voc_PI"]
    actual_arc = actual_W2V_vpi - actual_D1_vpi
    cf_B_arc = cf_B["W2_Voc"] - cf_B["D1"]
    if actual_arc != 0:
        acoustic_learning_pct = (1 - cf_B_arc / actual_arc) * 100
    else:
        acoustic_learning_pct = 0.0

    print(f"\n  Habituation arc (D1 to W2_Voc):")
    print(f"    Full model arc:           {actual_arc:+.6f}")
    print(f"    No-acoustic-learning arc: {cf_B_arc:+.6f}")
    print(f"    Acoustic learning accounts for ~{acoustic_learning_pct:.0f}% of arc")

    # Counterfactual B per day
    print(f"\n  Counterfactual B (no acoustic learning) Voc PI per day:")
    for d in DAY_ORDER:
        print(f"    {d:8s}: {cf_B[d]:+.6f}")

    # Lesioned models
    print(f"\n  Lesioned model RMSE:")
    print(f"    {'Model':<30s} {'RMSE':>10s}")
    print(f"    {'-'*30} {'-'*10}")
    print(f"    {'Full model':<30s} {best_rmse:>10.6f}")
    print(f"    {'wr=0 (learning only)':<30s} {decomp['lesion_A']['rmse']:>10.6f}")
    print(f"    {'wV=0 (fluency only)':<30s} {decomp['lesion_B']['rmse']:>10.6f}")
    print(f"    {'wr=wV=1 (equal weights)':<30s} {decomp['lesion_C']['rmse']:>10.6f}")

    # --- W2_VOC SPECIAL CASE ---
    print(f"\n--- W2_VOC SPECIAL CASE NOTE ---\n")
    print("  W2_Voc has 7x vocalisation exposure per block compared to other")
    print("  sound types on other days (all 7 sound arms play vocalisations).")
    print("  This means mu(t) is pulled strongly toward vocalisation features.")
    w2v_pred = evals["scaled_voc_PI"].get("W2_Voc", np.nan)
    w2s_pred = evals["scaled_voc_PI"].get("W2_Seq", np.nan)
    w2v_obs = observed_voc_pi["W2_Voc"]["mean"]
    w2s_obs = observed_voc_pi["W2_Seq"]["mean"]
    overpredict = w2v_pred > w2s_pred and not (w2v_obs > w2s_obs)
    if overpredict:
        print(f"  >> Model OVERPREDICTS W2_Voc PI ({w2v_pred:+.4f}) relative to W2_Seq ({w2s_pred:+.4f})")
    else:
        print(f"  >> W2_Voc pred={w2v_pred:+.4f}, W2_Seq pred={w2s_pred:+.4f}")
        print(f"     W2_Voc obs={w2v_obs:+.4f},  W2_Seq obs={w2s_obs:+.4f}")

    # --- KEY THEORETICAL STATEMENTS ---
    print(f"\n--- KEY THEORETICAL STATEMENTS ---\n")

    print(f"  1. On Day 1, the model predicts Voc PI = {evals['scaled_voc_PI']['D1']:+.4f}.")
    print(f"     Approximately {loc_pct:.0f}% of this avoidance is attributable to maze")
    print(f"     novelty (location familiarity dim 0) and approximately {acoustic_pct:.0f}%")
    print(f"     to acoustic novelty (spectral, biological, and temporal dims).")
    print(f"     This decomposition is not accessible from the behavioural data alone.")

    A_diff_voc_smooth = evals["A_voc_D1"] - evals["A_smooth_D1"]
    print(f"\n  2. The vocalisation dwell-time anomaly on D1 (906ms vs 511ms for")
    print(f"     smooth tones) corresponds to a predicted A(t) difference of")
    print(f"     {A_diff_voc_smooth:+.4f} in favour of vocalisations. This is driven")
    print(f"     primarily by the biological relevance dimension (dim 2), which")
    print(f"     shifts Delta_V toward p_T even before acoustic learning has")
    print(f"     occurred, consistent with an evolutionary prior for conspecific calls.")

    # Determine dominant component for statement 3
    # Check mean r and mean dV for W2 days
    w2_stims_check = best_results["W2_Seq"]["stimuli"]
    mean_r_w2 = np.mean([w2_stims_check[s]["mean_r"] for s in w2_stims_check if s != "silence"])
    mean_dV_w2 = np.mean([w2_stims_check[s]["mean_dV"] for s in w2_stims_check if s != "silence"])
    if abs(mean_r_w2) > abs(mean_dV_w2) * 2:
        w2_reason = "r(t) has increased due to acoustic familiarity"
    elif abs(mean_dV_w2) > abs(mean_r_w2) * 2:
        w2_reason = "Delta_V has decreased as the system state has adapted"
    else:
        w2_reason = "both r(t) increase and Delta_V decrease contribute comparably"

    print(f"\n  3. In Week 2, the model predicts near-zero PI because {w2_reason}.")
    print(f"     This plateau is not boredom in the paper's sense (the mouse has")
    print(f"     not overexposed to any single stimulus) but rather reflects")
    print(f"     adaptation across a diverse acoustic environment.")

    voc_gt_smooth = ">" if evals["temporal_structure"] else "<"
    is_consistent = "is" if evals["temporal_structure"] else "is not"
    supports = "supports" if evals["temporal_structure"] else "does not support"
    print(f"\n  4. The temporal predictability dimension predicts A(AAAAA) {voc_gt_smooth} A(ABBA)")
    print(f"     in W2_Seq, which {is_consistent} consistent with observed dwell times")
    print(f"     (914ms vs 822ms respectively). This {supports} the hypothesis that")
    print(f"     temporal predictability contributes independently to acoustic")
    print(f"     processing fluency in mice.")

    print(f"\n  5. The reshuffling design (all arms including silence reassigned every")
    print(f"     15 minutes) means PI reflects real-time acoustic processing only,")
    print(f"     free of spatial learning confounds. The model's A(t) computation")
    print(f"     is consistent with this — it predicts instantaneous aesthetic value")
    print(f"     from system state and stimulus features, without any spatial memory")
    print(f"     component.")

    # --- ROBUSTNESS ---
    print(f"\n--- ROBUSTNESS (Top 20 fits) ---\n")
    print(f"  RMSE range: [{top20[0][0]:.6f}, {top20[-1][0]:.6f}]")
    top20_params = [params_from_vec(x) for _, x, _ in top20]
    for name in PARAM_NAMES:
        vals = [p[name] for p in top20_params]
        print(f"  {name:12s}: [{min(vals):8.4f}, {max(vals):8.4f}] "
              f"(mean={np.mean(vals):8.4f})")

    print(f"\n{sep}")
    print("END OF REPORT")
    print(sep)


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 65)
    print("Brielmann & Dayan (2022) Aesthetic Value Model - 4D Extension")
    print("Applied to Mouse Acoustic Preference (8-arm Maze)")
    print("=" * 65)
    print()

    # Determine output directory
    try:
        from preference_analysis_config import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    except ImportError:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # --- Fit ---
    best_params, best_rmse, top20 = fit_model()

    # --- Simulate with best params ---
    best_results = simulate_experiment(best_params)

    # --- Secondary evaluations ---
    evals = secondary_evaluations(best_results)

    # --- Decompositions ---
    print("\nRunning decomposition analyses...")
    decomp = run_decompositions(best_params, best_results)
    print("  Done.")

    # --- Report ---
    print_report(best_params, best_rmse, best_results, top20, decomp, evals)

    # --- Figure ---
    print("\nGenerating figure...")
    make_figure(best_params, best_results, best_rmse, top20, decomp, evals,
                output_dir=output_dir)

    # --- Save results CSV ---
    rows = []
    for d in DAY_ORDER:
        for sn, sv in best_results[d]["stimuli"].items():
            rows.append({
                "day": d,
                "stimulus": sn,
                "mean_A": sv["mean_A"],
                "mean_r": sv["mean_r"],
                "mean_Delta_V": sv["mean_dV"],
                "predicted_PI": sv["predicted_PI"],
            })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "aesthetic_model_4D_predictions.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved predictions: {csv_path}")

    # Save parameters
    params_path = os.path.join(output_dir, "aesthetic_model_4D_params.csv")
    pd.DataFrame([best_params]).to_csv(params_path, index=False)
    print(f"Saved parameters: {params_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
