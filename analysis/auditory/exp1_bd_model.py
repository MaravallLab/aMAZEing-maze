#!/usr/bin/env python3
"""
Experiment 1: the Model of Brielmann and Dayan, Fitted
======================================================

Does the model of Brielmann and Dayan (2022), with the stimuli described only by
their measured acoustics, account for how the mice of Experiment 1 treated each
stimulus, and does it single out the vocalisation?

The listener's system state is a point mu in a space of two features, spectral
and temporal complexity (stimulus_complexity.py), standardised over the stimuli
of the experiment; its covariance is fixed and equal in every direction, so it
drops out of every comparison and is absorbed into the weights. Each exposure to
a stimulus s gives

    fluency   r(s)  = -|s - mu|^2
    learning  dV(s) = -(|mu' - mu_T|^2 - |mu - mu_T|^2),  mu' = mu + alpha (s - mu)
    value     A(s)  = w_r r(s) + w_V dV(s)

and then moves the state to mu'. mu_T is the mean of the environment the listener
expects in the long run. The state starts there, so that before the experiment
the listener's model matches its world. The days run in the order the mice heard
them (temporal envelope modulation, the three interval sets, the sequences, the
vocalisations), and within a block each stimulus is met as often as the mice
visited its arm, on average over the sessions of that set.

A recording never repeats within a visit (playback restarted at every entry), so
its temporal complexity is log2 of its length in cycles of 2 kHz, the floor of
the band its spectral complexity is measured over; the vocalisation arm of days
1 to 5 takes the mean of the three recordings it played (Table 4.2). The silent
arm has no features: the index of a stimulus is modelled as A(s) - c, with c the
value of the silent arm, constant over days.

The data are the mean index of every stimulus against the silent arm on every
day (stimuli heard by at least five mice, as in Figure 4.10). Nested models are
fitted by least squares from many starting points:

    intercept   c                                   1 parameter
    voc only    c, w_voc                            2
    fluency     c, w_r, alpha, mu_T (2)             5
    B and D     c, w_r, w_V, alpha, mu_T (2)        6
    B and D + voc   the same and w_voc              7
    B and D, centred   B and D with mu_T fixed at the centre of the stimuli,
                    0 on both standardised features   4

where w_voc is an extra value for a vocalisation that its acoustics do not give.
They are compared by AIC and by leaving out one day at a time: each model is
fitted to the other four days and predicts the left out one.

Usage:
    python exp1_bd_model.py [DATA_DIR] [--thesis-figures DIR]

Reads stimulus_pi/stimulus_pi.csv, complexity/complexity.csv,
complexity/vocalisation_complexity.csv and lmm/arm_blocks.csv from
DATA_DIR/BATCH_ANALYSIS, and writes to DATA_DIR/BATCH_ANALYSIS/bd_model/:
the fits and their comparison, fig_exp1_bd_model (the fit), and
fig_exp1_bd_state with state_by_day.csv (how the fitted model reads the
experiment: the path of the state, and for each day the fluency and learning
parts of the value of its synthetic sounds and its vocalisation).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimulus_complexity as sc
from exp1_stimulus_pi import COLOURS, label

DAYS = ["TEM", "Intervals 1", "Intervals 2", "Intervals 3", "Sequences", "Vocalisations"]
OBSERVED = ["TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations"]
BLOCKS = 4
VOC_DAYS_1_TO_5 = ["run1_day2_male_w_female_oestrus", "run3_day2_male_w_female_oestrus",
                   "run3_day2-2_male_w_female_oestrus"]
FULL_RECORDING_S = 300.0   # run1 day2 was played in full, five minutes, until January 2026
FLOOR_HZ = 2000.0
N_STARTS = 40
CV_STARTS = 12
SEED = 1


# -- the stimuli and the data ---------------------------------------------------
def features(base):
    """Spectral and temporal complexity of every stimulus, keyed by (set, stimulus)."""
    rows = []
    arms = pd.read_csv(os.path.join(base, "lmm", "arm_blocks.csv"))
    voc = pd.read_csv(os.path.join(base, "complexity", "vocalisation_complexity.csv")).set_index("stimulus")
    voc_bits = np.log2(voc["seconds"] * FLOOR_HZ)
    played = voc.loc[VOC_DAYS_1_TO_5].copy()
    played_bits = voc_bits.loc[VOC_DAYS_1_TO_5].copy()
    played_bits.loc["run1_day2_male_w_female_oestrus"] = np.log2(FULL_RECORDING_S * FLOOR_HZ)
    for (set_name, stim, fam), _ in arms[arms["family"] != "silent"].groupby(["set", "stimulus", "family"]):
        if fam == "vocalisation" and set_name != "Vocalisations":
            spec, bits = played["spectral_audible"].mean(), played_bits.mean()
        elif set_name == "Vocalisations":
            spec, bits = voc.loc[stim, "spectral_audible"], voc_bits.loc[stim]
        else:
            d = sc.describe_exp1(stim, fam)
            spec, bits = d["spectral"], d["temporal_bits"]
        rows.append({"set": set_name, "stimulus": stim, "family": fam,
                     "spectral": spec, "temporal_bits": bits,
                     "voc": float(fam == "vocalisation" or set_name == "Vocalisations")})
    f = pd.DataFrame(rows).dropna(subset=["spectral", "temporal_bits"])
    # standardised over the stimuli of the experiment
    for col in ("spectral", "temporal_bits"):
        f[col + "_z"] = (f[col] - f[col].mean()) / f[col].std(ddof=0)
    return f


def exposures(base, f):
    """Visits to each stimulus in a block, on average over the sessions of its set."""
    arms = pd.read_csv(os.path.join(base, "lmm", "arm_blocks.csv"))
    sessions = arms.groupby("set")["session"].nunique()
    v = arms[arms["family"] != "silent"].groupby(["set", "stimulus"])["visits"].sum()
    out = {}
    for s in DAYS:
        if s not in sessions:
            continue
        per = (v.loc[s] / sessions[s] / BLOCKS).round().clip(lower=1).astype(int)
        out[s] = per
    return out


def observations(base, f):
    st = pd.read_csv(os.path.join(base, "stimulus_pi", "stimulus_pi.csv")).dropna(subset=["pi"])
    st = st[st["set"].isin(OBSERVED)]
    st = st[st.groupby(["set", "stimulus"])["pi"].transform("size") >= 5]
    obs = st.groupby(["set", "stimulus"], as_index=False).agg(
        pi=("pi", "mean"), se=("pi", lambda x: x.std(ddof=1) / np.sqrt(len(x))), n=("pi", "size"))
    return obs.merge(f, on=["set", "stimulus"])


# -- the model ------------------------------------------------------------------
MODELS = {
    "intercept": ["c"],
    "voc only": ["c", "w_voc"],
    "fluency": ["c", "w_r", "alpha", "muT_s", "muT_t"],
    "B and D": ["c", "w_r", "w_V", "alpha", "muT_s", "muT_t"],
    "B and D, centred": ["c", "w_r", "w_V", "alpha"],
    "B and D + voc": ["c", "w_r", "w_V", "alpha", "muT_s", "muT_t", "w_voc"],
}
NAMES = {"intercept": "intercept only", "voc only": "vocalisation only", "fluency": "fluency",
         "B and D": "Brielmann and Dayan", "B and D, centred": "B and D, world at centre",
         "B and D + voc": "B and D and vocalisation"}
BOUNDS = {"c": (-1.0, 1.0), "w_r": (0.0, 5.0), "w_V": (0.0, 200.0), "alpha": (1e-4, 0.3),
          "muT_s": (-3.0, 6.0), "muT_t": (-3.0, 6.0), "w_voc": (-1.0, 1.0)}


def plan(f, expo):
    """The order of the exposures within a block of each day, built once: each
    stimulus met as often as the mice visited it, its visits spread through the
    block in turn with the others."""
    vec = {(r.set, r.stimulus): (r.spectral_z, r.temporal_bits_z) for r in f.itertuples()}
    out = []
    for day in DAYS:
        if day not in expo:
            continue
        counts = {s: int(n) for s, n in expo[day].items() if (day, s) in vec}
        order = [s for k in range(max(counts.values())) for s in sorted(counts) if counts[s] > k]
        out.append((day, [(s,) + vec[(day, s)] for s in order]))
    return out


def simulate(p, steps, record=False):
    """Mean value of each stimulus on each day, and the path of the state."""
    ts, tt = p.get("muT_s", 0.0), p.get("muT_t", 0.0)
    ms, mt = ts, tt
    alpha, w_r, w_V = p.get("alpha", 0.0), p.get("w_r", 0.0), p.get("w_V", 0.0)
    values, path = {}, [(ms, mt)]
    for day, order in steps:
        acc = {}
        for _ in range(BLOCKS):
            for stim, ss, st in order:
                ns, nt = ms + alpha * (ss - ms), mt + alpha * (st - mt)
                r = -((ss - ms) ** 2 + (st - mt) ** 2)
                dV = -(((ns - ts) ** 2 + (nt - tt) ** 2) - ((ms - ts) ** 2 + (mt - tt) ** 2))
                acc.setdefault(stim, []).append(w_r * r + w_V * dV)
                ms, mt = ns, nt
            path.append((ms, mt))
        for stim, a in acc.items():
            values[(day, stim)] = sum(a) / len(a)
    return (values, np.array(path)) if record else values


def predict(p, obs, steps):
    values = simulate(p, steps) if any(k in p for k in ("w_r", "w_V")) else {}
    out = []
    for r in obs.itertuples():
        a = values.get((r.set, r.stimulus), 0.0)
        out.append(a - p["c"] + p.get("w_voc", 0.0) * r.voc)
    return np.array(out)


def fit(model, obs, steps, rng, starts=None):
    names = MODELS[model]
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])

    def resid(x):
        return predict(dict(zip(names, x)), obs, steps) - obs["pi"].to_numpy()

    best = None
    starts = 1 if model in ("intercept", "voc only") else (starts or N_STARTS)
    for k in range(starts):
        x0 = lo + rng.uniform(0.05, 0.95, len(names)) * (hi - lo)
        if names[0] == "c":
            x0[0] = 0.0
        try:
            res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    return dict(zip(names, best.x)), 2 * best.cost


def aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k


# -- figure -----------------------------------------------------------------------
def fig_fit(obs, preds, comparison, path):
    fig = plt.figure(figsize=(6.3, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.6, wspace=0.12)
    order = obs.assign(o=obs["set"].map(OBSERVED.index)).sort_values(["o", "voc", "stimulus"])
    x = np.arange(len(order))
    ax = fig.add_subplot(gs[0, :])
    ax.errorbar(x, order["pi"], yerr=order["se"], fmt="o", ms=3, color="black", lw=0.8,
                capsize=1.5, label="mice, mean and standard error", zorder=3)
    for name, colour, ls in (("B and D", "#1565C0", "-"), ("B and D + voc", "#2E7D32", (0, (4, 2)))):
        ax.plot(x, preds[name].loc[order.index], color=colour, lw=1.1, ls=ls, marker="D", ms=2.5,
                label=f"{name.replace('B and D', 'Brielmann and Dayan')}")
    for k, s in enumerate(OBSERVED):
        idx = np.where(order["set"].to_numpy() == s)[0]
        if k:
            ax.axvline(idx[0] - 0.5, color="#BDBDBD", lw=0.6)
        ax.text(idx.mean(), 0.33, s, ha="center", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels([label(s) for s in order["stimulus"]], rotation=70, ha="right", fontsize=5)
    for t, v in zip(ax.get_xticklabels(), order["voc"]):
        if v:
            t.set_color("#2E7D32")
    ax.set_ylabel("Index against the silent arm")
    ax.set_ylim(-0.4, 0.38)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right", ncol=3)
    ax.set_title("(a) every stimulus, observed and fitted", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    names = list(comparison.index)
    ax.barh(range(len(names)), comparison["cv_rmse"], color="#90A4AE", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([NAMES.get(n, n) for n in names], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Error predicting a\nleft out day (RMSE)")
    ax.set_title("(b) prediction", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ax.barh(range(len(names)), comparison["aic"] - comparison["aic"].min(), color="#B0BEC5", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("AIC above the best")
    ax.set_title("(c) fit", loc="left")
    for a in fig.axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -- how the fitted model reads the experiment ------------------------------------
DAY_SHORT = {"TEM": "TEM", "Intervals 1": "Int 1", "Intervals 2": "Int 2",
             "Intervals 3": "Int 3", "Sequences": "Seq", "Vocalisations": "Voc"}
SYNTH, VOCAL = "#455A64", "#2E7D32"


def components(p, steps):
    """The two parts of the value of every stimulus on every day, as simulate
    computes them: fluency (w_r r, the cost of being unfamiliar to the current
    state) and learning (w_V dV, the gain from moving the state towards the
    expected world), each averaged over the stimulus's exposures that day."""
    ts, tt = p["muT_s"], p["muT_t"]
    ms, mt = ts, tt
    alpha, w_r, w_V = p["alpha"], p["w_r"], p["w_V"]
    acc = {}
    for day, order in steps:
        for _ in range(BLOCKS):
            for stim, ss, st in order:
                ns, nt = ms + alpha * (ss - ms), mt + alpha * (st - mt)
                r = -((ss - ms) ** 2 + (st - mt) ** 2)
                dV = -(((ns - ts) ** 2 + (nt - tt) ** 2) - ((ms - ts) ** 2 + (mt - tt) ** 2))
                acc.setdefault((day, stim), []).append((w_r * r, w_V * dV))
                ms, mt = ns, nt
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def state_by_day(p, path_traj, steps, f, obs, preds):
    """For every day: where the state was while the day ran (the mean of its
    position at the start of each block), how far it lay from that day's
    synthetic sounds and from its vocalisation, the two parts of their value
    in the model of Brielmann and Dayan, and their mean index, observed and
    fitted."""
    parts = components(p, steps)
    rows = []
    for i, (day, _) in enumerate(steps):
        state = path_traj[BLOCKS * i: BLOCKS * i + BLOCKS].mean(axis=0)
        here = f[f["set"] == day]
        dist = np.hypot(here["spectral_z"] - state[0], here["temporal_bits_z"] - state[1])
        row = {"day": i + 1, "set": day, "state_spectral_z": state[0], "state_temporal_z": state[1]}
        for kind, mask in (("synthetic", here["voc"] == 0), ("vocalisation", here["voc"] == 1)):
            d = dist[mask.to_numpy()]
            row[f"dist_{kind}"] = d.mean() if len(d) else np.nan
            vals = np.array([parts[(day, s)] for s in here.loc[mask, "stimulus"] if (day, s) in parts])
            row[f"fluency_{kind}"] = vals[:, 0].mean() if len(vals) else np.nan
            row[f"learning_{kind}"] = vals[:, 1].mean() if len(vals) else np.nan
            o = obs[(obs["set"] == day) & (obs["voc"] == float(kind == "vocalisation"))]
            row[f"pi_{kind}_observed"] = o["pi"].mean() if len(o) else np.nan
            row[f"pi_{kind}_fitted"] = preds["B and D"].loc[o.index].mean() if len(o) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fig_state(path_traj, steps, f, days, path):
    """The fitted model of Brielmann and Dayan, read as a story: where the mouse's
    expectation starts and how it moves (a), the cost of each day's sounds being
    unfamiliar to it (b), the gain from sounds that move it back towards the world
    it expects (c), and their sum, the preference, against the mice (d). Panels b
    to d share one scale, so that b and c add up to d."""
    fig = plt.figure(figsize=(6.3, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32)

    # (a) the map
    ax = fig.add_subplot(gs[0, 0])
    start = path_traj[0]
    for r in f.itertuples():
        colour = VOCAL if r.voc else COLOURS.get(r.family, "#424242")
        ax.scatter(r.temporal_bits_z, r.spectral_z, s=9, color=colour, lw=0, alpha=0.75, zorder=2)
    ax.plot(path_traj[:, 1], path_traj[:, 0], color="black", lw=0.8, zorder=3)
    ends = [path_traj[BLOCKS * (i + 1)] for i in range(len(steps))]
    for i, e in enumerate(ends):
        ax.scatter(e[1], e[0], s=34, color="white", edgecolor="black", lw=0.7, zorder=4)
        ax.text(e[1], e[0], str(i + 1), ha="center", va="center", fontsize=5, zorder=5)
    ax.scatter(start[1], start[0], marker="*", s=70, color="#E65100", zorder=5)
    ax.annotate("expected world (fitted),\nwhere the expectation starts",
                (start[1], start[0]), xytext=(start[1] + 0.3, start[0]),
                fontsize=5.5, color="#E65100", va="center")
    voc = f[f["voc"] == 1]
    ax.text(voc["temporal_bits_z"].max() + 0.2, voc["spectral_z"].mean(), "vocalisations",
            fontsize=5.5, color=VOCAL, ha="left", va="center")
    syn = f[f["voc"] == 0]
    ax.text(syn["temporal_bits_z"].min(), syn["spectral_z"].min() - 0.45,
            "synthetic sounds", fontsize=5.5, color=SYNTH, ha="left", va="center")
    ax.set_xlim(min(f["temporal_bits_z"].min(), start[1]) - 0.3,
                max(f["temporal_bits_z"].max(), start[1]) + 1.5)
    ax.set_ylim(min(f["spectral_z"].min(), start[0]) - 0.75,
                max(f["spectral_z"].max(), start[0]) + 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Temporal complexity (z)")
    ax.set_ylabel("Spectral complexity (z)")
    ax.set_title("(a) where the expectation goes", loc="left")

    # (b) to (d) on one scale, so that b and c add up to d
    x = days["day"].to_numpy()
    ticks = [f"{d}\n{DAY_SHORT[s]}" for d, s in zip(x, days["set"])]
    lim = (min(days[[c for c in days if c.startswith(("fluency", "learning", "pi_"))]].min().min(), -0.3) - 0.03,
           max(days[[c for c in days if c.startswith(("fluency", "learning", "pi_"))]].max().max(), 0.1) + 0.05)
    kinds = (("synthetic", SYNTH, "synthetic sounds"), ("vocalisation", VOCAL, "vocalisation"))

    def day_axis(ax, title, ylabel):
        ax.axhline(0, color="black", lw=0.5, ls=(0, (4, 2)))
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=5.5)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylim(*lim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    for pos, part, title, ylabel in (
            (gs[0, 1], "fluency", "(b) the cost of being unfamiliar", "Fluency part of value"),
            (gs[1, 0], "learning", "(c) the gain from learning", "Learning part of value")):
        ax = fig.add_subplot(pos)
        for kind, colour, name in kinds:
            seen = days.dropna(subset=[f"{part}_{kind}"])
            ax.plot(seen["day"], seen[f"{part}_{kind}"], color=colour, marker="o", ms=3, lw=1,
                    label=name)
        day_axis(ax, title, ylabel)
        if part == "fluency":
            ax.legend(fontsize=5.5, frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[1, 1])
    for kind, colour, _ in kinds:
        seen = days.dropna(subset=[f"pi_{kind}_observed"])
        ax.plot(seen["day"], seen[f"pi_{kind}_fitted"], color=colour, lw=1)
        ax.scatter(seen["day"], seen[f"pi_{kind}_observed"], s=12, color=colour, zorder=3)
    ax.plot([], [], color="black", lw=1, label="model")
    ax.scatter([], [], s=12, color="black", label="mice")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    day_axis(ax, "(d) the preference that follows", "Index against the silent arm")

    for a in fig.axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default="D:/AuditoryExperiments/8_arms_w_voc")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    base = os.path.join(args.data_dir, "BATCH_ANALYSIS")
    out = os.path.join(base, "bd_model")
    os.makedirs(out, exist_ok=True)

    f = features(base)
    expo = exposures(base, f)
    obs = observations(base, f)
    steps = plan(f, expo)
    rng = np.random.default_rng(SEED)
    lines = []

    def say(t=""):
        lines.append(t)
        print(t)

    say(f"{len(obs)} stimulus indices over {obs['set'].nunique()} days, "
        f"{int(obs['voc'].sum())} of them vocalisations")
    fits, preds, rows = {}, {}, []
    n = len(obs)
    for model in MODELS:
        p, rss = fit(model, obs, steps, rng)
        fits[model] = p
        preds[model] = pd.Series(predict(p, obs, steps), index=obs.index)
        # leave one day out
        cv = []
        for day in OBSERVED:
            train, test = obs[obs["set"] != day], obs[obs["set"] == day]
            pt, _ = fit(model, train, steps, rng, starts=CV_STARTS)
            cv += list(predict(pt, test, steps) - test["pi"].to_numpy())
        rows.append({"model": model, "k": len(MODELS[model]), "rss": rss,
                     "rmse": np.sqrt(rss / n), "aic": aic(rss, n, len(MODELS[model])),
                     "cv_rmse": np.sqrt(np.mean(np.square(cv)))})
        say(f"\n{model}: " + ", ".join(f"{k} = {v:.4g}" for k, v in p.items()))
        say(f"  RMSE {np.sqrt(rss / n):.4f}, AIC {aic(rss, n, len(MODELS[model])):.1f}, "
            f"left out day RMSE {rows[-1]['cv_rmse']:.4f}")
    comparison = pd.DataFrame(rows).set_index("model")
    say("\n" + comparison.round(4).to_string())

    # where the vocalisation sits for the fitted B and D model
    bd = fits["B and D"]
    say("\nB and D, fitted against observed, the vocalisation on each day:")
    for r in obs[obs["voc"] == 1].itertuples():
        say(f"  {r.set:13s} {label(r.stimulus):12s} observed {r.pi:+.3f}  "
            f"B and D {preds['B and D'][r.Index]:+.3f}  B and D + voc {preds['B and D + voc'][r.Index]:+.3f}")
    resid = obs.assign(res=obs["pi"] - preds["B and D"])
    say("\nB and D, mean residual: vocalisations "
        f"{resid.loc[resid['voc'] == 1, 'res'].mean():+.3f}, synthetic sounds "
        f"{resid.loc[resid['voc'] == 0, 'res'].mean():+.3f}")
    say("mean residual by day: " + ", ".join(
        f"{d} {resid.loc[resid['set'] == d, 'res'].mean():+.3f}" for d in OBSERVED))

    _, path_traj = simulate(bd, steps, record=True)
    days = state_by_day(bd, path_traj, steps, f, obs, preds)
    days.to_csv(os.path.join(out, "state_by_day.csv"), index=False)
    obs.assign(**{f"pred_{k}": v for k, v in preds.items()}).to_csv(
        os.path.join(out, "predictions.csv"), index=False)
    comparison.to_csv(os.path.join(out, "comparison.csv"))
    pd.DataFrame(fits).T.to_csv(os.path.join(out, "parameters.csv"))
    f.to_csv(os.path.join(out, "features.csv"), index=False)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42}):
        targets = [out] + ([args.thesis_figures] if args.thesis_figures else [])
        for t in targets:
            fig_fit(obs, preds, comparison, os.path.join(t, "fig_exp1_bd_model"))
            fig_state(path_traj, steps, f, days, os.path.join(t, "fig_exp1_bd_state"))
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
