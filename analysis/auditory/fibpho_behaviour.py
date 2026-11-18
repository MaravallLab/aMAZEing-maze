"""
Fiber Photometry: Dopamine Against Where the Mouse Was
======================================================

Uses the maze's log of every zone the mouse entered to place each moment
of the dopamine recording in one of three states:

    in an arm       ROI1 to ROI8, labelled by the stimulus that arm played
    in an entrance  the entrance zones the maze also tracks
    outside         anywhere else, exploring the rest of the maze

and asks how dopamine differs between them, how it unfolds through each
block, and how the response to each stimulus changes from its first to
its last exposure in a session.

delta F/F is z-scored within each block. A transient is a peak of at
least TRANSIENT_Z (block standard deviations) prominence, at least 1 s
from the next, and takes the state the mouse was in at its peak.

The response to an exposure (an arm entry) is the peak z in the 5 s after
it. Its reference is the same measure at random moments when the mouse
was outside the arms, in the same blocks.

Usage:
    python fibpho_behaviour.py [FIBPHO_ANALYSIS]

    reads FIBPHO_ANALYSIS (written by fibpho_alignment.py) and the maze
    logs in the folder above it.

Outputs (FIBPHO_ANALYSIS/behaviour/):
    timelines/<day>_<mouse>.png/.pdf  every block of a session: dopamine
                                      over where the mouse was, transients
                                      marked
    fig_state_level.png/.pdf          mean dopamine by state and stimulus
    fig_state_transients.png/.pdf     transients per minute by state
    fig_exposures.png/.pdf            first, mean and last exposure per
                                      stimulus, against exploring
    fig_block_course.png/.pdf         behaviour and dopamine through the
                                      sound blocks, in 2-minute bins
    states.csv, exposures.csv, block_course.csv
    exposure_stats.txt                do stimuli differ, does the response
                                      change with exposure, is it above
                                      exploring

    With --thesis-figures DIR, the exposure figure is also written to DIR
    as fig_fp_exposures, in the thesis style.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fibpho_alignment as fa

TRANSIENT_Z = 3.0          # prominence of a transient, in block SDs
EXPOSURE_WIN_S = 5.0       # window after an entry for its peak
N_REFERENCE = 2000         # random outside moments for the reference
COURSE_BIN_MIN = 2.0       # bins through a sound block
MIN_EXPOSURES = 3          # a stimulus needs this many exposures to enter a test
PLOT_DS = 10               # plot every 10th sample of the ~100 Hz trace

OUTSIDE, ENTRANCE = "outside", "entrance"
STATE_COLOURS = {OUTSIDE: "#FFFFFF", ENTRANCE: "#F2E394"}
MOUSE_COLOURS = {"K30_5": "#1565C0", "K30_7": "#E65100"}

DAYS = ["temporal_envelope_modulation", "intervals", "sequences", "vocalisations"]
DAY_TITLE = {"temporal_envelope_modulation": "Envelope modulation",
             "intervals": "Intervals", "sequences": "Sequences",
             "vocalisations": "Vocalisations"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# -- per session -------------------------------------------------------

def load_session(fa_root, row):
    """Everything for one session, in the recording's clock."""
    folder = os.path.join(fa_root, row["day"], row["mouse"])
    blocks = pd.read_csv(os.path.join(folder, "blocks.csv"))
    npz = np.load(os.path.join(folder, "dff_downsampled.npz"))
    t, dff, fs = npz["time_s"].astype(float), npz["dff"].astype(float), float(npz["fs"])

    # the full maze log, entrances included
    session_dir = os.path.join(os.path.dirname(fa_root), row["day"], row["session"])
    log = pd.read_csv(glob.glob(os.path.join(session_dir, "*_detailed_visits.csv"))[0])
    log["t0"] = log["sound_on_time"] - row["offset_s"]
    log["t1"] = log["t0"] + log["time_spent_seconds"]
    entrance = log["ROI_visited"].astype(str).str.startswith("entrance")
    log["state"] = [ENTRANCE if e else fa.stimulus_label(s, tr)
                    for e, s, tr in zip(entrance, log["stimulus"], log["trial_ID"])]
    log["in_arm"] = ~entrance
    log = log.sort_values("t0").reset_index(drop=True)

    # state of every sample: the last zone entered that the mouse is still in
    state = np.full(len(t), OUTSIDE, dtype=object)
    for _, v in log.iterrows():
        state[(t >= v["t0"]) & (t < v["t1"])] = v["state"]

    # block of every sample, and z within each block
    block = np.zeros(len(t), int)
    z = np.full_like(dff, np.nan)
    for _, b in blocks.iterrows():
        m = (t >= b["start_tdt"]) & (t < b["end_tdt"])
        block[m] = int(b["trial_ID"])
        ok = m & np.isfinite(dff)
        if ok.sum() > 10:
            z[ok] = (dff[ok] - dff[ok].mean()) / dff[ok].std()

    # transients, block by block so none straddles a boundary
    peaks = []
    for _, b in blocks.iterrows():
        idx = np.flatnonzero((block == b["trial_ID"]) & np.isfinite(z))
        if len(idx) < fs * 5:
            continue
        p, _ = find_peaks(z[idx], prominence=TRANSIENT_Z, distance=int(fs))
        peaks.extend(idx[p])
    return {"t": t, "z": z, "fs": fs, "state": state, "block": block,
            "blocks": blocks, "log": log, "peaks": np.array(peaks, int)}


def arm_states(state):
    return sorted({s for s in set(state) if s not in (OUTSIDE, ENTRANCE)})


def state_table(s):
    """Time, mean level and transients per state, per sound block."""
    rows = []
    sound_ids = s["blocks"].loc[~s["blocks"]["is_silent"], "trial_ID"]
    peak_mask = np.zeros(len(s["t"]), bool)
    peak_mask[s["peaks"]] = True
    for tid in sound_ids:
        in_block = (s["block"] == tid) & np.isfinite(s["z"])
        groups = {OUTSIDE: s["state"] == OUTSIDE, ENTRANCE: s["state"] == ENTRANCE}
        arms = np.isin(s["state"], [a for a in arm_states(s["state"]) if a != "silent arm"])
        groups["all sound arms"] = arms
        for a in arm_states(s["state"]):
            groups[a] = s["state"] == a
        for name, g in groups.items():
            m = in_block & g
            rows.append({"trial_ID": int(tid), "state": name,
                         "seconds": m.sum() / s["fs"],
                         "level_z": s["z"][m].mean() if m.any() else np.nan,
                         "transients": int(peak_mask[m].sum())})
    return pd.DataFrame(rows)


def exposures(s, rng):
    """Peak z after every arm entry, and the outside reference."""
    fs, z, t = s["fs"], s["z"], s["t"]
    w = int(EXPOSURE_WIN_S * fs)
    sound_ids = set(s["blocks"].loc[~s["blocks"]["is_silent"], "trial_ID"])
    rows = []
    arms = s["log"][s["log"]["in_arm"] & s["log"]["trial_ID"].isin(sound_ids)]
    for order, (label, g) in enumerate(arms.groupby("state")):
        g = g.sort_values("t0").reset_index(drop=True)
        for k, v in g.iterrows():
            i = int(round(v["t0"] * fs))
            seg = z[i:i + w] if 0 <= i and i + w <= len(z) else np.array([np.nan])
            rows.append({"stimulus": label, "exposure": k + 1, "of": len(g),
                         "trial_ID": int(v["trial_ID"]), "visit_s": v["time_spent_seconds"],
                         "peak_z": np.nanmax(seg) if np.isfinite(seg).any() else np.nan})
    # reference: random moments outside the arms in the sound blocks
    ok = np.flatnonzero(np.isin(s["block"], list(sound_ids)) & (s["state"] == OUTSIDE)
                        & np.isfinite(z))
    ok = ok[ok + w < len(z)]
    picks = rng.choice(ok, min(N_REFERENCE, len(ok)), replace=False)
    ref = np.array([np.nanmax(z[i:i + w]) for i in picks])
    return pd.DataFrame(rows), ref


def block_course(s):
    """Behaviour and dopamine in COURSE_BIN_MIN bins through each sound block."""
    rows = []
    fs = s["fs"]
    peak_mask = np.zeros(len(s["t"]), bool)
    peak_mask[s["peaks"]] = True
    for _, b in s["blocks"][~s["blocks"]["is_silent"]].iterrows():
        edges = np.arange(b["start_tdt"], b["end_tdt"] + 1e-6, COURSE_BIN_MIN * 60)
        entries = s["log"][s["log"]["in_arm"]]["t0"].to_numpy()
        for k, (a, c) in enumerate(zip(edges[:-1], edges[1:])):
            m = (s["t"] >= a) & (s["t"] < c) & np.isfinite(s["z"])
            out = m & (s["state"] == OUTSIDE)
            arm = m & ~np.isin(s["state"], [OUTSIDE, ENTRANCE])
            minutes = m.sum() / fs / 60
            rows.append({"trial_ID": int(b["trial_ID"]), "bin": k,
                         "minute": (k + 0.5) * COURSE_BIN_MIN,
                         "pct_time_in_arms": 100 * arm.sum() / max(m.sum(), 1),
                         "entries_per_min": ((entries >= a) & (entries < c)).sum() / minutes
                         if minutes else np.nan,
                         "level_outside_z": s["z"][out].mean() if out.any() else np.nan,
                         "level_arms_z": s["z"][arm].mean() if arm.any() else np.nan,
                         "transients_per_min": peak_mask[m].sum() / minutes if minutes else np.nan,
                         "transients_outside_per_min": peak_mask[out].sum() / (out.sum() / fs / 60)
                         if out.sum() > fs * 10 else np.nan})
    return pd.DataFrame(rows)


# -- figures -----------------------------------------------------------

def save(fig, path):
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight")
    fa.safe_savefig(fig, path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}.png / .pdf")


def fig_timeline(s, title, path):
    """One row per block: dopamine over a background of where the mouse was."""
    colours = fa.label_colours(arm_states(s["state"]))
    colours.update(STATE_COLOURS)
    blocks = s["blocks"]
    longest = ((blocks["end_tdt"] - blocks["start_tdt"]).max()) / 60
    fig, axes = plt.subplots(len(blocks), 1, figsize=(14, 1.15 * len(blocks)),
                             sharex=True, squeeze=False)
    ylim = np.nanpercentile(np.abs(s["z"]), 99.8)
    for ax, (_, b) in zip(axes[:, 0], blocks.iterrows()):
        m = s["block"] == b["trial_ID"]
        t0 = b["start_tdt"]
        for _, v in s["log"][(s["log"]["t1"] > t0) & (s["log"]["t0"] < b["end_tdt"])].iterrows():
            ax.axvspan((v["t0"] - t0) / 60, (v["t1"] - t0) / 60, color=colours[v["state"]],
                       alpha=0.8 if v["in_arm"] else 0.6, lw=0, zorder=1)
        tm = (s["t"][m] - t0) / 60
        ax.plot(tm[::PLOT_DS], s["z"][m][::PLOT_DS], color="#1a1a2e", lw=0.4, zorder=2)
        pk = s["peaks"][s["block"][s["peaks"]] == b["trial_ID"]]
        for i in pk:
            st = s["state"][i]
            ax.plot((s["t"][i] - t0) / 60, ylim * 0.92, marker="v", ms=3.5, zorder=3,
                    color="black" if st == OUTSIDE else ("#8D6E00" if st == ENTRANCE
                                                         else colours[st]),
                    markeredgecolor="black", markeredgewidth=0.3)
        ax.set_ylim(-ylim * 0.6, ylim)
        ax.set_xlim(0, longest)
        kind = "habituation" if b["trial_ID"] == 1 else ("silent" if b["is_silent"] else "sound")
        ax.set_ylabel(f"T{int(b['trial_ID'])}\n{kind}", rotation=0, ha="right", va="center",
                      fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1, 0].set_xlabel("Minutes into the block")
    shown = [l for l in colours if l != OUTSIDE and (s["state"] == l).any()]
    handles = [mpatches.Patch(color=colours[l], label=l.replace("_", " ")) for l in shown]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="grey", label="outside the arms"))
    handles.append(plt.Line2D([], [], marker="v", ls="", color="black",
                              label=f"transient (≥{TRANSIENT_Z:g} SD), coloured by where"))
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 1.0 + 0.25 / len(blocks)))
    fig.suptitle(f"{title}: dopamine (z within block) over where the mouse was",
                 fontsize=10, y=1.0 + 0.6 / len(blocks))
    fig.tight_layout()
    save(fig, path)


def grid(mice, title, figsize=None):
    fig, axes = plt.subplots(len(mice), len(DAYS),
                             figsize=figsize or (3.8 * len(DAYS), 3.1 * len(mice)), squeeze=False)
    for c, day in enumerate(DAYS):
        axes[0, c].set_title(DAY_TITLE[day])
    for r, mouse in enumerate(mice):
        axes[r, 0].annotate(mouse, (-0.42, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=11)
    return fig, axes


def state_order(states):
    fixed = [OUTSIDE, ENTRANCE, "all sound arms", "silent arm"]
    return [f for f in fixed if f in states] + sorted(s for s in states if s not in fixed)


def short(label):
    return label.replace("_", " ") if len(label) <= 16 else label[:14] + "..."


def fig_state(tables, mice, measure, out):
    if measure == "level":
        title = ("Mean dopamine by where the mouse was, in the sound blocks "
                 "(z within block; dots: blocks; numbers: seconds there)")
    else:
        title = (f"Transients (≥{TRANSIENT_Z:g} SD) per minute by where the mouse was, in the "
                 "sound blocks (bars: exact Poisson 95% interval; numbers: seconds there)")
    fig, axes = grid(mice, title)
    for (day, mouse), tab in tables.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        order = [st for st in state_order(set(tab["state"]))
                 if tab.loc[tab["state"] == st, "seconds"].sum() >= 5]
        colours = fa.label_colours(order)
        colours.update({OUTSIDE: "#424242", ENTRANCE: "#C9B037", "all sound arms": "#1565C0"})
        for i, st in enumerate(order):
            g = tab[tab["state"] == st]
            secs = g["seconds"].sum()
            if measure == "level":
                per_block = g["level_z"]
                pooled = np.average(g["level_z"].fillna(0), weights=g["seconds"]) if secs else np.nan
            else:
                per_block = g["transients"] / (g["seconds"] / 60).replace(0, np.nan)
                k = g["transients"].sum()
                pooled = k / (secs / 60)
                # exact Poisson 95% interval: a few seconds in an arm give a wide one
                lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
                hi = chi2.ppf(0.975, 2 * k + 2) / 2
                ax.errorbar(i, pooled, yerr=[[pooled - lo / (secs / 60)],
                                             [hi / (secs / 60) - pooled]],
                            color="black", lw=0.8, capsize=2, zorder=4)
            ax.bar(i, pooled, color=colours[st], alpha=0.75, width=0.7)
            if measure == "level":
                ax.scatter(np.full(len(per_block), i), per_block, s=6, color="black",
                           zorder=3, lw=0)
            ax.annotate(f"{secs:.0f}", (i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=5,
                        color="#555555")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short(o) for o in order], rotation=60, ha="right", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Level (z)" if measure == "level" else "Transients per minute")
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_state_level" if measure == "level" else "fig_state_transients"))


def fig_exposures(expo, refs, mice, out):
    fig, axes = grid(mice, f"Peak dopamine in the {EXPOSURE_WIN_S:g} s after entering an arm: "
                           "first exposure, mean of all, last exposure; grey band: the same "
                           "measure while exploring outside the arms (median, quartiles)")
    for (day, mouse), e in expo.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        ref = refs[(day, mouse)]
        lo, med, hi = np.percentile(ref, [25, 50, 75])
        ax.axhspan(lo, hi, color="#BDBDBD", alpha=0.5, lw=0)
        ax.axhline(med, color="#757575", lw=0.8)
        e = e.dropna(subset=["peak_z"])
        order = state_order(set(e["stimulus"]))
        colours = fa.label_colours(order)
        for i, st in enumerate(order):
            g = e[e["stimulus"] == st].sort_values("exposure")
            if g.empty:
                continue
            first, last, mean = g["peak_z"].iloc[0], g["peak_z"].iloc[-1], g["peak_z"].mean()
            ax.plot([i - 0.25, i, i + 0.25], [first, mean, last], color=colours[st], lw=0.8)
            ax.scatter([i - 0.25], [first], marker="o", s=16, color=colours[st], zorder=3)
            ax.scatter([i], [mean], marker="D", s=14, facecolor="white",
                       edgecolor=colours[st], zorder=3)
            ax.scatter([i + 0.25], [last], marker="s", s=14, color=colours[st], zorder=3)
            ax.annotate(str(len(g)), (i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=5,
                        color="#555555")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short(o) for o in order], rotation=60, ha="right", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Peak z, {EXPOSURE_WIN_S:g} s after entry")
    handles = [plt.Line2D([], [], marker="o", ls="", color="grey", label="first exposure"),
               plt.Line2D([], [], marker="D", ls="", markerfacecolor="white",
                          markeredgecolor="grey", label="mean of all exposures"),
               plt.Line2D([], [], marker="s", ls="", color="grey", label="last exposure"),
               mpatches.Patch(color="#BDBDBD", label="exploring outside the arms")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_exposures"))


VOC_SHORT = {"dyadic_social_interaction": "dyadic", "male_male_interaction": "M+M",
             "male_w_female_oestrus": "M+F", "run1_day2_male_w_female_oestrus": "urine r1",
             "run3_day2_male_w_female_oestrus": "urine r3",
             "run3_day2-2_male_w_female_oestrus": "urine r3b",
             "mousetube3BB89473_socialcomparison": "mouseTube"}
INTERVAL_SHORT = {"perf_5": "5th", "perf_4": "4th", "maj_3": "maj 3rd", "maj_2": "maj 2nd",
                  "min_7": "min 7th", "tritone": "tritone"}


def thesis_label(label):
    """A short name for a stimulus, for the thesis figure."""
    if label in VOC_SHORT:
        return VOC_SHORT[label]
    if label in INTERVAL_SHORT:
        return INTERVAL_SHORT[label]
    if label.endswith(" kHz"):                      # 'rough_complex 20 kHz' -> 'complex 20'
        kind, freq = label[:-4].rsplit(" ", 1)
        return f"{kind.replace('rough_complex', 'complex')} {freq}"
    return {"silent arm": "silent", "vocalisation": "voc"}.get(label, label)


def exposure_class(label, day):
    if label == "silent arm":
        return label
    if label == "vocalisation" or day == "vocalisations":
        return "vocalisation"
    return "synthetic sound"


def exposure_table(expo, refs):
    """Every exposure, its peak measured from the median while exploring."""
    e = pd.concat([x.assign(day=d, mouse=m, session=f"{m} {d}") for (d, m), x in expo.items()])
    e = e.dropna(subset=["peak_z"]).reset_index(drop=True)
    med = {k: np.median(v) for k, v in refs.items()}
    e["rel"] = [p - med[(d, m)] for p, d, m in zip(e["peak_z"], e["day"], e["mouse"])]
    e["cls"] = [exposure_class(s, d) for s, d in zip(e["stimulus"], e["day"])]
    e["log2_exposure"] = np.log2(e["exposure"])
    return e


def exposure_stats(e, out_path):
    """Do stimuli differ, does the response change with exposure, and is it above exploring?"""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import wilcoxon
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    say("Response: peak z in the 5 s after entering an arm, minus the median of the "
        "same measure at random moments outside the arms in the same session.")
    say(f"{len(e)} exposures, {e['session'].nunique()} sessions.\n")

    from statsmodels.stats.multitest import multipletests
    say("1. Do the stimuli differ? Per day, response ~ C(stimulus) + C(mouse), "
        f"stimuli with at least {MIN_EXPOSURES} exposures, type II ANOVA; "
        "the four days are one family of tests, Holm corrected across them")
    days = []
    for day in DAYS:
        d = e[e["day"] == day]
        keep = d.groupby("stimulus")["rel"].transform("size") >= MIN_EXPOSURES
        d = d[keep]
        if d["stimulus"].nunique() < 2:
            continue
        fit = smf.ols("rel ~ C(stimulus) + C(mouse)", d).fit()
        a = sm.stats.anova_lm(fit, typ=2)
        f, p = a.loc["C(stimulus)", "F"], a.loc["C(stimulus)", "PR(>F)"]
        df1, df2 = int(a.loc["C(stimulus)", "df"]), int(a.loc["Residual", "df"])
        days.append((day, d, f, p, df1, df2))
    p_holm = multipletests([p for *_, p, _, _ in days], method="holm")[1] if days else []
    for (day, d, f, p, df1, df2), ph in zip(days, p_holm):
        say(f"  {DAY_TITLE[day]:20s} {d['stimulus'].nunique()} stimuli, {len(d)} exposures: "
            f"F({df1}, {df2}) = {f:.2f}, p = {p:.3f}, Holm p = {ph:.3f}")
        means = (d.groupby("stimulus")["rel"].agg(["mean", "sem", "size"])
                  .sort_values("mean", ascending=False))
        say("    " + means.round(3).to_string().replace("\n", "\n    "))

    say("\n2. Does the response change with exposure?")
    # session as a fixed effect: a random intercept for it is estimated at zero
    # here, and that singular fit gives no usable intercept
    m = smf.ols("rel ~ log2_exposure + C(cls, Treatment('silent arm')) + C(session)", e).fit()
    say("  OLS, response ~ log2(exposure number) + class + session:")
    say(m.summary2().tables[1].loc[["log2_exposure",
                                    "C(cls, Treatment('silent arm'))[T.synthetic sound]",
                                    "C(cls, Treatment('silent arm'))[T.vocalisation]"]]
        .round(3).to_string())
    pairs = []
    for (s, st), g in e.groupby(["session", "stimulus"]):
        if len(g) >= 2:
            g = g.sort_values("exposure")
            pairs.append({"session": s, "stimulus": st, "first": g["rel"].iloc[0],
                          "last": g["rel"].iloc[-1], "mean": g["rel"].mean()})
    pairs = pd.DataFrame(pairs)
    w = wilcoxon(pairs["first"], pairs["last"])
    say(f"  first against last exposure, {len(pairs)} stimuli across sessions: "
        f"first {pairs['first'].mean():+.3f}, last {pairs['last'].mean():+.3f}, "
        f"mean of all {pairs['mean'].mean():+.3f}; Wilcoxon W = {w.statistic:.0f}, "
        f"p = {w.pvalue:.3f}")

    say("\n3. Is the response above or below exploring? response ~ 0 + class + (1 | session)")
    m = smf.mixedlm("rel ~ 0 + C(cls)", e, groups=e["session"]).fit()
    say(m.summary().tables[1].to_string())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return pairs


def fig_exposures_thesis(e, refs, mice, path):
    """First, mean and last exposure per stimulus, measured from exploring."""
    fig, axes = plt.subplots(len(mice), len(DAYS), figsize=(6.3, 2.0 * len(mice) + 0.4),
                             squeeze=False, sharey=True)
    for c, day in enumerate(DAYS):
        axes[0, c].set_title(f"({'abcdefgh'[c]}) {DAY_TITLE[day].lower()}", loc="left")
        for r in range(1, len(mice)):
            axes[r, c].set_title(f"({'abcdefgh'[c + 4 * r]})", loc="left")
    for (day, mouse), _ in refs.items():
        ax = axes[mice.index(mouse), DAYS.index(day)]
        ref = refs[(day, mouse)] - np.median(refs[(day, mouse)])
        lo, hi = np.percentile(ref, [25, 75])
        ax.axhspan(lo, hi, color="#D6D6D6", lw=0)
        ax.axhline(0, color="#9E9E9E", lw=0.6)
        d = e[(e["day"] == day) & (e["mouse"] == mouse)]
        order = state_order(set(d["stimulus"]))
        # colours from every stimulus of the day, so a stimulus keeps its colour
        # in both mice even when one of them never entered some arm
        colours = fa.label_colours(state_order(set(e.loc[e["day"] == day, "stimulus"])))
        for i, st in enumerate(order):
            g = d[d["stimulus"] == st].sort_values("exposure")
            first, last = g["rel"].iloc[0], g["rel"].iloc[-1]
            mean = g["rel"].mean()
            sem = g["rel"].std() / np.sqrt(len(g)) if len(g) > 1 else 0
            col = colours[st]
            ax.plot([i - 0.25, i, i + 0.25], [first, mean, last], color=col, lw=0.6)
            ax.scatter([i - 0.25], [first], marker="o", s=9, color=col, zorder=3, lw=0)
            ax.errorbar([i], [mean], yerr=[sem], fmt="D", ms=3, mfc="white", mec=col,
                        ecolor=col, elinewidth=0.6, capsize=1.2, zorder=3)
            ax.scatter([i + 0.25], [last], marker="s", s=8, color=col, zorder=3, lw=0)
            ax.annotate(str(len(g)), (i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 1.5), textcoords="offset points", ha="center",
                        fontsize=4.5, color="#616161")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([thesis_label(o) for o in order], rotation=60, ha="right",
                           fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_xlim(-0.6, len(order) - 0.4)
    for r, mouse in enumerate(mice):
        axes[r, 0].set_ylabel(f"{mouse.replace('_', '.')}\nPeak dopamine after entry\n"
                              "(z, from exploring)", fontsize=6)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#616161", ms=3,
                          label="first exposure"),
               plt.Line2D([], [], marker="D", ls="", mfc="white", mec="#616161", ms=3,
                          label="mean of all, ± SEM"),
               plt.Line2D([], [], marker="s", ls="", color="#616161", ms=3,
                          label="last exposure"),
               mpatches.Patch(color="#D6D6D6", label="exploring, quartiles")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path)


def fig_course(course, mice, out):
    measures = [("pct_time_in_arms", "Time in arms (%)"),
                ("entries_per_min", "Arm entries per minute"),
                ("level_outside_z", "Dopamine outside arms (z)"),
                ("level_arms_z", "Dopamine in arms (z)"),
                ("transients_per_min", "Transients per minute"),
                ("transients_outside_per_min", "Transients per minute\noutside the arms")]
    fig, axes = plt.subplots(len(measures), len(DAYS), figsize=(3.4 * len(DAYS), 1.7 * len(measures)),
                             sharex=True, squeeze=False)
    for c, day in enumerate(DAYS):
        axes[0, c].set_title(DAY_TITLE[day])
    for (day, mouse), cdf in course.items():
        for r, (col, label) in enumerate(measures):
            ax = axes[r, DAYS.index(day)]
            g = cdf.groupby("minute")[col].agg(["mean", "sem"])
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=MOUSE_COLOURS.get(mouse, "k"),
                        lw=1.2, capsize=2, marker="o", ms=3, label=mouse)
            axes[r, 0].set_ylabel(label, fontsize=7)
            if col.startswith("level"):
                ax.axhline(0, color="grey", lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Minutes into the sound block")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Behaviour and dopamine through the sound blocks ({COURSE_BIN_MIN:g}-minute "
                 "bins; mean ± SEM over the four sound blocks)", fontsize=11)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_block_course"))


# -- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?",
                        default=os.path.join(fa.DEFAULT_ROOT, "FIBPHO_ANALYSIS"),
                        help="the folder fibpho_alignment.py wrote")
    parser.add_argument("--thesis-figures", metavar="DIR",
                        help="also write the thesis version of the exposure figure "
                             "as fig_fp_exposures")
    args = parser.parse_args()
    out = os.path.join(args.root, "behaviour")
    os.makedirs(os.path.join(out, "timelines"), exist_ok=True)
    rng = np.random.default_rng(0)

    summary = pd.read_csv(os.path.join(args.root, "alignment_summary.csv"))
    tables, expo, refs, course = {}, {}, {}, {}
    print("timelines:")
    for _, row in summary.iterrows():
        key = (row["day"], row["mouse"])
        s = load_session(args.root, row)
        fig_timeline(s, f"{row['mouse']}, {DAY_TITLE[row['day']].lower()}",
                     os.path.join(out, "timelines", f"{row['day']}_{row['mouse']}"))
        tables[key] = state_table(s)
        expo[key], refs[key] = exposures(s, rng)
        course[key] = block_course(s)
    mice = sorted({m for _, m in tables})

    print("summaries:")
    fig_state(tables, mice, "level", out)
    fig_state(tables, mice, "transients", out)
    fig_exposures(expo, refs, mice, out)
    fig_course(course, mice, out)

    states = pd.concat([t.assign(day=d, mouse=m) for (d, m), t in tables.items()])
    states.to_csv(os.path.join(out, "states.csv"), index=False)
    pd.concat([e.assign(day=d, mouse=m) for (d, m), e in expo.items()]).to_csv(
        os.path.join(out, "exposures.csv"), index=False)
    pd.concat([c.assign(day=d, mouse=m) for (d, m), c in course.items()]).to_csv(
        os.path.join(out, "block_course.csv"), index=False)

    # where the transients happened, against where the time went
    pooled = states[states["state"].isin([OUTSIDE, ENTRANCE, "all sound arms", "silent arm"])]
    p = pooled.groupby(["mouse", "day", "state"])[["seconds", "transients"]].sum()
    p["share_of_time_%"] = 100 * p["seconds"] / p.groupby(["mouse", "day"])["seconds"].transform("sum")
    p["share_of_transients_%"] = 100 * p["transients"] / p.groupby(["mouse", "day"])["transients"].transform("sum")
    p["per_min"] = p["transients"] / (p["seconds"] / 60)
    pd.set_option("display.width", 200)
    print("\nsound blocks: where the time went and where the transients were")
    print(p.round(1).to_string())
    e = exposure_table(expo, refs)
    print("\npeak z after entry minus the outside median, first exposures against the rest:")
    print(e.assign(first=e["exposure"] == 1).groupby("first")["rel"]
          .agg(["mean", "median", "size"]).round(3).to_string())
    print()
    exposure_stats(e, os.path.join(out, "exposure_stats.txt"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "pdf.fonttype": 42}):
            fig_exposures_thesis(e, refs, mice,
                                 os.path.join(args.thesis_figures, "fig_fp_exposures"))
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
