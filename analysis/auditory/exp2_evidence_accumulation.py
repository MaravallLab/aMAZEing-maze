#!/usr/bin/env python3
"""
Experiment 2: how many tone transitions does it take to tell the cages apart?
=============================================================================

EXPLORATORY, VIVA VERSION ONLY. Not part of the preregistered or confirmatory
analyses, and not reported in the submitted thesis (it appears only in
future_main.pdf, the viva version).

The question, in the words of the thesis: if you hear a piano melody, how many
notes do you need before you can say whether it is classical or jazz? Here the
two styles are the two grammars, one heard in the enriched cage (EE) and one in
the standard cage (SC). Many single steps are shared by the two grammars (both
have the rare steps -1 and -2, both have the step +2 as a secondary step, and
one grammar's dominant step is the other's secondary step), so a single
transition is rarely conclusive. Evidence has to pile up.

Part 1, simulation (no mouse data). A listener hears the melodies of one arm
and after every transition updates its belief that the arm belongs to the EE
cage. Written as log odds, every transition adds a fixed amount of evidence
(the log of how much more likely that step is under one cage's grammar than
under the other's), so the belief moves like a ball rolling towards one of
two walls: a drift diffusion process. The walls are at 95 and 5 percent belief.
The listener decides when the ball first touches a wall.

  memory window W  the listener keeps only the last W transitions in mind (the
                   log odds are the sum over those W transitions only). W runs
                   from 1 transition to a whole melody (11), two melodies (22)
                   and the whole visit. A leaky accumulator, which forgets old
                   evidence a little at every step, behaves much like a short
                   window: a leak with a time constant of about W transitions
                   gives similar curves. The two cannot be told apart here.
  transition       the step by step listener: each transition weighed against
                   the full grammars heard in training (the listener of the
                   fitted model, without its persistence term)
  level            a listener that also infers the level of the melody: each
                   cage is a mixture of its three level restricted chains (one
                   third each), the level held fixed over the window (an arm
                   plays one level), 1 percent of the full grammar mixed into
                   each restricted chain so that no step is impossible

Melodies are sampled from the level restricted chains as at test (12 tones,
11 transitions per melody). The step from the last tone of one melody to the
first tone of the next is not a grammar step and is not counted. Visits last
up to 5 melodies (55 transitions). For each level the script gives the share
of visits that reach the right wall, the wrong wall or neither, and the
transitions to a decision, against transitions heard and against W. Averaged
over both cages (the arm plays the EE or the SC grammar) and both
counterbalancing groups (15 and 17 mice), fixed seed.

Part 2, mouse data (test day 1). Each arm entry starts a fresh melody
(experiments.GrammarStimulus.render: melody, 2 s gap, repeat), so the length
of a visit gives the transitions the mouse heard: a transition counts once its
second tone has finished (0.2 s per tone within a melody, nothing new during
the 2 s gap). For each level, the visits that lasted at least n transitions
are pooled over mice and the share of them made to the EE grammar arm is
computed, for n = 0 (all entries), 1, 2, ... If a mouse recognises the cage
after about n* transitions and then stays longer in the EE arm (or leaves the
SC arm sooner), this share stays at its entry value up to n* and rises after
it. A second, more local measure is the leaving rate: of the visits still
going after n transitions, the share that ended before the next one, for the
EE and the SC arm. The transition at which the SC arm starts to be left more
often marks when the melody starts to matter. Both measures are given for all
visits and for the first visit to each arm in each block (before the mouse can
know from memory what an arm plays; few visits). The onset is compared with
Part 1. 95 percent intervals come from a
bootstrap over mice (resampling whole mice). No p values: this is exploratory
and the curves are read by eye. Day 1 has 32 mice and most visits are short
(median under 1 s), so only a rough range can be read off.

Usage:
    python exp2_evidence_accumulation.py [GRAMMAR_DIR]

GRAMMAR_DIR defaults to D:/AuditoryExperiments/grammar. Without it, only Part 1
runs.

Output (in exp2_evidence_accumulation/ next to this script):
    simulation.txt, simulation_curves.csv,
    simulation_summary.csv                     Part 1
    mice.txt, mice_survivor_share.csv,
    mice_leave_rates.csv                       Part 2 (summaries only, no raw data)
    evidence_accumulation.png / .pdf           the figure
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from amazeing.auditory.grammar_stimuli import config as gcfg  # noqa: E402
from exp2_listener_check import EPS, GROUP_N, LEVELS, restricted  # noqa: E402

OUT = os.path.join(HERE, "exp2_evidence_accumulation")
SEED = 20261119
WALL = 0.95
N_MEL = 5                                   # melodies per simulated visit
T_MEL = gcfg.MELODY_LENGTH - 1              # 11 transitions per melody
HORIZON = N_MEL * T_MEL                     # 55 transitions
WINDOWS = [1, 2, 3, 4, 5, 6, 8, 11, 22, HORIZON]
N_SIM = 4000                                # simulated visits per arm grammar and level
LISTENERS = ("transition", "level")
N_BOOT = 2000

TONE_S = gcfg.TONE_DURATION_MS / 1000       # 0.15
STEP_S = (gcfg.TONE_DURATION_MS + gcfg.INTER_TONE_GAP_MS) / 1000   # 0.2
CYCLE_S = gcfg.MELODY_CYCLE_MS / 1000       # 4.4
THRESH = list(range(0, 23)) + [33]          # transitions heard, Part 2


def wlabel(w):
    return "whole visit" if w == HORIZON else f"{w}"


# --------------------------------------------------------------------------- Part 1

def sample_visits(R, n, rng):
    """n visits of N_MEL melodies from chain R: tones array (n, N_MEL, 12)."""
    x = np.empty((n, N_MEL, gcfg.MELODY_LENGTH), dtype=int)
    x[:, :, 0] = rng.integers(gcfg.N_TONES, size=(n, N_MEL))
    cum = np.cumsum(R, axis=1)
    cum[:, -1] = 1.0
    for k in range(1, gcfg.MELODY_LENGTH):
        u = rng.random((n, N_MEL))
        x[:, :, k] = (u[..., None] >= cum[x[:, :, k - 1]]).sum(-1)
    return x


def step_loglik(tones, chains):
    """Log likelihood of every transition under each chain: (n, HORIZON, k)."""
    a, c = tones[:, :, :-1].reshape(len(tones), -1), tones[:, :, 1:].reshape(len(tones), -1)
    return np.stack([np.log(M[a, c]) for M in chains], axis=-1)


def window_sums(ll, w):
    """Sum of the last w per transition values, after each transition."""
    cs = np.cumsum(ll, axis=1)
    cs = np.concatenate([np.zeros_like(cs[:, :1]), cs], axis=1)
    t = np.arange(1, HORIZON + 1)
    return cs[:, t] - cs[:, np.maximum(t - w, 0)]


def window_logodds(tones, Mee, Msc, listener, w):
    """Log odds EE vs SC after each transition, from the last w transitions."""
    if listener == "transition":
        win = window_sums(step_loglik(tones, [Mee, Msc]), w)          # (n, T, 2)
        return win[..., 0] - win[..., 1]
    chains = [(1 - EPS) * restricted(M, lv) + EPS * M for M in (Mee, Msc) for lv in LEVELS]
    win = window_sums(step_loglik(tones, chains), w)                   # (n, T, 6)
    lee = np.logaddexp.reduce(win[..., :3], axis=-1)    # equal level weights cancel
    lsc = np.logaddexp.reduce(win[..., 3:], axis=-1)
    return lee - lsc


def first_hit(logodds_right):
    """Transition of the first wall crossing (1 based) and which wall."""
    up = np.log(WALL / (1 - WALL))
    hit_r, hit_w = logodds_right >= up, logodds_right <= -up
    any_hit = hit_r | hit_w
    first = np.where(any_hit.any(1), any_hit.argmax(1), -1)
    out = np.full(len(first), "none", dtype=object)
    rows = np.arange(len(first))
    out[(first >= 0) & hit_r[rows, first]] = "right"
    out[(first >= 0) & hit_w[rows, first]] = "wrong"
    return out, np.where(first >= 0, first + 1, 0)


def simulate(rng):
    """Outcome and decision time per simulated visit, with weights."""
    rows = []
    tot = sum(GROUP_N.values())
    for g, n_mice in GROUP_N.items():
        Mee = gcfg.GRAMMARS[gcfg.COUNTERBALANCE[g]["EE"]]
        Msc = gcfg.GRAMMARS[gcfg.COUNTERBALANCE[g]["SC"]]
        for cage, M in (("EE", Mee), ("SC", Msc)):
            for lv in LEVELS:
                tones = sample_visits(restricted(M, lv), N_SIM, rng)
                for listener in LISTENERS:
                    for w in WINDOWS:
                        lo = window_logodds(tones, Mee, Msc, listener, w)
                        lo = lo if cage == "EE" else -lo
                        out, t = first_hit(lo)
                        rows.append(pd.DataFrame({
                            "group": g, "cage": cage, "level": lv, "listener": listener,
                            "window": w, "outcome": out, "t": t,
                            "weight": n_mice / tot / 2 / N_SIM}))
    return pd.concat(rows, ignore_index=True)


def curves(sim):
    """Cumulative share at the right and wrong wall by transitions heard."""
    out = []
    n = np.arange(0, HORIZON + 1)
    for (listener, w, lv), d in sim.groupby(["listener", "window", "level"], sort=False):
        t, wt = d["t"].to_numpy(), d["weight"].to_numpy()
        r = np.array([wt[(d["outcome"].to_numpy() == "right") & (t <= k)].sum() for k in n])
        wr = np.array([wt[(d["outcome"].to_numpy() == "wrong") & (t <= k)].sum() for k in n])
        out.append(pd.DataFrame({"listener": listener, "window": w, "level": lv, "transitions": n,
                                 "right": r, "wrong": wr, "undecided": 1 - r - wr}))
    return pd.concat(out, ignore_index=True)


def wquant(t, wt, q):
    o = np.argsort(t)
    c = np.cumsum(wt[o]) / wt.sum()
    return int(t[o][np.searchsorted(c, q)])


def summarise(sim):
    lines = ["EXPLORATORY, viva version only: transitions a listener needs to tell the cages apart",
             "",
             f"Evidence accumulation with walls at {WALL:.0%} and {1 - WALL:.0%} belief that the arm",
             f"belongs to the EE cage. Visits of up to {N_MEL} melodies ({HORIZON} transitions).",
             f"{N_SIM} simulated visits per group, cage and level; averaged over both cages and",
             f"both groups (weights {GROUP_N[1]} and {GROUP_N[2]} mice); seed {SEED}.",
             "W is the memory window in transitions. Right, wrong, undecided: share of visits",
             "ending at each wall by the end of the visit. Transitions to the right wall:",
             "median [quartiles] among visits that reached it.",
             ""]
    rows = []
    for listener in LISTENERS:
        lines.append(f"{listener} listener")
        for lv in LEVELS:
            for w in WINDOWS:
                d = sim[(sim["listener"] == listener) & (sim["level"] == lv) & (sim["window"] == w)]
                share = {o: d.loc[d["outcome"] == o, "weight"].sum() for o in ("right", "wrong", "none")}
                rr = d[d["outcome"] == "right"]
                if len(rr):
                    t, wt = rr["t"].to_numpy(), rr["weight"].to_numpy()
                    q = [wquant(t, wt, p) for p in (0.25, 0.5, 0.75)]
                    tt = f"{q[1]:2d} [{q[0]}, {q[2]}]"
                else:
                    q, tt = [np.nan] * 3, "-"
                lines.append(f"  {lv:9s} W {wlabel(w):>11s}  right {share['right']:6.1%}  "
                             f"wrong {share['wrong']:6.1%}  undecided {share['none']:6.1%}  "
                             f"transitions to the right wall {tt}")
                rows.append({"listener": listener, "level": lv, "window": w, **share,
                             "t_q25": q[0], "t_median": q[1], "t_q75": q[2]})
            lines.append("")
    return lines, pd.DataFrame(rows)


# --------------------------------------------------------------------------- Part 2

def transitions_heard(dur):
    """Transitions heard in a visit of dur seconds that started a fresh melody."""
    dur = np.asarray(dur, dtype=float)
    cyc = np.floor(dur / CYCLE_S)
    rem = dur - cyc * CYCLE_S
    within = np.clip(np.floor((rem - TONE_S) / STEP_S), 0, T_MEL)   # tone k ends at 0.2k + 0.15
    return (cyc * T_MEL + within).astype(int)


def load_visits(root):
    from model_validation.data_loading import load_arm_blocks, load_day
    res = load_arm_blocks(root)
    day1 = load_day(res.arm_blocks, "primary")
    day1 = day1[day1.groupby("mouse")["time_spent_s"].transform("sum") > 0]
    keep = set(day1["timestamp"] + "/" + day1["mouse"])
    arms = day1[day1["arm_type"] == "grammar"].set_index(["mouse", "timestamp", "trial_id", "roi"])
    rows = []
    for s in res.sessions:
        if s.excluded or s.day != "day_1" or f"{s.timestamp}/{s.mouse}" not in keep or not s.visits_path:
            continue
        v = pd.read_csv(s.visits_path)
        v = v[v["trial_ID"] % 2 == 0]
        for r in v.itertuples(index=False):
            key = (s.mouse, s.timestamp, int(r.trial_ID), str(r.ROI_visited))
            if key not in arms.index:
                continue
            a = arms.loc[key]
            rows.append({"mouse": s.mouse, "group": a["group"], "block": int(r.trial_ID) // 2,
                         "roi": key[3], "environment": a["environment"], "level": a["tier"],
                         "start": r.sound_on_time, "dur": r.time_spent_seconds})
    vis = pd.DataFrame(rows).sort_values(["mouse", "start"])
    vis["n_heard"] = transitions_heard(vis["dur"])
    vis["first_in_block"] = ~vis.duplicated(["mouse", "block", "roi"])
    # sanity check: visit table against the trials table
    tot = vis.groupby(["mouse", "block", "roi"])["dur"].sum()
    ref = arms.reset_index().assign(block=lambda d: d["trial_id"] // 2) \
        .groupby(["mouse", "block", "roi"])["time_spent_s"].sum()
    j = pd.concat([tot, ref], axis=1).fillna(0)
    mism = float((j["dur"] - j["time_spent_s"]).abs().sum() / j["time_spent_s"].sum())
    return vis, day1["mouse"].nunique(), mism


def survivor_share(vis, rng):
    """EE share of visits lasting at least n transitions, per level, with a
    bootstrap over mice."""
    mice = vis["mouse"].unique()
    idx = rng.integers(len(mice), size=(N_BOOT, len(mice)))
    rows = []
    for lv in LEVELS:
        d = vis[vis["level"] == lv]
        c = np.zeros((len(mice), len(THRESH), 2))
        for i, m in enumerate(mice):
            dm = d[d["mouse"] == m]
            for j, n in enumerate(THRESH):
                ok = dm["n_heard"] >= n
                c[i, j, 0] = (ok & (dm["environment"] == "EE")).sum()
                c[i, j, 1] = (ok & (dm["environment"] == "SC")).sum()
        tot = c.sum(0)
        share = tot[:, 0] / tot.sum(1)
        b = c[idx].sum(1)                                      # (boot, thresh, 2)
        with np.errstate(invalid="ignore"):
            bs = b[..., 0] / b.sum(-1)
        diff = bs - bs[:, :1]
        for j, n in enumerate(THRESH):
            rows.append({"level": lv, "transitions": n, "n_EE": int(tot[j, 0]), "n_SC": int(tot[j, 1]),
                         "ee_share": share[j],
                         "lo": np.nanpercentile(bs[:, j], 2.5), "hi": np.nanpercentile(bs[:, j], 97.5),
                         "rise": share[j] - share[0],
                         "rise_lo": np.nanpercentile(diff[:, j], 2.5),
                         "rise_hi": np.nanpercentile(diff[:, j], 97.5)})
    return pd.DataFrame(rows)


def leave_rates(vis, rng):
    """Per level and n: of the visits still going after n transitions heard,
    the share that ended before the next transition, EE and SC arms, and the
    difference SC minus EE with a bootstrap over mice."""
    mice = vis["mouse"].unique()
    idx = rng.integers(len(mice), size=(N_BOOT, len(mice)))
    rows = []
    for lv in LEVELS:
        d = vis[vis["level"] == lv]
        ns = list(range(1, T_MEL + 1))
        c = np.zeros((len(mice), len(ns), 2, 2))           # mouse, n, cage, (at risk, left)
        for i, m in enumerate(mice):
            dm = d[d["mouse"] == m]
            for j, n in enumerate(ns):
                for k, env in enumerate(("EE", "SC")):
                    x = dm.loc[(dm["environment"] == env) & (dm["n_heard"] >= n), "n_heard"]
                    c[i, j, k] = [len(x), (x == n).sum()]
        tot = c.sum(0)
        b = c[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            h = tot[..., 1] / tot[..., 0]
            hb = b[..., 1] / b[..., 0]
        dd = hb[..., 1] - hb[..., 0]
        for j, n in enumerate(ns):
            rows.append({"level": lv, "transitions": n, "at_risk_EE": int(tot[j, 0, 0]),
                         "at_risk_SC": int(tot[j, 1, 0]), "leave_EE": h[j, 0], "leave_SC": h[j, 1],
                         "sc_minus_ee": h[j, 1] - h[j, 0],
                         "lo": np.nanpercentile(dd[:, j], 2.5), "hi": np.nanpercentile(dd[:, j], 97.5)})
    return pd.DataFrame(rows)


def onset(tab, lv):
    """First threshold at which the rise over the entry share has a 95 percent
    interval above zero, and stays above zero at the next threshold too."""
    t = tab[tab["level"] == lv].reset_index(drop=True)
    for j in range(1, len(t) - 1):
        if t.loc[j, "rise_lo"] > 0 and t.loc[j + 1, "rise_lo"] > 0:
            return int(t.loc[j, "transitions"])
    return None


def leave_report(name, t):
    lines = [f"{name}: of the visits still going after n transitions, the share that ended",
             "before the next transition (leaving rate), EE and SC arms, and SC minus EE",
             "[95% bootstrap interval over mice]. Visits that heard 11 transitions include",
             "all that ended in the 2 s gap after the first melody.",
             ]
    for lv in LEVELS:
        lines.append(f"  {lv}")
        for r in t[t["level"] == lv].itertuples():
            lines.append(f"    n = {r.transitions:2d}  at risk EE {r.at_risk_EE:4d} SC {r.at_risk_SC:4d}  "
                         f"leave EE {r.leave_EE:.3f} SC {r.leave_SC:.3f}  "
                         f"SC minus EE {r.sc_minus_ee:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")
    return lines + [""]


def mice_report(vis, n_mice, mism, tab, tab_first, lr_all, lr_first):
    lines = ["EXPLORATORY, viva version only: at what visit length does the EE preference appear?",
             "",
             f"Test day 1, {n_mice} mice, {len(vis)} visits to grammar arms "
             f"(visit table and trials table differ by {mism:.1%} of total time).",
             "Transitions heard: a transition counts once its second tone has ended (0.2 s per",
             "tone within a melody; nothing new during the 2 s gap between melodies).",
             ""]
    q = vis["n_heard"].quantile([0.25, 0.5, 0.75]).to_list()
    lines.append(f"Visit length: median {vis['dur'].median():.2f} s.")
    lines.append(f"Visit length in transitions heard: median {q[1]:.0f}, quartiles {q[0]:.0f} and {q[2]:.0f};"
                 f" {np.mean(vis['n_heard'] >= T_MEL):.1%} of visits heard a whole melody,"
                 f" {np.mean(vis['n_heard'] == 0):.1%} heard no transition.")
    lines.append("")
    for name, t, v in (("All visits", tab, vis),
                       ("First visit to each arm in each block", tab_first, vis[vis["first_in_block"]])):
        lines.append(f"{name} ({len(v)}): EE share of the visits that lasted at least n transitions,")
        lines.append("pooled over mice [95% bootstrap interval over mice]; rise = share minus the")
        lines.append("share at n = 0 (all entries) [95% interval].")
        for lv in LEVELS:
            o = onset(t, lv)
            lines.append(f"  {lv}  (onset: {o if o is not None else 'none'})")
            for r in t[t["level"] == lv].itertuples():
                lines.append(f"    n >= {r.transitions:2d}  visits EE {r.n_EE:4d} SC {r.n_SC:4d}  "
                             f"EE share {r.ee_share:.3f} [{r.lo:.3f}, {r.hi:.3f}]  "
                             f"rise {r.rise:+.3f} [{r.rise_lo:+.3f}, {r.rise_hi:+.3f}]")
        lines.append("")
    lines += leave_report("All visits", lr_all)
    lines += leave_report("First visit to each arm in each block", lr_first)
    lines += ["Onset: the first n at which the rise has a 95 percent interval above zero at n and",
              "at the next n. Exploratory, no correction for the many n and levels: read as a",
              "rough range only."]
    return lines


# --------------------------------------------------------------------------- figure

def figure(cur, tab, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    rows = 3 if tab is not None else 2
    fig, axes = plt.subplots(rows, 3, figsize=(7.0, 2.0 * rows + 0.3), sharex="col")
    show = [1, 2, 3, 5, 11, 22, HORIZON]
    cmap = plt.get_cmap("viridis")
    col = {w: cmap(i / (len(show) - 1) * 0.9) for i, w in enumerate(show)}
    names = {"transition": "Step by step listener", "level": "Listener that infers the level"}
    for i, listener in enumerate(LISTENERS):
        for j, lv in enumerate(LEVELS):
            ax = axes[i, j]
            for w in show:
                d = cur[(cur["listener"] == listener) & (cur["window"] == w) & (cur["level"] == lv)]
                ax.plot(d["transitions"], d["right"], color=col[w], lw=1.1,
                        label=f"W = {wlabel(w)}")
                ax.plot(d["transitions"], d["wrong"], color=col[w], lw=0.9, ls=(0, (3, 2)))
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, 33)
            if i == 0:
                ax.set_title(lv, fontsize=8)
            last = cur[(cur["listener"] == listener) & (cur["level"] == lv)
                       & (cur["transitions"] == HORIZON)]
            if (last["undecided"] > 0.999).all():
                ax.text(16.5, 0.12, "never reaches a wall", ha="center", va="center",
                        color="#757575", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{names[listener]}\nshare of visits decided")
            for x in (T_MEL, 2 * T_MEL):
                ax.axvline(x, color="#BDBDBD", lw=0.5, zorder=0)
    axes[0, 2].legend(frameon=False, fontsize=6, loc="upper right",
                      title="memory window\n(solid right wall,\ndashed wrong wall)", title_fontsize=6)
    if tab is not None:
        for j, lv in enumerate(LEVELS):
            ax = axes[2, j]
            d = tab[(tab["level"] == lv) & (tab["transitions"] <= 22)]
            ax.fill_between(d["transitions"], d["lo"], d["hi"], color="#1565C0", alpha=0.18, lw=0)
            ax.plot(d["transitions"], d["ee_share"], "o-", color="#1565C0", ms=2.5, lw=1)
            ax.axhline(0.5, color="black", lw=0.6, ls=(0, (4, 2)))
            ax.axhline(d["ee_share"].iloc[0], color="#1565C0", lw=0.6, ls=(0, (1, 1.5)))
            ax.set_ylim(0.3, 0.85)
            ax.set_xlabel("tone transitions heard")
            for x in (T_MEL, 2 * T_MEL):
                ax.axvline(x, color="#BDBDBD", lw=0.5, zorder=0)
            if j == 0:
                ax.set_ylabel("Mice, day 1\nEE share of visits\nlasting at least n transitions")
    else:
        for ax in axes[-1]:
            ax.set_xlabel("tone transitions heard")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sim = simulate(rng)
    cur = curves(sim)
    lines, summ = summarise(sim)
    cur.to_csv(os.path.join(OUT, "simulation_curves.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "simulation_summary.csv"), index=False)
    with open(os.path.join(OUT, "simulation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    root = sys.argv[1] if len(sys.argv) > 1 else "D:/AuditoryExperiments/grammar"
    tab = None
    if os.path.isdir(root):
        vis, n_mice, mism = load_visits(root)
        tab = survivor_share(vis, np.random.default_rng(SEED))
        tab_first = survivor_share(vis[vis["first_in_block"]], np.random.default_rng(SEED))
        pd.concat([tab.assign(visits="all"), tab_first.assign(visits="first in block")]) \
            .to_csv(os.path.join(OUT, "mice_survivor_share.csv"), index=False)
        lr_all = leave_rates(vis, np.random.default_rng(SEED))
        lr_first = leave_rates(vis[vis["first_in_block"]], np.random.default_rng(SEED))
        pd.concat([lr_all.assign(visits="all"), lr_first.assign(visits="first in block")]) \
            .to_csv(os.path.join(OUT, "mice_leave_rates.csv"), index=False)
        ml = mice_report(vis, n_mice, mism, tab, tab_first, lr_all, lr_first)
        with open(os.path.join(OUT, "mice.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ml) + "\n")
        print("\n".join(ml))
    else:
        print(f"{root} not found: Part 2 (mouse data) skipped")
    figure(cur, tab, os.path.join(OUT, "evidence_accumulation"))


if __name__ == "__main__":
    main()
