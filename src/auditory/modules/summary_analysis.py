"""
Cross-session summary analysis for grammar experiments.

Discovers session folders under a root path (day folder or grammar folder),
loads their trials CSVs, computes EE vs SC and per-tier preference metrics,
and generates summary figures per mouse, per day, and overall.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple

from modules.analysis import STIM_COLORS


ENV_COLORS  = {"EE": "#1565C0",  "SC": "#B71C1C"}
TIERS       = ["dominant", "secondary", "rare"]
ENVS        = ["EE", "SC"]

# All 6 tier × env categories, in display order
TIER_ENV_ORDER = [
    ("dominant",  "EE"),
    ("secondary", "EE"),
    ("rare",      "EE"),
    ("dominant",  "SC"),
    ("secondary", "SC"),
    ("rare",      "SC"),
]


def _day_sort_key(day: str) -> int:
    if day == "habituation":
        return -1
    m = re.search(r'\d+', day)
    return int(m.group()) if m else 999


def _discover_sessions(root: str) -> List[Dict]:
    """Walk root, return one dict per session folder that has a trials CSV."""
    sessions = []
    for dirpath, _dirs, _files in os.walk(root):
        trials_files = glob.glob(os.path.join(dirpath, "trials_*.csv"))
        if not trials_files:
            continue

        folder_name = os.path.basename(dirpath)
        mouse_match = re.search(r'(mouse\w+)', folder_name, re.IGNORECASE)
        mouse_id = mouse_match.group(0) if mouse_match else folder_name

        parent = os.path.basename(os.path.dirname(dirpath))
        day_label = parent if re.match(r'(day_\d+|habituation)', parent, re.IGNORECASE) else "unknown"

        trials_path = max(trials_files, key=os.path.getmtime)
        try:
            df = pd.read_csv(trials_path)
        except Exception:
            continue

        sessions.append({
            "path":      dirpath,
            "mouse_id":  mouse_id,
            "day_label": day_label,
            "trials_df": df,
        })
    return sessions


def _is_grammar_session(trials_df: pd.DataFrame) -> bool:
    return (
        "frequency" in trials_df.columns
        and (trials_df["frequency"] == "grammar").any()
    )


def _compute_metrics(trials_df: pd.DataFrame) -> Dict:
    """Return a flat dict of EE/SC and per-tier time (min), visits, and PI."""
    df = trials_df.copy()
    df["time_spent"]       = pd.to_numeric(df["time_spent"],       errors="coerce").fillna(0)
    df["visitation_count"] = pd.to_numeric(df["visitation_count"], errors="coerce").fillna(0)

    active = df[(df["trial_ID"] % 2 == 0) & (df["frequency"] == "grammar")]

    result: Dict = {}

    # ── environment totals ──────────────────────────────────────────────
    for env in ENVS:
        sub = active[active["environment_association"] == env]
        result[f"{env}_time"]   = float(sub["time_spent"].sum() / 60)
        result[f"{env}_visits"] = int(sub["visitation_count"].sum())

    total = result["EE_time"] + result["SC_time"]
    result["PI"] = (
        (result["EE_time"] - result["SC_time"]) / total if total > 0 else np.nan
    )

    # ── tier × environment breakdown ────────────────────────────────────
    for tier in TIERS:
        for env in ENVS:
            sub = active[
                (active["environment_association"] == env) &
                (active["tier"] == tier)
            ]
            key = f"{tier}_{env}"
            result[f"{key}_time"]   = float(sub["time_spent"].sum() / 60)
            result[f"{key}_visits"] = int(sub["visitation_count"].sum())

    return result


class SummaryAnalyzer:
    """Aggregate analysis across multiple sessions."""

    def __init__(self, root_path: str):
        self.root_path   = os.path.normpath(root_path)
        self.sessions:   List[Dict]              = []
        self.metrics_df: Optional[pd.DataFrame]  = None
        self._load_sessions()

    def _load_sessions(self):
        all_sessions  = _discover_sessions(self.root_path)
        self.sessions = [s for s in all_sessions if _is_grammar_session(s["trials_df"])]
        if not self.sessions:
            print("  [summary] No grammar sessions found.")
            return
        n_days = len({s["day_label"] for s in self.sessions})
        n_mice = len({s["mouse_id"]  for s in self.sessions})
        print(f"  [summary] Found {len(self.sessions)} session(s) across "
              f"{n_days} day(s) and {n_mice} mouse/mice.")

    def generate_report(self) -> str:
        """Compute metrics, produce all figures, save to root_path."""
        if not self.sessions:
            return self.root_path

        for s in self.sessions:
            s["metrics"] = _compute_metrics(s["trials_df"])

        rows = [
            {"mouse_id": s["mouse_id"], "day_label": s["day_label"], **s["metrics"]}
            for s in self.sessions
        ]
        self.metrics_df = pd.DataFrame(rows)

        days = sorted(self.metrics_df["day_label"].unique(), key=_day_sort_key)
        mice = sorted(self.metrics_df["mouse_id"].unique())
        multi_day = len(days) > 1

        figs: List[Tuple[str, plt.Figure]] = []

        # ── EE vs SC ──────────────────────────────────────────────────
        figs.append(self._fig_ee_sc_per_mouse(days, mice))
        figs.append(self._fig_preference_index(days, mice))
        figs.append(self._fig_group_summary(days))
        if multi_day:
            figs.append(self._fig_cross_day_pi(days, mice))

        # ── Predictive complexity (tier) ──────────────────────────────
        figs.append(self._fig_tier_breakdown_per_mouse(days, mice))
        figs.append(self._fig_group_tier_breakdown(days))
        if multi_day:
            figs.append(self._fig_cross_day_tiers(days, mice))

        for fname, fig in figs:
            out = os.path.join(self.root_path, fname)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [summary] Saved {fname}")

        return self.root_path

    # ══════════════════════════════════════════════════════════════════
    # EE vs SC figures
    # ══════════════════════════════════════════════════════════════════

    def _fig_ee_sc_per_mouse(self, days, mice) -> Tuple[str, plt.Figure]:
        df = self.metrics_df
        n  = max(len(days), 1)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
        fig.suptitle("EE vs SC time per mouse", fontsize=12)

        x, w = np.arange(len(mice)), 0.35
        for ax, day in zip(axes[0], days):
            dd = df[df["day_label"] == day]
            ee = self._col_per_mouse(dd, mice, "EE_time")
            sc = self._col_per_mouse(dd, mice, "SC_time")
            ax.bar(x - w / 2, ee, w, color=ENV_COLORS["EE"], label="EE", edgecolor="white")
            ax.bar(x + w / 2, sc, w, color=ENV_COLORS["SC"], label="SC", edgecolor="white")
            ax.set_xticks(x);  ax.set_xticklabels(mice, rotation=30, ha="right", fontsize=8)
            ax.set_title(day);  ax.set_xlabel("Mouse");  ax.legend(fontsize=8)

        axes[0][0].set_ylabel("Time on grammar arms (min)")
        plt.tight_layout()
        return ("summary_A_ee_sc_per_mouse.png", fig)

    def _fig_preference_index(self, days, mice) -> Tuple[str, plt.Figure]:
        df = self.metrics_df
        n  = max(len(days), 1)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True, squeeze=False)
        fig.suptitle(
            "EE preference index per mouse\n(+1 = all time EE arms,  −1 = all time SC arms)",
            fontsize=11,
        )
        for ax, day in zip(axes[0], days):
            dd  = df[df["day_label"] == day]
            pis = self._col_per_mouse(dd, mice, "PI", fill=np.nan)
            col = [ENV_COLORS["EE"] if (not np.isnan(v) and v > 0) else ENV_COLORS["SC"]
                   for v in pis]
            ax.bar(mice, pis, color=col, edgecolor="white")
            ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
            ax.set_ylim(-1.15, 1.15)
            ax.set_xticklabels(mice, rotation=30, ha="right", fontsize=8)
            ax.set_title(day);  ax.set_xlabel("Mouse")

        axes[0][0].set_ylabel("Preference Index (EE − SC) / (EE + SC)")
        leg = [mpatches.Patch(color=ENV_COLORS["EE"], label="EE preference"),
               mpatches.Patch(color=ENV_COLORS["SC"], label="SC preference")]
        axes[0][-1].legend(handles=leg, loc="upper right", fontsize=8)
        plt.tight_layout()
        return ("summary_B_preference_index.png", fig)

    def _fig_group_summary(self, days) -> Tuple[str, plt.Figure]:
        df  = self.metrics_df
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Group summary: EE vs SC grammar arms", fontsize=12)
        x, w = np.arange(len(days)), 0.35

        for metric, ax, ylabel in [
            ("time",   axes[0], "Mean time on grammar arms (min)"),
            ("visits", axes[1], "Mean visit count"),
        ]:
            for i, env in enumerate(ENVS):
                col   = f"{env}_{metric}"
                means = [df.loc[df["day_label"] == d, col].mean() for d in days]
                sems  = [df.loc[df["day_label"] == d, col].sem()  for d in days]
                ax.bar(x + i * w, means, w, yerr=sems,
                       color=ENV_COLORS[env], label=env, edgecolor="white",
                       capsize=4, error_kw={"linewidth": 1.5})
            ax.set_xticks(x + w / 2);  ax.set_xticklabels(days, rotation=20, ha="right")
            ax.set_ylabel(ylabel);  ax.set_title(ylabel);  ax.legend(title="Grammar env.")

        plt.tight_layout()
        return ("summary_C_group_summary.png", fig)

    def _fig_cross_day_pi(self, days, mice) -> Tuple[str, plt.Figure]:
        df    = self.metrics_df
        day_x = {d: i for i, d in enumerate(days)}
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle("EE preference index across days", fontsize=12)

        cmap = plt.cm.tab10
        for idx, mouse in enumerate(mice):
            md = (df[df["mouse_id"] == mouse]
                  .sort_values("day_label", key=lambda s: s.map(_day_sort_key)))
            xs = [day_x[d] for d in md["day_label"]]
            ax.plot(xs, md["PI"].values, marker="o", label=mouse,
                    color=cmap(idx / max(len(mice) - 1, 1)),
                    linewidth=1.5, markersize=7, alpha=0.75)

        means = [df.loc[df["day_label"] == d, "PI"].mean() for d in days]
        sems  = [df.loc[df["day_label"] == d, "PI"].sem()  for d in days]
        ax.errorbar(range(len(days)), means, yerr=sems, fmt="-s",
                    color="black", linewidth=2.5, markersize=9,
                    label="group mean ± SEM", zorder=5)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(days)));  ax.set_xticklabels(days)
        ax.set_ylabel("Preference Index");  ax.set_ylim(-1.15, 1.15)
        ax.set_title("Each line = one mouse,  black = group mean ± SEM")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        return ("summary_D_cross_day_pi.png", fig)

    # ══════════════════════════════════════════════════════════════════
    # Predictive complexity (tier) figures
    # ══════════════════════════════════════════════════════════════════

    def _fig_tier_breakdown_per_mouse(self, days, mice) -> Tuple[str, plt.Figure]:
        """Stacked bars per mouse: EE bar (dom/sec/rare) and SC bar (dom/sec/rare)."""
        df = self.metrics_df
        n  = max(len(days), 1)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
        fig.suptitle(
            "Predictive complexity breakdown per mouse\n"
            "Left bar = EE arms, right bar = SC arms",
            fontsize=12,
        )
        x, w = np.arange(len(mice)), 0.35

        for ax, day in zip(axes[0], days):
            dd = df[df["day_label"] == day]
            ee_bot = np.zeros(len(mice))
            sc_bot = np.zeros(len(mice))

            for tier, env in TIER_ENV_ORDER:
                col    = f"{tier}_{env}_time"
                color  = STIM_COLORS.get(f"{tier} {env}", "#ccc")
                vals   = self._col_per_mouse(dd, mice, col)
                offset = -w / 2 if env == "EE" else w / 2
                bottom = ee_bot if env == "EE" else sc_bot

                ax.bar(x + offset, vals, w, bottom=bottom,
                       color=color, edgecolor="white", linewidth=0.3)
                if env == "EE":
                    ee_bot += vals
                else:
                    sc_bot += vals

            ax.set_xticks(x);  ax.set_xticklabels(mice, rotation=30, ha="right", fontsize=8)
            ax.set_title(day);  ax.set_xlabel("Mouse")

        axes[0][0].set_ylabel("Time on grammar arms (min)")

        leg = [mpatches.Patch(color=STIM_COLORS.get(f"{t} {e}", "#ccc"),
                              label=f"{t} {e}")
               for t, e in TIER_ENV_ORDER]
        axes[0][-1].legend(handles=leg, bbox_to_anchor=(1.02, 1),
                           loc="upper left", fontsize=8)
        plt.tight_layout()
        return ("summary_E_tier_breakdown_per_mouse.png", fig)

    def _fig_group_tier_breakdown(self, days) -> Tuple[str, plt.Figure]:
        """Group mean ± SEM for all 6 tier × environment categories, one panel per day."""
        df     = self.metrics_df
        labels = [f"{t} {e}" for t, e in TIER_ENV_ORDER]
        colors = [STIM_COLORS.get(lbl, "#ccc") for lbl in labels]
        n      = max(len(days), 1)

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
        fig.suptitle("Group predictive complexity breakdown (mean ± SEM)", fontsize=12)

        for ax, day in zip(axes[0], days):
            dd    = df[df["day_label"] == day]
            means, sems = [], []
            for tier, env in TIER_ENV_ORDER:
                col = f"{tier}_{env}_time"
                means.append(dd[col].mean() if col in dd else 0)
                sems.append(dd[col].sem()   if col in dd else 0)

            xp = np.arange(len(labels))
            ax.bar(xp, means, color=colors, edgecolor="white",
                   yerr=sems, capsize=4, error_kw={"linewidth": 1.5})

            # draw a vertical separator between EE and SC groups
            ax.axvline(2.5, color="gray", linewidth=1, linestyle=":")

            ax.set_xticks(xp);  ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(day);  ax.set_xlabel("Tier × environment")

        axes[0][0].set_ylabel("Mean time on arms (min)")
        plt.tight_layout()
        return ("summary_F_group_tier_breakdown.png", fig)

    def _fig_cross_day_tiers(self, days, mice) -> Tuple[str, plt.Figure]:
        """Group mean ± SEM per tier across days, split into EE (left) and SC (right)."""
        df  = self.metrics_df
        x   = np.arange(len(days))

        tier_line_colors = {
            ("dominant",  "EE"): "#1565C0",
            ("secondary", "EE"): "#42A5F5",
            ("rare",      "EE"): "#90CAF9",
            ("dominant",  "SC"): "#B71C1C",
            ("secondary", "SC"): "#EF5350",
            ("rare",      "SC"): "#FFAB91",
        }

        fig, (ax_ee, ax_sc) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(
            "Predictive complexity across days — group mean ± SEM",
            fontsize=12,
        )

        for env, ax in [("EE", ax_ee), ("SC", ax_sc)]:
            for tier in TIERS:
                col    = f"{tier}_{env}_time"
                color  = tier_line_colors[(tier, env)]
                means  = [df.loc[df["day_label"] == d, col].mean() for d in days]
                sems   = [df.loc[df["day_label"] == d, col].sem()  for d in days]
                ax.errorbar(x, means, yerr=sems, fmt="-o",
                            color=color, label=tier,
                            linewidth=2, markersize=7, capsize=4)

                # individual mouse lines (thin, same colour, no label)
                for mouse in mice:
                    md = (df[df["mouse_id"] == mouse]
                          .sort_values("day_label", key=lambda s: s.map(_day_sort_key)))
                    xs_m = [x[days.index(d)] for d in md["day_label"] if d in days]
                    ys_m = [md.loc[md["day_label"] == d, col].values[0]
                            for d in md["day_label"] if d in days]
                    if xs_m:
                        ax.plot(xs_m, ys_m, color=color, linewidth=0.6,
                                alpha=0.35, marker="o", markersize=3)

            ax.set_xticks(x);  ax.set_xticklabels(days)
            ax.set_title(f"{env}-associated arms")
            ax.set_xlabel("Day");  ax.legend(title="Tier")

        ax_ee.set_ylabel("Mean time (min)")
        ax_sc.tick_params(labelleft=True)
        plt.tight_layout()
        return ("summary_G_cross_day_tiers.png", fig)

    # ── helpers ────────────────────────────────────────────────────────

    def _col_per_mouse(self, day_df: pd.DataFrame, mice: list, col: str,
                       fill: float = 0.0) -> np.ndarray:
        """Return array of col values aligned to mice list; fill if mouse absent."""
        out = []
        for m in mice:
            rows = day_df.loc[day_df["mouse_id"] == m, col]
            out.append(float(rows.values[0]) if len(rows) > 0 else fill)
        return np.array(out)
