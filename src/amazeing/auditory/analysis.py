import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple


# Consistent colour palette for stimulus categories
STIM_COLORS = {
    "dominant EE":   "#1565C0",
    "secondary EE":  "#42A5F5",
    "rare EE":       "#BBDEFB",
    "dominant SC":   "#B71C1C",
    "secondary SC":  "#EF5350",
    "rare SC":       "#FFCDD2",
    "vocalisation":  "#2E7D32",
    "silent":        "#9E9E9E",
    "unknown":       "#BDBDBD",
}

TIER_ORDER = ["dominant", "secondary", "rare"]
ENV_ORDER  = ["EE", "SC"]


_ABBREV = {
    "dominant EE":   "dom EE",
    "secondary EE":  "sec EE",
    "rare EE":       "rare EE",
    "dominant SC":   "dom SC",
    "secondary SC":  "sec SC",
    "rare SC":       "rare SC",
    "vocalisation":  "voc",
    "silent":        "sil",
}


def _abbrev_stim(label: str) -> str:
    return _ABBREV.get(label, label)


def _stim_label(row) -> str:
    freq = row.get("frequency", "")
    if freq == "grammar":
        tier = row.get("tier", "-")
        env  = row.get("environment_association", "-")
        return f"{tier} {env}"
    if freq == "vocalisation":
        return "vocalisation"
    if freq == 0 or str(freq) == "0":
        return "silent"
    return str(freq)


def _legend_patches(labels):
    seen, patches = set(), []
    for lbl in labels:
        if lbl not in seen:
            patches.append(
                mpatches.Patch(color=STIM_COLORS.get(lbl, STIM_COLORS["unknown"]),
                               label=lbl)
            )
            seen.add(lbl)
    return patches


class SessionAnalyzer:

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.trials_df:   Optional[pd.DataFrame] = None
        self.visits_df:   Optional[pd.DataFrame] = None
        self.maze_df:     Optional[pd.DataFrame] = None
        self._session_title = os.path.basename(session_dir)
        self._load_data()

    # ── data loading ──────────────────────────────────────────────────

    def _load_data(self):
        def _newest(pattern):
            hits = glob.glob(os.path.join(self.session_dir, pattern))
            return max(hits, key=os.path.getmtime) if hits else None

        path = _newest("trials_*.csv")
        if path:
            self.trials_df = pd.read_csv(path)

        path = _newest("*_detailed_visits.csv")
        if path:
            self.visits_df = pd.read_csv(path)

        path = _newest("*_maze_entries.csv")
        if path:
            self.maze_df = pd.read_csv(path)

    def _is_grammar(self) -> bool:
        return (
            self.trials_df is not None
            and "grammar" in self.trials_df.columns
        )

    # ── public entry point ────────────────────────────────────────────

    def generate_report(self):
        if self.trials_df is None:
            print("  [analysis] No trials CSV found — skipping.")
            return

        figures: List[Tuple[str, plt.Figure]] = (
            self._grammar_figures() if self._is_grammar()
            else self._generic_figures()
        )

        for fname, fig in figures:
            out = os.path.join(self.session_dir, fname)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [analysis] Saved {fname}")

    # ── grammar experiment figures ────────────────────────────────────

    def _grammar_figures(self) -> List[Tuple[str, plt.Figure]]:
        df = self.trials_df.copy()
        df["ROIs"] = df["ROIs"].astype(str)
        df["stim_label"] = df.apply(_stim_label, axis=1)
        df["time_spent"]  = pd.to_numeric(df["time_spent"],  errors="coerce").fillna(0)
        df["visitation_count"] = pd.to_numeric(df["visitation_count"], errors="coerce").fillna(0)

        # Active blocks only (trial_ID 2, 4, 6, 8)
        active = df[df["trial_ID"] % 2 == 0].copy()

        figs = []
        figs.append(self._fig_arm_totals(active))
        figs.append(self._fig_ee_vs_sc(active))
        figs.append(self._fig_block_evolution(active))
        if self.visits_df is not None and not self.visits_df.empty:
            f = self._fig_visit_duration(active)
            if f:
                figs.append(f)
        if self.maze_df is not None and not self.maze_df.empty:
            f = self._fig_maze_time()
            if f:
                figs.append(f)
        figs.append(self._fig_location_preference(active))
        return figs

    # ── Figure 1: total time and visit count per arm ──────────────────

    def _fig_arm_totals(self, active: pd.DataFrame) -> Tuple[str, plt.Figure]:
        totals = (
            active.groupby(["ROIs", "stim_label"])
            .agg(time_min=("time_spent", lambda x: x.sum() / 60),
                 visits=("visitation_count", "sum"))
            .reset_index()
            .sort_values("ROIs")
        )

        rois   = totals["ROIs"].astype(str).tolist()
        colors = [STIM_COLORS.get(s, STIM_COLORS["unknown"])
                  for s in totals["stim_label"]]

        fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Arm summary — {self._session_title}", fontsize=10)

        ax_t.bar(rois, totals["time_min"], color=colors, edgecolor="white", linewidth=0.5)
        ax_t.set_xlabel("Arm (ROI)")
        ax_t.set_ylabel("Total time (min)")
        ax_t.set_title("Time spent per arm")
        ax_t.legend(handles=_legend_patches(totals["stim_label"].unique()),
                    bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        ax_v.bar(rois, totals["visits"], color=colors, edgecolor="white", linewidth=0.5)
        ax_v.set_xlabel("Arm (ROI)")
        ax_v.set_ylabel("Total visits")
        ax_v.set_title("Visit count per arm")

        plt.tight_layout()
        return ("fig1_arm_totals.png", fig)

    # ── Figure 2: EE vs SC preference by tier ─────────────────────────

    def _fig_ee_vs_sc(self, active: pd.DataFrame) -> Tuple[str, plt.Figure]:
        grammar_rows = active[active["frequency"] == "grammar"].copy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
        fig.suptitle(f"EE vs SC preference — {self._session_title}", fontsize=10)

        for ax, metric, ylabel in zip(
            [axes[0], axes[1]],
            ["time_spent", "visitation_count"],
            ["Total time (min)", "Total visits"],
        ):
            x = np.arange(len(TIER_ORDER))
            width = 0.35
            for i, env in enumerate(ENV_ORDER):
                sub = grammar_rows[grammar_rows["environment_association"] == env]
                by_tier = (
                    sub.groupby("tier")[metric]
                    .sum()
                    .reindex(TIER_ORDER, fill_value=0)
                )
                vals = by_tier.values / 60 if metric == "time_spent" else by_tier.values
                bar_colors = [STIM_COLORS.get(f"{t} {env}", "#ccc") for t in TIER_ORDER]
                bars = ax.bar(x + i * width, vals, width,
                              color=bar_colors, edgecolor="white", label=env)

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(TIER_ORDER)
            ax.set_xlabel("Predictability tier")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(title="Environment")

        plt.tight_layout()
        return ("fig2_ee_vs_sc.png", fig)

    # ── Figure 3: preference evolution across active blocks ───────────

    def _fig_block_evolution(self, active: pd.DataFrame) -> Tuple[str, plt.Figure]:
        block_stim = (
            active.groupby(["trial_ID", "stim_label"])["time_spent"]
            .sum()
            .reset_index()
        )
        block_stim["time_min"] = block_stim["time_spent"] / 60

        present = block_stim["stim_label"].unique()
        stim_order = [s for s in STIM_COLORS if s in present]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"Block-by-block evolution — {self._session_title}", fontsize=10)

        for stim in stim_order:
            sub = block_stim[block_stim["stim_label"] == stim].sort_values("trial_ID")
            if sub.empty:
                continue
            ax.plot(sub["trial_ID"], sub["time_min"],
                    marker="o", label=stim,
                    color=STIM_COLORS.get(stim, "#999"))

        ax.set_xlabel("Active block (trial ID)")
        ax.set_ylabel("Time spent (min)")
        ax.set_title("Time per stimulus category across active blocks")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        return ("fig3_block_evolution.png", fig)

    # ── Figure 4: visit duration distribution per stimulus ────────────

    def _fig_visit_duration(self, active: pd.DataFrame) -> Optional[Tuple[str, plt.Figure]]:
        visits = self.visits_df.copy()

        # exclude entrance ROIs
        visits = visits[~visits["ROI_visited"].isin(["entrance1", "entrance2"])]
        visits = visits[visits["time_spent_seconds"] > 0.1]
        if visits.empty:
            return None

        # attach stimulus label via (trial_ID, ROI) lookup
        # cast to str so numeric ROI names ("1","2"...) don't cause a type mismatch
        visits["ROI_visited"] = visits["ROI_visited"].astype(str)
        label_map = (
            active[["trial_ID", "ROIs", "stim_label"]]
            .drop_duplicates()
            .assign(ROIs=lambda d: d["ROIs"].astype(str))
        )
        visits = visits.merge(
            label_map,
            left_on=["trial_ID", "ROI_visited"],
            right_on=["trial_ID", "ROIs"],
            how="left",
        )
        visits["stim_label"] = visits["stim_label"].fillna("unknown")

        stims = [s for s in STIM_COLORS if s in visits["stim_label"].unique()]
        if not stims:
            return None

        data   = [visits[visits["stim_label"] == s]["time_spent_seconds"].values for s in stims]
        colors = [STIM_COLORS.get(s, "#999") for s in stims]

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.suptitle(f"Visit duration distribution — {self._session_title}", fontsize=10)

        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)

        ax.set_xticks(range(1, len(stims) + 1))
        ax.set_xticklabels(stims, rotation=25, ha="right")
        ax.set_ylabel("Visit duration (s)")
        ax.set_title("Visit duration per stimulus type")
        plt.tight_layout()
        return ("fig4_visit_duration.png", fig)

    # ── Figure 5: time in maze per block ──────────────────────────────

    def _fig_maze_time(self) -> Optional[Tuple[str, plt.Figure]]:
        exits = self.maze_df[
            self.maze_df["event"].isin(["exited", "session_end_still_inside"])
        ].copy()
        if exits.empty:
            return None

        by_trial = exits.groupby("trial_ID")["time_in_maze_seconds"].sum() / 60

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"Maze occupancy per block — {self._session_title}", fontsize=10)

        bar_colors = ["#90A4AE" if tid % 2 != 0 else "#455A64"
                      for tid in by_trial.index]
        ax.bar(by_trial.index.astype(str), by_trial.values, color=bar_colors)
        ax.set_xlabel("Trial (block)")
        ax.set_ylabel("Time in maze (min)")
        ax.set_title("Time inside maze per block  (dark = active, light = silent)")

        legend_patches = [
            mpatches.Patch(color="#455A64", label="active block"),
            mpatches.Patch(color="#90A4AE", label="silent block"),
        ]
        ax.legend(handles=legend_patches)
        plt.tight_layout()
        return ("fig5_maze_time.png", fig)

    # ── Figure 6: location preference (arm × block heatmap) ──────────

    def _fig_location_preference(self, active: pd.DataFrame) -> Tuple[str, plt.Figure]:
        # Try to sort ROI labels numerically (they're "1"-"8")
        def _roi_sort_key(r):
            try:
                return int(r)
            except ValueError:
                return r

        roi_order  = sorted(active["ROIs"].unique(), key=_roi_sort_key)
        block_order = sorted(active["trial_ID"].unique())

        # time matrix (rows=ROIs, cols=blocks)
        time_matrix  = np.zeros((len(roi_order), len(block_order)))
        stim_matrix  = [[""] * len(block_order) for _ in roi_order]

        for bi, block in enumerate(block_order):
            for ri, roi in enumerate(roi_order):
                mask = (active["trial_ID"] == block) & (active["ROIs"] == roi)
                rows = active[mask]
                if not rows.empty:
                    time_matrix[ri, bi] = rows["time_spent"].sum() / 60
                    stim_matrix[ri][bi] = _abbrev_stim(rows["stim_label"].iloc[0])

        # ── layout: heatmap (wide) + totals bar (narrow) ─────────────
        fig, (ax_h, ax_b) = plt.subplots(
            1, 2, figsize=(13, 5),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        fig.suptitle(f"Location preference — {self._session_title}", fontsize=10)

        vmax = max(time_matrix.max(), 0.01)
        im = ax_h.imshow(time_matrix, aspect="auto", cmap="YlOrRd",
                         vmin=0, vmax=vmax)

        ax_h.set_xticks(range(len(block_order)))
        ax_h.set_xticklabels([f"Block {b}" for b in block_order])
        ax_h.set_yticks(range(len(roi_order)))
        ax_h.set_yticklabels([f"Arm {r}" for r in roi_order])
        ax_h.set_xlabel("Active block")
        ax_h.set_title("Time per arm per block (min)\n— cell label = stimulus assigned —")

        for ri in range(len(roi_order)):
            for bi in range(len(block_order)):
                lbl  = stim_matrix[ri][bi]
                val  = time_matrix[ri, bi]
                text_color = "white" if val > vmax * 0.6 else "black"
                ax_h.text(bi, ri, lbl, ha="center", va="center",
                          fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax_h, label="Time (min)", shrink=0.8)

        # side bar: total time per arm across all active blocks
        arm_totals = time_matrix.sum(axis=1)
        ax_b.barh(range(len(roi_order)), arm_totals, color="#607D8B")
        ax_b.set_yticks(range(len(roi_order)))
        ax_b.set_yticklabels([f"Arm {r}" for r in roi_order])
        ax_b.invert_yaxis()
        ax_b.set_xlabel("Total time (min)")
        ax_b.set_title("Total across\nall active blocks")

        plt.tight_layout()
        return ("fig6_location_preference.png", fig)

    # ── generic fallback (non-grammar experiments) ────────────────────

    def _generic_figures(self) -> List[Tuple[str, plt.Figure]]:
        df = self.trials_df.copy()
        df["time_spent"] = pd.to_numeric(df["time_spent"], errors="coerce").fillna(0)
        active = df[df["trial_ID"] % 2 == 0]

        by_roi = active.groupby("ROIs")["time_spent"].sum().sort_index() / 60

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"Arm summary — {self._session_title}", fontsize=10)
        ax.bar(by_roi.index.astype(str), by_roi.values, color="#607D8B")
        ax.set_xlabel("Arm (ROI)")
        ax.set_ylabel("Total time (min)")
        ax.set_title("Time spent per arm (active blocks only)")
        plt.tight_layout()
        return [("fig1_arm_time.png", fig)]
