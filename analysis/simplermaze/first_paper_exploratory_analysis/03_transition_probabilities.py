"""
03 — Within-trial transition probabilities from DLC tracking.

For each trial in sessions with DLC data:
  1. Assign each frame to an ROI (entrance1/2, rewA-D) or 'corridor'
  2. Collapse consecutive identical states -> state-change sequence
  3. Build per-trial transition matrix
  4. Aggregate: Hit vs Miss comparison

Outputs:
  - transition_matrix_hit.png / .pdf   — average transition heatmap for Hits
  - transition_matrix_miss.png / .pdf  — average transition heatmap for Misses
  - transition_matrix_diff.png / .pdf  — Hit minus Miss difference
  - transition_summary.csv             — per-trial metrics (perseveration, exploration entropy)
  - transition_combined.png / .pdf     — side-by-side Hit / Miss / Diff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, entropy as shannon_entropy

from config import (MOUSE_ID, MOUSE_DIR, FPS, BODYPART,
                    LIKELIHOOD_THRESH, get_sessions_with_dlc)
from utils import (load_trials, classify_trial, load_dlc, load_rois,
                   compute_state_sequence, collapse_state_sequence,
                   transition_matrix)

OUTPUT_DIR = os.path.join(MOUSE_DIR, f"MOUSE_{MOUSE_ID}_TOTAL_ANALYSIS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# canonical state order for consistent matrices
STATE_LABELS = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD", "corridor"]


# ── 1. compute per-trial transitions ───────────────────────────────

sessions = get_sessions_with_dlc()
print(f"Processing transitions for {len(sessions)} sessions")

trial_records = []
hit_matrices = []
miss_matrices = []

for sess in sessions:
    print(f"\n--- Session {sess.session_id} ---")

    df_trials = load_trials(sess.trial_csv)
    rois = load_rois(sess.roi_csv) if sess.roi_csv else {}
    tracking = load_dlc(sess.dlc_csv, bodypart=BODYPART,
                        likelihood_thresh=LIKELIHOOD_THRESH)

    if not rois:
        print("  No ROI file, skipping")
        continue

    # determine frame columns
    if "start_trial_frame" in df_trials.columns:
        fc_start, fc_end = "start_trial_frame", "end_trial_frame"
    elif "start_frame" in df_trials.columns:
        fc_start, fc_end = "start_frame", "end_frame"
    else:
        print("  No frame columns, skipping")
        continue

    for idx, row in df_trials.iterrows():
        try:
            f_start = int(row[fc_start])
            f_end = int(row[fc_end])
        except (ValueError, TypeError):
            continue
        if f_start >= f_end or (f_end - f_start) < 10:
            continue

        outcome = classify_trial(row)
        if outcome == "no_choice":
            continue
        status = "Hit" if outcome == "correct" else "Miss"

        # extract trial tracking
        trial_track = tracking.iloc[f_start:f_end]
        if len(trial_track) < 10:
            continue

        # frame-by-frame ROI assignment
        raw_states = compute_state_sequence(trial_track, rois, buffer=5)

        # filter out 'unknown' frames (low-likelihood DLC points)
        raw_states_clean = [s for s in raw_states if s != "unknown"]
        if len(raw_states_clean) < 5:
            continue

        # collapse consecutive identical states
        collapsed = collapse_state_sequence(raw_states_clean)

        # build transition matrix for this trial
        mat, _ = transition_matrix(collapsed, state_labels=STATE_LABELS,
                                   normalise=True)

        # count raw transitions (for aggregation)
        mat_raw, _ = transition_matrix(collapsed, state_labels=STATE_LABELS,
                                       normalise=False)

        if status == "Hit":
            hit_matrices.append(mat_raw)
        else:
            miss_matrices.append(mat_raw)

        # compute per-trial metrics
        n_transitions = len(collapsed) - 1
        n_unique_states = len(set(collapsed) - {"corridor", "unknown"})

        # perseveration: fraction of transitions that return to the same
        # ROI the mouse just left (A -> corridor -> A)
        n_persev = 0
        for i in range(len(collapsed) - 2):
            if collapsed[i] == collapsed[i + 2] and collapsed[i] != "corridor":
                n_persev += 1
        persev_rate = n_persev / max(n_transitions, 1)

        # exploration entropy: entropy of the visit distribution
        # (which ROIs were visited and how evenly)
        roi_visits = [s for s in collapsed if s not in ("corridor", "unknown")]
        if roi_visits:
            visit_counts = pd.Series(roi_visits).value_counts()
            probs = visit_counts.values / visit_counts.values.sum()
            expl_entropy = shannon_entropy(probs)
        else:
            expl_entropy = 0.0

        trial_records.append({
            "mouse_id": MOUSE_ID,
            "session_id": sess.session_id,
            "trial_id": row.get("trial_ID", idx),
            "status": status,
            "n_state_changes": n_transitions,
            "n_unique_rois": n_unique_states,
            "perseveration_rate": persev_rate,
            "exploration_entropy": expl_entropy,
            "state_sequence": " -> ".join(collapsed),
        })

df_transitions = pd.DataFrame(trial_records)
df_transitions.to_csv(os.path.join(OUTPUT_DIR, "transition_summary.csv"), index=False)
print(f"\nProcessed {len(df_transitions)} trials")
print(df_transitions.groupby("status")[
    ["n_state_changes", "n_unique_rois", "perseveration_rate", "exploration_entropy"]
].mean().round(3))


# ── 2. aggregate transition matrices ────────────────────────────────

def aggregate_matrices(matrices):
    """Sum raw count matrices and normalise rows."""
    if not matrices:
        return np.zeros((len(STATE_LABELS), len(STATE_LABELS)))
    total = np.sum(matrices, axis=0)
    row_sums = total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return total / row_sums

avg_hit = aggregate_matrices(hit_matrices)
avg_miss = aggregate_matrices(miss_matrices)
diff = avg_hit - avg_miss


# ── 3. plot transition heatmaps ─────────────────────────────────────

short_labels = ["ent1", "ent2", "rewA", "rewB", "rewC", "rewD", "corr"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Hit matrix
sns.heatmap(avg_hit, ax=axes[0], annot=True, fmt=".2f", cmap="Greens",
            xticklabels=short_labels, yticklabels=short_labels,
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
axes[0].set_title(f"Hit trials (n={len(hit_matrices)})", fontsize=12)
axes[0].set_xlabel("To")
axes[0].set_ylabel("From")

# Miss matrix
sns.heatmap(avg_miss, ax=axes[1], annot=True, fmt=".2f", cmap="Reds",
            xticklabels=short_labels, yticklabels=short_labels,
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
axes[1].set_title(f"Miss trials (n={len(miss_matrices)})", fontsize=12)
axes[1].set_xlabel("To")
axes[1].set_ylabel("From")

# Difference (Hit - Miss)
max_abs = max(np.abs(diff).max(), 0.01)
sns.heatmap(diff, ax=axes[2], annot=True, fmt=".2f", cmap="RdBu_r",
            xticklabels=short_labels, yticklabels=short_labels,
            vmin=-max_abs, vmax=max_abs, center=0,
            cbar_kws={"shrink": 0.8})
axes[2].set_title("Difference (Hit - Miss)", fontsize=12)
axes[2].set_xlabel("To")
axes[2].set_ylabel("From")

plt.suptitle(f"Mouse {MOUSE_ID} — Transition probabilities", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "transition_combined.png"),
            dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "transition_combined.pdf"),
            bbox_inches="tight")


# ── 4. perseveration / exploration comparison ───────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
palette = {"Hit": "#2CA02C", "Miss": "#D62728"}

compare_metrics = [
    ("perseveration_rate", "Perseveration rate"),
    ("exploration_entropy", "Exploration entropy"),
    ("n_unique_rois", "Unique ROIs visited"),
]

for ax, (col, label) in zip(axes2, compare_metrics):
    sns.violinplot(data=df_transitions, x="status", y=col, ax=ax,
                   palette=palette, hue="status", inner="quart",
                   alpha=0.4, legend=False)
    sns.stripplot(data=df_transitions, x="status", y=col, ax=ax,
                  color="black", alpha=0.3, size=4, legend=False)

    # stats
    h = df_transitions[df_transitions["status"] == "Hit"][col].dropna()
    m = df_transitions[df_transitions["status"] == "Miss"][col].dropna()
    if len(h) > 0 and len(m) > 0:
        _, p = mannwhitneyu(h, m, alternative="two-sided")
        sig = "*" if p < 0.05 else "NS"
        ax.set_title(f"{label}\nMWU p={p:.3f} ({sig})", fontsize=10)
    else:
        ax.set_title(label, fontsize=10)

    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle(f"Mouse {MOUSE_ID} — Exploration metrics", fontsize=13, y=1.02)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "exploration_metrics.png"),
             dpi=200, bbox_inches="tight")
fig2.savefig(os.path.join(OUTPUT_DIR, "exploration_metrics.pdf"),
             bbox_inches="tight")

# ── 5. ROI-only transitions (corridor removed) ─────────────────────
# Instead of frame-by-frame, we just look at which ROI the mouse goes
# to after leaving another ROI (skipping corridor entirely).

ROI_LABELS = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
ROI_SHORT = ["ent1", "ent2", "rewA", "rewB", "rewC", "rewD"]


def roi_only_sequence(collapsed_states):
    """Filter a collapsed state sequence to only named ROIs (drop corridor/unknown)."""
    return [s for s in collapsed_states if s in ROI_LABELS]


# rebuild matrices from raw trial data
hit_roi_matrices = []
miss_roi_matrices = []

for sess in sessions:
    df_trials = load_trials(sess.trial_csv)
    rois = load_rois(sess.roi_csv) if sess.roi_csv else {}
    tracking = load_dlc(sess.dlc_csv, bodypart=BODYPART,
                        likelihood_thresh=LIKELIHOOD_THRESH)
    if not rois:
        continue

    if "start_trial_frame" in df_trials.columns:
        fc_start, fc_end = "start_trial_frame", "end_trial_frame"
    elif "start_frame" in df_trials.columns:
        fc_start, fc_end = "start_frame", "end_frame"
    else:
        continue

    for idx, row in df_trials.iterrows():
        try:
            f_start, f_end = int(row[fc_start]), int(row[fc_end])
        except (ValueError, TypeError):
            continue
        if f_start >= f_end or (f_end - f_start) < 10:
            continue

        outcome = classify_trial(row)
        if outcome == "no_choice":
            continue
        status = "Hit" if outcome == "correct" else "Miss"

        trial_track = tracking.iloc[f_start:f_end]
        if len(trial_track) < 10:
            continue

        raw_states = compute_state_sequence(trial_track, rois, buffer=5)
        clean = [s for s in raw_states if s != "unknown"]
        if len(clean) < 5:
            continue

        collapsed = collapse_state_sequence(clean)
        roi_seq = roi_only_sequence(collapsed)

        if len(roi_seq) < 2:
            continue

        mat_raw, _ = transition_matrix(roi_seq, state_labels=ROI_LABELS,
                                        normalise=False)
        if status == "Hit":
            hit_roi_matrices.append(mat_raw)
        else:
            miss_roi_matrices.append(mat_raw)

avg_hit_roi = aggregate_matrices(hit_roi_matrices)
avg_miss_roi = aggregate_matrices(miss_roi_matrices)
diff_roi = avg_hit_roi - avg_miss_roi

fig3, axes3 = plt.subplots(1, 3, figsize=(20, 5.5))

sns.heatmap(avg_hit_roi, ax=axes3[0], annot=True, fmt=".2f", cmap="Greens",
            xticklabels=ROI_SHORT, yticklabels=ROI_SHORT,
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
axes3[0].set_title(f"Hit trials (n={len(hit_roi_matrices)})", fontsize=12)
axes3[0].set_xlabel("To")
axes3[0].set_ylabel("From")

sns.heatmap(avg_miss_roi, ax=axes3[1], annot=True, fmt=".2f", cmap="Reds",
            xticklabels=ROI_SHORT, yticklabels=ROI_SHORT,
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
axes3[1].set_title(f"Miss trials (n={len(miss_roi_matrices)})", fontsize=12)
axes3[1].set_xlabel("To")
axes3[1].set_ylabel("From")

max_abs_roi = max(np.abs(diff_roi).max(), 0.01)
sns.heatmap(diff_roi, ax=axes3[2], annot=True, fmt=".2f", cmap="RdBu_r",
            xticklabels=ROI_SHORT, yticklabels=ROI_SHORT,
            vmin=-max_abs_roi, vmax=max_abs_roi, center=0,
            cbar_kws={"shrink": 0.8})
axes3[2].set_title("Difference (Hit - Miss)", fontsize=12)
axes3[2].set_xlabel("To")
axes3[2].set_ylabel("From")

plt.suptitle(f"Mouse {MOUSE_ID} — ROI-to-ROI transitions (corridor removed)",
             fontsize=14, y=1.02)
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "transition_roi_only.png"),
             dpi=200, bbox_inches="tight")
fig3.savefig(os.path.join(OUTPUT_DIR, "transition_roi_only.pdf"),
             bbox_inches="tight")

print(f"\nAll outputs saved to {OUTPUT_DIR}")
plt.show()
