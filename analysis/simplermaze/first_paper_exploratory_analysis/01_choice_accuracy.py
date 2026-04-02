"""
01 — Choice accuracy across sessions.

For each session, recomputes trial outcomes from
first_reward_area_visited[-1] vs rew_location (sanity-checking the
hit/miss/incorrect columns which may have misdetections).

Outputs:
  - Grouped bar chart: correct / incorrect / no-choice per session
  - Binomial GLMM: correct ~ session + (1 | mouse_id)  [via rpy2 + lme4]
  - Summary CSV with per-session counts
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import (MOUSE_ID, MOUSE_DIR, discover_sessions)
from utils import load_trials, classify_trial, has_entered_any_arm


# ── 1. load & classify ──────────────────────────────────────────────

sessions = discover_sessions()
print(f"Found {len(sessions)} sessions for mouse {MOUSE_ID}")

rows = []
for sess in sessions:
    df = load_trials(sess.trial_csv)
    for _, trial in df.iterrows():
        outcome = classify_trial(trial)
        entered_arm = has_entered_any_arm(trial)
        rows.append({
            "mouse_id": MOUSE_ID,
            "session_id": sess.session_id,
            "rew_location": trial.get("rew_location", ""),
            "first_visited": trial.get("first_reward_area_visited", ""),
            "outcome": outcome,
            "entered_any_arm": entered_arm,
        })

all_trials = pd.DataFrame(rows)

# ── 2. summary counts ───────────────────────────────────────────────

# sort sessions in logical order (hab, 3.1, 3.2, ...)
def session_sort_key(sid):
    if sid == "hab":
        return (0, 0)
    return (1, float(sid))

session_order = sorted(all_trials["session_id"].unique(), key=session_sort_key)

summary = []
for sid in session_order:
    mask = all_trials["session_id"] == sid
    sub = all_trials[mask]
    n_total = len(sub)
    n_correct = (sub["outcome"] == "correct").sum()
    n_incorrect = (sub["outcome"] == "incorrect").sum()
    n_no_choice = (sub["outcome"] == "no_choice").sum()
    summary.append({
        "session": sid,
        "total_trials": n_total,
        "correct": n_correct,
        "incorrect": n_incorrect,
        "no_choice": n_no_choice,
        "pct_correct": round(100 * n_correct / n_total, 1) if n_total > 0 else 0,
    })

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# ── 3. plot (line plot) ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(session_order))

correct_vals = summary_df["correct"].values
incorrect_vals = summary_df["incorrect"].values
no_choice_vals = summary_df["no_choice"].values

ax.plot(x, correct_vals, "o-", color="#2CA02C", linewidth=2, markersize=7,
        label="Correct", zorder=3)
ax.plot(x, incorrect_vals, "s-", color="#D62728", linewidth=2, markersize=7,
        label="Incorrect", zorder=3)
ax.plot(x, no_choice_vals, "^--", color="#7F7F7F", linewidth=1.5, markersize=6,
        label="No choice", alpha=0.7, zorder=2)

ax.fill_between(x, correct_vals, alpha=0.08, color="#2CA02C")
ax.fill_between(x, incorrect_vals, alpha=0.08, color="#D62728")

ax.set_xlabel("Session", fontsize=12)
ax.set_ylabel("Number of trials", fontsize=12)
ax.set_title(f"Mouse {MOUSE_ID} — Choice accuracy across sessions", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(session_order)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

OUTPUT_DIR = os.path.join(MOUSE_DIR, f"MOUSE_{MOUSE_ID}_TOTAL_ANALYSIS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, "choice_accuracy_across_sessions.png"), dpi=200)
fig.savefig(os.path.join(OUTPUT_DIR, "choice_accuracy_across_sessions.pdf"))
summary_df.to_csv(os.path.join(OUTPUT_DIR, "choice_accuracy_summary.csv"), index=False)
print(f"\nSaved to {OUTPUT_DIR}")

# ── 4. binomial GLMM ────────────────────────────────────────────────
# correct ~ session + (1 | mouse_id)
# For a single mouse this is just a logistic regression, but the
# structure generalises when you add more mice.

# filter to trials where mouse made a choice (exclude no_choice)
choice_trials = all_trials[all_trials["outcome"] != "no_choice"].copy()
choice_trials["correct_int"] = (choice_trials["outcome"] == "correct").astype(int)
choice_trials["session_num"] = choice_trials["session_id"].apply(
    lambda s: 0.0 if s == "hab" else float(s)
)

print(f"\n--- Binomial GLMM ({len(choice_trials)} trials with a choice) ---")

HAS_RPY2 = False
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    HAS_RPY2 = True
except Exception:
    pass

if HAS_RPY2:
    try:
        lme4 = importr("lme4")
        base = importr("base")

        r_df = pandas2ri.py2rpy(choice_trials[["correct_int", "session_num", "mouse_id"]])
        ro.globalenv["df"] = r_df

        unique_mice = choice_trials["mouse_id"].nunique()

        if unique_mice > 1:
            ro.r("""
                library(lme4)
                model <- glmer(correct_int ~ session_num + (1 | mouse_id),
                               data = df, family = binomial)
                model_summary <- summary(model)
            """)
        else:
            ro.r("""
                model <- glm(correct_int ~ session_num,
                             data = df, family = binomial)
                model_summary <- summary(model)
            """)

        print(ro.r("model_summary"))
        coefs = ro.r("coef(model_summary)")
        coef_df = pandas2ri.rpy2py(coefs)
        print("\nModel coefficients:")
        print(coef_df)
    except Exception as e:
        print(f"R GLMM failed ({e}), falling back to statsmodels")
        HAS_RPY2 = False

if not HAS_RPY2:
    print("Using statsmodels logistic regression")
    import statsmodels.api as sm

    X = sm.add_constant(choice_trials["session_num"])
    y = choice_trials["correct_int"]
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    print(result.summary2())
