# What a session saves

A session writes everything it produces into one folder, including a manifest that records the configuration used and the units of every column.

Sessions are saved under:

```
base_output_path / experiment_mode / [--day label] / time_<timestamp>_<mouseID> /
```

Examples:

```
maze_recordings/grammar/habituation/time_2026-05-23_10_00_00_mouse1/
maze_recordings/grammar/day_1/time_2026-05-23_14_30_00_mouse1/
maze_recordings/grammar/day_2/time_2026-05-24_10_00_00_mouse1/
```

Figures are generated **automatically at the end of every session** and saved inside the session folder alongside the CSVs. To regenerate them for sessions already collected:

```bash
# Single session
amaze-analyse-session "C:\path\to\session_folder"

# Multiple sessions at once
amaze-analyse-session "C:\path\to\session1" "C:\path\to\session2"
```

| File | What it shows |
|------|--------------|
| `fig1_arm_totals.png` | Total time (min) and visit count per arm, coloured by stimulus type |
| `fig2_ee_vs_sc.png` | EE vs SC arms grouped by predictability tier - time and visits |
| `fig3_block_evolution.png` | Time per stimulus category across active blocks - checks whether preference shifts |
| `fig4_visit_duration.png` | Boxplot of individual visit durations per stimulus type |
| `fig5_maze_time.png` | Total time inside the maze per trial block |
| `fig6_location_preference.png` | Heatmap of time per arm per block with stimulus labels - distinguishes location bias from stimulus preference |

After running all mice for a day (or across multiple days), generate summary figures with `run_summary_analysis.py`. Saved into the folder you pass.

```bash
# All mice on one day
amaze-summary --day "C:\...\maze_recordings\grammar\day_1"

# All mice across all days collected so far
amaze-summary --all "C:\...\maze_recordings\grammar"
```

Silent-baseline sessions are automatically excluded - only active test-day sessions contribute.

**EE vs SC preference:**

| File | What it shows |
|------|--------------|
| `summary_A_ee_sc_per_mouse.png` | EE vs SC total time - one pair of bars per mouse, one panel per day |
| `summary_B_preference_index.png` | EE preference index (−1 to +1) per mouse; positive = EE preference |
| `summary_C_group_summary.png` | Group mean ± SEM time and visit count on EE vs SC arms per day |
| `summary_D_cross_day_pi.png` | PI trajectory per mouse + group mean ± SEM across days *(multi-day only)* |

**Predictive complexity (dominant / secondary / rare):**

| File | What it shows |
|------|--------------|
| `summary_E_tier_breakdown_per_mouse.png` | Stacked bars per mouse: EE bar and SC bar each split by tier (dark → light = dominant → rare) |
| `summary_F_group_tier_breakdown.png` | Group mean ± SEM for all 6 tier × environment combinations |
| `summary_G_cross_day_tiers.png` | Per-tier preference across days - group mean ± SEM for each complexity level, EE and SC panels *(multi-day only)* |

---
