import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.stats import mannwhitneyu
from analysisfunc_config import Paths

# 1. SETUP & LOAD
OUTPUT_DIR = os.path.join(Paths.session_path, "master_analysis")
CSV_PATH = os.path.join(OUTPUT_DIR, "master_behavioral_data.csv")

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found. Run the master CSV script first.")
    exit()

df = pd.read_csv(CSV_PATH)

# Set global style for static plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# --- 2. THE "QUICK LOOK" (Seaborn FacetGrid) ---
# This shows Speed vs Duration distributions for Hits and Misses
def plot_speed_duration_facets(df):
    g = sns.JointGrid(data=df, x="p1_duration_s", y="p1_mean_speed_cm_s", hue="status")
    g.plot_joint(sns.scatterplot, s=100, alpha=.7)
    g.plot_marginals(sns.kdeplot, fill=True)
    g.fig.suptitle("Phase 1: Speed vs Duration (Search Strategy)", y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, "p1_joint_plot.png"), bbox_inches='tight')
    plt.close()

# --- 3. CORRELATION HEATMAP (Matplotlib/Seaborn) ---
# See which metrics are actually linked (e.g., does high entropy mean low speed?)
def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Metric Correlation Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), bbox_inches='tight')
    plt.close()

# --- 4. THE INTERACTIVE MULTI-CHART (Plotly) ---
# Combining Speed, Duration, and Entropy in one interactive file
def plot_interactive_summary(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "P1 Duration: Hit vs Miss", 
            "P1 Speed vs Entropy", 
            "P2 Duration: Hit vs Miss", 
            "P1 Speed vs P2 Speed"
        )
    )

    # 4a. P1 Duration Box Plot
    for status in ['Hit', 'Miss']:
        sub = df[df['status'] == status]
        fig.add_trace(go.Box(y=sub['p1_duration_s'], name=status, showlegend=False), row=1, col=1)

    # 4b. Speed vs Entropy Scatter (Are 'messy' paths slower?)
    fig.add_trace(
        go.Scatter(
            x=df['p1_entropy'], y=df['p1_mean_speed_cm_s'], 
            mode='markers', marker=dict(color=df['status'].map({'Hit': 'green', 'Miss': 'red'})),
            text=df['trial_id'], name="Speed vs Entropy"
        ), row=1, col=2
    )

    # 4c. P2 Duration Box Plot
    for status in ['Hit', 'Miss']:
        sub = df[df['status'] == status]
        fig.add_trace(go.Box(y=sub['p2_duration_s'], name=status, showlegend=False), row=2, col=1)

    # 4d. Search Speed vs Reward Speed (Consistency Check)
    fig.add_trace(
        go.Scatter(
            x=df['p1_mean_speed_cm_s'], y=df['p2_mean_speed_cm_s'], 
            mode='markers', marker=dict(symbol='diamond'), name="P1 vs P2 Speed"
        ), row=2, col=2
    )

    fig.update_layout(height=800, width=1000, title_text="Master Trial Analysis Summary", template="plotly_white")
    fig.write_html(os.path.join(OUTPUT_DIR, "interactive_master_summary.html"))

# --- 5. THE PHASE TRANSITION RADAR (Plotly) ---
# Compares the "Average Trial" profile for Hits vs Misses
def plot_radar_comparison(df):
    categories = ['p1_duration_s', 'p1_mean_speed_cm_s', 'p1_entropy', 'p2_duration_s', 'p2_mean_speed_cm_s']
    
    # Normalize data for radar (0 to 1 scale)
    df_norm = df[categories].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_norm['status'] = df['status']
    avg_stats = df_norm.groupby('status').mean().reset_index()

    fig = go.Figure()
    for i, row in avg_stats.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values,
            theta=categories,
            fill='toself',
            name=row['status']
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, title="Behavioral Profile: Hits vs Misses (Normalized)"
    )
    fig.write_html(os.path.join(OUTPUT_DIR, "behavioral_radar_profile.html"))

# EXECUTE ALL
print("Generating figures...")
plot_speed_duration_facets(df)
plot_correlation_matrix(df)
plot_interactive_summary(df)
plot_radar_comparison(df)
print(f"All plots saved in {OUTPUT_DIR}")



#---------------------------------------------------------
# Metrics to test
metrics = [
    'p1_duration_s', 'p1_mean_speed_cm_s', 'p1_entropy',
    'p2_duration_s', 'p2_mean_speed_cm_s', 'p2_entropy'
]

# 2. STATISTICAL TESTING (Mann-Whitney U)
def run_stats(df, metrics):
    stats_results = []
    hits = df[df['status'] == 'Hit']
    misses = df[df['status'] == 'Miss']
    
    print("\n--- Statistical Comparison: Hit vs Miss ---")
    for m in metrics:
        # Run test
        stat, p = mannwhitneyu(hits[m], misses[m], alternative='two-sided')
        
        # Determine significance level
        sig = "NS"
        if p < 0.001: sig = "***"
        elif p < 0.01: sig = "**"
        elif p < 0.05: sig = "*"
        
        stats_results.append({'Metric': m, 'p-value': p, 'Significance': sig})
        print(f"{m:20} | p = {p:.4f} ({sig})")
    
    # Save stats to CSV
    pd.DataFrame(stats_results).to_csv(os.path.join(OUTPUT_DIR, "statistical_results.csv"), index=False)
    return stats_results

# 3. SIGNIFICANCE-ANNOTATED BOXPLOTS
def plot_annotated_boxplots(df, metrics, stats):
    # Create a grid of plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        sns.boxplot(data=df, x='status', y=m, ax=axes[i], palette={'Hit': 'green', 'Miss': 'red'}, order=['Hit', 'Miss'])
        sns.stripplot(data=df, x='status', y=m, ax=axes[i], color='black', alpha=0.3)
        
        # Add p-value annotation from stats list
        p_val = next(item['p-value'] for item in stats if item['Metric'] == m)
        sig = next(item['Significance'] for item in stats if item['Metric'] == m)
        
        axes[i].set_title(f"{m}\n(p={p_val:.4f} {sig})")
        axes[i].set_xlabel("")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "significant_metrics_comparison.png"))
    plt.show()

# EXECUTE
stats_list = run_stats(df, metrics)
plot_annotated_boxplots(df, metrics, stats_list)