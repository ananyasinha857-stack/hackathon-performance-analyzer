
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

PALETTE = {
    "primary":   "#5B4FCF",   # us
    "accent":    "#1D9E75",   # teal winners
    "neutral":   "#888780",   # others
    "bg":        "#F8F8F6",
    "text":      "#2C2C2A",
    "warn":      "#D4537E"
}

CRITERIA = ["innovation", "technical_complexity", "presentation",
            "feasibility", "social_impact"]

WEIGHTS = {
    "innovation":            0.25,
    "technical_complexity":  0.25,
    "presentation":          0.20,
    "feasibility":           0.15,
    "social_impact":         0.15,
}

OUR_TEAM = "Commit Issues"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlepad":    12,
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "text.color":       PALETTE["text"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
})


def load_data(path: str = "data/scores.csv"):
    df = pd.read_csv(path)

    df["weighted_score"] = sum(
        df[c] * w for c, w in WEIGHTS.items()
    )

    df["judge_std"] = df[["judge_1", "judge_2", "judge_3"]].std(axis=1)

    df["criteria_avg"] = df[CRITERIA].mean(axis=1)

    return df


def our_team_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pull all rows for OUR_TEAM and compute performance stats."""
    team_df = df[df["team_name"] == OUR_TEAM].copy()
    team_df["gap_to_winner"] = team_df.groupby("hackathon")["weighted_score"].transform("max") - team_df["weighted_score"]
    return team_df[["hackathon", "rank", "weighted_score", "gap_to_winner"] + CRITERIA]


def criteria_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix between judging criteria."""
    return df[CRITERIA].corr()


def winning_team_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average criteria scores for rank-1 teams vs everyone else."""
    df = df.copy()
    df["group"] = df["rank"].apply(lambda r: "Top 3" if r <= 3 else "Rest")
    return df.groupby("group")[CRITERIA].mean()


def judge_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Which hackathons had the most/least judge disagreement?"""
    return (
        df.groupby("hackathon")["judge_std"]
        .agg(["mean", "max", "min"])
        .rename(columns={"mean": "avg_disagreement", "max": "worst_case", "min": "best_case"})
        .round(2)
    )


def criteria_variance(df: pd.DataFrame) -> pd.Series:
    """Which criteria had the highest spread across all teams?"""
    return df[CRITERIA].std().sort_values(ascending=False)


def color_bars(values, highlight_team=None, team_names=None):
    colors = []
    for i, v in enumerate(values):
        if team_names is not None and team_names[i] == OUR_TEAM:
            colors.append(PALETTE["primary"])
        elif i == 0:
            colors.append(PALETTE["accent"])
        else:
            colors.append(PALETTE["neutral"])
    return colors


def plot_leaderboard(df: pd.DataFrame, hackathon: str, ax: plt.Axes):
    """Horizontal bar chart of weighted scores for one hackathon."""
    sub = df[df["hackathon"] == hackathon].sort_values("weighted_score")
    teams = sub["team_name"].tolist()
    scores = sub["weighted_score"].tolist()

    bars = ax.barh(teams, scores, color=[
        PALETTE["primary"] if t == OUR_TEAM else
        PALETTE["accent"] if i == len(teams) - 1 else PALETTE["neutral"]
        for i, t in enumerate(teams)
    ], height=0.6)

    ax.set_xlim(5.5, 10)
    ax.set_xlabel("Weighted score")
    ax.set_title(f"{hackathon} — leaderboard")

    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=9)


def plot_radar(df: pd.DataFrame, hackathon: str, ax: plt.Axes):
    """Radar chart comparing your team vs hackathon average."""
    sub = df[df["hackathon"] == hackathon]
    your_row = sub[sub["team_name"] == OUR_TEAM][CRITERIA].values.flatten()
    avg_row  = sub[CRITERIA].mean().values

    N = len(CRITERIA)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    your_vals = your_row.tolist() + [your_row[0]]
    avg_vals  = avg_row.tolist()  + [avg_row[0]]

    ax.set_facecolor(PALETTE["bg"])
    ax.plot(angles, your_vals, color=PALETTE["primary"], linewidth=2, label=OUR_TEAM)
    ax.fill(angles, your_vals, color=PALETTE["primary"], alpha=0.15)
    ax.plot(angles, avg_vals, color=PALETTE["neutral"], linewidth=1.5,
            linestyle="--", label="Hackathon avg")
    ax.fill(angles, avg_vals, color=PALETTE["neutral"], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in CRITERIA], fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7,
                       color=PALETTE["neutral"])
    ax.set_title(f"{hackathon} — your profile vs avg", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)


def plot_criteria_heatmap(df: pd.DataFrame, ax: plt.Axes):
    """Heatmap of all teams × criteria scores."""
    pivot = df.set_index("team_name")[CRITERIA]
    sns.heatmap(
        pivot, ax=ax, cmap="YlGnBu", annot=True, fmt=".1f",
        linewidths=0.4, linecolor="#ddd",
        cbar_kws={"shrink": 0.7, "label": "Score"},
        annot_kws={"size": 8}
    )
    ax.set_title("All teams — criteria scores heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([c.replace("_", "\n") for c in CRITERIA], fontsize=9)


def plot_your_journey(df: pd.DataFrame, ax: plt.Axes):
    """Line chart of your team's criteria scores across hackathons."""
    your = df[df["team_name"] == OUR_TEAM].set_index("hackathon")[CRITERIA]
    hackathon_order = ["SIH 25", "Inspiron 26", "HackGenX 26"]
    your = your.reindex([h for h in hackathon_order if h in your.index])

    colors_line = [PALETTE["primary"], PALETTE["accent"], PALETTE["warn"],
                   "#D4537E", "#378ADD"]

    for i, col in enumerate(CRITERIA):
        ax.plot(your.index, your[col], marker="o", label=col.replace("_", " "),
                color=colors_line[i], linewidth=2, markersize=6)

    ax.set_ylim(7, 10.2)
    ax.set_ylabel("Score")
    ax.set_title(f"{OUR_TEAM} — performance journey across hackathons")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.3)


def plot_criteria_variance(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: which criteria had the most spread (hardest to predict)?"""
    var = criteria_variance(df).reset_index()
    var.columns = ["criteria", "std"]
    colors = [PALETTE["warn"] if v == var["std"].max() else PALETTE["neutral"]
              for v in var["std"]]

    bars = ax.bar(var["criteria"].str.replace("_", "\n"), var["std"],
                  color=colors, width=0.5)
    ax.set_ylabel("Std deviation")
    ax.set_title("Criteria variance — which was hardest to predict?")

    for bar, val in zip(bars, var["std"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", fontsize=9)


def plot_judge_consistency(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart of avg judge disagreement per hackathon."""
    cons = judge_consistency(df).reset_index()
    colors = [PALETTE["primary"] if h in ["HackGenX 26", "SIH 25", "Inspiron 26"]
              else PALETTE["neutral"] for h in cons["hackathon"]]

    bars = ax.bar(cons["hackathon"], cons["avg_disagreement"],
                  color=colors, width=0.4)
    ax.set_ylabel("Avg std across judges")
    ax.set_title("Judge consistency per hackathon\n(lower = judges agreed more)")

    for bar, val in zip(bars, cons["avg_disagreement"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.2f}", ha="center", fontsize=9)


def plot_top3_vs_rest(df: pd.DataFrame, ax: plt.Axes):
    """Grouped bar chart: Top 3 vs rest across criteria."""
    profile = winning_team_profile(df).T.reset_index()
    profile.columns = ["criteria", "Rest", "Top 3"]

    x = np.arange(len(profile))
    w = 0.35
    ax.bar(x - w/2, profile["Top 3"], width=w, label="Top 3",
           color=PALETTE["accent"], alpha=0.9)
    ax.bar(x + w/2, profile["Rest"],  width=w, label="Rest",
           color=PALETTE["neutral"], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(profile["criteria"].str.replace("_", "\n"), fontsize=9)
    ax.set_ylim(6.5, 10)
    ax.set_ylabel("Avg score")
    ax.set_title("Top 3 vs rest — where winners pull ahead")
    ax.legend(fontsize=9)


# ── Main report generator ─────────────────────────────────────────────────────

def generate_report(data_path: str = "data/scores.csv",
                    output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(data_path)

    print("=" * 55)
    print("  HACKATHON PERFORMANCE ANALYZER")
    print(f"  Team: {OUR_TEAM}")
    print("=" * 55)

    # ── Print summary table ────────────────────────────────────
    summary = our_team_summary(df)
    print("\n OUR TEAM SUMMARY\n")
    print(summary.to_string(index=False))

    # ── Print insights ─────────────────────────────────────────
    print("\n\n KEY INSIGHTS\n")

    best_hackathon = summary.loc[summary["weighted_score"].idxmax(), "hackathon"]
    best_score     = summary["weighted_score"].max()
    print(f"  → Best weighted score: {best_score:.2f} at {best_hackathon}")

    best_criteria = summary[CRITERIA].mean().idxmax().replace("_", " ")
    weak_criteria = summary[CRITERIA].mean().idxmin().replace("_", " ")
    print(f"  → Consistently strongest criteria: {best_criteria}")
    print(f"  → Consistently weakest criteria:   {weak_criteria}")

    print("\n")

    # ── Fig 1: Per-hackathon overview (leaderboard + radar × 3) ───────────────
    hackathons = df["hackathon"].unique()
    fig1, axes1 = plt.subplots(len(hackathons), 2,
                               figsize=(14, 5 * len(hackathons)))
    fig1.suptitle("Hackathon Performance Analyzer — Per-event breakdown",
                  fontsize=15, fontweight="bold", y=1.01)

    for i, h in enumerate(hackathons):
        plot_leaderboard(df, h, axes1[i][0])
        plot_radar(df[df["hackathon"] == h], h,
                   plt.subplot(len(hackathons), 2, i * 2 + 2,
                               projection="polar"))
        axes1[i][1].remove() if not hasattr(axes1[i][1], 'remove') else None

    # rebuild with polar axes properly
    fig1.clear()
    for i, h in enumerate(hackathons):
        ax_bar = fig1.add_subplot(len(hackathons), 2, i * 2 + 1)
        ax_pol = fig1.add_subplot(len(hackathons), 2, i * 2 + 2,
                                  projection="polar")
        plot_leaderboard(df, h, ax_bar)
        plot_radar(df[df["hackathon"] == h], h, ax_pol)

    fig1.tight_layout(pad=2.5)
    path1 = os.path.join(output_dir, "1_per_event_breakdown.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight",
                 facecolor=PALETTE["bg"])
    plt.close(fig1)
    print(f"  ✓ Saved: {path1}")

    # ── Fig 2: Your team journey + criteria heatmap ───────────────────────────
    fig2, (ax_j, ax_h) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Deep dive — your team & cross-hackathon patterns",
                  fontsize=14, fontweight="bold")
    plot_your_journey(df, ax_j)
    plot_criteria_heatmap(df, ax_h)
    fig2.tight_layout(pad=2.5)
    path2 = os.path.join(output_dir, "2_journey_and_heatmap.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight",
                 facecolor=PALETTE["bg"])
    plt.close(fig2)
    print(f"  ✓ Saved: {path2}")

    # ── Fig 3: Structural insights ────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("Structural insights — what drives winning?",
                  fontsize=14, fontweight="bold")
    plot_criteria_variance(df, axes3[0])
    plot_judge_consistency(df, axes3[1])
    plot_top3_vs_rest(df, axes3[2])
    fig3.tight_layout(pad=2.5)
    path3 = os.path.join(output_dir, "3_structural_insights.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight",
                 facecolor=PALETTE["bg"])
    plt.close(fig3)
    print(f"  ✓ Saved: {path3}")

    print("\n✅ All charts saved to", output_dir)
    print("=" * 55)


if __name__ == "__main__":
    generate_report() 