"""
Script 3: Statistical Analyses and Manuscript Figures
=====================================================
Reads `outputs/cln_summary_dat.csv` (N = 162) and runs:
  1. Subgroup Behavioral Task Comparisons (PRL Win-Switch, PRL Lose-Stay,
     Social JTC Advice Influence & Confidence across the 4 groups)
  2. Dimensional OLS Regressions (Table 2: predicting Composite_Psychosis,
     Composite_Conspiracy, and Combined_Severity from task metrics,
     working memory, demographics, Locus of Control, and SFQ)
  3. Social Functioning Double Dissociation Regression (predicting Score_SFQ
     from Composite_Psychosis and Composite_Conspiracy controlling for Age,
     Sex, and Working Memory)
  4. Manuscript Figures saved to `figures/`:
     - figures/ols_coefficients_forest.png
     - figures/sfq_double_dissociation_fit.png

Outputs:
  - outputs/Table_2_OLS_Regressions.csv
  - outputs/Table_3_SFQ_Double_Dissociation.csv
  - figures/ols_coefficients_forest.png
  - figures/sfq_double_dissociation_fit.png
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")

FEATURES_CSV = os.path.join(OUT_DIR, "cln_summary_dat.csv")
TABLE2_CSV = os.path.join(OUT_DIR, "Table_2_OLS_Regressions.csv")
TABLE3_CSV = os.path.join(OUT_DIR, "Table_3_SFQ_Double_Dissociation.csv")
FOREST_PNG = os.path.join(FIG_DIR, "ols_coefficients_forest.png")
SFQ_PNG = os.path.join(FIG_DIR, "sfq_double_dissociation_fit.png")


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return ""


def run_subgroup_comparisons(df: pd.DataFrame):
    print("=" * 80)
    print("1. SUBGROUP BEHAVIORAL TASK COMPARISONS")
    print("=" * 80)
    task_cols = [
        ("PRL Win-Switch Rate", "PRL_Win_Switch_Rate"),
        ("PRL Lose-Stay Rate", "PRL_Lose_Stay_Rate"),
        ("Social JTC Influence Rate", "Social_Influence_Rate"),
        ("Social JTC Mean Confidence", "Social_Task_Mean_Confidence"),
    ]
    for label, col in task_cols:
        grp_stats = df.groupby("Group")[col].agg(["mean", "std"])
        m_2x2 = smf.ols(f"{col} ~ High_Psychosis + High_Conspiracy", data=df).fit()
        print(f"\n{label} ({col}):")
        for g, row in grp_stats.iterrows():
            print(f"  {g:28s}: {row['mean']:.4f} (SD = {row['std']:.4f})")
        print(
            f"  Main effect Psychosis p = {m_2x2.pvalues['High_Psychosis']:.4f}, "
            f"Main effect Conspiracy p = {m_2x2.pvalues['High_Conspiracy']:.4f}"
        )


def run_dimensional_ols_models(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("2. DIMENSIONAL PREDICTIVE MODELING (TABLE 2 OLS REGRESSIONS)")
    print("=" * 80)

    df = df.copy()
    df["Combined_Severity"] = (df["Composite_Psychosis"] + df["Composite_Conspiracy"]) / 2.0
    df["Sex_Male"] = (df["Sex"] == "Male").astype(int)

    predictors = [
        "PRL_Win_Switch_Rate",
        "PRL_Lose_Stay_Rate",
        "Social_Influence_Rate",
        "Social_Task_Mean_Confidence",
        "DigitSpan_Forwards_Max",
        "Age",
        "Sex_Male",
        "Score_LocusOfControl",
        "Score_SFQ",
    ]
    rhs = " + ".join(predictors)

    models = {
        "Model 1: Psychosis": smf.ols(f"Composite_Psychosis ~ {rhs}", data=df).fit(),
        "Model 2: Conspiracy": smf.ols(f"Composite_Conspiracy ~ {rhs}", data=df).fit(),
        "Model 3: Combined": smf.ols(f"Combined_Severity ~ {rhs}", data=df).fit(),
    }

    pretty_names = {
        "PRL_Lose_Stay_Rate": "PRL Lose-Stay Rate",
        "PRL_Win_Switch_Rate": "PRL Win-Switch Rate",
        "Social_Task_Mean_Confidence": "Social JTC Confidence",
        "Score_LocusOfControl": "Locus of Control (Rotter)",
        "Score_SFQ": "Social Functioning (SFQ)",
        "Social_Influence_Rate": "Social JTC Influence",
        "DigitSpan_Forwards_Max": "Digit Span Forwards",
        "Age": "Age",
        "Sex_Male": "Sex (Male)",
    }

    table2_rows = []
    for col in [
        "PRL_Lose_Stay_Rate",
        "Social_Task_Mean_Confidence",
        "Score_LocusOfControl",
        "Score_SFQ",
        "Social_Influence_Rate",
        "PRL_Win_Switch_Rate",
        "DigitSpan_Forwards_Max",
        "Age",
        "Sex_Male",
    ]:
        row = {"Predictor": pretty_names[col]}
        for m_name, m_fit in models.items():
            beta = m_fit.params[col]
            pval = m_fit.pvalues[col]
            stars = sig_stars(pval)
            row[m_name] = f"{beta:+.3f} {stars} (p={pval:.3f})".strip()
        table2_rows.append(row)

    fit_row = {"Predictor": "Model Fit (R^2)"}
    for m_name, m_fit in models.items():
        fit_row[m_name] = f"{m_fit.rsquared * 100:.1f}% (p={m_fit.f_pvalue:.4f})"
    table2_rows.append(fit_row)

    table2_df = pd.DataFrame(table2_rows)
    table2_df.to_csv(TABLE2_CSV, index=False)
    print(table2_df.to_string(index=False))
    print(f"\nSaved Table 2 to: {TABLE2_CSV}")

    return models


def run_sfq_double_dissociation(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("3. DOUBLE DISSOCIATION ON SOCIAL FUNCTIONING (SFQ)")
    print("=" * 80)

    df = df.copy()
    df["Sex_Male"] = (df["Sex"] == "Male").astype(int)

    m_sfq = smf.ols(
        "Score_SFQ ~ Composite_Psychosis + Composite_Conspiracy + Age + Sex_Male + DigitSpan_Forwards_Max",
        data=df,
    ).fit()

    print(
        f"Model R^2 = {m_sfq.rsquared * 100:.2f}% "
        f"(Adjusted R^2 = {m_sfq.rsquared_adj * 100:.2f}%, F p-value = {m_sfq.f_pvalue:.6f})"
    )
    summary_df = pd.DataFrame(
        {
            "Term": m_sfq.params.index,
            "Beta": m_sfq.params.values,
            "SE": m_sfq.bse.values,
            "t": m_sfq.tvalues.values,
            "p_value": m_sfq.pvalues.values,
            "CI_2.5%": m_sfq.conf_int()[0].values,
            "CI_97.5%": m_sfq.conf_int()[1].values,
        }
    )
    summary_df.to_csv(TABLE3_CSV, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved Table 3 (SFQ Double Dissociation) to: {TABLE3_CSV}")
    return m_sfq


def plot_figures(df: pd.DataFrame, models: dict):
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------------------
    # Figure A: Forest Plot of OLS Regression Coefficients
    # -------------------------------------------------------------------------
    plot_order = [
        ("PRL_Win_Switch_Rate", "PRL Win-Switch Rate"),
        ("PRL_Lose_Stay_Rate", "PRL Lose-Stay Rate"),
        ("Social_Influence_Rate", "Social JTC Influence"),
        ("Social_Task_Mean_Confidence", "Social JTC Confidence"),
        ("DigitSpan_Forwards_Max", "Digit Span Forwards"),
        ("Age", "Age"),
        ("Sex_Male", "Sex (Male)"),
        ("Score_LocusOfControl", "Locus of Control"),
        ("Score_SFQ", "Social Functioning (SFQ)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    y_positions = np.arange(len(plot_order))[::-1]
    offsets = {
        "Model 1: Psychosis": 0.22,
        "Model 2: Conspiracy": 0.0,
        "Model 3: Combined": -0.22,
    }
    colors = {
        "Model 1: Psychosis": "#c44e52",
        "Model 2: Conspiracy": "#4c72b0",
        "Model 3: Combined": "#8172b3",
    }

    for m_name, m_fit in models.items():
        ci = m_fit.conf_int()
        betas = [m_fit.params[k] for k, _ in plot_order]
        err_low = [m_fit.params[k] - ci.loc[k, 0] for k, _ in plot_order]
        err_high = [ci.loc[k, 1] - m_fit.params[k] for k, _ in plot_order]
        ax.errorbar(
            betas,
            y_positions + offsets[m_name],
            xerr=[err_low, err_high],
            fmt="o",
            color=colors[m_name],
            label=m_name,
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )

    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in plot_order], fontsize=11)
    ax.set_xlabel("Standardized Regression Coefficient (beta) with 95% CI", fontsize=12)
    ax.set_title(
        "Predicting Clinical and Conspiracy Traits from Cognitive Task Metrics",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(title="Dependent Variable", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FOREST_PNG, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure B: Double Dissociation on Social Functioning (SFQ)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, dpi=300)

    sns.regplot(
        data=df,
        x="Composite_Psychosis",
        y="Score_SFQ",
        ax=axes[0],
        color="#8c1d20",
        scatter_kws={"alpha": 0.45, "s": 35, "color": "#c44e52"},
        line_kws={"linewidth": 2, "color": "#8c1d20"},
    )
    axes[0].set_title("Clinical Impairment Profile", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Composite Psychosis Severity (z-score)", fontsize=11)
    axes[0].set_ylabel("Social Functioning Score (SFQ)\n(Higher = More Impairment)", fontsize=11)

    sns.regplot(
        data=df,
        x="Composite_Conspiracy",
        y="Score_SFQ",
        ax=axes[1],
        color="#1d3d63",
        scatter_kws={"alpha": 0.45, "s": 35, "color": "#4c72b0"},
        line_kws={"linewidth": 2, "color": "#1d3d63"},
    )
    axes[1].set_title("Conspiracist Coping Profile", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Composite Conspiracy Score (z-score)", fontsize=11)
    axes[1].set_ylabel("")

    fig.suptitle(
        "Double Dissociation of Clinical vs. Conspiracy Traits predicting Social Functioning (SFQ)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(SFQ_PNG, dpi=300)
    plt.close(fig)
    print(f"\nSaved figures to:\n  - {FOREST_PNG}\n  - {SFQ_PNG}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    run_subgroup_comparisons(df)
    models = run_dimensional_ols_models(df)
    run_sfq_double_dissociation(df)
    plot_figures(df, models)
