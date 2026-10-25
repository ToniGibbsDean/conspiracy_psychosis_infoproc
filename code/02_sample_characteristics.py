"""
Script 2: Generate Sample Characteristics Table (Table 1)
=========================================================
Reads the cleaned dataset `outputs/cln_summary_dat.csv` (N = 162)
along with supplementary demographic fields (education, ethnicity, psychiatric
history) from `data/data_exp_229356-vall_questionnaires.csv`.

Produces:
  - outputs/Table_1_Sample_Characteristics.csv
    (Formatted Table 1 across Overall and the 4 Psychosis x Conspiracy groups,
     with one-way ANOVA / Chi-square group comparison p-values)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")

FEATURES_CSV = os.path.join(OUT_DIR, "cln_summary_dat.csv")
QS_PATH = os.path.join(DATA_DIR, "data_exp_229356-vall_questionnaires.csv")
OUT_TABLE1_CSV = os.path.join(OUT_DIR, "Table_1_Sample_Characteristics.csv")

GROUPS = [
    "1. Controls",
    "2. Psychosis only",
    "3. Conspiracy only",
    "4. Psychosis + Conspiracy",
]


def attach_supplementary_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Higher Education, White/Caucasian, and Psychiatric Dx indicators from Gorilla Demographics."""
    q_cols = ["Participant Public ID", "Task Name", "Question Key", "Response"]
    q_df = pd.read_csv(QS_PATH, usecols=lambda c: c in q_cols, low_memory=False)
    demo = q_df[
        (q_df["Task Name"] == "Demographics -FINAL SUMMER 2026")
        & (q_df["Participant Public ID"].isin(df["Participant Public ID"]))
    ].copy()

    edu = (
        demo[demo["Question Key"] == "education"]
        .groupby("Participant Public ID")["Response"]
        .first()
    )
    eth_white = (
        demo[demo["Question Key"] == "ethnicity-7"]
        .groupby("Participant Public ID")["Response"]
        .first()
    )
    psych_dx = (
        demo[demo["Question Key"] == "psychiatric diagnosis "]
        .groupby("Participant Public ID")["Response"]
        .first()
    )

    df = df.copy()
    df["Higher_Ed"] = df["Participant Public ID"].map(edu).isin(
        ["College", "University", "Graduate school", "Professional school", "Technical school"]
    ).astype(int)
    df["White_Caucasian"] = df["Participant Public ID"].map(eth_white).notna().astype(int)
    df["Psych_Dx"] = (
        df["Participant Public ID"]
        .map(psych_dx)
        .astype(str)
        .str.startswith("Yes")
        .astype(int)
    )
    df["Sex_Female"] = (df["Sex"] == "Female").astype(int)
    return df


def format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.3f} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f}"


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_continuous_row(label: str, col: str):
        overall_str = f"{df[col].mean():.1f} ({df[col].std():.1f})"
        grp_strs = []
        grp_arrays = []
        for g in GROUPS:
            sub = df[df["Group"] == g][col].dropna()
            grp_strs.append(f"{sub.mean():.1f} ({sub.std():.1f})")
            grp_arrays.append(sub.values)
        _, p_val = stats.f_oneway(*grp_arrays)
        rows.append([label, overall_str] + grp_strs + [format_p(p_val)])

    def add_binary_row(label: str, col: str):
        overall_str = f"{df[col].mean() * 100:.1f}%"
        grp_strs = []
        contingency = []
        for g in GROUPS:
            sub = df[df["Group"] == g][col].dropna()
            grp_strs.append(f"{sub.mean() * 100:.1f}%")
            pos = int((sub == 1).sum())
            neg = int((sub == 0).sum())
            contingency.append([pos, neg])
        _, p_val, _, _ = stats.chi2_contingency(np.array(contingency))
        rows.append([label, overall_str] + grp_strs + [format_p(p_val)])

    # Demographics
    add_continuous_row("Age (years) [Mean (SD)]", "Age")
    add_binary_row("Sex (% Female)", "Sex_Female")
    add_binary_row("Higher Ed (%)", "Higher_Ed")
    add_binary_row("White/Caucasian (%)", "White_Caucasian")
    add_binary_row("Psychiatric Dx (%)", "Psych_Dx")

    # Clinical & Belief Questionnaires
    add_continuous_row("SPQ (Schizotypy)", "Score_SPQ")
    add_continuous_row("GPTS (Paranoia)", "Score_GPTS")
    add_continuous_row("PDI (Delusions)", "Score_PDI")
    add_continuous_row("CMQ (Conspiracy Mentality)", "Score_CMQ")
    add_continuous_row("GCB (Conspiracy Beliefs)", "Score_GCB")
    add_continuous_row("Locus of Control", "Score_LocusOfControl")
    add_continuous_row("SFQ (Social Functioning)", "Score_SFQ")

    # Working Memory
    add_continuous_row("Digit Span Forwards Max", "DigitSpan_Forwards_Max")
    add_continuous_row("Digit Span Backwards Max", "DigitSpan_Backwards_Max")

    columns = [
        "Metric",
        f"Overall (N={len(df)})",
        f"Controls (N={(df['Group'] == GROUPS[0]).sum()})",
        f"Psychosis-like Only (N={(df['Group'] == GROUPS[1]).sum()})",
        f"Conspiracy Only (N={(df['Group'] == GROUPS[2]).sum()})",
        f"Psychosis + Conspiracy (N={(df['Group'] == GROUPS[3]).sum()})",
        "Group Comparison",
    ]
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df = attach_supplementary_demographics(df)
    table1_df = build_table1(df)

    table1_df.to_csv(OUT_TABLE1_CSV, index=False)

    print("Sample Characteristics Table (Table 1):")
    print(table1_df.to_string(index=False))
    print(f"\nExported to:\n  - {OUT_TABLE1_CSV}")
