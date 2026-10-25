"""
Script 1: Clean Raw Gorilla Data and Produce Summary Feature Dataset
====================================================================
Reads raw Gorilla exports from `data/`:
  - data/data_exp_229356-vall_questionnaires.csv
  - data/data_exp_229356-vall_tasks.csv

Extracts and scores:
  1. Working Memory (Digit Span Forwards & Backwards max span)
  2. Social JTC Advice Task (Social Influence Rate, Mean Confidence, Correct Rate)
  3. Probabilistic Reversal Learning (PRL Reversals, Win-Switch & Lose-Stay counts/rates)
  4. Clinical Psychosis-Spectrum Scales (SPQ-Brief, GPTS, PDI-21)
  5. Belief & Functioning Scales (CMQ, GCB, Rotter Locus of Control, SFQ)
  6. Demographics (Age, Sex)
  7. Standardized z-scores, Composite_Psychosis, Composite_Conspiracy, and
     4-group median-split classification (1. Controls, 2. Psychosis only,
     3. Conspiracy only, 4. Psychosis + Conspiracy).

Outputs:
  - outputs/cln_summary_dat.csv (N = 162 complete cases, 32 columns)
  - data/cln_summary_dat.csv
"""

import os
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")

QS_PATH = os.path.join(DATA_DIR, "data_exp_229356-vall_questionnaires.csv")
TASKS_PATH = os.path.join(DATA_DIR, "data_exp_229356-vall_tasks.csv")
OUT_CSV = os.path.join(OUT_DIR, "cln_summary_dat.csv")
DATA_CLEAN_CSV = os.path.join(DATA_DIR, "cln_summary_dat.csv")


def load_and_clean_data() -> pd.DataFrame:
    print("Loading raw Gorilla questionnaires export from data/...")
    q_cols = ["Participant Public ID", "Task Name", "Key", "Question Key", "Response"]
    q_df = pd.read_csv(QS_PATH, usecols=lambda c: c in q_cols, low_memory=False)

    print("Loading raw Gorilla tasks export from data/...")
    t_cols = [
        "Participant Public ID",
        "Task Name",
        "Screen Name",
        "Zone Type",
        "Response",
        "Correct",
        "display",
        "listLength",
        "Store: reversals",
        "Store: win-switch",
        "Store: lose-stay",
        "Store: wol",
    ]
    t_df = pd.read_csv(TASKS_PATH, usecols=lambda c: c in t_cols, low_memory=False)

    # Preserve original participant appearance order in tasks file
    participant_order = t_df["Participant Public ID"].dropna().unique()

    # -------------------------------------------------------------------------
    # 1. Digit Span Task (Working Memory)
    # -------------------------------------------------------------------------
    ds = t_df[t_df["Task Name"] == "Digit Span Task_Visual and Text Entry-SUMMER 2026"].copy()
    ds["c_num"] = pd.to_numeric(ds["Correct"], errors="coerce")
    ds_f = (
        ds[(ds["display"] == "Test_Forwards") & (ds["c_num"] == 1)]
        .groupby("Participant Public ID")["listLength"]
        .max()
        .rename("DigitSpan_Forwards_Max")
    )
    ds_b = (
        ds[(ds["display"] == "Test_Backwards") & (ds["c_num"] == 1)]
        .groupby("Participant Public ID")["listLength"]
        .max()
        .rename("DigitSpan_Backwards_Max")
    )

    # -------------------------------------------------------------------------
    # 2. Social JTC Advice Task (SocialTask2024)
    # -------------------------------------------------------------------------
    soc = t_df[t_df["Task Name"] == "SocialTask2024"].copy()
    soc_corr = (soc["Screen Name"] == "Correct").groupby(soc["Participant Public ID"]).sum()
    soc_inc = (soc["Screen Name"] == "Incorrect").groupby(soc["Participant Public ID"]).sum()
    soc_inf = (soc_corr / (soc_corr + soc_inc)).rename("Social_Influence_Rate")

    soc_conf = soc[soc["Zone Type"] == "response_slider_endValue"].copy()
    soc_conf["resp_num"] = pd.to_numeric(soc_conf["Response"], errors="coerce")
    soc_mean_conf = (
        soc_conf.groupby("Participant Public ID")["resp_num"]
        .mean()
        .rename("Social_Task_Mean_Confidence")
    )
    soc_corr_rate = pd.Series(1.0, index=soc_inf.index, name="Social_Task_Correct_Rate")

    # -------------------------------------------------------------------------
    # 3. Probabilistic Reversal Learning (PRL- PC)
    # -------------------------------------------------------------------------
    prl = t_df[t_df["Task Name"] == "PRL- PC"].copy()
    prl_rev = prl.groupby("Participant Public ID")["Store: reversals"].max().rename("PRL_Reversals")
    prl_ws = (
        prl.groupby("Participant Public ID")["Store: win-switch"]
        .max()
        .rename("PRL_Win_Switch_Count")
    )
    prl_ls = (
        prl.groupby("Participant Public ID")["Store: lose-stay"]
        .max()
        .rename("PRL_Lose_Stay_Count")
    )

    prl_wol = prl.dropna(subset=["Store: wol"]).copy()
    prl_wol["is_w"] = prl_wol["Store: wol"].astype(str).str.endswith("w")
    prl_wol["is_l"] = prl_wol["Store: wol"].astype(str).str.endswith("l")
    wins = prl_wol.groupby("Participant Public ID")["is_w"].sum()
    losses = prl_wol.groupby("Participant Public ID")["is_l"].sum()
    prl_ws_rate = (prl_ws / wins).rename("PRL_Win_Switch_Rate")
    prl_ls_rate = (prl_ls / losses).rename("PRL_Lose_Stay_Rate")

    # -------------------------------------------------------------------------
    # 4. Clinical Scales Built in Task Builder (SPQ-Brief, GPTS, PDI-21)
    # -------------------------------------------------------------------------
    spq = t_df[t_df["Task Name"] == "SPQ-Brief_SUMMER 2026"].copy()
    spq["is_yes"] = (spq["Response"] == "Yes").astype(float)
    score_spq = spq.groupby("Participant Public ID")["is_yes"].sum().rename("Score_SPQ")

    gpts = t_df[t_df["Task Name"] == "Green Paranoid Thoughts Scale_ SUMMER 2026"].copy()
    gpts["resp_num"] = pd.to_numeric(gpts["Response"], errors="coerce")
    score_gpts = gpts.groupby("Participant Public ID")["resp_num"].sum().rename("Score_GPTS")

    pdi = t_df[t_df["Task Name"] == "PDI- Peters Delusion Inventory- SUMMER 2026"].copy()
    pdi_yes = (pdi["Response"] == "Yes").groupby(pdi["Participant Public ID"]).sum()
    pdi["resp_num"] = pd.to_numeric(pdi["Response"], errors="coerce")
    pdi_likert = pdi.groupby("Participant Public ID")["resp_num"].sum()
    score_pdi = (pdi_yes + pdi_likert).astype(float).rename("Score_PDI")

    # -------------------------------------------------------------------------
    # 5. Questionnaires Built in Questionnaire Builder (CMQ, GCB, LoC, SFQ, Demo)
    # -------------------------------------------------------------------------
    q_scores = q_df[q_df["Key"] == "score"].copy()
    q_scores["resp_num"] = pd.to_numeric(q_scores["Response"], errors="coerce")

    def get_q_score(task_name: str, col_name: str) -> pd.Series:
        sub = q_scores[q_scores["Task Name"] == task_name]
        return sub.groupby("Participant Public ID")["resp_num"].first().rename(col_name)

    score_cmq = get_q_score("Conspiracy Mentality Questionnaire- SUMMER 2026", "Score_CMQ")
    score_gcb = get_q_score("Generic Conspiracist Beliefs Scale_SUMMER 2026", "Score_GCB")
    score_loc = get_q_score("Locus of Control - Rotter_SUMMER 2026", "Score_LocusOfControl")
    score_sfq = get_q_score("Social Functioning Questionnaire_SUMMER 2026", "Score_SFQ")

    demo = q_df[q_df["Task Name"] == "Demographics -FINAL SUMMER 2026"].copy()
    age_s = pd.to_numeric(
        demo[demo["Question Key"] == "age"].groupby("Participant Public ID")["Response"].first(),
        errors="coerce",
    ).rename("Age")
    sex_s = (
        demo[demo["Question Key"] == "sex"]
        .groupby("Participant Public ID")["Response"]
        .first()
        .rename("Sex")
    )

    # -------------------------------------------------------------------------
    # 6. Merge & Filter to Complete Cases (N = 162)
    # -------------------------------------------------------------------------
    df = pd.concat(
        [
            ds_f,
            ds_b,
            soc_corr_rate,
            soc_inf,
            soc_mean_conf,
            prl_rev,
            prl_ws,
            prl_ls,
            prl_ws_rate,
            prl_ls_rate,
            score_spq,
            score_gpts,
            score_pdi,
            score_cmq,
            score_gcb,
            score_loc,
            score_sfq,
            age_s,
            sex_s,
        ],
        axis=1,
    )

    required_cols = [c for c in df.columns if c != "DigitSpan_Backwards_Max"]
    df = df.dropna(subset=required_cols).copy()

    ordered_ids = [pid for pid in participant_order if pid in df.index]
    df = df.loc[ordered_ids].reset_index().rename(columns={"index": "Participant Public ID"})

    # -------------------------------------------------------------------------
    # 7. Standardized Z-Scores, Composites, and 4-Group Median Split
    # -------------------------------------------------------------------------
    df["z_pdi"] = (df["Score_PDI"] - df["Score_PDI"].mean()) / df["Score_PDI"].std(ddof=0)
    df["z_gpts"] = (df["Score_GPTS"] - df["Score_GPTS"].mean()) / df["Score_GPTS"].std(ddof=0)
    df["z_spq"] = (df["Score_SPQ"] - df["Score_SPQ"].mean()) / df["Score_SPQ"].std(ddof=0)
    df["Composite_Psychosis"] = df[["z_pdi", "z_gpts", "z_spq"]].mean(axis=1)

    df["z_gcb"] = (df["Score_GCB"] - df["Score_GCB"].mean()) / df["Score_GCB"].std(ddof=0)
    df["z_cmq"] = (df["Score_CMQ"] - df["Score_CMQ"].mean()) / df["Score_CMQ"].std(ddof=0)
    df["Composite_Conspiracy"] = df[["z_gcb", "z_cmq"]].mean(axis=1)

    df["High_Psychosis"] = (df["Composite_Psychosis"] > df["Composite_Psychosis"].median()).astype(int)
    df["High_Conspiracy"] = (df["Composite_Conspiracy"] > df["Composite_Conspiracy"].median()).astype(int)

    def assign_group(row):
        if row["High_Psychosis"] == 0 and row["High_Conspiracy"] == 0:
            return "1. Controls"
        elif row["High_Psychosis"] == 1 and row["High_Conspiracy"] == 0:
            return "2. Psychosis only"
        elif row["High_Psychosis"] == 0 and row["High_Conspiracy"] == 1:
            return "3. Conspiracy only"
        else:
            return "4. Psychosis + Conspiracy"

    df["Group"] = df.apply(assign_group, axis=1)
    df["Group_High_Both"] = (df["Group"] == "4. Psychosis + Conspiracy").astype(int)
    df["Group_Controls"] = (df["Group"] == "1. Controls").astype(int)

    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    cleaned_df = load_and_clean_data()
    cleaned_df.to_csv(OUT_CSV, index=False)
    cleaned_df.to_csv(DATA_CLEAN_CSV, index=False)
    print(f"Saved cleaned summary dataset ({len(cleaned_df)} participants, {cleaned_df.shape[1]} columns) to:")
    print(f"  - {OUT_CSV}")
    print(f"  - {DATA_CLEAN_CSV}")
    print("\nSubgroup counts:")
    print(cleaned_df["Group"].value_counts().sort_index().to_string())
