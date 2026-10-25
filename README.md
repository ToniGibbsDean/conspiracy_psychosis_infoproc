# Conspiracy Mentality & Psychosis-Spectrum Information Processing (`conspiracy_psychosis_infoproc`)

This repository contains the raw Gorilla data, data-cleaning pipeline, statistical analysis scripts, summary tables, and figures for the study examining the dimensional and subgroup boundaries between **clinical psychosis-like traits** (schizotypy, paranoia, delusion proneness) and **conspiracy beliefs** across belief-updating and social-learning paradigms ($N = 162$ complete cases).

---

## 1. Directory Structure

```text
conspiracy_psychosis_infoproc/
├── README.md                                     # Project documentation & data dictionary
├── code/
│   ├── 01_clean_data.py                          # Step 1: Raw Gorilla CSVs -> cln_summary_dat.csv
│   ├── 02_sample_characteristics.py              # Step 2: cln_summary_dat.csv -> Table 1 (Demographics & Scales)
│   └── 03_run_analyses.py                        # Step 3: cln_summary_dat.csv -> Regressions (Tables 2-3) & Figures
├── data/
│   ├── data_exp_229356-vall_questionnaires.csv   # Raw Gorilla Questionnaire Builder export (long format)
│   ├── data_exp_229356-vall_tasks.csv            # Raw Gorilla Task Builder export (long format)
│   └── cln_summary_dat.csv                       # Cleaned participant-level summary dataset (N=162, 32 cols)
├── outputs/
│   ├── cln_summary_dat.csv                       # Cleaned participant-level summary dataset (N=162, 32 cols)
│   ├── Table_1_Sample_Characteristics.csv        # Sample characteristics across Overall & 4 subgroups
│   ├── Table_2_OLS_Regressions.csv               # Dimensional OLS models predicting Psychosis, Conspiracy, Combined
│   └── Table_3_SFQ_Double_Dissociation.csv       # OLS regression testing double dissociation on Social Functioning
└── figures/
    ├── ols_coefficients_forest.png               # Forest plot of regression betas & 95% CIs (Models 1–3)
    └── sfq_double_dissociation_fit.png           # Two-panel regression fit: Clinical Impairment vs. Conspiracist Coping
```

---

## 2. Analysis Pipeline (`code/`)

Run the scripts in numerical order from the project's virtual environment:

```bash
/Users/tg625/Documents/PDA/Directory/AgenticProjects/RedCap/venv/bin/python3 code/01_clean_data.py
/Users/tg625/Documents/PDA/Directory/AgenticProjects/RedCap/venv/bin/python3 code/02_sample_characteristics.py
/Users/tg625/Documents/PDA/Directory/AgenticProjects/RedCap/venv/bin/python3 code/03_run_analyses.py
```

### Script 1: `code/01_clean_data.py`
- **Reads:** `data/data_exp_229356-vall_questionnaires.csv` and `data/data_exp_229356-vall_tasks.csv`.
- **What it does:**
  1. Scores the three behavioral tasks:
     - **Digit Span (`Digit Span Task_Visual and Text Entry-SUMMER 2026`):** Maximum list length answered correctly on Forwards and Backwards blocks.
     - **Social JTC Advice Task (`SocialTask2024`):** Social Influence Rate (proportion of trials aligning with advisor feedback), Mean Decision Confidence (0–100 slider at end of trial), and task completion/correctness check.
     - **Probabilistic Reversal Learning (`PRL- PC`):** Total reversals (`Store: reversals`), Win-Switch count & rate (`Store: win-switch` divided by total win trials in `Store: wol`), and Lose-Stay count & rate (`Store: lose-stay` divided by total loss trials in `Store: wol`).
  2. Scores the clinical and belief questionnaires:
     - *Note on Gorilla structure:* `SPQ-Brief_SUMMER 2026`, `Green Paranoid Thoughts Scale_ SUMMER 2026` (GPTS), and `PDI- Peters Delusion Inventory- SUMMER 2026` (PDI-21) were built as Gorilla **Task Builder** nodes and are extracted from `data_exp_229356-vall_tasks.csv`.
     - `Conspiracy Mentality Questionnaire- SUMMER 2026` (CMQ), `Generic Conspiracist Beliefs Scale_SUMMER 2026` (GCB), `Locus of Control - Rotter_SUMMER 2026`, `Social Functioning Questionnaire_SUMMER 2026` (SFQ), and `Demographics -FINAL SUMMER 2026` were built as **Questionnaire Builder** nodes and are extracted from `data_exp_229356-vall_questionnaires.csv`.
  3. Filters to complete cases ($N = 162$), standardizes scale scores ($z$-scores), computes `Composite_Psychosis` (mean of `z_pdi`, `z_gpts`, `z_spq`) and `Composite_Conspiracy` (mean of `z_gcb`, `z_cmq`), and classifies participants into 4 subgroups via median splits.
- **Writes:** `outputs/cln_summary_dat.csv` (and a copy to `data/cln_summary_dat.csv`).

### Script 2: `code/02_sample_characteristics.py`
- **Reads:** `outputs/cln_summary_dat.csv` and supplementary demographic items (`education`, `ethnicity-7`, `psychiatric diagnosis`) from `data/data_exp_229356-vall_questionnaires.csv`.
- **What it does:** Generates **Table 1** comparing demographics, clinical/conspiracy scales, and working memory across the Overall sample ($N=162$) and the 4 subgroups (`1. Controls`, `2. Psychosis only`, `3. Conspiracy only`, `4. Psychosis + Conspiracy`), computing one-way ANOVA $p$-values for continuous variables and $\chi^2$ test $p$-values for categorical variables.
- **Writes:** `outputs/Table_1_Sample_Characteristics.csv`.

### Script 3: `code/03_run_analyses.py`
- **Reads:** `outputs/cln_summary_dat.csv`.
- **What it does:**
  1. **Subgroup Behavioral Comparisons:** Evaluates group means and $2 \times 2$ main effects (Psychosis $\times$ Conspiracy) on PRL Win-Switch Rate, PRL Lose-Stay Rate, Social Influence Rate, and Social JTC Confidence.
  2. **Dimensional Predictive Modeling (Table 2):** Runs three ordinary least squares (OLS) regressions predicting `Composite_Psychosis` (Model 1), `Composite_Conspiracy` (Model 2), and `Combined_Severity` (Model 3) from task metrics (`PRL_Win_Switch_Rate`, `PRL_Lose_Stay_Rate`, `Social_Influence_Rate`, `Social_Task_Mean_Confidence`), working memory (`DigitSpan_Forwards_Max`), demographics (`Age`, `Sex_Male`), `Score_LocusOfControl`, and `Score_SFQ`.
  3. **Social Functioning Double Dissociation (Table 3):** Runs an OLS regression predicting `Score_SFQ` (where higher scores = greater social impairment) simultaneously from `Composite_Psychosis` ($\beta = +1.089, p < 0.001$) and `Composite_Conspiracy` ($\beta = -0.423, p = 0.043$), adjusting for Age, Sex, and Working Memory.
  4. **Generates Figures:** Saves high-resolution (300 DPI) manuscript plots to `figures/`.
- **Writes:**
  - `outputs/Table_2_OLS_Regressions.csv`
  - `outputs/Table_3_SFQ_Double_Dissociation.csv`
  - `figures/ols_coefficients_forest.png`
  - `figures/sfq_double_dissociation_fit.png`

---

## 3. Data Dictionary for `cln_summary_dat.csv` ($N = 162$, 32 Columns)

| Column Name | Description |
| :--- | :--- |
| `Participant Public ID` | Unique 24-character participant identifier (Prolific/Gorilla ID) |
| `DigitSpan_Forwards_Max` | Maximum list length recalled correctly on Digit Span Forwards |
| `DigitSpan_Backwards_Max` | Maximum list length recalled correctly on Digit Span Backwards |
| `Social_Task_Correct_Rate` | Task completion / validity rate on `SocialTask2024` |
| `Social_Influence_Rate` | Proportion of trials on `SocialTask2024` aligning with social advice |
| `Social_Task_Mean_Confidence` | Mean end-of-trial confidence slider rating (0–100) on `SocialTask2024` |
| `PRL_Reversals` | Total reversals achieved on the Probabilistic Reversal Learning (`PRL- PC`) task |
| `PRL_Win_Switch_Count` | Count of switches immediately following positive feedback (wins) on `PRL- PC` |
| `PRL_Lose_Stay_Count` | Count of stays immediately following negative feedback (losses) on `PRL- PC` |
| `PRL_Win_Switch_Rate` | `PRL_Win_Switch_Count` divided by total positive-feedback trials |
| `PRL_Lose_Stay_Rate` | `PRL_Lose_Stay_Count` divided by total negative-feedback trials |
| `Score_SPQ` | Schizotypal Personality Questionnaire-Brief (SPQ-Brief) total score (Yes count, 0–22) |
| `Score_GPTS` | Green Paranoid Thoughts Scale (GPTS) total score |
| `Score_PDI` | Peters Delusion Inventory (PDI-21) total score (Yes endorsements + Distress + Preoccupation + Conviction ratings) |
| `Score_CMQ` | Conspiracy Mentality Questionnaire (CMQ) total score |
| `Score_GCB` | Generic Conspiracist Beliefs Scale (GCB) total score |
| `Score_LocusOfControl` | Rotter Locus of Control total score (higher = more external locus of control) |
| `Score_SFQ` | Social Functioning Questionnaire (SFQ) total score (higher = greater social impairment) |
| `Age` | Self-reported age (years) from Gorilla Demographics |
| `Sex` | Self-reported biological sex (`Female`, `Male`, `I prefer not to answer`) |
| `z_pdi`, `z_gpts`, `z_spq` | Sample-standardized $z$-scores for PDI-21, GPTS, and SPQ-Brief |
| `Composite_Psychosis` | Mean of `z_pdi`, `z_gpts`, and `z_spq` |
| `z_gcb`, `z_cmq` | Sample-standardized $z$-scores for GCB and CMQ |
| `Composite_Conspiracy` | Mean of `z_gcb` and `z_cmq` |
| `High_Psychosis` | Binary median-split indicator (`1` if `Composite_Psychosis > median`, else `0`) |
| `High_Conspiracy` | Binary median-split indicator (`1` if `Composite_Conspiracy > median`, else `0`) |
| `Group` | 4-level subgroup classification: `1. Controls` ($n=51$), `2. Psychosis only` ($n=30$), `3. Conspiracy only` ($n=30$), `4. Psychosis + Conspiracy` ($n=51$) |
| `Group_High_Both` | Binary indicator (`1` if `4. Psychosis + Conspiracy`, else `0`) |
| `Group_Controls` | Binary indicator (`1` if `1. Controls`, else `0`) |
