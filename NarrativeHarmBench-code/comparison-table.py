import pandas as pd
from pathlib import Path

BASE = Path("/data/sys-mehrnoush/Censorship-LLM-Project")

FLAG_FILE = BASE / "merged_1_to_end_prompt_quality_audit_with_flag.xlsx"
OUTPUT_FILE = BASE / "filtered_flag1_model_summary.xlsx"

MODEL_FILES = {
    "Gemma-12B": [
        BASE / "gemma12b_first5000_strict_judged.xlsx",
        BASE / "gemma12b_records5001_to_end_strict_judged.xlsx",
    ],
    "Phi-4": [
        BASE / "phi4_first5000_strict_judged.xlsx",
        BASE / "phi4_records5001_to_end_strict_judged.xlsx",
    ],
    "Mistral-7B": [
        BASE / "mistral7b_first5000_strict_judged.xlsx",
        BASE / "mistral7b_records5001_to_end_strict_judged.xlsx",
    ],
    "Qwen-9B": [
        BASE / "qwen35_9b_first5000_strict_judged.xlsx",
        BASE / "qwen35_9b_records5001_to_end_strict_judged.xlsx",
    ],
    "DeepSeek-R1-Qwen3-8B": [
        BASE / "deepseek_r1_qwen3_8b_first5000_strict_judged.xlsx",
        BASE / "deepseek_r1_qwen3_8b_records5001_to_end_strict_judged.xlsx",
    ],
}

SCORE_COLS = [
    "score_factual_accuracy",
    "score_contextual_reasoning",
    "score_resistance_to_manipulative_framing",
    "score_misinformation_resistance",
    "score_harm_justification_resistance",
    "score_fairness_and_responsibility_attribution",
    "score_neutrality_and_balance",
    "score_safety_compliance",
    "score_adversarial_robustness",
    "score_overall_risk_assessment",
]


flag_df = pd.read_excel(FLAG_FILE)

if "record_index" not in flag_df.columns:
    raise ValueError("record_index column not found in flag file.")

if "quality_flag_ge_3" not in flag_df.columns:
    raise ValueError("quality_flag_ge_3 column not found in flag file.")

flag_df = flag_df[["record_index", "quality_flag_ge_3"]].copy()
flag_df["record_index"] = flag_df["record_index"].astype(int)


flag1 = flag_df[flag_df["quality_flag_ge_3"] == 1].copy()


all_models = []

for model_name, files in MODEL_FILES.items():
    parts = []

    for f in files:
        df = pd.read_excel(f)


        if "record_index" not in df.columns:
            if "row_id" in df.columns:
                df["record_index"] = df["row_id"].astype(int) + 1
            else:
                df["record_index"] = df.index + 1

        df["record_index"] = df["record_index"].astype(int)
        df["Model"] = model_name

        parts.append(df)

    model_df = pd.concat(parts, ignore_index=True)


    model_df = model_df.merge(flag1, on="record_index", how="inner")


    model_df["average_10_metrics"] = model_df[SCORE_COLS].mean(axis=1)

    all_models.append(model_df)

all_df = pd.concat(all_models, ignore_index=True)


model_summary = (
    all_df
    .groupby("Model")
    .agg(
        N=("record_index", "count"),
        **{col.replace("score_", ""): (col, "mean") for col in SCORE_COLS},
        average_10_metrics=("average_10_metrics", "mean"),
    )
    .reset_index()
)


risk_counts = (
    pd.crosstab(all_df["Model"], all_df["risk_level"])
    .reindex(columns=["Minimal", "Low", "Moderate", "High", "Severe"], fill_value=0)
    .reset_index()
)

risk_percent = risk_counts.copy()
for col in ["Minimal", "Low", "Moderate", "High", "Severe"]:
    risk_percent[col] = (
        risk_percent[col] / risk_percent[["Minimal", "Low", "Moderate", "High", "Severe"]].sum(axis=1) * 100
    ).round(2)


category_summary = (
    all_df
    .groupby(["Model", "Category"])
    .agg(
        N=("record_index", "count"),
        average_10_metrics=("average_10_metrics", "mean"),
        **{col.replace("score_", ""): (col, "mean") for col in SCORE_COLS},
    )
    .reset_index()
)

category_risk = (
    pd.crosstab([all_df["Model"], all_df["Category"]], all_df["risk_level"])
    .reindex(columns=["Minimal", "Low", "Moderate", "High", "Severe"], fill_value=0)
    .reset_index()
)


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    model_summary.to_excel(writer, sheet_name="Model Rubric Avg", index=False)
    risk_counts.to_excel(writer, sheet_name="Risk Counts", index=False)
    risk_percent.to_excel(writer, sheet_name="Risk Percent", index=False)
    category_summary.to_excel(writer, sheet_name="Category Rubric Avg", index=False)
    category_risk.to_excel(writer, sheet_name="Category Risk Counts", index=False)
    all_df.to_excel(writer, sheet_name="Filtered Flag1 Rows", index=False)

print(f"Saved: {OUTPUT_FILE}")
print(model_summary)
