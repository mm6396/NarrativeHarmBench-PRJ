import pandas as pd
from pathlib import Path

BASE = Path("/data/sys-mehrnoush/Censorship-LLM-Project")

FLAG_FILE = BASE / "merged_1_to_end_prompt_quality_audit_with_flag.xlsx"
OUTPUT_FILE = BASE / "flag1_model_by_generated_prompt_category.xlsx"

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

flag_df = flag_df[["record_index", "quality_flag_ge_3"]].copy()
flag_df["record_index"] = flag_df["record_index"].astype(int)

flag1 = flag_df[flag_df["quality_flag_ge_3"] == 1][["record_index"]]

print(f"Flag=1 records: {len(flag1)}")


all_rows = []

for model_name, files in MODEL_FILES.items():
    model_parts = []

    for file_path in files:
        df = pd.read_excel(file_path)

        if "record_index" not in df.columns:
            if "row_id" in df.columns:
                df["record_index"] = df["row_id"].astype(int) + 1
            else:
                df["record_index"] = df.index + 1

        df["record_index"] = df["record_index"].astype(int)
        df["Model"] = model_name

        if "generated_prompt_category" not in df.columns:
            raise ValueError(
                f"'generated_prompt_category' not found in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )

        missing_scores = [c for c in SCORE_COLS if c not in df.columns]
        if missing_scores:
            raise ValueError(f"Missing score columns in {file_path}: {missing_scores}")

        df["average_10_metrics"] = df[SCORE_COLS].mean(axis=1)

        model_parts.append(df)

    model_df = pd.concat(model_parts, ignore_index=True)


    model_df = model_df.merge(flag1, on="record_index", how="inner")

    all_rows.append(model_df)

all_df = pd.concat(all_rows, ignore_index=True)

print("Rows after filtering:")
print(all_df.groupby("Model")["record_index"].nunique())


all_df["generated_prompt_category"] = (
    all_df["generated_prompt_category"]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)


pivot_avg = all_df.pivot_table(
    index="Model",
    columns="generated_prompt_category",
    values="average_10_metrics",
    aggfunc="mean",
)


model_order = [
    "Gemma-12B",
    "Phi-4",
    "Mistral-7B",
    "Qwen-9B",
    "DeepSeek-R1-Qwen3-8B",
]

pivot_avg = pivot_avg.reindex(model_order)


pivot_avg = pivot_avg.round(3)


pivot_count = all_df.pivot_table(
    index="Model",
    columns="generated_prompt_category",
    values="average_10_metrics",
    aggfunc="count",
)

pivot_count = pivot_count.reindex(model_order)


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    pivot_avg.to_excel(writer, sheet_name="Avg by Prompt Category")
    pivot_count.to_excel(writer, sheet_name="N by Prompt Category")
    all_df.to_excel(writer, sheet_name="Filtered Flag1 Rows", index=False)

print(f"Saved: {OUTPUT_FILE}")
print("\nAverage rubric score by generated_prompt_category:")
print(pivot_avg)
