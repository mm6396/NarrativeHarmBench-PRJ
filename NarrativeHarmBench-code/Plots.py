import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/data/sys-mehrnoush/Censorship-LLM-Project")

FLAG_FILE = BASE / "merged_1_to_end_prompt_quality_audit_with_flag.xlsx"

paths = {
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

OUTPUT_PNG = BASE / "event_category_model_heatmap_flag1.png"
OUTPUT_PDF = BASE / "event_category_model_heatmap_flag1.pdf"
OUTPUT_TABLE = BASE / "event_category_model_heatmap_flag1_values.xlsx"

score_cols = [
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


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


flag_df = pd.read_excel(FLAG_FILE)

if "record_index" not in flag_df.columns:
    raise ValueError("record_index column not found in flag file.")

if "quality_flag_ge_3" not in flag_df.columns:
    raise ValueError("quality_flag_ge_3 column not found in flag file.")

flag_df = flag_df[["record_index", "quality_flag_ge_3"]].copy()
flag_df["record_index"] = flag_df["record_index"].astype(int)

flag1 = flag_df[flag_df["quality_flag_ge_3"] == 1][["record_index"]]

print(f"Flag=1 records: {len(flag1)}")


frames = []

for model, model_paths in paths.items():
    for file_path in model_paths:
        df = pd.read_excel(file_path)

        df["Model"] = model

        if "record_index" not in df.columns:
            if "row_id" in df.columns:
                df["record_index"] = df["row_id"].astype(int) + 1
            else:
                df["record_index"] = df.index + 1

        df["record_index"] = df["record_index"].astype(int)


        df = df.merge(flag1, on="record_index", how="inner")

        if "Category" in df.columns:
            category_col = "Category"
        elif "category" in df.columns:
            category_col = "category"
        else:
            raise ValueError(
                f"No category column found in {file_path}. Columns: {list(df.columns)}"
            )

        df["Category_clean"] = (
            df[category_col]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
        )

        missing_scores = [c for c in score_cols if c not in df.columns]
        if missing_scores:
            raise ValueError(f"Missing score columns in {file_path}: {missing_scores}")

        df["average_10_metrics"] = df[score_cols].mean(axis=1)

        frames.append(
            df[["Model", "record_index", "Category_clean", "average_10_metrics"]]
        )

all_df = pd.concat(frames, ignore_index=True)

print("Filtered rows by model:")
print(all_df.groupby("Model")["record_index"].nunique())


pivot = all_df.pivot_table(
    index="Category_clean",
    columns="Model",
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

pivot = pivot[[m for m in model_order if m in pivot.columns]]

pivot["Overall"] = pivot.mean(axis=1)
pivot = pivot.sort_values("Overall", ascending=True)
plot_data = pivot.drop(columns=["Overall"])


model_label_map = {
    "Gemma-12B": "Gemma",
    "Phi-4": "Phi",
    "Mistral-7B": "Mistral",
    "Qwen-9B": "Qwen",
    "DeepSeek-R1-Qwen3-8B": "DeepSeek-R1-Qwen",
}

plot_data = plot_data.rename(columns=model_label_map)
plot_data.to_excel(OUTPUT_TABLE)


short_labels = {
    "Protests, Privacy & Human Rights Events": "Protests / Human Rights",
    "Corruption & Political Scandals": "Corruption / Scandals",
    "AI Scandals & Deepfake Incidents": "AI Scandals / Deepfakes",
    "Defamation, Lawsuits & Political Speech Cases": "Defamation / Political Speech",
    "Hate Speech & Extremism": "Hate Speech / Extremism",
    "Disinformation & Election Manipulation": "Disinformation / Elections",
    "War Thunder & Military Manual Leaks": "Military Manual Leaks",
    "Health Care Fraud & Abuse": "Health Care Fraud",
    "AI Jailbreaks, Prompt Injection & Exploits": "AI Jailbreaks / Prompt Injection",
    "Sexual Abuse & Sex Trafficking": "Sexual Abuse / Trafficking",
}

y_labels = [short_labels.get(x, x) for x in plot_data.index]


fig, ax = plt.subplots(figsize=(16, 8.5))

im = ax.imshow(plot_data.values, aspect="auto", cmap="viridis")

ax.set_xticks(np.arange(len(plot_data.columns)))
ax.set_xticklabels(
    plot_data.columns,
    rotation=25,
    ha="center",
    fontsize=25,
)

ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(
    y_labels,
    fontsize=22,


    multialignment="center"
)


ax.set_xlabel("LLM", fontsize=25)
ax.set_ylabel("Event Category", fontsize=22)


for i in range(plot_data.shape[0]):
    for j in range(plot_data.shape[1]):
        ax.text(
            j,
            i,
            f"{plot_data.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=25,
            color="black",
        )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(
    "Avg. of 10 Rubric Metrics",
    rotation=270,
    labelpad=32,
    fontsize=25,
)
cbar.ax.tick_params(labelsize=26)

fig.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")

print(f"Saved PNG to: {OUTPUT_PNG}")
print(f"Saved PDF to: {OUTPUT_PDF}")
print(f"Saved values to: {OUTPUT_TABLE}")
