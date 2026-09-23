import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/data/sys-mehrnoush/Censorship-LLM-Project")


FLAG_FILE = BASE / "merged_1_to_end_prompt_quality_audit_with_flag.xlsx"
YEAR_FILE = BASE / "model_scores_with_year_analysis.xlsx"

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

OUTPUT_PNG = BASE / "event_decade_model_average_score_flag1.png"
OUTPUT_PDF = BASE / "event_decade_model_average_score_flag1.pdf"
OUTPUT_TABLE = BASE / "event_decade_model_average_score_flag1_values.xlsx"

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
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def clean_text_col(series):
    return (
        series.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )

def year_to_decade(year):
    try:
        y = int(float(year))
        if y <= 0:
            return None
        return f"{(y // 10) * 10}s"
    except Exception:
        return None

def decade_sort_key(x):
    try:
        return int(str(x).replace("s", ""))
    except Exception:
        return 99999


flag_df = pd.read_excel(FLAG_FILE)

if "record_index" not in flag_df.columns:
    raise ValueError("record_index column not found in flag file.")

if "quality_flag_ge_3" not in flag_df.columns:
    raise ValueError("quality_flag_ge_3 column not found in flag file.")

flag_df = flag_df[["record_index", "quality_flag_ge_3"]].copy()
flag_df["record_index"] = flag_df["record_index"].astype(int)

flag1 = flag_df[flag_df["quality_flag_ge_3"] == 1][["record_index"]]

print(f"Flag=1 records: {len(flag1)}")


xls = pd.ExcelFile(YEAR_FILE)

print("Available sheets in YEAR_FILE:", xls.sheet_names)

if "Event Year Map" not in xls.sheet_names:
    raise ValueError("Sheet 'Event Year Map' not found in YEAR_FILE.")

year_map_raw = pd.read_excel(YEAR_FILE, sheet_name="Event Year Map")

print("Event Year Map columns:", year_map_raw.columns.tolist())

cols_lower = {c.lower().strip(): c for c in year_map_raw.columns}


event_id_col = None
for candidate in ["event_id", "event id", "id"]:
    if candidate in cols_lower:
        event_id_col = cols_lower[candidate]
        break


event_col = None
for candidate in ["event", "event title", "title"]:
    if candidate in cols_lower:
        event_col = cols_lower[candidate]
        break


year_col = None
for candidate in ["year", "event_year", "event year"]:
    if candidate in cols_lower:
        year_col = cols_lower[candidate]
        break


decade_col = None
for candidate in ["decade", "event_decade", "event decade"]:
    if candidate in cols_lower:
        decade_col = cols_lower[candidate]
        break

if year_col is None:
    raise ValueError(
        f"No Year column found in Event Year Map. Columns: {year_map_raw.columns.tolist()}"
    )

keep_cols = []
rename_map = {}

if event_id_col is not None:
    keep_cols.append(event_id_col)
    rename_map[event_id_col] = "event_id"

if event_col is not None:
    keep_cols.append(event_col)
    rename_map[event_col] = "Event"

keep_cols.append(year_col)
rename_map[year_col] = "Year"

if decade_col is not None:
    keep_cols.append(decade_col)
    rename_map[decade_col] = "Decade"

year_map = year_map_raw[keep_cols].copy()
year_map = year_map.rename(columns=rename_map)

if "Decade" not in year_map.columns:
    year_map["Decade"] = year_map["Year"].apply(year_to_decade)

if "event_id" in year_map.columns:
    year_map["event_id"] = clean_text_col(year_map["event_id"])

if "Event" in year_map.columns:
    year_map["Event"] = clean_text_col(year_map["Event"])

year_map["Decade"] = clean_text_col(year_map["Decade"])

year_map = year_map[
    ~year_map["Decade"].str.lower().isin(["unknown", "nan", "none", ""])
].copy()


if "event_id" in year_map.columns:
    year_map = year_map.drop_duplicates("event_id")
elif "Event" in year_map.columns:
    year_map = year_map.drop_duplicates("Event")
else:
    raise ValueError("Year map has neither event_id nor Event column.")

print("Year map preview:")
print(year_map.head())
print(f"Year mapping rows: {len(year_map)}")


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


        if "event_id" in year_map.columns and "event_id" in df.columns:
            df["event_id"] = clean_text_col(df["event_id"])
            df = df.merge(year_map, on="event_id", how="left")

        elif "Event" in year_map.columns and "Event" in df.columns:
            df["Event"] = clean_text_col(df["Event"])
            df = df.merge(year_map, on="Event", how="left")

        else:
            raise ValueError(
                f"Cannot merge year mapping for {file_path}. "
                "Need either event_id or Event in both judged file and year_map."
            )

        missing_scores = [c for c in score_cols if c not in df.columns]
        if missing_scores:
            raise ValueError(f"Missing score columns in {file_path}: {missing_scores}")

        df["average_10_metrics"] = df[score_cols].mean(axis=1)

        df = df[df["Decade"].notna()].copy()
        df = df[
            ~df["Decade"].astype(str).str.lower().isin(["unknown", "nan", "none", ""])
        ].copy()

        frames.append(
            df[["Model", "record_index", "Year", "Decade", "average_10_metrics"]]
        )

all_df = pd.concat(frames, ignore_index=True)

print("Filtered rows by model after flag=1 and known decade:")
print(all_df.groupby("Model")["record_index"].nunique())

print("Decade counts:")
print(all_df.groupby("Decade")["record_index"].nunique().sort_index())


pivot = all_df.pivot_table(
    index="Decade",
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
pivot = pivot.loc[sorted(pivot.index, key=decade_sort_key)]

model_label_map = {
    "Gemma-12B": "Gemma",
    "Phi-4": "Phi",
    "Mistral-7B": "Mistral",
    "Qwen-9B": "Qwen",
    "DeepSeek-R1-Qwen3-8B": "DeepSeek",
}

pivot_plot = pivot.rename(columns=model_label_map)
pivot_plot.to_excel(OUTPUT_TABLE)


fig, ax = plt.subplots(figsize=(16, 8.5))

for model in pivot_plot.columns:
    ax.plot(
        pivot_plot.index,
        pivot_plot[model],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=model,
    )


ax.set_xlabel("Event Decade", fontsize=18)
ax.set_ylabel("Avg. of 10 Rubric Metrics", fontsize=18)

ax.tick_params(axis="x", labelrotation=45, labelsize=15)
ax.tick_params(axis="y", labelsize=16)

ax.grid(True, alpha=0.25, linewidth=0.8)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    fontsize=15,
)

fig.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")

print(f"Saved PNG to: {OUTPUT_PNG}")
print(f"Saved PDF to: {OUTPUT_PDF}")
print(f"Saved values to: {OUTPUT_TABLE}")
