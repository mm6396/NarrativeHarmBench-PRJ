import pandas as pd
from pathlib import Path

BASE = Path("/data/sys-mehrnoush/Censorship-LLM-Project")

FLAG_FILE = BASE / "merged_1_to_end_prompt_quality_audit_with_flag.xlsx"
OUTPUT_XLSX = BASE / "flag1_event_category_distribution.xlsx"
OUTPUT_TEX = BASE / "flag1_event_category_distribution.tex"


df = pd.read_excel(FLAG_FILE)

if "quality_flag_ge_3" not in df.columns:
    raise ValueError("Column 'quality_flag_ge_3' not found.")

if "Category" in df.columns:
    category_col = "Category"
elif "category" in df.columns:
    category_col = "category"
else:
    raise ValueError(f"No category column found. Columns: {list(df.columns)}")


df_flag1 = df[df["quality_flag_ge_3"] == 1].copy()

df_flag1[category_col] = (
    df_flag1[category_col]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)


summary = (
    df_flag1[category_col]
    .value_counts()
    .rename_axis("Event Category")
    .reset_index(name="Count")
)

summary["Percent"] = summary["Count"] / summary["Count"].sum() * 100
summary["Percent"] = summary["Percent"].round(2)

total = pd.DataFrame({
    "Event Category": ["Total"],
    "Count": [summary["Count"].sum()],
    "Percent": [100.00],
})

summary_with_total = pd.concat([summary, total], ignore_index=True)


summary_with_total.to_excel(OUTPUT_XLSX, index=False)


latex_df = summary_with_total.copy()
latex_df["Percent"] = latex_df["Percent"].map(lambda x: f"{x:.2f}\\%")
latex_df["Count"] = latex_df["Count"].map(lambda x: f"{int(x):,}")

latex_table = latex_df.to_latex(
    index=False,
    escape=False,
    column_format="lrr",
    caption="Distribution of event categories after filtering to prompts with overall prompt quality $\\geq 3$.",
    label="tab:flag1_event_category_distribution",
)

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(summary_with_total)
print(f"Saved Excel to: {OUTPUT_XLSX}")
print(f"Saved LaTeX to: {OUTPUT_TEX}")
