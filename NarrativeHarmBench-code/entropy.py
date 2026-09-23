import pandas as pd
import numpy as np
import math

INPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/model_scores_with_year_analysis.xlsx"
OUTPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/event_category_entropy_by_model.xlsx"

model_sheets = ["Gemma-12B", "Phi-4", "Mistral-7B", "Qwen-9B"]

def shannon_entropy(values):
    counts = values.value_counts(dropna=True)
    probs = counts / counts.sum()
    return -sum(p * math.log2(p) for p in probs if p > 0)

rows = []

for model in model_sheets:
    df = pd.read_excel(INPUT_FILE, sheet_name=model)

    if "Category" in df.columns:
        category_col = "Category"
    elif "category" in df.columns:
        category_col = "category"
    else:
        raise ValueError(f"No category column found in sheet {model}")

    if "risk_level" not in df.columns:
        raise ValueError(f"No risk_level column found in sheet {model}")

    df[category_col] = (
        df[category_col]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )

    for category, g in df.groupby(category_col):
        entropy = shannon_entropy(g["risk_level"])
        max_entropy = math.log2(5)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else np.nan

        risk_counts = g["risk_level"].value_counts()

        rows.append({
            "Model": model,
            "Event Category": category,
            "N": len(g),
            "Entropy": round(entropy, 4),
            "Normalized Entropy": round(normalized_entropy, 4),
            "Minimal": int(risk_counts.get("Minimal", 0)),
            "Low": int(risk_counts.get("Low", 0)),
            "Moderate": int(risk_counts.get("Moderate", 0)),
            "High": int(risk_counts.get("High", 0)),
            "Severe": int(risk_counts.get("Severe", 0)),
        })

entropy_df = pd.DataFrame(rows)

pivot_entropy = entropy_df.pivot_table(
    index="Event Category",
    columns="Model",
    values="Entropy",
    aggfunc="mean"
).reset_index()

pivot_norm_entropy = entropy_df.pivot_table(
    index="Event Category",
    columns="Model",
    values="Normalized Entropy",
    aggfunc="mean"
).reset_index()

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    entropy_df.to_excel(writer, sheet_name="Long Format", index=False)
    pivot_entropy.to_excel(writer, sheet_name="Entropy Pivot", index=False)
    pivot_norm_entropy.to_excel(writer, sheet_name="Normalized Entropy Pivot", index=False)

print(f"Saved: {OUTPUT_FILE}")
