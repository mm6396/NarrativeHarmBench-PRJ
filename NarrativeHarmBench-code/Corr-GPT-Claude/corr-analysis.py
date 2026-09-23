import pandas as pd


df = pd.read_csv("/data/sys-mehrnoush/Censorship-LLM-Project/Corr-GPT-Claude/correlation_results.csv")


results = df[
    [
        "model",
        "dimension",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p"
    ]
].copy()


print("\nGPT vs Claude Correlation Results\n")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


overall = results[
    results["dimension"].str.lower() == "overall score"
]

print("\n\n===== Overall Score Only =====\n")

print(
    overall.to_string(
        index=False,
        float_format=lambda x: f"{x:.7f}"
    )
)


print("\n\n===== Average Correlation =====")

print(f"Mean Pearson r:    {df['pearson_r'].mean():.7f}")
print(f"Mean Spearman rho: {df['spearman_rho'].mean():.7f}")


results.to_csv(
    "gpt_claude_correlations_7decimal.csv",
    index=False,
    float_format="%.4f"
)

print("\nSaved as: gpt_claude_correlations_4decimal.csv")
