import os
import re
import pandas as pd
from scipy.stats import pearsonr, spearmanr


GPT_DIR = "/data/sys-mehrnoush/Censorship-LLM-Project/Corr-GPT-Claude/GPT-Annotation"

CLAUDE_DIR = "/data/sys-mehrnoush/Censorship-LLM-Project/Corr-GPT-Claude/claude-annotation 2"


DIMENSION_MAP = {

    "overall_quality_score":
        ("claude_score", "Overall score"),

    "factual_accuracy":
        ("I_Factual_Accuracy", "Factual accuracy"),

    "contextual_reasoning":
        ("II_Contextual_Reasoning", "Contextual reasoning"),

    "resistance_to_manipulative_framing":
        (
            "III_Resistance_to_Manipulative_Framing",
            "Resist. manipulative framing"
        ),

    "misinformation_resistance":
        (
            "IV_Misinformation_Resistance",
            "Misinformation resistance"
        ),

    "harm_justification_resistance":
        (
            "V_Harm_Justification_Resistance",
            "Harm-justification resistance"
        ),

    "fairness_responsibility_attribution":
        (
            "VI_Fairness_and_Responsibility_Attribution",
            "Fairness/responsibility attrib."
        ),

    "neutrality_balance":
        (
            "VII_Neutrality_and_Balance",
            "Neutrality & balance"
        ),

    "safety_compliance":
        (
            "VIII_Safety_Compliance",
            "Safety compliance"
        ),

    "adversarial_robustness":
        (
            "IX_Adversarial_Robustness",
            "Adversarial robustness"
        ),

    "overall_risk_assessment":
        (
            "X_Overall_Risk_Assessment",
            "Overall risk assessment"
        ),
}


MODEL_NAMES = {

    "deepseek": "DeepSeek",

    "gemma12b": "Gemma 12B",

    "mistral7b": "Mistral 7B",

    "phi4": "Phi-4",

    "qwen35_9b": "Qwen3.5 9B",
}


def discover_pairs():

    gpt_files = {
        f
        for f in os.listdir(GPT_DIR)
        if f.endswith(".xlsx") and "(1)" not in f
    }

    claude_files = {
        f
        for f in os.listdir(CLAUDE_DIR)
        if f.endswith(".xlsx")
    }

    pairs = []

    for gf in sorted(gpt_files):

        match = re.match(
            r"(?P<model>.+?)_batch(?P<batch>\d+)_safety_evaluated\.xlsx",
            gf
        )

        if not match:
            continue

        model = match.group("model")
        batch = match.group("batch")

        claude_filename = f"{model}_batch{batch}_scored.xlsx"

        if claude_filename in claude_files:

            pairs.append(
                (
                    model,
                    batch,
                    os.path.join(GPT_DIR, gf),
                    os.path.join(CLAUDE_DIR, claude_filename)
                )
            )

        else:

            print(
                f"[skip] No Claude counterpart for {gf} "
                f"(looked for {claude_filename})"
            )

    return pairs


def load_merged(gpt_path, claude_path):

    gpt_df = pd.read_excel(
        gpt_path,
        sheet_name="Sheet1"
    )

    claude_df = pd.read_excel(
        claude_path,
        sheet_name="Sheet1"
    )


    join_keys = {
        "record_index",
        "event_id"
    }

    overlap = (
        set(gpt_df.columns)
        & set(claude_df.columns)
    ) - join_keys

    if overlap:

        gpt_df = gpt_df.drop(
            columns=list(overlap)
        )

    merged = gpt_df.merge(
        claude_df,
        on=[
            "record_index",
            "event_id"
        ],
        how="inner"
    )

    return merged


def correlations_for_model(model_key, frames):


    full = pd.concat(
        frames,
        ignore_index=True
    )

    rows = []

    for gpt_col, (claude_col, label) in DIMENSION_MAP.items():


        x = pd.to_numeric(
            full[gpt_col],
            errors="coerce"
        )

        y = pd.to_numeric(
            full[claude_col],
            errors="coerce"
        )


        valid = x.notna() & y.notna()

        x = x[valid]
        y = y[valid]

        n = len(x)


        if n < 2:

            pear_r = float("nan")
            pear_p = float("nan")

            spear_r = float("nan")
            spear_p = float("nan")

        else:

            pear_r, pear_p = pearsonr(
                x,
                y
            )

            spear_r, spear_p = spearmanr(
                x,
                y
            )

        rows.append({

            "model":
                MODEL_NAMES.get(
                    model_key,
                    model_key
                ),

            "dimension":
                label,

            "n":
                n,

            "pearson_r":
                round(pear_r, 7),

            "pearson_p":
                round(pear_p, 7),

            "spearman_rho":
                round(spear_r, 7),

            "spearman_p":
                round(spear_p, 7),

            "gpt_mean":
                round(x.mean(), 7),

            "claude_mean":
                round(y.mean(), 7),
        })

    return pd.DataFrame(rows)


def main():


    pairs = discover_pairs()

    print(
        f"\nFound {len(pairs)} matching GPT-Claude file pairs.\n"
    )


    frames_by_model = {}

    for model, batch, gpt_path, claude_path in pairs:

        merged = load_merged(
            gpt_path,
            claude_path
        )

        frames_by_model.setdefault(
            model,
            []
        ).append(
            merged
        )

        print(
            f"[ok] {model} batch{batch}: "
            f"{len(merged)} matched rows"
        )


    all_results = []

    for model_key, frames in frames_by_model.items():

        model_results = correlations_for_model(
            model_key,
            frames
        )

        all_results.append(
            model_results
        )


    if not all_results:

        print(
            "\nNo matching data found."
        )

        return None


    results = pd.concat(
        all_results,
        ignore_index=True
    )


    output_file = "correlation_results.csv"

    results.to_csv(
        output_file,
        index=False,
        float_format="%.4f"
    )


    overall = (

        results[
            results["dimension"]
            == "Overall score"
        ][
            [
                "model",
                "n",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
                "gpt_mean",
                "claude_mean"
            ]
        ]

        .sort_values(
            "spearman_rho",
            ascending=False
        )
    )


    print(
        "\n"
        "==============================================="
    )

    print(
        "Overall-score GPT vs Claude correlation"
    )

    print(
        "===============================================\n"
    )


    print(

        overall.to_string(

            index=False,

            formatters={

                "pearson_r":
                    lambda x: f"{x:.4f}",

                "pearson_p":
                    lambda x: f"{x:.4f}",

                "spearman_rho":
                    lambda x: f"{x:.4f}",

                "spearman_p":
                    lambda x: f"{x:.4f}",

                "gpt_mean":
                    lambda x: f"{x:.4f}",

                "claude_mean":
                    lambda x: f"{x:.4f}",
            }
        )
    )


    overall_output = "correlation_overall_only.csv"

    overall.to_csv(
        overall_output,
        index=False,
        float_format="%.4f"
    )


    print(
        "\n"
        "==============================================="
    )

    print(
        "Average Overall-Score Correlation"
    )

    print(
        "===============================================\n"
    )


    print(
        f"Mean Pearson r    : "
        f"{overall['pearson_r'].mean():.4f}"
    )

    print(
        f"Mean Spearman rho : "
        f"{overall['spearman_rho'].mean():.4f}"
    )


    print(
        "\nFiles written:"
    )

    print(
        f"1. {output_file}"
    )

    print(
        f"2. {overall_output}"
    )


    return results


if __name__ == "__main__":

    results = main()
