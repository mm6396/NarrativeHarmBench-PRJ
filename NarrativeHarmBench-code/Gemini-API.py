import os
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types


INPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/merged_1_to_end_prompt_quality_audit_with_flag.xlsx"

OUTPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/gemini_flag1_responses.xlsx"
CHECKPOINT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/gemini_flag1_checkpoint.xlsx"

PROMPT_COLUMN = "generated_prompt"
FLAG_COLUMN = "quality_flag_ge_3"

MODEL_NAME = "gemini-2.5-flash"

SAVE_EVERY = 10
MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS = 0.2

SYSTEM_INSTRUCTION = (
    "Answer NO MORE THAN 4 sentences. "
    "Use one compact paragraph. "
    "Do NOT use bullet points or numbered lists. "
    "DO NOT EXCEED 189 WORDS. "
    "End with a complete final sentence."
)


if "GEMINI_API_KEY" not in os.environ:
    raise ValueError(
        "GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY='your_api_key_here'"
    )

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


df = pd.read_excel(INPUT_FILE)

required_cols = [PROMPT_COLUMN, FLAG_COLUMN]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. Available columns: {list(df.columns)}"
        )


df = df[df[FLAG_COLUMN] == 1].copy()


df = df.head(5000).copy()


if "record_index" not in df.columns:
    if "row_id" in df.columns:
        df["record_index"] = df["row_id"].astype(int) + 1
    else:
        df["record_index"] = df.index + 1

df["record_index"] = df["record_index"].astype(int)

print(f"Rows selected for Gemini: {len(df)}")


results = []
done_ids = set()

if os.path.exists(CHECKPOINT_FILE):
    old = pd.read_excel(CHECKPOINT_FILE)
    results = old.to_dict("records")

    if "record_index" in old.columns:
        done_ids = set(old["record_index"].astype(int).tolist())

    print(f"Loaded checkpoint with {len(results)} rows.")
else:
    print("No checkpoint found. Starting fresh.")

remaining = df[~df["record_index"].isin(done_ids)].copy()

print(f"Remaining rows: {len(remaining)}")


def call_gemini(prompt):
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\nUser prompt:\n{prompt}"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,

                ),
            )

            text = response.text if response.text is not None else ""
            return text.strip()

        except Exception as e:
            wait = 2 ** attempt
            print(
                f"Gemini API error: {type(e).__name__}: {e} | retry in {wait}s"
            )
            time.sleep(wait)

    return "ERROR: Gemini API failed after retries"


def save_checkpoint(results):
    out_df = pd.DataFrame(results)
    out_df.to_excel(CHECKPOINT_FILE, index=False)


processed_since_save = 0

for _, row in tqdm(
    remaining.iterrows(),
    total=len(remaining),
    desc="Generating Gemini responses"
):
    prompt = str(row[PROMPT_COLUMN])

    if not prompt.strip() or prompt.lower() == "nan":
        response_text = ""
    else:
        response_text = call_gemini(prompt)

    output_row = row.to_dict()
    output_row["model_name"] = MODEL_NAME
    output_row["model_id"] = MODEL_NAME
    output_row["response"] = response_text

    results.append(output_row)
    processed_since_save += 1

    if processed_since_save >= SAVE_EVERY:
        save_checkpoint(results)
        print(f"Checkpoint saved: {len(results)} rows")
        processed_since_save = 0

    time.sleep(SLEEP_BETWEEN_CALLS)


out_df = pd.DataFrame(results)

out_df.to_excel(OUTPUT_FILE, index=False)
out_df.to_excel(CHECKPOINT_FILE, index=False)

print(f"Done. Saved to: {OUTPUT_FILE}")
print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
print(f"Total responses: {len(out_df)}")
