import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


import sys
import transformers

print("PYTHON:", sys.executable)
print("TRANSFORMERS:", transformers.__version__)
print("TRANSFORMERS PATH:", transformers.__file__)

INPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/MERGED_augmented_PROMPT.xlsx"
OUTPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/gemma12b_responses.xlsx"
CHECKPOINT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/gemma12b_checkpoint.xlsx"

PROMPT_COLUMN = "generated_prompt"

MODEL_NAME = "gemma3_12b"
MODEL_ID = "google/gemma-3-12b-it"

MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
SAVE_EVERY = 500

LENGTH_SYSTEM_PROMPT = "Answer in 120-180 words.Stop after a complete final sentence.Do not exceed 180 words."


print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())


df = pd.read_excel(INPUT_FILE)

if PROMPT_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{PROMPT_COLUMN}' not found. Available columns: {list(df.columns)}"
    )


processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).eval()


results = []
done_row_ids = set()

if os.path.exists(CHECKPOINT_FILE):
    old = pd.read_excel(CHECKPOINT_FILE)
    results = old.to_dict("records")

    if "row_id" in old.columns:
        done_row_ids = set(old["row_id"].tolist())

    print(f"Loaded checkpoint with {len(results)} rows.")
else:
    print("No checkpoint found. Starting fresh.")


remaining_rows = []

for row_id, row in df.iterrows():
    if row_id in done_row_ids:
        continue

    prompt = row[PROMPT_COLUMN]

    if pd.isna(prompt) or str(prompt).strip() == "":
        continue

    remaining_rows.append((row_id, row))

print(f"Remaining rows to process: {len(remaining_rows)}")


def ask_gemma_batch(prompts):
    texts = []

    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": LENGTH_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": str(prompt),
            },
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        texts.append(text)

    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    responses = []

    for i in range(len(prompts)):
        generated_tokens = outputs[i][prompt_length:]

        response = processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        responses.append(response)

    return responses


processed_since_save = 0

for start in tqdm(range(0, len(remaining_rows), BATCH_SIZE), desc="Generating"):

    batch = remaining_rows[start:start + BATCH_SIZE]
    prompts = [str(row[PROMPT_COLUMN]) for _, row in batch]

    try:
        responses = ask_gemma_batch(prompts)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print("OOM detected. Falling back to single-sample inference.")

            responses = []

            for prompt in prompts:
                try:
                    single_response = ask_gemma_batch([prompt])[0]
                except Exception as inner_e:
                    single_response = f"ERROR: {type(inner_e).__name__}: {inner_e}"

                responses.append(single_response)
        else:
            responses = [f"ERROR: {type(e).__name__}: {e}" for _ in batch]

    except Exception as e:
        responses = [f"ERROR: {type(e).__name__}: {e}" for _ in batch]

    for (row_id, row), response in zip(batch, responses):
        output_row = row.to_dict()

        output_row["row_id"] = row_id
        output_row["model_name"] = MODEL_NAME
        output_row["model_id"] = MODEL_ID
        output_row["response"] = response

        results.append(output_row)
        processed_since_save += 1

    if processed_since_save >= SAVE_EVERY:
        pd.DataFrame(results).to_excel(CHECKPOINT_FILE, index=False)
        print(f"Checkpoint saved: {len(results)} rows")
        processed_since_save = 0


out_df = pd.DataFrame(results)
out_df.to_excel(OUTPUT_FILE, index=False)
out_df.to_excel(CHECKPOINT_FILE, index=False)

print(f"Done. Saved to: {OUTPUT_FILE}")
