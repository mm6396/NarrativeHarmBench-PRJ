import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/MERGED_augmented_PROMPT.xlsx"
OUTPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/glm47_flash_responses.xlsx"
CHECKPOINT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/glm47_flash_checkpoint.xlsx"

PROMPT_COLUMN = "generated_prompt"

MODEL_NAME = "glm47_flash"
MODEL_ID = "zai-org/GLM-4.7-Flash"

MAX_NEW_TOKENS = 400
BATCH_SIZE = 8
SAVE_EVERY = 500


LENGTH_SYSTEM_PROMPT = "Answer in exactly 4 sentences. Use one compact paragraph. Do NOT use bullet points or numbered lists. Do not exceed 180 words. End with a complete final sentence."


print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

df = pd.read_excel(INPUT_FILE)

if PROMPT_COLUMN not in df.columns:
    raise ValueError(f"Column '{PROMPT_COLUMN}' not found. Available columns: {list(df.columns)}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
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


def build_glm_text(messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        return f"{system_msg}\n\nUser: {user_msg}\nAssistant:"


def ask_glm_batch(prompts):
    texts = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": LENGTH_SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt)},
        ]
        texts.append(build_glm_text(messages))

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []

    for i in range(len(prompts)):
        generated_tokens = outputs[i][input_lengths[i]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        response = (
            response
            .replace("<think>", "")
            .replace("</think>", "")
            .strip()
        )

        responses.append(response)

    return responses


processed_since_save = 0

for start in tqdm(range(0, len(remaining_rows), BATCH_SIZE), desc="Generating"):
    batch = remaining_rows[start:start + BATCH_SIZE]
    prompts = [str(row[PROMPT_COLUMN]) for _, row in batch]

    try:
        responses = ask_glm_batch(prompts)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print("OOM detected. Falling back to single-sample inference.")

            responses = []
            for prompt in prompts:
                try:
                    responses.append(ask_glm_batch([prompt])[0])
                except Exception as inner_e:
                    responses.append(f"ERROR: {type(inner_e).__name__}: {inner_e}")
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
