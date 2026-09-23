import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


INPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/MERGED_augmented_PROMPT.xlsx"
OUTPUT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/deepseek_r1_qwen3_8b_responses.xlsx"
CHECKPOINT_FILE = "/data/sys-mehrnoush/Censorship-LLM-Project/deepseek_r1_qwen3_8b_checkpoint.xlsx"

PROMPT_COLUMN = "generated_prompt"

MODEL_NAME = "DeepSeek-R1-0528-Qwen3-8B"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
SAVE_EVERY = 500

SYSTEM_PROMPT = (
    "Answer in exactly 4 sentences. Use one compact paragraph. "
    "Do NOT use bullet points or numbered lists. Do NOT exceed 180 words. "
    "End with a complete final sentence. Do not include reasoning tags."
)

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())


df = pd.read_excel(INPUT_FILE)

if PROMPT_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{PROMPT_COLUMN}' not found. Available columns: {list(df.columns)}"
    )


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
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval()


results = []
done_row_ids = set()

if os.path.exists(CHECKPOINT_FILE):
    old = pd.read_excel(CHECKPOINT_FILE)
    results = old.to_dict("records")

    if "row_id" in old.columns:
        done_row_ids = set(old["row_id"].dropna().astype(int).tolist())

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


def clean_deepseek_response(text):
    text = str(text).strip()


    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


    text = text.replace("<think>", "").replace("</think>", "").strip()

    return text


def build_chat_prompt(user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(user_prompt)},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def ask_deepseek_batch(prompts):
    texts = [build_chat_prompt(prompt) for prompt in prompts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_len = inputs["input_ids"].shape[1]


    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []

    for i in range(len(prompts)):
        generated_tokens = outputs[i][input_len:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        responses.append(clean_deepseek_response(response))

    return responses


processed_since_save = 0

for start in tqdm(range(0, len(remaining_rows), BATCH_SIZE), desc="Generating"):
    batch = remaining_rows[start:start + BATCH_SIZE]
    prompts = [str(row[PROMPT_COLUMN]) for _, row in batch]

    try:
        responses = ask_deepseek_batch(prompts)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print("OOM detected. Falling back to single-sample inference.")

            responses = []

            for prompt in prompts:
                try:
                    single_response = ask_deepseek_batch([prompt])[0]
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

if "row_id" in out_df.columns:
    out_df = out_df.sort_values("row_id").reset_index(drop=True)

out_df.to_excel(OUTPUT_FILE, index=False)
out_df.to_excel(CHECKPOINT_FILE, index=False)

print(f"Done. Saved to: {OUTPUT_FILE}")
