
"""
patois_translator.py
--------------------
Reusable module for translating Jamaican Patois → English using OpenAI API.

Usage example:
--------------
from patois_translator import translate_patois_csv

translate_patois_csv(
    input_path="input.csv",
    output_path="output.csv",
    batch_size=100,
    budget_usd=5.0,
    model="gpt-4o-mini",
)
"""

import os
import time
import math
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# def translate_patois_csv(
#     input_path: str,
#     output_path: str,
#     column_to_translate : str,
#     batch_size: int = 100,
#     budget_usd: float = 5.0,
#     model: str = "gpt-4o-mini",
#     api_key: str | None = None,
# ) -> pd.DataFrame:
#     """
#     Translates Jamaican Patois sentences in a CSV to English.

#     Parameters
#     ----------
#     input_path : str
#         Path to input CSV file (must include a 'patois_sentence' column).
#     output_path : str
#         File path to save translated results.
#     batch_size : int, optional
#         Number of sentences to translate per API call.
#     budget_usd : float, optional
#         Maximum estimated budget before stopping.
#     model : str, optional
#         OpenAI model to use (default 'gpt-4o-mini').
#     include_notes : bool, optional
#         If True, includes idiom notes in the output CSV.
#     api_key : str, optional
#         Explicit OpenAI API key (defaults to OPENAI_API_KEY env var).

#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame with original sentences, translations, and notes.
#     """

#     client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

#     # ---- Pricing constants ----
#     INPUT_RATE = 0.15 / 1_000_000   # $ per input token
#     OUTPUT_RATE = 0.60 / 1_000_000  # $ per output token

#     # ---- Read CSV ----
#     new_outout_column = "english_"+column_to_translate
#     df = pd.read_csv(input_path)
#     if column_to_translate not in df.columns:
#         raise ValueError(f"CSV must contain a '{column_to_translate}' column.")
#     if new_outout_column not in df.columns:
#         df[new_outout_column] = ""

#     total_input_tokens = 0
#     total_output_tokens = 0

#     def translate_batch(batch_texts):
#         """Translate one batch via API."""
#         system_prompt = (
#             "You are a translator from Jamaican Patois to natural Standard American English. "
#             "Keep meaning and tone faithful."
#         )
#         joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch_texts))
#         prompt = (
#             "Translate the following Jamaican Patois sentences into English. "
#              "Return a JSON object with key 'translations' whose value is an array of strings, "
#             "one per input line, same order.\n\n"
#             f"{joined}"
#         )

#         resp = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0,
#             response_format={"type": "json_object"},   # JSON-mode (v2.x syntax)
#         )

#         content = resp.choices[0].message.content
#         try:
#             data = json.loads(content)
#             # Model can return {"translations": [...]} or a bare list → handle both
#             translations = data["translations"] if isinstance(data, dict) and "translations" in data else data
#             if not isinstance(translations, list):
#                 raise ValueError("Unexpected JSON format.")
#         except Exception as e:
#             raise ValueError(f"Bad JSON from model: {e}\nRaw:\n{content[:400]}")
        
#          # Token accounting
#         usage = resp.usage or {}
        
#         in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
#         out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
#         new_eng_df = pd.DataFrame({"english": translations})
        
#         print(new_eng_df.head())

#         return in_tok, out_tok, new_eng_df


#     batches = math.ceil(len(df) / batch_size)
#     print(f"Translating {len(df)} sentences in {batches} batches...")

#     for i in tqdm(range(batches), desc="Translating"):
#         start, end = i * batch_size, (i + 1) * batch_size
#         subset = df.iloc[start:end]
#         texts = subset[column_to_translate].tolist()

#         try:
#             in_tok, out_tok, results = translate_batch(texts)
#         except Exception as e:
#             print(f"\n Batch {i+1} failed: {e}")
#             time.sleep(5)
#             continue

#         total_input_tokens += in_tok
#         total_output_tokens += out_tok
#         est_cost = total_input_tokens * INPUT_RATE + total_output_tokens * OUTPUT_RATE

#         print(f"\nBatch {i+1}: {in_tok+out_tok} tokens → est. ${est_cost:.3f}")
#         if est_cost > budget_usd:
#             print("Budget limit reached — stopping further translation.")
#             break

#         df.loc[start:end-1, new_outout_column] = results["english"]

#         df.to_csv(output_path, index=False)
#         time.sleep(1)

#     total_cost = total_input_tokens * INPUT_RATE + total_output_tokens * OUTPUT_RATE
#     print(f"\n Done! Total estimated cost: ${total_cost:.2f}")
#     print(f"Output saved to {output_path}")

#     return df

def translate_patois_csv(
    input_path: str,
    output_path: str,
    column_to_translate: str,
    batch_size: int = 100,
    budget_usd: float = 5.0,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> pd.DataFrame:

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # Pricing (USD/token)
    INPUT_RATE  = 0.15 / 1_000_000
    OUTPUT_RATE = 0.60 / 1_000_000

    out_col = f"english_{column_to_translate}"

    # ---- Read & clean ----
    df = pd.read_csv(input_path)
    # drop "Unnamed" junk columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if column_to_translate not in df.columns:
        raise ValueError(f"CSV must contain column '{column_to_translate}'.")
    if out_col not in df.columns:
        df[out_col] = ""

    # coerce to strings and handle NaN
    df[column_to_translate] = df[column_to_translate].astype(str).fillna("")
    df[out_col] = df[out_col].astype(str).fillna("")

    # translate only rows still empty (supports resume)
    pending_idx = df.index[df[out_col].str.strip() == ""].tolist()
    if not pending_idx:
        print("Nothing to translate.")
        return df

    total_in = total_out = 0
    batches = math.ceil(len(pending_idx) / batch_size)
    print(f"Translating {len(pending_idx)} rows in {batches} batches…")

    def translate_batch(texts: list[str]) -> tuple[int, int, list[str]]:
        """
        Always returns a list of len(texts), using original text as fallback.
        """
        # JSON input payload — less ambiguity than numbered lines
        payload = {"items": texts}

        system_prompt = (
            "You are a reliable translator from Jamaican Patois to natural Standard American English. "
            "You MUST return strict JSON. For any text you cannot translate, return the original text unchanged. "
            "Preserve order exactly."
        )
        user_prompt = (
            "Translate the following JSON array of Patois sentences. "
            "Return a JSON object with key 'translations' whose value is an array of strings "
            "of EXACTLY the same length and order as the input.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},  # JSON mode (SDK 2.x)
        )

        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
            translations = data["translations"]
        except Exception as e:
            # If parsing fails, fall back to originals for safety
            print(f"JSON parse issue, using originals. Error: {e}\nRaw: {content[:300]}")
            translations = texts

        # Hard guarantees: list, correct length, strings only
        if not isinstance(translations, list):
            translations = texts
        # pad/trim to exact length
        if len(translations) != len(texts):
            if len(translations) < len(texts):
                translations = translations + texts[len(translations):]
            else:
                translations = translations[:len(texts)]
        # ensure all strings, fallback to original if empty/None
        clean = []
        for src, tgt in zip(texts, translations):
            s = "" if tgt is None else str(tgt).strip()
            clean.append(s if s != "" else src)

        usage = resp.usage
        in_tok  = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return in_tok, out_tok, clean

    for b in tqdm(range(batches), desc="Translating"):
        start = b * batch_size
        end   = min(start + batch_size, len(pending_idx))
        idxs  = pending_idx[start:end]
        texts = df.loc[idxs, column_to_translate].tolist()

        try:
            in_tok, out_tok, translations = translate_batch(texts)
        except Exception as e:
            print(f"\nBatch {b+1} failed: {e}")
            time.sleep(2)
            continue

        total_in  += in_tok
        total_out += out_tok
        est_cost = total_in * INPUT_RATE + total_out * OUTPUT_RATE
        print(f"\nBatch {b+1}: tokens_in/out={in_tok}/{out_tok} → est ${est_cost:.3f}")
        if est_cost > budget_usd:
            print("Budget limit reached — stopping.")
            break

        # write results for exactly these rows
        df.loc[idxs, out_col] = translations

        # save progress every batch
        df.to_csv(output_path, index=False)
        time.sleep(0.5)

    total_cost = total_in * INPUT_RATE + total_out * OUTPUT_RATE
    print(f"\nDone. Estimated total cost: ${total_cost:.2f}")
    print(f"Output saved to {output_path}")
    return df

