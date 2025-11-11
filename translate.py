
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
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def translate_patois_csv(
    input_path: str,
    output_path: str,
    batch_size: int = 100,
    budget_usd: float = 5.0,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Translates Jamaican Patois sentences in a CSV to English.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (must include a 'patois_sentence' column).
    output_path : str
        File path to save translated results.
    batch_size : int, optional
        Number of sentences to translate per API call.
    budget_usd : float, optional
        Maximum estimated budget before stopping.
    model : str, optional
        OpenAI model to use (default 'gpt-4o-mini').
    include_notes : bool, optional
        If True, includes idiom notes in the output CSV.
    api_key : str, optional
        Explicit OpenAI API key (defaults to OPENAI_API_KEY env var).

    Returns
    -------
    pandas.DataFrame
        DataFrame with original sentences, translations, and notes.
    """

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # ---- Pricing constants ----
    INPUT_RATE = 0.15 / 1_000_000   # $ per input token
    OUTPUT_RATE = 0.60 / 1_000_000  # $ per output token

    # ---- Read CSV ----
    df = pd.read_csv(input_path)
    if "patois_sentence" not in df.columns:
        raise ValueError("CSV must contain a 'patois_sentence' column.")
    if "english_translation" not in df.columns:
        df["english_translation"] = ""

    total_input_tokens = 0
    total_output_tokens = 0

    def translate_batch(batch_texts):
        """Translate one batch via API."""
        system_prompt = (
            "You are a translator from Jamaican Patois to natural Standard English. "
            "Keep meaning and tone faithful."
        )
        joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch_texts))
        prompt = (
            "Translate the following Jamaican Patois sentences into English. "
            "Return JSON array with object {english} preserving order:\n\n"
            f"{joined}"
        )
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json"},
        )
        usage = resp.usage
        in_tok, out_tok = usage.input_tokens, usage.output_tokens
        content = resp.output[0].content[0].text
        return in_tok, out_tok, pd.read_json(content)

    batches = math.ceil(len(df) / batch_size)
    print(f"Translating {len(df)} sentences in {batches} batches...")

    for i in tqdm(range(batches), desc="Translating"):
        start, end = i * batch_size, (i + 1) * batch_size
        subset = df.iloc[start:end]
        texts = subset["patois_sentence"].tolist()

        try:
            in_tok, out_tok, results = translate_batch(texts)
        except Exception as e:
            print(f"\n Batch {i+1} failed: {e}")
            time.sleep(5)
            continue

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        est_cost = total_input_tokens * INPUT_RATE + total_output_tokens * OUTPUT_RATE

        print(f"\nBatch {i+1}: {in_tok+out_tok} tokens → est. ${est_cost:.3f}")
        if est_cost > budget_usd:
            print("🚨 Budget limit reached — stopping further translation.")
            break

        df.loc[start:end-1, "english_translation"] = results["english"]

        df.to_csv(output_path, index=False)
        time.sleep(1)

    total_cost = total_input_tokens * INPUT_RATE + total_output_tokens * OUTPUT_RATE
    print(f"\n Done! Total estimated cost: ${total_cost:.2f}")
    print(f"Output saved to {output_path}")

    return df
