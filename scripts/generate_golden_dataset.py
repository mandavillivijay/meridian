"""
One-off script to populate the golden dataset using the DeepSeek API.

DeepSeek's API is OpenAI-compatible, so this uses the `openai` SDK pointed
at DeepSeek's endpoint — no Anthropic or OpenAI account required.

Usage:
    pip install openai
    export DEEPSEEK_API_KEY=sk-...
    python scripts/generate_golden_dataset.py

Reads:  datasets/golden_50_template.json  (prompts + intent labels, no outputs)
Writes: datasets/golden_50.json           (fully populated, ready for MERIDIAN)

Models compared:
    old  →  deepseek-chat      (DeepSeek-V3, general-purpose chat)
    new  →  deepseek-reasoner  (DeepSeek-R1, reasoning model)

This is intentionally a one-time data collection step outside the MERIDIAN
library. MERIDIAN itself has zero API dependencies.
"""

import json
import os
import sys
import time
from pathlib import Path

TEMPLATE_PATH = Path("datasets/golden_50_template.json")
OUTPUT_PATH = Path("datasets/golden_50.json")

OLD_MODEL = "deepseek-chat"
NEW_MODEL = "deepseek-reasoner"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_TOKENS = 512
RETRY_DELAY = 5


def call(client, model: str, prompt: str) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if "rate" in str(exc).lower() and attempt < 2:
                print(f"  Rate limited, retrying in {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)
            else:
                raise
    raise RuntimeError("Exhausted retries")


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("Set DEEPSEEK_API_KEY before running this script.")

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Run: pip install openai")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(template)} prompts from {TEMPLATE_PATH}")
    print(f"Old model : {OLD_MODEL}")
    print(f"New model : {NEW_MODEL}\n")

    results = []
    for i, item in enumerate(template, 1):
        prompt = item["prompt"]
        intent = item["intent"]
        print(f"[{i:02d}/{len(template)}] {intent:<20} {prompt[:60]}")

        old_output = call(client, OLD_MODEL, prompt)
        print(f"  old -> {old_output[:80].replace(chr(10), ' ')}")

        new_output = call(client, NEW_MODEL, prompt)
        print(f"  new -> {new_output[:80].replace(chr(10), ' ')}\n")

        results.append({
            "prompt": prompt,
            "intent": intent,
            "old_output": old_output,
            "new_output": new_output,
        })

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(results)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
