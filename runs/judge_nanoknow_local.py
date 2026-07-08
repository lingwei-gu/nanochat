#!/usr/bin/env python3
"""Judge NanoKnow generation JSONL files with a local instruct model."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_judge_output(text: str) -> dict:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return {
            "correct": bool(data.get("correct", False)),
            "explanation": str(data.get("explanation", ""))[:500],
            "raw": text,
        }
    except Exception:
        lower = text.lower()
        if re.search(r'"?correct"?\s*:\s*true', lower):
            return {"correct": True, "explanation": "parsed from raw text", "raw": text[:1000]}
        if re.search(r'"?correct"?\s*:\s*false', lower):
            return {"correct": False, "explanation": "parsed from raw text", "raw": text[:1000]}
        return {"correct": False, "explanation": "JSON_PARSE_ERROR", "raw": text[:1000]}


def build_prompt(row: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an impartial judge for closed-book question answering. "
                "Decide whether the candidate answer correctly answers the question. "
                "Accept aliases and minor wording differences. "
                "If the candidate contains the correct answer but also contradicts it, mark incorrect. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {row['question']}\n"
                f"Reference answers: {json.dumps(row['answers'], ensure_ascii=False)}\n"
                f"Candidate answer: {row['prediction']}\n\n"
                'Respond exactly as: {"correct": <true|false>, "explanation": "<short reason>"}'
            ),
        },
    ]


def render_chat(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.open("r", encoding="utf-8")]
    done = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
    rows_to_score = rows[done:]
    print(f"Loaded {len(rows)} rows from {input_path}; resuming at {done}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.eval()
    model_device = next(model.parameters()).device

    with output_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(rows_to_score), args.batch_size):
            batch = rows_to_score[start : start + args.batch_size]
            prompts = [render_chat(tokenizer, build_prompt(row)) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model_device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(
                generated[:, encoded.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            for row, completion in zip(batch, completions):
                judged = dict(row)
                judged["judge"] = parse_judge_output(completion)
                out.write(json.dumps(judged, ensure_ascii=False) + "\n")
                out.flush()
            scored = done + start + len(batch)
            print(f"[{scored:05d}/{len(rows):05d}] wrote {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
