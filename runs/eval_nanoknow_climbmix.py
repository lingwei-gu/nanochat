#!/usr/bin/env python3
"""Evaluate a nanochat checkpoint on NanoKnow ClimbMix QA splits.

The script samples supported/unsupported questions from NQ and SQuAD, generates
closed-book direct answers with a nanochat SFT checkpoint, and optionally scores
the answers with an OpenAI LLM judge.
"""

import argparse
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init
from nanochat.engine import Engine


DEFAULT_BASE_DIR = "/home/l39gu/nanochat-runs/d36_4b_climbmix"
DEFAULT_NANOKNOW_DIR = "/home/l39gu/projects/NanoKnow"


def load_answers(path: Path) -> dict[str, list[str]]:
    answers = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            answers[str(row["qid"])] = [str(a) for a in row["answer"]]
    return answers


def load_topics(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, question = line.split("\t", 1)
            rows.append({"qid": str(qid), "question": question})
    return rows


def sample_rows(rows: list[dict[str, str]], n: int, rng: random.Random) -> list[dict[str, str]]:
    if n < 0 or n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def simple_tokens(text: str) -> list[str]:
    # Mirrors pyserini.eval.evaluate_dpr_retrieval.SimpleTokenizer closely
    # enough for NanoKnow exact-match scoring without requiring pyserini.
    return re.findall(r"[\w]+|[^\s]", text, flags=re.UNICODE | re.MULTILINE)


def exact_contains(prediction: str, answers: list[str]) -> bool:
    pred_tokens = [token.lower() for token in simple_tokens(normalize_text(prediction))]
    if not pred_tokens:
        return False
    for answer in answers:
        gold_tokens = [token.lower() for token in simple_tokens(normalize_text(answer))]
        if not gold_tokens:
            continue
        for i in range(0, len(pred_tokens) - len(gold_tokens) + 1):
            if gold_tokens == pred_tokens[i : i + len(gold_tokens)]:
                return True
        if gold_tokens == pred_tokens:
            return True
    return False


def clean_generation(text: str) -> str:
    for marker in [
        "<|assistant_end|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|bos|>",
    ]:
        text = text.replace(marker, "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def make_chat_direct_prompt(question: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer the question directly with the shortest correct answer. "
                    "Do not explain.\n\n"
                    f"Question: {question}"
                ),
            },
            {"role": "assistant", "content": ""},
        ]
    }


def generate_answers(records: list[dict], args) -> list[dict]:
    os.environ["NANOCHAT_BASE_DIR"] = args.base_dir
    if args.device_type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None and args.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    _ddp, _rank, _local_rank, _world_size, device = compute_init(args.device_type)
    model, tokenizer, meta = load_model(
        "sft",
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)

    generated = []
    for i, record in enumerate(records, 1):
        if args.prompt_style == "nanoknow":
            prompt_text = f"Question: {record['question']}\n\nAnswer:"
            prompt_tokens, _ = tokenizer.render_conversation({"messages": [{"role": "user", "content": prompt_text}]})
        else:
            prompt_tokens = tokenizer.render_for_completion(make_chat_direct_prompt(record["question"]))
        results, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        completion_tokens = results[0][len(prompt_tokens) :]
        prediction = clean_generation(tokenizer.decode(completion_tokens))
        row = dict(record)
        row["prediction"] = prediction
        row["exact_contains"] = exact_contains(prediction, row["answers"])
        generated.append(row)
        print(
            f"[{i:03d}/{len(records):03d}] "
            f"{row['dataset']}/{row['support']} qid={row['qid']} "
            f"exact={row['exact_contains']} pred={prediction[:120]!r}",
            flush=True,
        )

    print(f"Loaded checkpoint metadata step={meta.get('step')}", flush=True)
    return generated


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"Judge did not return JSON: {text[:200]}")


def judge_records(records: list[dict], args) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set; skipping GPT judge.", flush=True)
        for row in records:
            row["judge"] = {"skipped": True, "reason": "OPENAI_API_KEY not set"}
        return records

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed in this environment") from exc

    client = OpenAI(api_key=api_key)
    for i, row in enumerate(records, 1):
        prompt = (
            "You are an impartial judge for closed-book question answering.\n"
            "Determine whether the candidate answer is semantically correct for the question.\n"
            "Accept aliases, minor wording differences, and answers that contain the core reference answer.\n"
            "Reject contradictions, unrelated answers, and missing answers.\n"
            "Return strict JSON only, exactly: "
            '{"correct": <bool>, "explanation": "<short reason>"}\n\n'
            f"Question: {row['question']}\n"
            f"Reference answers: {json.dumps(row['answers'], ensure_ascii=False)}\n"
            f"Candidate answer: {row['prediction']}\n"
        )
        response = client.responses.create(
            model=args.judge_model,
            input=prompt,
            reasoning={"effort": args.model_reasoning_effort},
            text={"format": {"type": "json_object"}},
        )
        text = getattr(response, "output_text", None)
        if text is None:
            text = response.output[0].content[0].text
        row["judge"] = parse_judge_json(text)
        print(
            f"[judge {i:03d}/{len(records):03d}] "
            f"{row['dataset']}/{row['support']} qid={row['qid']} "
            f"correct={row['judge'].get('correct')}",
            flush=True,
        )
        if args.judge_sleep > 0:
            time.sleep(args.judge_sleep)
    return records


def summarize(records: list[dict]) -> dict:
    groups = {}
    for row in records:
        key = f"{row['dataset']}_{row['support']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, rows in sorted(groups.items()):
        exact = sum(bool(r.get("exact_contains")) for r in rows)
        judged = [r for r in rows if isinstance(r.get("judge"), dict) and "correct" in r["judge"]]
        item = {
            "n": len(rows),
            "exact_contains_correct": exact,
            "exact_contains_accuracy": exact / len(rows) if rows else None,
        }
        if judged:
            correct = sum(bool(r["judge"].get("correct")) for r in judged)
            item["judge_n"] = len(judged)
            item["judge_correct"] = correct
            item["judge_accuracy"] = correct / len(judged)
        summary[key] = item
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nanoknow-dir", default=DEFAULT_NANOKNOW_DIR)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--model-tag", default="d36_4b_climbmix_w3")
    parser.add_argument("--step", type=int, default=246)
    parser.add_argument("--per-split", type=int, default=10)
    parser.add_argument("--datasets", default="nq,squad", help="Comma-separated subset of datasets: nq,squad")
    parser.add_argument("--supports", default="supported,unsupported", help="Comma-separated subset of supports: supported,unsupported")
    parser.add_argument("--num-shards", type=int, default=1, help="Split selected records into this many deterministic shards")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to evaluate")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--output-dir", default="/home/l39gu/nanochat-runs/d36_4b_climbmix/nanoknow_eval")
    parser.add_argument("--device-type", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt-style", choices=["chat_direct", "nanoknow"], default="chat_direct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--input-generations", default="", help="Existing JSONL generations to score instead of generating")
    parser.add_argument("--judge-model", default="gpt5.5")
    parser.add_argument("--model-reasoning-effort", default="low")
    parser.add_argument("--judge-sleep", type=float, default=0.0)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_filter = [item.strip() for item in args.datasets.split(",") if item.strip()]
    support_filter = [item.strip() for item in args.supports.split(",") if item.strip()]
    if not dataset_filter or any(item not in {"nq", "squad"} for item in dataset_filter):
        raise ValueError(f"Invalid --datasets value: {args.datasets!r}")
    if not support_filter or any(item not in {"supported", "unsupported"} for item in support_filter):
        raise ValueError(f"Invalid --supports value: {args.supports!r}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    split_suffix = ""
    if set(dataset_filter) != {"nq", "squad"} or set(support_filter) != {"supported", "unsupported"}:
        split_suffix = "_" + "-".join(dataset_filter) + "_" + "-".join(support_filter)
    shard_suffix = f"_shard{args.shard_index:03d}-of-{args.num_shards:03d}" if args.num_shards > 1 else ""
    prefix = f"nanoknow_climbmix_sft_step{args.step:06d}{split_suffix}{shard_suffix}_n{args.per_split}_seed{args.seed}"
    raw_path = outdir / f"{prefix}.generations.jsonl"
    judged_path = outdir / f"{prefix}.judged.jsonl"
    summary_path = outdir / f"{prefix}.summary.json"

    if args.input_generations:
        raw_path = Path(args.input_generations)
        prefix = raw_path.name.removesuffix(".generations.jsonl")
        judged_path = raw_path.with_name(f"{prefix}.judged.jsonl")
        summary_path = raw_path.with_name(f"{prefix}.summary.json")
        generated = [json.loads(line) for line in raw_path.open("r", encoding="utf-8")]
        print(f"Loaded generations: {raw_path}", flush=True)
    else:
        nanoknow = Path(args.nanoknow_dir)
        rng = random.Random(args.seed)
        configs = [(dataset, support) for dataset in dataset_filter for support in support_filter]
        records = []
        for dataset, support in configs:
            answers = load_answers(nanoknow / "questions-and-qrels" / dataset / f"answers.nanoknow-{dataset}.jsonl")
            topics = load_topics(
                nanoknow
                / "questions-and-qrels"
                / dataset
                / "climbmix"
                / f"topics.nanoknow-{dataset}-climbmix.{support}.tsv"
            )
            for row in sample_rows(topics, args.per_split, rng):
                row = dict(row)
                row["dataset"] = dataset
                row["support"] = support
                row["answers"] = answers.get(row["qid"], [])
                if not row["answers"]:
                    raise KeyError(f"Missing answers for {dataset} qid={row['qid']}")
                records.append(row)
        if args.num_shards > 1:
            before = len(records)
            records = [row for i, row in enumerate(records) if i % args.num_shards == args.shard_index]
            print(
                f"Selected shard {args.shard_index}/{args.num_shards}: "
                f"{len(records)} of {before} records",
                flush=True,
            )

        generated = generate_answers(records, args)
        with raw_path.open("w", encoding="utf-8") as f:
            for row in generated:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote generations: {raw_path}", flush=True)

    if args.judge:
        generated = judge_records(generated, args)
        with judged_path.open("w", encoding="utf-8") as f:
            for row in generated:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote judged records: {judged_path}", flush=True)

    summary = summarize(generated)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
