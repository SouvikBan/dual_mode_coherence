#!/usr/bin/env python3
"""Generate Gemma-2-9B continuations from sentence-ID contexts."""

import argparse
import csv
import json
import multiprocessing as mp
import re
from collections import defaultdict
from pathlib import Path


MODEL = "google/gemma-2-9b"

STRATEGIES = {
    "ancestral": {"top_k": 0},
    "temp_075": {"top_k": 0, "temperature": 0.75},
    "temp_125": {"top_k": 0, "temperature": 1.25},
    "nucleus_08": {"top_k": 0, "top_p": 0.80},
    "nucleus_085": {"top_k": 0, "top_p": 0.85},
    "nucleus_09": {"top_k": 0, "top_p": 0.90},
    "nucleus_095": {"top_k": 0, "top_p": 0.95},
    "typical_02": {"top_k": 0, "typical_p": 0.20},
    "typical_03": {"top_k": 0, "typical_p": 0.30},
    "typical_085": {"top_k": 0, "typical_p": 0.85},
    "typical_095": {"top_k": 0, "typical_p": 0.95},
}


def story_number(path: Path) -> int:
    return int(re.search(r"story_(\d+)_", path.name).group(1))


def read_sentences(path: Path) -> list[tuple[int, str]]:
    sentences = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            sentences[int(row["sent_id"])].append(row["form"])

    sent_ids = sorted(sentences)
    if sent_ids != list(range(len(sent_ids))):
        raise ValueError(f"{path}: sent_id values must start at 0 and be contiguous")
    return [(sent_id, " ".join(sentences[sent_id])) for sent_id in sent_ids]


def make_contexts(sentences: list[tuple[int, str]]):
    previous = []
    for sent_id, target in sentences:
        context = " " if sent_id == 0 else " ".join(previous)
        yield sent_id, context
        previous.append(target)


def sample(model, tokenizer, inputs, strategy, samples, batch_size, max_new_tokens):
    raws = []
    input_length = inputs["input_ids"].shape[1]
    for start in range(0, samples, batch_size):
        current_batch = min(batch_size, samples - start)
        generated = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=current_batch,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **strategy,
        )
        raws.extend(
            tokenizer.decode(tokens[input_length:], skip_special_tokens=True)
            for tokens in generated
        )
    return raws


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=4)
    return parser.parse_args()


def generate(csv_paths, args, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    for csv_path in csv_paths:
        story_id = story_number(csv_path)
        output_path = args.output_dir / f"story_{story_id}_raw.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for sent_id, context in make_contexts(read_sentences(csv_path)):
                inputs = tokenizer(context, return_tensors="pt").to(device)
                for strategy_id, (name, strategy) in enumerate(STRATEGIES.items()):
                    set_seed(args.seed + story_id * 100_000 + sent_id * 100 + strategy_id)
                    raws = sample(
                        model,
                        tokenizer,
                        inputs,
                        strategy,
                        args.samples,
                        args.batch_size,
                        args.max_new_tokens,
                    )
                    output.write(json.dumps({
                        "sent_id": sent_id,
                        "strategy": name,
                        "raws": raws,
                    }, ensure_ascii=False) + "\n")
        print(f"GPU {device.index}: wrote {output_path}")


def run_gpu(gpu_id, csv_paths, args):
    import torch

    torch.cuda.set_device(gpu_id)
    generate(csv_paths, args, torch.device(f"cuda:{gpu_id}"))


def main():
    args = parse_args()
    all_csv_paths = sorted(
        args.csv_dir.glob("story_*_entity_annotated.csv"),
        key=story_number,
    )
    if not all_csv_paths:
        raise FileNotFoundError(f"no annotated story CSVs in {args.csv_dir}")

    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=run_gpu,
            args=(gpu_id, all_csv_paths[gpu_id::args.gpus], args),
        )
        for gpu_id in range(args.gpus)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failed = [gpu_id for gpu_id, worker in enumerate(workers) if worker.exitcode]
    if failed:
        raise SystemExit(f"generation failed on GPUs: {failed}")


if __name__ == "__main__":
    main()
