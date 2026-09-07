#!/usr/bin/env python3
"""Generate one shared alternative pool for the five targets of each CLASP ID."""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


LANGUAGES = ("English", "Czech", "German", "Spanish", "French")
REQUIRED_COLUMNS = {"ID", "Language", "Sentence", "Pre-Context"}

# The same decoding strategies used for the Natural Stories alternatives.
STRATEGIES = {
    "ancestral": {},
    "temp_075": {"temperature": 0.75},
    "temp_125": {"temperature": 1.25},
    "nucleus_08": {"top_p": 0.80},
    "nucleus_085": {"top_p": 0.85},
    "nucleus_09": {"top_p": 0.90},
    "nucleus_095": {"top_p": 0.95},
    "typical_02": {"typical_p": 0.20},
    "typical_03": {"typical_p": 0.30},
    "typical_085": {"typical_p": 0.85},
    "typical_095": {"typical_p": 0.95},
}


def clean(text):
    """Collapse CSV whitespace without changing the words."""
    return " ".join((text or "").split())


def load_clasp(path, allow_partial=False):
    """Load one context and five ordered targets for every numeric CLASP ID."""
    groups = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                clasp_id = int(row["ID"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid ID {row.get('ID')!r}"
                ) from error

            language = clean(row["Language"])
            sentence = clean(row["Sentence"])
            context = clean(row["Pre-Context"])
            if not language or not sentence:
                raise ValueError(f"line {line_number}: empty Language or Sentence")
            groups[clasp_id].append((language, sentence, context))

    if not allow_partial and set(groups) != set(range(100)):
        missing = sorted(set(range(100)) - set(groups))
        extra = sorted(set(groups) - set(range(100)))
        raise ValueError(f"expected IDs 0..99; missing={missing}, extra={extra}")

    records = []
    for clasp_id in sorted(groups):
        rows = groups[clasp_id]
        by_language = {language: (sentence, context) for language, sentence, context in rows}
        if len(rows) != 5 or set(by_language) != set(LANGUAGES):
            found = [language for language, _, _ in rows]
            raise ValueError(
                f"ID {clasp_id}: expected exactly {list(LANGUAGES)}, found {found}"
            )

        contexts = {context for _, context in by_language.values()}
        if len(contexts) != 1:
            raise ValueError(f"ID {clasp_id}: the five rows have different contexts")

        records.append({
            "clasp_id": clasp_id,
            "context": contexts.pop(),
            "targets": [
                {"language": language, "text": by_language[language][0]}
                for language in LANGUAGES
            ],
        })
    return records


def stable_seed(model_name, clasp_id, strategy):
    """Give every ID/strategy pair a repeatable but independent seed."""
    key = f"{model_name}|{clasp_id}|{strategy}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def generate_raws(model, tokenizer, prompt_ids, strategy, count, batch_size,
                  max_new_tokens, seed):
    """Generate `count` continuations in small batches to control GPU memory."""
    import torch
    from transformers import set_seed

    attention_mask = torch.ones_like(prompt_ids)
    generation_args = {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        **STRATEGIES[strategy],
    }

    raws = []
    for batch_number, start in enumerate(range(0, count, batch_size)):
        size = min(batch_size, count - start)
        set_seed((seed + batch_number) % (2**32))
        with torch.inference_mode():
            output = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                num_return_sequences=size,
                **generation_args,
            )
        continuation = output[:, prompt_ids.shape[1]:]
        raws.extend(
            tokenizer.decode(tokens, skip_special_tokens=True).strip()
            for tokens in continuation
        )
    return raws


def write_jsonl_atomic(path, rows):
    """Write a complete ID file, then rename it into place."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def worker(device, records, args):
    """Load one model on one device and generate the IDs assigned to it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device).eval()

    model_limit = getattr(model.config, "max_position_embeddings", None)
    for index, record in enumerate(records, start=1):
        output_path = args.output_dir / f"clasp_{record['clasp_id']}_raw.jsonl"
        if output_path.exists() and not args.overwrite:
            print(f"[{device}] skip {output_path.name}", flush=True)
            continue

        # The paper conditions blank-context items on one space.
        prompt = record["context"] or " "
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if model_limit and prompt_ids.shape[1] + args.max_new_tokens > model_limit:
            raise ValueError(
                f"CLASP ID {record['clasp_id']} exceeds the model window: "
                f"{prompt_ids.shape[1]} prompt + {args.max_new_tokens} new > {model_limit}"
            )

        rows = []
        for strategy in args.strategies:
            raws = generate_raws(
                model,
                tokenizer,
                prompt_ids,
                strategy,
                args.raws_per_strategy,
                args.batch_size,
                args.max_new_tokens,
                stable_seed(args.model, record["clasp_id"], strategy),
            )
            rows.append({
                "clasp_id": record["clasp_id"],
                "context": record["context"],
                "targets": record["targets"],
                "strategy": strategy,
                "raws": raws,
            })

        write_jsonl_atomic(output_path, rows)
        print(
            f"[{device}] {index}/{len(records)} wrote {output_path.name}",
            flush=True,
        )


def main():
    """Validate arguments, split CLASP IDs across devices, and start workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clasp-data", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU IDs, or 'cpu'")
    parser.add_argument("--raws-per-strategy", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.raws_per_strategy < 1 or args.batch_size < 1:
        parser.error("--raws-per-strategy and --batch-size must be positive")

    records = load_clasp(args.clasp_data, args.allow_partial)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus.strip().lower() == "cpu":
        devices = ["cpu"]
    else:
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpu_ids:
            parser.error("--gpus must contain at least one GPU ID, or 'cpu'")
        devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]

    print(
        f"{len(records)} IDs x {len(args.strategies)} strategies x "
        f"{args.raws_per_strategy} raws = "
        f"{len(records) * len(args.strategies) * args.raws_per_strategy} generations",
        flush=True,
    )

    shards = [records[index::len(devices)] for index in range(len(devices))]
    if len(devices) == 1:
        worker(devices[0], shards[0], args)
        return

    import torch.multiprocessing as multiprocessing

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(device, shard, args))
        for device, shard in zip(devices, shards)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise SystemExit(f"generation worker(s) failed with exit codes {failed}")


if __name__ == "__main__":
    main()
