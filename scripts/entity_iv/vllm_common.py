"""Shared sampling and GPU-worker setup for the two raw generators."""

import hashlib
import json
import multiprocessing as mp
import os


STRATEGIES = {
    "ancestral": {},
    "temp_075": {"temperature": 0.75},
    "temp_125": {"temperature": 1.25},
    "nucleus_08": {"top_p": 0.80},
    "nucleus_085": {"top_p": 0.85},
    "nucleus_09": {"top_p": 0.90},
    "nucleus_095": {"top_p": 0.95},
    "typical_02": {"extra_args": {"typical_p": 0.20}},
    "typical_03": {"extra_args": {"typical_p": 0.30}},
    "typical_085": {"extra_args": {"typical_p": 0.85}},
    "typical_095": {"extra_args": {"typical_p": 0.95}},
}


def add_generation_args(parser):
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--gpus", type=int, default=4,
                        help="Number of visible GPUs; one model replica per GPU")
    parser.add_argument("--samples", "--raws-per-strategy", dest="samples",
                        type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Maximum active sequences per GPU (vLLM max_num_seqs)")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Optional context-window cap; defaults to model limit")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES),
                        default=list(STRATEGIES))
    parser.add_argument("--overwrite", action="store_true")


def validate_args(parser, args):
    for name in ("gpus", "samples", "batch_size", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_model_len is not None and args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be in (0, 1)")
    if len(set(args.strategies)) != len(args.strategies):
        parser.error("--strategies must not contain duplicates")


def run_workers(worker, jobs, args):
    """Spawn independent workers, respecting an existing CUDA_VISIBLE_DEVICES."""
    if not jobs:
        print("No files to generate. Use a new output directory or --overwrite.")
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = ([x.strip() for x in visible.split(",") if x.strip()]
               if visible is not None else [str(i) for i in range(args.gpus)])
    if len(devices) < args.gpus or len(set(devices[:args.gpus])) < args.gpus:
        raise ValueError(f"Need {args.gpus} distinct GPUs; CUDA_VISIBLE_DEVICES={visible!r}")
    devices = devices[:min(args.gpus, len(jobs))]
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(gpu, jobs[i::len(devices)], args))
        for i, gpu in enumerate(devices)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [(gpu, p.exitcode) for gpu, p in zip(devices, processes) if p.exitcode]
    if failed:
        raise SystemExit(f"Generation worker(s) failed (GPU, exit code): {failed}")


def create_llm(gpu, args):
    # Set visibility before importing vLLM or torch in this spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM

    processors = []
    if any(name.startswith("typical_") for name in args.strategies):
        from typical_sampling import TypicalLogitsProcessor
        processors.append(TypicalLogitsProcessor)
    return LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=True,
        generation_config="vllm",
        logits_processors=processors,
        seed=args.seed,
    )


def encode_context(llm, context, max_new_tokens):
    # Match tokenizer(context) in the originals, including BOS if configured.
    token_ids = llm.get_tokenizer().encode(context or " ", add_special_tokens=True)
    limit = llm.model_config.max_model_len
    if len(token_ids) + max_new_tokens > limit:
        raise ValueError(
            f"Context exceeds model window: {len(token_ids)} prompt + "
            f"{max_new_tokens} new > {limit}"
        )
    return token_ids


def sample_raws(llm, token_ids, strategy, key, args, strip=False):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    # n > 1 gets seed + sample_index internally in vLLM. The base seed depends
    # on the context and strategy, not which GPU receives the job.
    seed_key = f"{args.seed}|{args.model}|{key}|{strategy}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:8], "big") % (2**63 - 1)
    options = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
    options.update(STRATEGIES[strategy])
    params = SamplingParams(
        n=args.samples, max_tokens=args.max_new_tokens, seed=seed,
        detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY,
        **options,
    )
    result = llm.generate(
        [{"prompt_token_ids": token_ids}], params, use_tqdm=False
    )[0]
    outputs = sorted(result.outputs, key=lambda output: output.index)
    if len(outputs) != args.samples:
        raise RuntimeError(f"{key}/{strategy}: expected {args.samples} outputs, got {len(outputs)}")
    tokenizer = llm.get_tokenizer()
    raws = [tokenizer.decode(output.token_ids, skip_special_tokens=True) for output in outputs]
    return [text.strip() for text in raws] if strip else raws


def write_jsonl_atomic(path, rows):
    """Consume rows as generated; publish the file only when it is complete."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
