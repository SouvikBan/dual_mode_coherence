#!/usr/bin/env python3
"""Run annotate_ns.py or annotate_clasp.py on several GPUs.

    python run_parallel.py ns    --gpus 0 1 2 3 --workers-per-gpu 2 -- <script arguments>
    python run_parallel.py clasp --gpus 0 1 2 3 --workers-per-gpu 2 -- <script arguments>

Each worker is a separate process pinned to one GPU (CUDA_VISIBLE_DEVICES).
It loads CorPipe and Stanza once and then takes jobs from the shared job
list: one Natural Stories sentence (all strategies), or one CLASP ID with
one strategy. Jobs with the longest context start first. A claim file in
OUT/.claims/<dataset>/ stops two workers from taking the same job.

Finished jobs are saved as part files, so rerunning the same command only
does what is missing (for example after a crash or a cancelled job). At the
end the launcher assembles the story/ID files and prints what, if anything,
is incomplete. Worker logs: OUT/logs/<dataset>_gpu<g>_w<k>.log.

Only one launcher per output folder at a time: it clears old claims at start.
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# One pipeline per dataset: the one that produced the released
# information-value files. The research checkout also has manual-target and
# all-CorPipe variants, which were the alternatives measured against these two.
SCRIPTS = {"ns": "annotate_ns", "clasp": "annotate_clasp"}
HERE = Path(__file__).resolve().parent


def detect_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [g for g in visible.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("no GPUs found; pass --gpus")


def progress(module, script_args, jobs):
    done = sum(module.job_done(script_args, job) for job in jobs)
    return f"{done}/{len(jobs)} jobs done"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=SCRIPTS)
    parser.add_argument("--gpus", nargs="+", help="GPU ids (default: all visible GPUs)")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="CPU threads per worker (OMP/MKL; also CorPipe --threads unless given)")
    parser.add_argument("--stagger", type=float, default=10,
                        help="seconds between worker starts (avoids simultaneous model loading)")
    parser.add_argument("--report-every", type=float, default=300, help="seconds between progress lines")
    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("put the filter script arguments after --")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    rest = argv[split + 1:]

    sys.path.insert(0, str(HERE))
    module = importlib.import_module(SCRIPTS[args.dataset])
    script_args = module.parse_args(rest)
    if script_args.claim_dir:
        parser.error("do not pass --claim-dir; the launcher sets it")

    claim_dir = script_args.out_dir / ".claims" / args.dataset
    shutil.rmtree(claim_dir, ignore_errors=True)  # claims left by an interrupted run
    if script_args.overwrite:
        module.clear_outputs(script_args)
        rest = [a for a in rest if a != "--overwrite"]
        script_args.overwrite = False
    if "--threads" not in rest:
        rest += ["--threads", str(args.cpu_threads)]

    jobs = module.list_jobs(script_args)
    todo = sum(not module.job_done(script_args, job) for job in jobs)
    gpus = args.gpus or detect_gpus()
    n_workers = min(len(gpus) * args.workers_per_gpu, todo)
    print(f"{args.dataset}: {len(jobs)} jobs, {todo} to do, {n_workers} workers on GPUs {gpus}", flush=True)

    workers = []
    if todo:
        log_dir = script_args.out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        slots = [(gpu, k) for k in range(args.workers_per_gpu) for gpu in gpus][:n_workers]
        try:
            for gpu, k in slots:
                name = f"gpu{gpu}_w{k}"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
                           OPENBLAS_NUM_THREADS=str(args.cpu_threads), TOKENIZERS_PARALLELISM="false")
                log = (log_dir / f"{args.dataset}_{name}.log").open("a")
                command = [sys.executable, str(HERE / f"{SCRIPTS[args.dataset]}.py"), *rest,
                           "--claim-dir", str(claim_dir), "--worker-name", name]
                workers.append((name, subprocess.Popen(command, env=env, stdout=log,
                                                       stderr=subprocess.STDOUT), log))
                print(f"started {name} (log {log.name})", flush=True)
                time.sleep(args.stagger)
            last = time.time()
            while any(process.poll() is None for _, process, _ in workers):
                time.sleep(5)
                if time.time() - last >= args.report_every:
                    running = sum(process.poll() is None for _, process, _ in workers)
                    print(f"{time.strftime('%F %T')}  {progress(module, script_args, jobs)}, "
                          f"{running} workers running", flush=True)
                    last = time.time()
        except KeyboardInterrupt:
            print("stopping workers ...", flush=True)
            for _, process, _ in workers:
                process.terminate()
            for _, process, _ in workers:
                process.wait()
            raise SystemExit(130)
        finally:
            for _, _, log in workers:
                log.close()

    failed_workers = [f"{name} (exit {process.returncode}, see {log.name})"
                      for name, process, log in workers if process.returncode]
    print(progress(module, script_args, jobs), flush=True)
    report = module.assemble_all(script_args)
    not_done = {name: status for name, status in report.items() if status != "done"}
    print(f"{len(report) - len(not_done)}/{len(report)} documents complete", flush=True)
    for name, status in not_done.items():
        print(f"  {name}: {status}", flush=True)
    for line in failed_workers:
        print(f"  worker failed: {line}", flush=True)
    if not_done or failed_workers:
        print("rerun the same command to retry what is missing", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
