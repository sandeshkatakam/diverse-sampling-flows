# Diverse Sampling Flows

Complete remote-CUDA implementation of the Gaussian-preserving prior-flow
experiment in [`experiment_method_remote_cuda.md`](experiment_method_remote_cuda.md).
The frozen primary generator is the pinned one-step Sana-Sprint 0.6B model;
only a preserving prior or learned preserving baseline is optimized.

The implementation reports failed gates, solver drift, unavailable optional
models, and negative results explicitly. A written source file, launched PID,
or nonsignificant marginal test is never reported as a completed experiment or
proof of invariance.

## Environment and cache

All commands are run from this repository. The shared Python 3.11 environment,
model cache, outputs, and logs remain inside the project directory.

```bash
bash launchers/setup.sh
source .venv/bin/activate
bash launchers/setup.sh --dry-run
python -m diverse_sampling_flows.cli.cache --config configs/main.yaml --all-encoders
```

The checked environment uses Python 3.11, PyTorch 2.7.1 with CUDA 12.6,
Diffusers 0.35.1, and Transformers 4.56.0. Exact resolved packages are in
`requirements/constraints.txt`. Normal experiment launchers never upgrade or
reinstall packages.

## Validation gates

Run CPU mathematics before any model process, then select one actually free
physical server GPU for the real-model gate:

```bash
bash scripts/inspect_gpus.sh
bash launchers/validate.sh --stage math --device cpu
bash launchers/validate.sh --stage model --gpu 0
```

The model launcher narrows `CUDA_VISIBLE_DEVICES` to the requested physical ID;
inside Python it is always logical `cuda:0`. It rejects IDs outside `0..7` and
any request that would widen a pre-existing visibility restriction. Generator
forwards require `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN, and
deterministic PyTorch algorithms. Training additionally locks math SDPA.
PyTorch 2.7.1 has no deterministic CUDA backward for antialiased bicubic
resize, so the backward path records that one exception and uses a declared
0.5% real replay-gradient tolerance. Official generator image parity remains
bitwise exact.

Current validation evidence and measured costs are summarized in
[`reports/validation/README.md`](reports/validation/README.md).

## Pilot and training

The bounded pilot uses prompt 1, initialization 17, 300 optimizer updates,
four independent sets per update, and one memory microbatch at a time:

```bash
bash launchers/pilot.sh --gpu 0

# Persistent remote session
tmux new-session -d -s dsf-pilot \
  'cd /absolute/path/to/diverse-sampling-flows && bash launchers/pilot.sh --gpu 0'

# Main training and exact resume
bash launchers/train.sh --gpu 0 --config configs/main.yaml \
  --method skew_trained --run-id retriever-s17
bash launchers/train.sh --gpu 0 --config configs/main.yaml \
  --method skew_trained --resume outputs/train/retriever-s17/latest.pt \
  --run-id retriever-s17-resumed
```

Checkpoints `initial.pt`, `latest.pt`, and validation-selected `best.pt` contain
model, EMA, optimizer, scheduler, global update, RNG namespaces/counters, call
counters, and best-validation metadata. Parameter initialization, calibration,
training, validation, refinement, test, IID reference, projection, and
visualization use separate recorded seed namespaces.

## Evaluation, refinement, and aggregation

Every generated slot and image is retained. Learned methods require an explicit
checkpoint; fixed methods do not.

```bash
bash launchers/evaluate.sh --gpu 0 --method iid --sets 256 \
  --run-id retriever-iid
bash launchers/evaluate.sh --gpu 0 --method skew_trained \
  --checkpoint outputs/train/retriever-s17/best.pt --sets 256 \
  --refinement --refinement-sets 256 --run-id retriever-skew

bash launchers/aggregate.sh \
  --input outputs/evaluate/retriever-iid/raw.jsonl \
  --input outputs/evaluate/retriever-skew/raw.jsonl \
  --output reports/pilot/method_table.json
```

Evaluation records the exact CLIP RBF-logdet objective, CLIP and DINOv2
pairwise diversity, per-slot SigLIP alignment, raw-slot prior marginals, seed
manifest, images, and optional 8/16/32-step refinement. Aggregation regenerates
JSON/CSV tables, confidence intervals, a method plot, complete image grids, and
explicit `not_run`/`unavailable` states from raw records.

## Opt-in sweep and concurrency

The full sweep is never started implicitly. Its default is a dry plan with
unique run IDs, physical GPU assignments, and output directories:

```bash
bash launchers/sweep.sh --gpus 0,1,2,3,4 --max-concurrency 5
# Only after reviewing the printed cost and assignments:
bash launchers/sweep.sh --gpus 0,1,2,3,4 --max-concurrency 5 --execute
```

Concurrency runs independent one-GPU processes; it does not change effective
batch size, optimizer updates, seeds, or the experimental protocol. The user
requested ceiling is five concurrent GPUs.

## Layout, tensors, and records

- `src/diverse_sampling_flows/`: generators, priors, conditioners, solvers,
  baselines, objectives, training, evaluation, reproducibility, and CLIs.
- `configs/`: strict typed module configuration with CLI-last overrides.
- `launchers/`: direct environment/GPU selection and persistent logging.
- `outputs/`: ignored raw runs, images, checkpoints, and resolved configs.
- `logs/`: ignored exact commands, server GPU UUIDs, PIDs, stdout/stderr, and
  exit statuses.
- `artifacts/`: ignored fixed calibration bandwidths and cache manifests.

Sets use `[B,K,C,H,W]`, where `B` contains independent rows and `K=4` contains
slots from one shared seed for the proposed method. The field itself sees
`[B,C,H,W]` and never couples independent rows. Flatten `B*K` only at frozen
generator/encoder boundaries and restore it before the set loss.
