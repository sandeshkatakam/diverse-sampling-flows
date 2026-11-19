# Validation and profiling record

Recorded on 2026-11-19 on eight NVIDIA A40 46 GB GPUs. All model evidence used
physical GPU 0 (`GPU-2c3e5a91-5cf0-8eb2-8e7e-2d94be4cb863`) mapped to logical
`cuda:0`. Raw run directories and launcher logs are intentionally ignored by
Git but remain under `outputs/` and `logs/`.

## Gates

| Gate | Result | Evidence |
|---|---:|---|
| Stage B mathematics | Passed | `outputs/validation/stage-b-math-20261119`; 20 focused tests, 1.90 s |
| Official SANA parity | Passed | `outputs/validation/stage-c-model-math-sdp-20261119`; max/mean error 0 for seeds 17 and 29 |
| RNG independence | Passed | repeated output remained bitwise equal after unrelated RNG draws |
| Full input gradient | Passed | finite norm 0.1610 through transformer, VAE, preprocessing, and CLIP |
| Frozen-weight gradients | Passed | zero generator and CLIP parameter gradients |
| Replay comparison | Passed | monolithic -7.5171, replay -7.4980, relative error 0.2538% under declared 0.5% tolerance |

Generator forward policy is strict deterministic cuBLAS/cuDNN/PyTorch. Gradient
runs disable flash and memory-efficient SDPA and use math SDPA. PyTorch reports
only antialiased bicubic resize backward as nondeterministic; this is the reason
for the separately declared real-gradient tolerance. Toy replay checks remain at
tight floating-point tolerance.

## Calibration and profiles

- Prompt-1 CLIP calibration used all 512 prespecified IID images and 4,096 fixed
  distinct pairs. `bandwidth_sq=0.04606286043022572`; provenance is in
  `artifacts/bandwidths/golden_retriever.json`.
- One effective-set replay update: 5.677 s, 7 generator/feature passes, 96 RK4
  field calls, peak allocated CUDA memory 25,960,532,480 bytes (24.17 GiB).
- One validation set: 1.688 s.
- One four-image IID set through SANA, CLIP, DINOv2, and SigLIP, including model
  loading and image persistence: 15 s launcher wall time.
- One-row 8/16/32 refinement: 6.644 s, 672 field calls, 12 generated images.
  The finest comparison measured normalized latent difference `1.73e-6` and
  CLIP-logdet difference `0.00417`, below declared `1e-3` and `0.02` tolerances.

At the unmodified four-set effective update, early measured updates take about
21–26 seconds. The 300-update pilot is projected at roughly 1.8–2.2 GPU-hours
including three 64-set validations. A 1,000-update main training run projects to
about 6 GPU-hours. The dry full plan contains 72 learned runs (eight prompts,
three initializations, three learned methods): roughly 430 GPU-hours, or about
86 wall hours at five-way concurrency before final evaluation. This projection
must be revised from completed pilot timing before opt-in execution.

## Execution status

The first persistent pilot was stopped after five updates and marked ineligible
because parameter initialization was not yet bound to its recorded namespace.
The implementation now derives and records `parameter_initialization`. A later
partial restart was also stopped before branch integration so the final clean
run can start from a committed tree with complete launch provenance. No result
from either partial run is eligible for reporting. The completed 300-update
pilot and full sweep therefore remain `not_run`; the full sweep stays opt-in.
