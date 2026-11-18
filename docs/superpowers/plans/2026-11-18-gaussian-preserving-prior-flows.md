# Gaussian-Preserving Prior Flows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and pilot the complete modular Gaussian-preserving prior-flow experiment specified in `experiment_method_remote_cuda.md`, while keeping the pretrained image generator frozen and making the full comparison sweep an explicit opt-in.

**Architecture:** A typed OmegaConf configuration composes a frozen tensor-only Sana-Sprint adapter, exact structured Gaussian-preserving priors or preserving baselines, differentiable image-feature objectives, replay-based training, and streaming evaluation. Thin Python CLIs provide the implementation boundary; shell launchers own environment activation, explicit CUDA visibility, logs, PIDs, and concurrency. CPU mathematical gates precede single-GPU model parity/gradient profiling and the bounded 300-update pilot.

**Tech Stack:** Python 3.11, PyTorch with CUDA 12.x wheels compatible with driver 560.35.03, Diffusers 0.35.1 when compatible, Transformers, Accelerate, OmegaConf, Safetensors, NumPy/SciPy/Pandas, scikit-learn, TensorBoard, Pillow, pytest, Ruff, and Hugging Face Hub.

**Spec:** `experiment_method_remote_cuda.md`

## Global Constraints

- Keep all source, environments, caches, outputs, and scripts inside the tracked project directory; never edit or rely on the nested untracked checkout.
- The primary generator is `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`, 1024x1024, deterministic one-step schedule `[1.57080, 0]`, guidance 4.5, standardized latent `[32,32,32]`, and pinned immutable revisions.
- The generator, VAE, text encoder, CLIP, DINOv2, and SigLIP parameters are frozen; only prior/baseline parameters train.
- Use `[B,K,C,H,W]` for sets and slots, `[B,C,H,W]` inside the field, `K=4`, and one shared seed per proposed-method set.
- Implement `w = Div K - Kz` analytically with context-only conditioning; do not materialize image-scale matrices or use stochastic divergence estimates.
- Use float32 prior/RK4 arithmetic, bfloat16 generator transformer/VAE where validated, float32 encoder features, and float64 small kernels and mathematical checks.
- Calibration, initialization, training, validation, refinement, final test, IID reference, projection, and visualization randomness have separate recorded namespaces.
- Full comparisons, architecture sweeps, and FLUX replication remain opt-in; implementation automatically proceeds only through validation and the 300-update pilot.
- Every model run sets and verifies `CUDA_VISIBLE_DEVICES`; logical `cuda:0` maps to one recorded server GPU ID and UUID.
- Do not claim invariance from finite diagnostics, empirical success from source-only checks, or completion for a process that merely started.

## Review Focus

- A pre-existing restricted `CUDA_VISIBLE_DEVICES` must be honored and invalid/widening selections must fail before Python starts; Task 10 tests visibility resolution and dry-run launch plans.
- A conditioner must never see active coordinates through normalization, pooling, residuals, or attention; Task 3 tests zero excluded-input derivatives and a deliberate leak failure.
- Resume must restore RNG namespaces and counters exactly rather than repeating or skipping training rows; Task 7 compares uninterrupted and resumed seed progressions.
- One-step generator output must be independent of scheduler RNG and numerically match the pinned official tensor order; Task 8 tests fixed-seed parity and repeated RNG perturbations.
- Cholesky or finite-value failures must be recorded and stop/reject the affected run without method-specific jitter changes; Tasks 5 and 7 test explicit failure records.

---

### Task 1: Project environment, package shell, and strict configuration

**Files:**
- Create: `pyproject.toml`, `requirements/base.in`, `requirements/constraints.txt`, `.gitignore`, `README.md`
- Create: `src/diverse_sampling_flows/__init__.py`, `src/diverse_sampling_flows/config.py`, `src/diverse_sampling_flows/types.py`
- Create: `configs/main.yaml`, `configs/model/sana_sprint_1step.yaml`, `configs/model/flux_schnell.yaml`, `configs/prior/masked_skew.yaml`, `configs/conditioner/{resnet,mlp,transformer}.yaml`, `configs/objective/clip_logdet.yaml`, `configs/training/default.yaml`, `configs/evaluation/default.yaml`, `configs/runtime/remote_cuda.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: Path, overrides: Sequence[str]) -> DictConfig`, structured dataclass schemas, and shared tensor-shape aliases used by all later tasks.

- [ ] Write failing tests proving relative include order, CLI-last overrides, unknown-key rejection, complete module sections, checkpoint revision requirements for executable model runs, and resolved-config serialization.
- [ ] Run `pytest tests/test_config.py -v` and confirm failures are missing imports/implementations.
- [ ] Add the package metadata, pinned input dependencies, strict structured OmegaConf loader, full module YAMLs, repository ignores, and README skeleton.
- [ ] Create `.venv` with Python 3.11, install a driver-compatible CUDA PyTorch build and project dependencies, resolve Diffusers/Transformers compatibility, and export exact pins to `requirements/constraints.txt` without upgrading on normal experiment reruns.
- [ ] Run `python -m pip check`, `python -c 'import torch, diffusers, transformers, omegaconf'`, `pytest tests/test_config.py -v`, and record exact Python/PyTorch/CUDA/package versions.
- [ ] Commit with `git add pyproject.toml requirements configs src tests README.md .gitignore && git commit -m 'build: add strict experiment configuration and environment'`.

### Task 2: Reproducibility utilities, metadata, logging, and atomic checkpoints

**Files:**
- Create: `src/diverse_sampling_flows/utils/{seeding,checkpointing,logging,metadata}.py`
- Create: `tests/test_reproducibility.py`, `tests/test_checkpointing.py`

**Interfaces:**
- Produces: `SeedStreams`, `atomic_torch_save`, `save_checkpoint`, `load_checkpoint`, `JsonlLogger`, `capture_runtime_metadata`, and immutable run IDs/output layouts.

- [ ] Write failing tests for disjoint deterministic namespaces, state/counter round trips, atomic replacement without partial targets, checkpoint completeness, token redaction, unique run IDs, and exact launch/config metadata.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement seed generators/manifests, RNG state capture/restore, atomic temporary-file checkpoint writes, JSONL/CSV/TensorBoard logging, dirty-diff/commit capture, prompt hashes, package/hardware/GPU metadata, and model revision/hash records.
- [ ] Run focused tests and `ruff check src tests`.
- [ ] Commit with `git add src/diverse_sampling_flows/utils tests && git commit -m 'feat: add reproducible run and checkpoint utilities'`.

### Task 3: Masks, interchangeable context-only conditioners, and exact masked-skew field

**Files:**
- Create: `src/diverse_sampling_flows/priors/{masks,masked_skew}.py`
- Create: `src/diverse_sampling_flows/conditioners/{base,mlp,resnet,transformer,factory}.py`
- Create: `tests/test_masks.py`, `tests/test_masked_skew.py`, `tests/test_conditioners.py`

**Interfaces:**
- Produces: `MaskGroup`, `build_channel_mask_groups`, `CoefficientConditioner.coefficients(...)`, `build_conditioner`, and `MaskedSkewField.forward(t, z, prompt_context=None) -> Tensor`.

- [ ] Write float64 failing tests for complete/non-overlapping active coverage, saved permutations, bounded-head values and non-dead initialization, analytic pair velocity, dense `Div K-Kz` agreement, `div w-z^T w`, exact rotation at `alpha=0`, nonzero radial tanh motion, and zero derivative with respect to excluded active inputs; include a leaky test double that must fail the derivative assertion.
- [ ] Run the focused tests and confirm failures.
- [ ] Implement masks and the shared bounded-head/ridge algebra with the exact sign and all groups reading the same unmodified state.
- [ ] Implement context-only MLP, residual CNN, and 2x2-patch transformer factories; keep bounds and velocity algebra outside architecture modules.
- [ ] Run focused tests in float64, then a `[B,32,32,32]` CPU smoke test for every conditioner.
- [ ] Commit with `git add src/diverse_sampling_flows/{priors,conditioners} tests && git commit -m 'feat: implement exact masked Gaussian-preserving field'`.

### Task 4: Pure RK4 solver and preserving baseline families

**Files:**
- Create: `src/diverse_sampling_flows/solvers/rk4.py`
- Create: `src/diverse_sampling_flows/baselines/{sampling,fixed_rotation,learned_orthogonal,conditional_rotation}.py`
- Create: `tests/test_rk4.py`, `tests/test_baselines.py`

**Interfaces:**
- Produces: `integrate_rk4(field, initial, checkpoint_times, steps_per_interval) -> Tensor[B,K,C,H,W]`, `sample_iid`, `sample_balanced_gaussian`, `fixed_rotation_slots`, `LearnedOrthogonalCoupling`, and `ConditionalRotationCoupling`.

- [ ] Write failing tests for exact checkpoint alignment and differentiability, decreasing harmonic-rotation and small-map Jacobian error at 8/16/32 steps, IID/balanced analytic covariance, quarter-turn uniqueness, orthogonality/determinant-one pair maps, sequential noncommuting residual maps, context-conditional Gaussian preservation, and the independent-seed negative control shape/recipe.
- [ ] Run focused tests and confirm failures.
- [ ] Implement the pure fixed-step solver without mutable scheduler state and all mandatory preserving prior constructions without dense image-scale matrices.
- [ ] Run focused tests, including a central finite-difference parameter-gradient test through a tiny solver.
- [ ] Commit with `git add src/diverse_sampling_flows/{solvers,baselines} tests && git commit -m 'feat: add RK4 integration and preserving baselines'`.

### Task 5: Differentiable objectives and frozen metric encoders

**Files:**
- Create: `src/diverse_sampling_flows/objectives/{clip_logdet,preprocessing,encoders}.py`
- Create: `src/diverse_sampling_flows/evaluation/diversity.py`
- Create: `tests/test_objectives.py`, `tests/test_diversity.py`

**Interfaces:**
- Produces: `FrozenImageEncoder`, `rbf_logdet_score(features, bandwidth_sq, epsilon)`, `calibrate_bandwidth`, pairwise diversity metrics, and SigLIP prompt-alignment summaries.

- [ ] Write failing tests for `[B,K,D]` normalization, exact float64 RBF/diagonal-normalized Cholesky logdet, score upper bound, fixed seeded calibration pairs/provenance, explicit Cholesky failure records, mean/minimum pair metrics, and differentiable tensor-only resize/crop/normalization.
- [ ] Run focused tests and confirm failures.
- [ ] Implement frozen CLIP projected-image, DINOv2 CLS, and SigLIP prompt-score adapters using tensor preprocessing and no PIL/NumPy/detach on training paths.
- [ ] Implement calibration and raw set/slot metrics with fixed bandwidth/jitter and saved provenance.
- [ ] Run focused tests and a local toy-encoder gradient smoke test.
- [ ] Commit with `git add src/diverse_sampling_flows/{objectives,evaluation} tests && git commit -m 'feat: add differentiable diversity objectives and metrics'`.

### Task 6: Streaming marginal and solver-refinement diagnostics

**Files:**
- Create: `src/diverse_sampling_flows/evaluation/{marginals,refinement,statistics}.py`
- Create: `tests/test_marginals.py`, `tests/test_refinement.py`

**Interfaces:**
- Produces: `PriorMarginalAccumulator`, projected sliced/MMD diagnostics, IID-null calibration, output-feature diagnostics, `run_refinement_study`, and small-dimensional numerical map identity checks.

- [ ] Write failing tests for per-raw-slot streaming means/variances/radial statistics, fixed independent 64-D projections, unbiased negative MMD retention, IID split null ranges including maximum-statistic alarms, normalized latent refinement differences, and Gaussian change-of-variables residuals without image-scale Jacobians.
- [ ] Run focused tests and confirm failures.
- [ ] Implement streaming prior/output diagnostics and 8/16/32 refinement reporting with declared `1e-3` latent and `0.02` CLIP tolerances.
- [ ] Run focused tests on seeded Gaussian data and a known drifting map.
- [ ] Commit with `git add src/diverse_sampling_flows/evaluation tests && git commit -m 'feat: add marginal and refinement diagnostics'`.

### Task 7: Feature replay, EMA, optimizer construction, and resumable trainer

**Files:**
- Create: `src/diverse_sampling_flows/training/{feature_replay,ema,optim,trainer}.py`
- Create: `tests/test_feature_replay.py`, `tests/test_trainer.py`

**Interfaces:**
- Produces: `feature_replay_backward`, `ExponentialMovingAverage`, `build_adamw`, `Trainer.train`, `Trainer.resume`, eligibility-based best-checkpoint selection, and generator/field-call counters.

- [ ] Write failing tests comparing monolithic and replay gradients through a tiny deterministic generator/encoder, excluding anchored slot zero, scaling accumulation once, weight-decay exclusions, cosine warmup, EMA updates, finite-failure stop records, validation-only selection, and uninterrupted-versus-resumed RNG/optimizer/scheduler progression.
- [ ] Run focused tests and confirm failures.
- [ ] Implement the full-set detached-feature gradient followed by deterministic per-slot VJP replay and one backward through all trainable prior checkpoints.
- [ ] Implement the trainer for skew and learned baselines with fresh seed rows, clipping/logging, EMA, initial/latest/best atomic checkpoints, and exact resume.
- [ ] Run focused tests and an end-to-end toy training smoke test showing finite loss and nonzero parameter updates.
- [ ] Commit with `git add src/diverse_sampling_flows/training tests && git commit -m 'feat: add replay training and exact resumption'`.

### Task 8: Frozen Sana-Sprint adapter and optional FLUX adapter

**Files:**
- Create: `src/diverse_sampling_flows/generators/{base,sana_sprint,flux_schnell}.py`
- Create: `tests/test_generator_contract.py`, `tests/test_sana_tensor_math.py`

**Interfaces:**
- Produces: `FrozenGeneratorAdapter`, `SanaSprintAdapter.from_pretrained`, cached conditioning, `generate(standardized_noise, differentiable)`, and an access-gated `FluxSchnellAdapter`.

- [ ] Write CPU failing tests for standard-noise shape, exact one-step scalar/tensor operation order, `sigma_data=0.5` versus VAE scaling `0.41407`, freeze/eval enforcement, differentiable/no-grad branches, conditioning-cache reuse, and deterministic terminal update independent of supplied RNG.
- [ ] Run CPU tests and confirm failures.
- [ ] Inspect the installed pinned Diffusers Sana-Sprint pipeline and SCM scheduler; record source hashes and adapt the spec formula only where pinned source requires a documented compatibility correction.
- [ ] Implement direct frozen component calls, official casts/conditioning/postprocessing, config assertions, and immutable revisions; implement FLUX packing/schedule separately and report gated access as unavailable rather than substituting a model.
- [ ] On one selected free GPU, cache exact revisions and compare adapter versus official tensor output for two seeds at target `1e-5`; test finite latent-to-transformer-to-VAE-to-CLIP gradients, frozen-weight non-gradients, and sampler RNG independence.
- [ ] Commit with `git add src/diverse_sampling_flows/generators tests && git commit -m 'feat: add frozen differentiable generator adapters'`.

### Task 9: Command-line workflows and aggregation

**Files:**
- Create: `src/diverse_sampling_flows/cli/{common,validate,cache,pilot,train,evaluate,aggregate}.py`
- Create: `tests/test_cli.py`, `tests/test_aggregation.py`

**Interfaces:**
- Produces: module CLIs for setup/cache, staged validation, training/resume, evaluation/refinement, and deterministic regeneration of tables/plots/grids from raw records.

- [ ] Write failing CLI tests for strict overrides, stage/device compatibility, run-directory isolation, checkpoint argument validation, method registry completeness, raw slot preservation, and non-fabricated unavailable/not-run report states.
- [ ] Run focused tests and confirm failures.
- [ ] Implement thin CLIs wiring Tasks 1–8, all mandatory methods, seed recipes, calibration, validation eligibility, raw result persistence, hierarchical/bootstrap aggregation, predetermined complete grids, and concise failure analysis.
- [ ] Run CLI help/import smoke tests plus toy CPU validate/train/evaluate/aggregate workflows.
- [ ] Commit with `git add src/diverse_sampling_flows/cli tests && git commit -m 'feat: add experiment command workflows'`.

### Task 10: Direct remote launchers and controlled GPU concurrency

**Files:**
- Create: `launchers/{common,setup,validate,pilot,train,evaluate,aggregate,sweep}.sh`
- Create: `scripts/{inspect_gpus,record_command}.sh`
- Create: `tests/test_launchers.py`

**Interfaces:**
- Produces: the exact shell interfaces required by Section 11, explicit host-GPU-to-logical-device resolution, persistent logs/PIDs/status files, and a dry-run-by-default opt-in sweep scheduler.

- [ ] Write failing tests for launcher help/dry-runs, shared `.venv` invocation, CPU math without CUDA, required `--gpu` for model/pilot, allowed-ID and pre-existing-visibility checks, UUID recording, distinct output directories, max-concurrency enforcement, dry-run sweep default, and child exit-status collection.
- [ ] Run focused tests and confirm failures.
- [ ] Implement POSIX/Bash launchers that never install per worker, set visibility before Python, use logical `cuda:0`, capture exact commands/logs/PIDs/statuses, and optionally emit a documented `tmux` command.
- [ ] Run launcher tests and shell syntax checks; dry-run a five-GPU sweep plan without executing it.
- [ ] Commit with `git add launchers scripts tests && git commit -m 'feat: add direct CUDA launch and sweep controls'`.

### Task 11: Validation gates, profiling, and bounded pilot

**Files:**
- Modify: `README.md`
- Create: `reports/validation/README.md` and generated run records under ignored `outputs/`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence for Stages B/C, a locked pilot recipe, a 300-update checkpoint/report if gates pass, and a costed but unexecuted full-sweep plan.

- [ ] Run `bash launchers/validate.sh --stage math --device cpu`; require all seven mathematical/gradient checks, record command, exit status, timing, and artifacts.
- [ ] Reinspect GPUs and run cache/model validation on one free A40; require adapter parity, finite input gradients, frozen weights, replay/monolithic agreement, and RNG independence.
- [ ] Profile one update, inference set, and refinement evaluation; record peak VRAM, wall time, transformer/generator/field calls, disk/cache use, and projected pilot/full-sweep cost.
- [ ] If any gate fails, diagnose with `superpowers:systematic-debugging`, fix through a failing regression test, rerun the gate, and do not start the pilot until it passes.
- [ ] Run `bash launchers/pilot.sh --gpu <selected-free-id>` for prompt 1/seed 17/300 updates in a persistent logged session; wait for completion, evaluate its validation-selected EMA checkpoint, and record negative/numerical results as readily as improvements.
- [ ] Produce the pilot comparison containing IID, fixed rotation, balanced Gaussian, learned orthogonal, learned conditional rotation, skew initial, skew trained, and independent-seed negative control with CLIP/DINO/SigLIP and prior/output marginal/refinement diagnostics.
- [ ] Lock or explicitly revise the recipe from validation evidence, then run `bash launchers/sweep.sh --gpus 0,1,2,3,7 --max-concurrency 5` without `--execute` and save its planned commands, assignments, run IDs, and projected cost.
- [ ] Complete README commands for activation, cache, validation, pilot, training/resume, evaluation, aggregation, opt-in sweep, logs, checkpoints, tensor axes, and known deviations.
- [ ] Commit with `git add README.md reports/validation && git commit -m 'docs: record validation and pilot evidence'`.

### Task 12: Final checklist audit and verification

**Files:**
- Create: `reports/completion_checklist.md`
- Modify: any files with audit defects.

**Interfaces:**
- Produces: a line-by-line Section 13 audit linking each requirement to code, tests, commands, artifacts, or an explicit unresolved deviation.

- [ ] Map all sixteen completion-checklist items and Stages A–D to evidence; mark blocked/not-run items with concrete causes rather than checking them.
- [ ] Run the complete unit/integration suite, Ruff, package build/import, launcher shell checks, strict config resolution, CPU mathematical gate, and the available GPU/model/pilot gates from a clean process.
- [ ] Run `git status --short`, inspect the dirty diff for secrets, model blobs, caches, generated images, unrelated files, and the nested untracked checkout; ensure ignores keep experimental artifacts out of commits.
- [ ] Use `superpowers:requesting-code-review` for an implementation/spec review and resolve all valid findings with tests.
- [ ] Use `superpowers:verification-before-completion`, rerun final commands, and report exact pass/fail counts, run/checkpoint paths, GPU mappings, cost/results, full-sweep not-run status, and every unresolved issue.
- [ ] Commit with `git add reports/completion_checklist.md && git commit -m 'docs: audit experiment completion checklist'`.

## Self-Review

- **Spec coverage:** Sections 0–3 map to Tasks 1, 8, 10, and 11; Sections 4–5 to Task 3; Section 6 to Task 5; Section 7 to Task 4 and Task 9; Section 8 to Tasks 4 and 7; Section 9 to Tasks 2, 7, 9, and 11; Section 10 to Tasks 5, 6, 9, and 11; Section 11 to Tasks 1–10; Sections 12–13 to Tasks 11–12. Optional FLUX is implemented but only executed if access and resources allow; optional publication baselines are reported unavailable rather than silently substituted.
- **Step scan:** Each task has a failing-test, implementation, verification, and commit cycle; model downloads and long execution occur only after pure mathematics and local contracts exist.
- **Type consistency:** Config, field, solver, generator, encoder, objective, replay, and trainer boundaries all preserve `[B,K,C,H,W]` externally and `[B,C,H,W]` within field/generator calls.
- **Review focus:** Device visibility, active-input leakage, resume seed progression, deterministic generator parity, and numerical failures each have an explicit owning test.
- **Proportion:** The plan decomposes the large research specification into twelve independently reviewable deliverables without copying implementation bodies.
