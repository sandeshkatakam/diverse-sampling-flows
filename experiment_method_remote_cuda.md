# Gaussian-preserving prior flows for diverse text-to-image generation

## Implementation brief for the coding agent

Implement a modular research codebase that learns a Gaussian-preserving prior velocity, keeps a pretrained text-to-image generator frozen, and improves a differentiable diversity objective evaluated on sets of generated images. This document contains the mathematical construction, default experiment choices, baselines, training procedure, evaluation protocol, and implementation order. It does not require any earlier chat or PDF.

The experiment tests a hypothesis; it must not assume the proposed method wins. Report negative results and numerical failures as clearly as improvements. The defaults below are proposed starting configurations, not experimentally validated hyperparameters.

## 0. General remote CUDA server: execution rules

**The deployment environment is a general remote Linux server with eight CUDA devices, indexed `0,1,2,3,4,5,6,7`. Execute commands directly on this server through the remote terminal. There is no login-node execution restriction or Slurm dependency.**

### 0.1 Implement and validate directly

When asked to implement this plan, perform environment setup, dependency installation, model downloads, import checks, smoke tests, toy mathematical tests, gradient checks, feature extraction, and bounded profiling directly on the server as needed. These routine implementation and validation steps are authorized as part of implementation; run them as components become available unless the session explicitly requests write-only work. Run the 300-update pilot after the mathematical and real-model validation gates pass.

Read any repository-local `AGENTS.md`, `INSTRUCTIONS.md`, or README that actually exists for project paths, environment conventions, and shared-server usage. This version does not depend on a separate cluster policy file or a hard-coded university path. Use a project-specific virtual environment or Conda environment; record compatible Python/PyTorch/CUDA versions and pin the validated dependencies. Respect existing model-access requirements.

Create ordinary shell/Python launchers for setup, validation, the pilot, training, evaluation, and aggregation. Do not generate batch-submission templates, require scheduler environment variables, or block CPU mathematical checks because a CUDA device is unnecessary for them. GPU model runs should check actual CUDA availability and selected-device memory and report a concrete hardware/environment problem if unavailable.

### 0.2 Select and record CUDA devices

Treat device IDs `0` through `7` as the available server device pool, not a requirement to use all eight at once. Inspect the actual devices, GPU UUIDs, free memory, utilization, and existing visibility restrictions before choosing a device. Use one available selected GPU for initial model validation, profiling, and each independent training run; keep tiny mathematical checks on CPU when appropriate.

Set `CUDA_VISIBLE_DEVICES` in the launcher environment before starting Python. Resolve the selected server index to its GPU UUID where possible, and verify the mapping rather than assuming enumeration orders match. A process exposing one selected GPU uses logical `cuda:0` internally, even when the selected server device is, for example, device `7`. Record both the server device ID/UUID and the process-visible mapping. Honor any pre-existing device restriction; do not widen it automatically. [R16]

Start with one experiment process at a time. After profiling, independent prompt/initialization/method runs may execute concurrently across selected devices `0–7`, with one process per GPU and a configurable concurrency limit of at most eight. Coordinate assignments within the launcher, respect existing server workloads, and account for CPU RAM, threads, disk space, and shared I/O as well as GPU memory. Prepare the environment and shared model caches before starting parallel workers. Merely listing eight devices must not activate distributed training or change batches, optimizer updates, seeds, or the experimental protocol.

### 0.3 Run scope and reporting

Direct execution is permitted during implementation. Keep the validation-first order in Section 12 and report actual commands, exit statuses, logs, and measured resource use. For an explicitly requested write-only session, or an execution check that could not be performed, report that check as **not run** and explain the concrete reason.

Preserve the original boundary on the complete experimental sweep: do not automatically launch the full main comparison, architecture sweep, or generator replication while implementing. First complete validation and the pilot, report projected cost, and provide a separate opt-in sweep command. A session that already authorizes the full sweep may proceed through those gates without asking again.

For long direct runs, provide a documented persistent-session or background-launch option, such as an existing `tmux` session or a launcher that captures output and exit status. Give every run its own output directory and log, record its PID, and retain resumable checkpoints. Do not treat a started process as a completed experiment.

## 1. Decisions and scope

| Component | Primary choice |
|---|---|
| Task | Generate four diverse images for the same fixed text prompt, with unchanged individual output marginals |
| Generator | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` |
| Sampling | Supported deterministic one-step configuration; 1024 by 1024 images |
| Conditioning | Class-oriented natural-language prompts; not an integer-label-only ImageNet model |
| Set size | `K=4`, configurable |
| Initial randomness for our method | One standard Gaussian seed per set, with four trajectory checkpoints |
| Learned field | Structured exact implementation of `Div K_phi - K_phi z` |
| Default conditioner | Small residual CNN; shared patch MLP and small transformer are architecture ablations |
| Training objective | Negative normalized log determinant of an RBF kernel on frozen CLIP image features |
| Training/evaluation encoder | `openai/clip-vit-large-patch14` |
| Independent diversity evaluator | `facebook/dinov2-base`, normalized CLS features |
| Independent prompt evaluator | `google/siglip-base-patch16-224` |
| First training setting | A separate prior-flow model per fixed prompt; fresh training seeds and disjoint validation/test seeds |
| Default solver | Differentiable fixed-step RK4; convergence checked at increasing step counts |
| Optional generator replication | `black-forest-labs/FLUX.1-schnell`, four deterministic steps |

### Why these choices

Sana-Sprint is a compact, flow-derived consistency-distilled generator with supported one-step generation. This makes repeated differentiation through a frozen generator a more practical first experiment than starting with a much larger multistep model. It is **not an undistilled flow-matching ODE sampler**. The preservation argument below needs a fixed deterministic generator map, so this distinction does not invalidate the experiment. Use the FLUX replication to test a second rectified-flow-based generator. Do not describe either choice as the best available model without an actual comparative study. [R1–R4]

Begin with prompt-specific training to isolate the coupling mechanism. One learned model is reused for arbitrarily many fresh seed sets for that prompt. This is not optimization separately for each returned seed. Training cost must be reported. A shared prompt-conditioned prior network is a later extension, not a zero-shot generalization claim of the first experiment.

Build the minimal end-to-end path first: one prompt, one conditioner, IID plus fixed rotations plus our method. Add the remaining mandatory baselines before drawing comparative conclusions. Architecture sweeps and a second generator follow only after the correctness gates pass.

## 2. Mathematical problem and the source of diversity

Use the following notation consistently:

- `theta`: all frozen generator parameters.
- `phi`: trainable prior-field parameters.
- `c`: a fixed text prompt.
- `K`: number of samples in one set; `K_phi` is the skew matrix field.
- `B`: number of independent sets in a training batch, not the number of images.

Let

\[
\gamma=\mathcal N(0,I_d),\qquad
G_\theta(z;c)=\text{the entire fixed numerical text-to-image pipeline},
\qquad p_G(\cdot\mid c)=(G_\theta(\cdot;c))_\#\gamma.
\]

The definition of `G_theta` includes the sampler, step count, guidance, precision, latent scaling, decoder, and tensor image postprocessing. Its distribution is the reference to preserve; it need not equal the true data distribution.

For each independent set, draw one seed and integrate one shared flow:

\[
Z\sim\gamma,\qquad Y_0=Z,\qquad
\dot Y_t=w_\phi(t,Y_t;c),\qquad t\in[0,K-1].
\]

Use fixed checkpoints

\[
\tau_i=i-1,\qquad
X_0^{(i)}=Y_{\tau_i},\qquad
X_1^{(i)}=G_\theta(X_0^{(i)};c),\quad i=1,\ldots,K.
\]

For `K=4`, the times are `[0,1,2,3]`. Slot 1 is the original Gaussian seed. Only `phi` is optimized:

\[
\min_\phi\;\mathbb E_{Z\sim\gamma}
 C\bigl(X_1^{(1)},\ldots,X_1^{(K)}\bigr).
\]

The cost is evaluated on **generated images**, not on prior distances.

**The shared seed is essential.** Applying fixed Gaussian-preserving maps separately to `K` independent seeds leaves the samples independent with the same product law, so it cannot improve the population distribution of a set-level diversity metric. Our method changes the joint dependence between slots while preserving each marginal.

For every fixed trained `phi` and prompt `c`, exact prior preservation implies

\[
X_0^{(i)}\sim\gamma
\quad\Longrightarrow\quad
X_1^{(i)}\sim p_G(\cdot\mid c).
\]

Freeze `phi` before drawing test seeds. Do not optimize parameters on a test seed and then invoke the fixed-parameter marginal guarantee. Do not choose checkpoint times, transform branches, or stopping rules using that seed's image reward. Present all `K` images; do not silently select or reject outputs.

## 3. Frozen generator adapter

### 3.1 Primary Sana-Sprint configuration

Use these explicit settings:

```yaml
model_id: Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
height: 1024
width: 1024
num_inference_steps: 1
max_timesteps: 1.57080
intermediate_timesteps: null
guidance_scale: 4.5
max_sequence_length: 300
standardized_latent_shape: [32, 32, 32]
transformer_dtype: bfloat16
sampler_arithmetic_dtype: float32
decoder_dtype: bfloat16
```

The standardized prior has `d=32768`. Apply the learned prior transform **before** the sampler's `sigma_data=0.5` scaling. The checkpoint's decoder scaling factor is `0.41407`; these are different quantities. Read and assert the values from the pinned checkpoint instead of assuming that other models share them. [R2, R3]

The chosen one-step schedule is `[1.57080, 0]`. Setting `intermediate_timesteps=None` is necessary when using one step in the inspected API. The usual two-step SCM sampler adds fresh noise at its intermediate update; `eta=0` does not in general disable SCM's extra noise. Do not quietly use that stochastic default while describing `G_theta` as a function of one initial seed. [R4]

A later stochastic experiment can preserve the initial Gaussian and keep every additional generator noise independent, or include all randomness in an enlarged Gaussian state. That is a separate configured experiment.

### 3.2 Tensor-only one-step mapping

The following describes the inspected Sana-Sprint tensor mapping. Verify its operation ordering against the installed pinned source during direct validation on the server.

Let `Z` be standardized noise, `S=1.57080`, `sigma=0.5`, and `kappa=0.41407`. With cached prompt embeddings and masks, compute

\[
x_S=\sigma Z,\qquad
r=\frac{\sin S}{\cos S+\sin S},\qquad
a=\sqrt{r^2+(1-r)^2},\qquad m=(x_S/\sigma)\,a.
\]

Evaluate the frozen transformer `F_theta` with its correct input dtype:

\[
u=F_\theta(m,r,c,g),\qquad g=4.5\times0.1,
\]

and convert its output as required by SCM:

\[
v=\sigma\frac{(1-2r)m+(1-2r+2r^2)u}{a},\qquad
\widehat x_0=\cos S\,x_S-\sin S\,v,
\qquad \ell=\widehat x_0/\sigma.
\]

Decode `ell/kappa`, then apply the fixed tensor image postprocessing. Preserve the official casts and conditioning conventions; do not algebraically simplify the implementation before establishing parity. At the terminal SCM update, the added-noise coefficient is zero and the decoder uses the denoised estimate. An unused RNG draw is allowed in a reference implementation; the returned one-step image must not depend on it. [R3, R4]

### 3.3 Differentiation requirements

The public inference pipeline uses `no_grad`; directly calling it is not the training path. Implement a small adapter from frozen transformer/decoder components and tensor operations.

- Set generator, decoder, text encoder, and metric-encoder parameters to `requires_grad_(False)` and `eval()`.
- Preserve gradients with respect to the input latent during training/replay.
- Cache text embeddings, attention masks, prompt preprocessing, instruction prefix, and guidance tensors. The text encoder can be removed from GPU memory afterward.
- Match official RGB range conversion, including any clamp, in the fixed `G_theta`. Use tensor resizing/normalization for metric inputs; no PIL, NumPy, uint8 conversion, or detach in a differentiable path.
- A `differentiable=False` inference branch may disable autograd; a `differentiable=True` branch must not.
- Do not use model quantization, new decoder tiling, CPU-offload hooks, a different attention backend, or a different resolution halfway through an experiment. If needed, validate and lock the new pipeline first, then rebuild its IID reference.
- All methods in one comparison must use the same frozen numerical generator.

Start from Python 3.11 and propose `diffusers==0.35.1`, whose versioned documentation includes Sana-Sprint. The versioned API documentation was inspected; the tagged source could not be fully retrieved here, and the implementation was not executed. Inspect installed source and validate compatibility during initial environment setup on the server. Resolve a compatible PyTorch/CUDA/Transformers environment for the actual server, lock exact versions, and record source/checkpoint revisions. If another Diffusers release is required, pin it before any experimental run; do not silently track `main`. [R4, R5]

### 3.4 Optional FLUX replication

Use `black-forest-labs/FLUX.1-schnell`, 1024 by 1024, four steps, guidance `0.0`, text length `256`, and deterministic sampling. Its unpacked standardized prior is `[16,128,128]`, followed by deterministic latent packing. Read decoder scale/shift from its checkpoint. Preserve its official schnell schedule and ensure any scheduler `stochastic_sampling` option is false. Do not reuse Sana scaling. The currently visible Hugging Face repository requires access acceptance despite its Apache license; use existing authorized access and report a block if unavailable. [R6]

If an undistilled flow model is specifically required, use `Efficient-Large-Model/Sana_600M_1024px_diffusers` as a separately profiled adapter with its documented sampler. Do not substitute it into the one-step experiment or imply identical output distributions. [R7]

## 4. Gaussian constraint and exact skew-field construction

### 4.1 Correct signs and constraint

For a fixed prompt, write the continuity equation for the prior density `q_t`:

\[
\partial_tq_t+\nabla\cdot(q_tw_\phi)=0.
\]

For standard Gaussian `gamma`, its score is `-z`. Under the smooth complete-flow assumptions, endpoint preservation is equivalent to

\[
\int_a^b
\left[\nabla\cdot w_\phi(t,Y_t)-Y_t^\top w_\phi(t,Y_t)\right]dt=0
\]

along every trajectory up to a Gaussian-null set, provided `q_a=gamma`.

Use the stronger pointwise condition

\[
\nabla\cdot w_\phi(t,z)-z^\top w_\phi(t,z)=0.
\]

The required parameterization is

\[
\boxed{
K_\phi=\frac{M_\phi-M_\phi^\top}{2},\qquad
w_\phi=\operatorname{Div}K_\phi+K_\phi\nabla\log\gamma
=\operatorname{Div}K_\phi-K_\phi z.
}
\]

Here `(Div K)_i = sum_j partial_{z_j} K_{ij}`. There is only one Gaussian score factor: `+K*(-z) = -K*z`. Do not implement `Div K - K*(-z)`.

Indeed,

\[
\gamma w_\phi=\operatorname{Div}(\gamma K_\phi),\qquad
\nabla\cdot(\gamma w_\phi)
=\sum_{i,j}\partial_{z_i}\partial_{z_j}(\gamma K_{\phi,ij})=0
\]

by skew symmetry and commuting mixed derivatives. This preserves the Gaussian throughout time in the exact ODE. [R8]

### 4.2 Why the image-scale implementation must be structured

A dense matrix at `d=32768` contains more than one billion entries per sample. Do not generate it. Do not estimate the divergence used in the actual velocity with Hutchinson noise.

Use the following sparse representation of the same `M -> K -> w` formula. It has exact analytic divergence and lets the neural architecture be changed independently. It is a **structured subclass** of the unrestricted smooth skew family; do not claim a finite mask/ridge network represents every Gaussian-preserving map.

### 4.3 One masked group

Partition coordinates into context `A` and active coordinates `B`. Pair the active coordinates into disjoint pairs `(u_j,v_j)`. A neural conditioner sees **only** context coordinates, time, and fixed external information:

\[
(c_j,\alpha_j,\beta_j,\delta_j,e_j)
=f_\phi(t,z_A)_j.
\]

For each pair define

\[
s_j=\beta_ju_j+\delta_jv_j+e_j,\qquad
h_j=c_j+\alpha_j\tanh s_j.
\]

Represent `M` implicitly with `M_uv=2h_j`, `M_vu=0`, and zero elsewhere in that group. Then

\[
K_{uv}=h_j,\qquad K_{vu}=-h_j.
\]

Because the coefficients do not depend on any active coordinate,

\[
\partial_{u_j}h_j=\alpha_j\beta_j(1-\tanh^2s_j),\qquad
\partial_{v_j}h_j=\alpha_j\delta_j(1-\tanh^2s_j).
\]

Therefore the exact active-pair velocity is

\[
\boxed{
w_{u_j}=\alpha_j\delta_j(1-\tanh^2s_j)-h_jv_j,\qquad
w_{v_j}=-\alpha_j\beta_j(1-\tanh^2s_j)+h_ju_j.
}
\]

The group contributes zero to its context coordinates. No full matrix, Jacobian, spatial-autograd call, or stochastic trace estimate is needed to evaluate this formula.

**Mask before every learned operation.** Do not feed the full latent into the conditioner and then detach or mask its output. That would omit real active-coordinate derivatives and invalidate the formula. The restriction includes attention queries, normalization inputs, residual paths, and pooled summaries. Fixed positions, time, mask IDs, and externally supplied prompt embeddings are allowed.

### 4.4 Cover all coordinates

For a tensor with 32 channels, use two groups at every spatial position:

1. Active channels `0:16`, context channels `16:32`; pair active channels `(0,1), (2,3), ..., (14,15)`.
2. Active channels `16:32`, context channels `0:16`; pair `(16,17), ..., (30,31)`.

Sum their fields, with both group weights equal to one. Each coordinate is active in one group. All groups read the same input state; never update that state in place while assembling the velocity. The context of one group can move under the other group: the pointwise weighted-divergence identity remains valid for the sum.

Use a shared conditioner backbone with a group-ID embedding and ordered context channels. It outputs `5 * (C/4)` coefficient maps per group for one ridge. Include optional fixed channel permutations or additional masks only through configuration. Pair/mask indices are saved with the run.

For more capacity, allow

\[
h_j=c_j+\sum_{r=1}^R\alpha_{jr}
\tanh(\beta_{jr}u_j+\delta_{jr}v_j+e_{jr}).
\]

Sum the corresponding derivative terms. The head then has `(1+4R)*(C/4)` output channels. Start with `R=1`; `R=2` is an ablation.

### 4.5 Bounded heads, initialization, and allowed training operations

Let `omega=2*pi/K`, so `omega=pi/2` for four slots. Use

\[
\begin{aligned}
c_j&=\omega+c_{\max}\tanh\widehat c_j,&c_{\max}&=\pi/4,\\
\alpha_j&=a_{\max}\tanh\widehat\alpha_j,&a_{\max}&=0.25,\\
\beta_j&=b_{\max}\tanh\widehat\beta_j,&b_{\max}&=1,\\
\delta_j&=b_{\max}\tanh\widehat\delta_j,\\
e_j&=e_{\max}\tanh\widehat e_j,&e_{\max}&=2.
\end{aligned}
\]

The bounds are configurable. Smooth bounded coefficients and these explicit active derivatives give `||w(z)|| <= C0 + C1 ||z||`; smooth conditioners therefore give a complete flow on finite intervals.

Initialize near the fixed quarter-turn trajectory: `c=omega`, small nonzero `alpha` near `0.02`, `beta` near `0.5`, and `delta=e=0`, with small nonzero head weights around the required biases. Avoid an all-zero field and avoid a completely zero ridge branch. Identical checkpoint images can give zero diversity gradient. The initial untrained field is a baseline to evaluate, not a result to omit.

Use smooth activations such as SiLU or tanh. Avoid BatchNorm, dropout, and batch-dependent computations in the prior field. Parameter-gradient clipping is allowed. Arbitrary clipping or normalization of the velocity/latent is not: it generally breaks the invariant-density identity. Do not project samples onto a sphere or normalize their empirical norms to "fix" drift.

The analytic masked construction can run under `no_grad` at inference. During training retain ordinary autograd through its coefficients and RK4. A separate dense low-dimensional reference may use `autograd.grad(..., create_graph=True)` to check `Div K`; it is not the image-scale implementation. With a completely generic neural `K`, training through its spatial derivatives would require higher derivatives; the explicit construction avoids that cost in the conditioner.

## 5. Interchangeable conditioner architectures

All conditioners implement the same interface:

`coefficients(t_normalized, context_tensor, group_id, optional_prompt_context)`

They output the raw coefficient maps; bounded-head transformations and velocity algebra belong to the field module, not to the architecture.

| Architecture | Concrete starting design | Role |
|---|---|---|
| Shared patch MLP | Per-location context channel vector, context-only global pooled vector, fixed coordinates, and time/group embeddings; 3 hidden layers of width 128, SiLU | Small, inexpensive architecture baseline |
| Residual CNN | Width 64, 4 residual blocks, two 3x3 convolutions per block, GroupNorm with positive epsilon and SiLU; time/group FiLM; final 1x1 coefficient head | Default image-scale conditioner |
| Small transformer | Tokenize only the context tensor into 2x2 patches; width 128, 4 blocks, 4 heads, MLP ratio 4, LayerNorm, GELU/SiLU, dropout 0; fixed positions and time/group conditioning; unpatchify coefficient head | Attention ablation after the default works |

These are standard architecture patterns chosen as useful starting points; they have not been shown superior for this particular objective. With a 32x32 Sana latent, 2x2 patches give 256 tokens, rather than one token per scalar coordinate. Use windowed/pooled attention if a larger generator makes token count excessive.

Keep masks, coefficient bounds, ridge count, time horizon, and solver fixed in the architecture comparison. Report parameter count, field-evaluation time, and total training cost. Do not imply equal width gives equal capacity. The default CNN is the first implementation; add the other factory options without changing the mathematical field.

## 6. Diversity cost and image encoders

### 6.1 Primary CLIP objective, also evaluated for every baseline

Freeze `openai/clip-vit-large-patch14`. Use its projected image embedding, not classification logits. Apply its fixed resize/crop/normalization using differentiable tensor operations, and normalize features:

\[
f_i=\frac{E_{\rm CLIP}(X_1^{(i)})}
{\max(\|E_{\rm CLIP}(X_1^{(i)})\|_2,10^{-8})}.
\]

For one set define

\[
L_{ij}=\exp\left(-\frac{\|f_i-f_j\|_2^2}{2\ell_c^2}\right),\qquad
\widetilde L=\frac{L+\varepsilon I_K}{1+\varepsilon},\qquad
F_{\rm CLIP}=\frac1K\log\det\widetilde L,
\qquad C=-F_{\rm CLIP}.
\]

Use `epsilon=1e-4` and minimize the average `C` over independent sets. The diagonal normalization makes the score bounded above by zero; larger/less negative scores mean more diversity under this metric. Zero is generally unattainable at finite bandwidth. This is a DPP-inspired differentiable score, **not exact sampling from a DPP**. The RBF kernel is used instead of the raw feature Gram determinant. [R9, R10]

Calibrate `ell_c` once for each prompt using 512 IID reference images generated from a separate calibration seed stream. Let `m_c` be the median squared feature distance over a fixed seeded sample of distinct pairs; set

\[
\ell_c^2=\max\left(\frac{m_c}{2\log2},10^{-4}\right).
\]

Save the calibration embeddings or their provenance, pair IDs, and bandwidth. Freeze it for every method and all training/evaluation. Do not learn the bandwidth or tune it on test outputs. If calibration is almost collapsed, report this fact before interpreting the metric.

Compute the small kernel and Cholesky log determinant in float64, even when embeddings are float32. Use `2*sum(log(diag(cholesky(tilde_L))))/K`. A failure triggers recomputation in float64 and a recorded numerical error if still invalid; do not silently alter jitter or the objective per method.

### 6.2 Companion metrics

Evaluate the exact same `F_CLIP` for IID, every fixed/learned baseline, the initialized field, and the trained field. Also report CLIP mean pairwise cosine distance and the minimum pairwise distance, because one highly different image can mask duplicates in an average.

The main independent diversity check uses `facebook/dinov2-base`, frozen normalized CLS features, with the same mean and minimum pairwise cosine-distance formulas. DINOv2 is an independent image representation, not a proof of human semantic diversity. [R11]

Use `google/siglip-base-patch16-224` to report mean and lower-tail image–prompt scores using its fixed logits for the actual prompt, never a softmax over whichever prompts happen to be in a batch. Report scores per raw slot and as set averages. These are alignment proxies, not calibrated probabilities or complete visual-quality measures. [R12]

Optional LPIPS adds a perceptual/texture diversity measure; it must not replace the semantic checks. Save a predetermined image-grid subset for blinded qualitative review of prompt relevance, artifacts, and duplicate content. Do not choose the best-looking seeds for the report.

Under exact marginals, the expectation of any additive single-image score is fixed:

\[
\mathbb E\left[\frac1K\sum_i q(X_1^{(i)},c)\right]
=\mathbb E_{X\sim p_G(\cdot\mid c)}q(X,c).
\]

Thus mean quality/alignment is a preservation diagnostic, not a promised additional improvement. Do not add an additive CLIP-quality reward and claim it can improve its population mean without changing marginals. Set-level quantities such as minimum alignment may change.

## 7. Baselines and fairness

All primary methods return exactly `K=4` images through the same `G_theta`, with identical prompt, resolution, sampler, guidance, precision, and metric preprocessing.

| Name | Construction | Training |
|---|---|---|
| `iid` | `K` independent standardized Gaussian priors | None |
| `fixed_rotation` | One seed; analytic pair rotations with angles `0, pi/2, pi, 3pi/2` | None |
| `balanced_gaussian` | `z_i=sqrt(K/(K-1))*(epsilon_i-mean_j epsilon_j)`, with independent Gaussian `epsilon_i` | None |
| `learned_orthogonal` | One seed; learned structured orthogonal maps with fixed pair permutations | Same image-space objective |
| `learned_conditional_rotation` | One seed; learned nonlinear conditional pair-rotation coupling layers | Same image-space objective |
| `skew_initial` | Proposed masked skew field before training, including its small nonzero residual | None |
| `skew_trained` | Proposed field after training/checkpoint selection | Same image-space objective |

The balanced tuple is a strong Gaussian-marginal baseline: each slot has covariance `I`, cross-slot covariance `-I/(K-1)`, and zero sum across slots. It uses more base randomness than the single-seed trajectory; this is permitted and should be stated. Our deterministic coupling class is not all possible joint laws with Gaussian marginals.

### 7.1 Learned orthogonal maps

Implement compositions of 2D pair rotations; no dense matrix exponential is needed. Each transition contains one fixed quarter-turn map `R0` followed by four learned residual rotation layers. A residual layer has the form `P_l^T R(angle_l) P_l`, where `P_l` is a fixed seeded permutation of flattened coordinates and `R` rotates disjoint pairs. Learn one angle per pair/layer/transition, independently of the current sample. The permutations mix spatial and channel coordinates and are saved. Every fixed parameter setting is orthogonal with determinant one.

Initialize residual angles at zero, so the composed transition initially equals `R0` even when the pairings differ. Anchor the first slot at `Z` and generate the remaining slots by successive learned transitions. Do not add angles across different pairings as if those rotations commuted, and do not accidentally apply a full `2*pi` and duplicate the first slot.

### 7.2 Learned nonlinear conditional rotations

Use four conditional coupling layers per transition, alternating the two channel masks. Predict each active pair's rotation angle from only the currently held coordinates, transition index, and external prompt information if enabled. Use the same conditioner family as our method, changing only its output head. For a split `(a,b)`, a layer is

\[
(a,b)\mapsto(a,Q_\phi(a)b),\qquad Q_\phi(a)^\top Q_\phi(a)=I.
\]

Because `b|a` is standard Gaussian and `a` is held fixed, the layer preserves the Gaussian. Composition preserves it too. For the default aligned pairings, each coordinate is active twice; initialize each active angle near `pi/4`, giving a transition near the quarter turn. If adding different pairings, instead use zero-angle residual layers around a fixed `R0` as above; angles of noncommuting rotations do not simply add. This is a serious nonlinear baseline with direct map evaluation and no ODE discretization error. It preserves individual radii, while the proposed field can change them. Allow four layers by default and a parameter-matched deeper variant if our network has substantially greater capacity; report the difference rather than hiding it.

### 7.3 Comparison rules

- Give learned methods the same objective, fresh-seed distribution, optimizer-step budget, effective number of training sets, and validation opportunities. Log actual generator calls, field/map calls, wall time, and GPU memory; equal updates do not imply equal compute.
- Evaluate the best prespecified validated checkpoint of each method, including trained rotation baselines. Do not compare a tuned method only against random rotations.
- Reuse the same base seed for coupled methods when forming paired differences. IID still needs independent seeds within its set; balanced Gaussian still needs its full independent noise tuple. Store these method-specific seed recipes explicitly.
- Preserve raw slot IDs. Any independent display permutation is separate from marginal evaluation.
- Direct rotation maps are algebraically Gaussian preserving in exact arithmetic. The ODE field is preserving in continuous time; its discretization must be measured.
- Include a negative control that applies the learned preserving field to independent seeds. Under accurate integration it should not improve the population set distribution relative to IID.

### 7.4 Optional practical comparators and related work

Include selecting four images from sixteen IID candidates by greedy maximization of the same fixed logdet score as a separate extra-compute comparator. Report all sixteen generator evaluations; selection generally changes individual output marginals. Deterministically break selection ties by candidate index. Do not place it in the exact-prior-preserving table.

For a publication-oriented extension, compare Particle Guidance and the current EDDY implementation on a common compatible multistep backbone with documented adaptations and actual evaluation counts. Their integration into the one-step pilot is not assumed. Report an unavailable baseline as unavailable, never substitute a different formula under the same method name. [R13, R14]

EDDY's September 2026 revision is particularly relevant: its current title is *Diverse Sampling in Diffusion Models with Divergence-Free Particle Guidance*. It discusses the inability of exact product-invariant dynamics to diversify an IID batch and marginal changes from correlated initialization/numerical approximations. Cite the revised version, not an earlier unqualified preservation claim. This motivates our independent shared-seed construction and solver-refinement checks; it does not establish our method's novelty or empirical superiority. [R14]

## 8. Training and backpropagation

### 8.1 Solver and numerical precision

Integrate in physical time `[0,K-1]`; normalize time by `K-1` only inside the network's time embedding. Do not accidentally rescale the velocity when changing the embedding.

Start with RK4 and eight steps per unit interval, so the four checkpoints align exactly with integration steps. This is a provisional training resolution, not a preservation guarantee. Validation compares 8, 16, and 32 steps per interval. Increase the training resolution and retrain if the apparent improvement relies on the coarse solver.

Use float32 for the prior state, conditioner, bounded heads, and RK4 accumulation. Use float64 for small mathematical validation cases. Use bfloat16 for the frozen generator transformer and initially its decoder, with sampler arithmetic in float32. Keep CLIP parameters/feature calculations in float32 initially; optimize mixed precision only after a gradient comparison. Keep logdet arithmetic in float64. Disable autocast around the prior field and the small kernel calculation.

Differentiate through the actual fixed-step solver. Do not detach between checkpoints or replace its gradient with a straight-through estimate. Prefer direct backpropagation; non-reentrant activation checkpointing is an optional memory tradeoff. Do not introduce an adjoint solver in the first implementation. A valid checkpointed function must be deterministic and have no mutable scheduler state. [R15]

### 8.2 Starting optimization settings

| Setting | Default |
|---|---|
| Optimizer | AdamW, trainable prior/baseline parameters only |
| Learning rate | `1e-4` |
| Betas / epsilon | `(0.9, 0.99)` / `1e-8` |
| Weight decay | `1e-4`; exclude biases and normalization parameters |
| Effective batch | 4 independent sets, each containing 4 images of one prompt |
| Memory microbatch | 1 set; accumulate 4 sets per optimizer update |
| Gradient clipping | Global parameter-gradient norm `1.0` |
| Main run length | 1000 optimizer updates per prompt and initialization |
| Warmup / schedule | 50 updates; cosine decay to `1e-5` |
| EMA | Decay `0.995`; apply the same policy to learned baselines |
| Validation | Every 100 updates on a fixed, disjoint validation manifest |
| Save | Initial, latest/resumable, and best eligible EMA checkpoint |
| Training initializations | `[17,18,19]` for the main comparison |

These are starting values. The pilot may revise them using validation evidence; then lock one declared training recipe before the main test runs. Give learned baselines the same tuning opportunity. All losses must be divided by the effective batch exactly once when accumulating gradients. Log unclipped gradient norms and skipped updates.

Use a fresh Gaussian seed for every training row. Maintain separate generators/seed namespaces for parameter initialization, training, bandwidth calibration, monitoring validation, refinement validation, final test, IID references, projections, and visualizations. Save RNG states and seed counters in checkpoints. Do not reset the training RNG each iteration.

The default loss is only the set diversity cost. Do not add latent-distance rewards or velocity penalties silently. An optional velocity-energy regularizer is a separately named ablation because it changes the optimization problem. A finite/NaN failure must be logged; repeated failure stops the job rather than repeatedly skipping until a favorable run appears.

### 8.3 End-to-end update

Conceptually, one update is:

```python
optimizer.zero_grad(set_to_none=True)
for microbatch in independent_training_sets:
    z = sample_standard_gaussian(training_rng, latent_shape)
    prior_checkpoints = integrate_rk4(prior_field, z, times=[0, 1, 2, 3])
    images = frozen_generator_with_input_grad(prior_checkpoints, fixed_prompt)
    features = frozen_clip_with_input_grad(images)
    loss = negative_rbf_logdet(features, fixed_bandwidth, fixed_jitter)
    (loss / gradient_accumulation_steps).backward()
clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
scheduler.step()
update_ema()
```

The first slot is independent of `phi`, but its image features influence the joint loss and therefore the gradients of the other slots. Do not remove it from the objective.

### 8.4 Default memory-saving feature replay

Holding all generator graphs can be expensive. Implement an exact first-order vector–Jacobian-product replay path, alongside the simple reference path above.

Let

\[
H_c(z)=\operatorname{normalize}\bigl(E_{\rm CLIP}(G_\theta(z;c))\bigr).
\]

1. Integrate the prior with autograd and retain its checkpoint tensors `z_i` and prior graph.
2. Compute all `f_i=H_c(z_i)` without retaining generator activation graphs. Detach them, make the feature tensors leaves, and differentiate the **complete set loss** to obtain `g_i = partial C / partial f_i`.
3. For each trainable-dependent slot, replay `H_c` on a detached copy of `z_i` with input gradients enabled. Compute `q_i = J_H(z_i)^T g_i` with `autograd.grad`, then release that generator graph.
4. Backpropagate through all `phi`-dependent prior checkpoints together with the detached `q_i`, using one `torch.autograd.backward` call. Exclude the anchored first checkpoint from both the `tensors` and `grad_tensors` lists because it normally has `requires_grad=False`. Apply accumulation scaling exactly once.

Mathematically,

\[
\nabla_\phi C
=\sum_i
\left(\frac{\partial z_i}{\partial\phi}\right)^\top
J_{H_c}(z_i)^\top g_i.
\]

This is the ordinary chain rule, not an approximate reward gradient. It requires identical deterministic sampler behavior, precision, preprocessing, model mode, and conditioning in the two passes. Do not reuse a stateful scheduler at its terminal index; reset it or use the pure tensor adapter.

The anchored first slot has no `phi` dependence, so its replay VJP can be skipped. For `K=4`, the default anchored construction uses four feature-pass forwards plus three replay forwards and three input VJPs per set. Count transformer evaluations inside each generator call as well. Compare this path against monolithic backprop on a small direct validation case before using it for training.

If the prior graph itself is large, checkpoint its deterministic RK4 intervals with `use_reentrant=False`. If the generator replay is large, checkpoint supported frozen transformer/decoder blocks. Do not wrap the replay pass in `no_grad`. Avoid automatic inference offload hooks until their behavior with input gradients is verified. [R15]

## 9. Prompts, seeds, and staged experiment size

Use the following fixed class-oriented prompts in the primary prompt-specific experiment:

1. `A photograph of a golden retriever.`
2. `A photograph of a domestic cat.`
3. `A photograph of a horse.`
4. `A photograph of a parrot.`
5. `A photograph of a butterfly.`
6. `A photograph of a red fox.`
7. `A photograph of a wooden chair.`
8. `A photograph of a sports car.`

These deliberately allow different instances, poses, compositions, and backgrounds while fixing the broad class. Save exact strings; do not rewrite prompts separately for different methods.

### Run stages

| Stage | Scope | Purpose |
|---|---|---|
| Initial validation | Small mathematical cases and 2 real-image sets | Verify signs, masks, gradients, adapter parity, and memory |
| Pilot | Prompt 1, seed 17, 300 optimizer updates | Establish a working learning signal and cost estimate |
| Primary comparison | Eight prompts, three training initializations, 1000 updates for each learned method | Compare trained couplings with fresh-seed evaluation |
| Architecture ablation | MLP/CNN/transformer on a declared prompt subset, same field and protocol | Isolate conditioner choice |
| Generator replication | FLUX on a declared prompt subset after profiling/access checks | Test transfer of the approach to another generator |

Per prompt, use 512 calibration images, 64 monitoring-validation sets, a separate 256-set refinement-validation pool, and 256 final test sets. The monitoring, refinement, and final test manifests must be disjoint. Generate an independent IID reference pool of 2048 images for image-marginal diagnostics. Share the same frozen reference across methods, but do not use it for optimization. It is acceptable to reuse that reference across training initializations; account for this reuse when estimating uncertainty.

For final prior diagnostics, use 8192 fresh seed rows in streaming batches without generating images. Preserve seed manifests rather than storing every large latent. The pilot may use fewer diagnostic rows, but clearly distinguish pilot from final results.

This is a nontrivial training sweep. Profile and report projected cost before launching it; do not assume a particular GPU-memory fit or promise a runtime that has not been measured. Independent one-GPU processes across prompt/initialization/method are the first parallelization choice. Assign each process a distinct selected device from `0–7`, and increase concurrency only after checking available GPU memory, CPU RAM, and I/O capacity. Distributed training is optional after the single-GPU path is correct.

### Optional amortized extension

After the primary experiment, support a single `w_phi(t,z;c)` conditioned on a cached prompt embedding. Project that fixed embedding into the conditioner; it is external information and does not invalidate Gaussian invariance for each fixed prompt. Train on a substantially larger prompt set and split train/validation/test by source/template/paraphrase group. Record the exact prompt dataset and split manifest. Do not call the eight-prompt, per-prompt experiment an unseen-prompt generalization result.

## 10. Evaluation: diversity, marginal preservation, and numerical stability

### 10.1 Report every metric for every primary method

For each fixed prompt and trained initialization, report:

- CLIP logdet score, exactly matching the training objective, including its mean, standard error, and distribution over sets.
- CLIP mean and minimum pairwise cosine distance.
- Independent DINOv2 mean and minimum pairwise cosine distance.
- SigLIP prompt-alignment mean, lower quantiles, and per-slot values.
- Generation time per set, prior transformation time, peak GPU memory, generator evaluations, prior evaluations, and training GPU-hours.
- Per-slot prior and output marginal diagnostics described below.

Use paired seed rows where applicable. Bootstrap whole sets, not the six pair distances within a set. For aggregated results, weight prompts equally and report variation across training initializations; use a hierarchical bootstrap that respects prompt/run/row structure and shared reference pools. Do not treat four dependent images as four independent rows.

Save raw set-level scores and slot-level features so all aggregate tables can be regenerated. Choose the qualitative-grid seed IDs before inspecting outputs. Show complete sets for IID, the strongest validated preserving baseline, initialized skew field, and trained skew field.

### 10.2 Prior marginals: inspect each checkpoint column separately

For every slot `i`, the samples across independent rows should follow `N(0,I_d)`. Do not pool all slots or all prompts to hide opposite drifts.

Compute in streaming batches:

1. Per-coordinate means/variances, with RMS errors and quantiles across coordinates.
2. Mean, variance, and quantiles of `||z||^2`; compare with `chi-square(d)`.
3. Standardized radial statistic `r=(||z||^2-d)/sqrt(2d)`; its mean should be zero and variance one.
4. A fixed 64-dimensional orthonormal projection, constructed once independently of outputs, including projected means and covariance errors.
5. Sliced distribution distances and RBF-MMD on the projected coordinates, with bandwidth calibrated on independent Gaussian reference data.
6. Between-slot differences and cross-slot dependence as descriptive coupling statistics; nonzero dependence is expected and is not itself marginal drift.

Calibrate the full diagnostic procedure against repeated equally sized Gaussian-versus-Gaussian samples, including the maximum across slots/statistics when setting an alarm threshold. Never use tiny unadjusted p-values across thousands of coordinates as a simplistic pass/fail test. Report effect sizes and null ranges. Finite diagnostics can detect drift; passing them does not prove equality of high-dimensional distributions.

### 10.3 Output marginals: compare with the fixed generator, not a new target

For every prompt and raw slot, compare generated samples to the independent IID reference from the identical `G_theta`:

- MMD and sliced distances on fixed DINOv2 features.
- Distributions of SigLIP prompt alignment and selected feature projections.
- Between-slot discrepancies, alongside each slot's discrepancy from the IID reference.

Use unbiased two-sample estimates where appropriate and calibrate them on equally sized IID-reference splits. If an unbiased MMD estimate is slightly negative, retain/report that fact rather than clipping it silently. Use the same feature encoder, preprocessing, and bandwidth for all methods. These are feature-space diagnostics and cannot prove equality of image laws.

Do not use FID computed from 256 images per slot as the principal preservation claim. A larger-sample FID/KID analysis is optional, with its own sample-size and bias accounting.

### 10.4 Solver refinement is a required result

Freeze each selected learned field and evaluate the same 256 rows from the separate refinement-validation manifest at 8, 16, and 32 RK4 steps per unit interval. Never use the final test manifest for this selection. Hold the generator configuration fixed. Record:

- Per-slot latent difference `||z_N-z_2N||/sqrt(d)` and its distribution.
- CLIP and DINO diversity changes for the same rows.
- All prior and output marginal diagnostics.
- Inference cost.

A practical initial convergence target is mean normalized latent difference below `1e-3` and mean CLIP logdet difference below `0.02` between the two finest settings; declare these tolerances before main testing. Also require the independent-diversity conclusion to agree at the two finest settings and no new systematic marginal drift. These are operational tolerances, not proofs of exact invariance.

Choose one common sufficiently converged evaluation resolution for the main proposed-method table; show its refinement table as well. If no resolution passes, report the problem. If training benefited from coarse-solver drift, increase its training resolution and retrain rather than presenting a changed inference procedure as the original result. Never tune resolution using final test scores.

An exactly zero analytic weighted divergence does not certify a numerical flow map. In small dimensions, additionally evaluate

\[
\delta_i(z)=\log|\det D\widehat\Phi_{\tau_i}(z)|
-\frac12\left(\|\widehat\Phi_{\tau_i}(z)\|^2-\|z\|^2\right).
\]

This checks the implemented map's Gaussian change-of-variables identity. Do not construct a dense image-scale Jacobian to repeat this check. Numerical drift must be distinguished from the intended change in joint dependence. [R14]

### 10.5 Interpreting success and failure

Evidence for a useful method is improved held-out CLIP logdet and independent DINO diversity relative to the strongest prespecified preserving baseline, with consistent prompt-alignment behavior, acceptable marginal diagnostics, and stable solver refinement. Report confidence intervals and per-prompt outcomes. A failure to reject marginal equality is not a proof, and a nonsignificant quality drop is not automatically noninferiority; report the uncertainty or use a prespecified equivalence margin.

If CLIP improves but DINO/visual review does not, report metric-specific improvement. If only latent distances improve, the generated-image hypothesis is unsupported. If improvement disappears with solver refinement, report numerical exploitation. If learned conditional rotations match or beat the ODE at lower cost, report that as a meaningful result. Do not weaken the baseline afterward.

## 11. Code organization and configuration

Use a standard `src/` Python package with explicit interfaces and small CLI entry points. Reuse existing project structure when sensible; do not overwrite unrelated code.

| Module | Responsibility |
|---|---|
| `generators/base.py`, `generators/sana_sprint.py` | Standardized-noise adapter, frozen components, conditioning cache, differentiable/inference modes |
| `generators/flux_schnell.py` | Optional replication adapter, separate scaling and packing |
| `priors/masked_skew.py` | Implicit `M`, skew construction, bounded heads, exact pairwise velocity |
| `priors/masks.py` | Saved masks, pairs, and fixed permutations |
| `conditioners/mlp.py`, `resnet.py`, `transformer.py` | Interchangeable coefficient networks |
| `solvers/rk4.py` | Pure differentiable fixed-step integration and checkpoint extraction |
| `baselines/` | IID, fixed rotations, balanced Gaussian, learned rotations, conditional couplings |
| `objectives/clip_logdet.py` | Frozen feature extraction interface, bandwidth calibration, exact set loss |
| `training/trainer.py`, `feature_replay.py` | Optimizer loop, accumulation, replay VJPs, EMA, resumption |
| `evaluation/diversity.py`, `marginals.py`, `refinement.py` | Set metrics, per-slot diagnostics, solver studies |
| `utils/seeding.py`, `checkpointing.py`, `logging.py` | Seed namespaces, atomic checkpoints, metadata and JSONL/CSV logs |
| `scripts/` | Thin setup/cache, validate, train, evaluate, and aggregate entry points |
| `launchers/` | Direct setup, validation, pilot, training, evaluation, aggregation, and opt-in sweep launchers with explicit GPU selection |
| `tests/` | Mathematical and gradient checks, run directly on CPU or a selected CUDA device as appropriate |

Document tensor shapes and which axes represent independent sets versus slots. The main latent/image interface uses `[B,K,C,H,W]`; flatten `B*K` only at the frozen generator/encoder boundary and restore it before the set loss. The field itself sees `[B,C,H,W]` and never couples independent rows.

Use OmegaConf with typed/dataclass validation. Keep one main YAML and separate module YAMLs. Resolve includes relative to the configuration directory, merge in a documented order, apply CLI overrides last, reject unknown keys, and save the fully resolved configuration. Do not scatter architectural or solver constants through scripts.

Example main configuration structure:

```yaml
includes:
  - model/sana_sprint_1step.yaml
  - prior/masked_skew.yaml
  - conditioner/resnet.yaml
  - objective/clip_logdet.yaml
  - training/default.yaml
  - evaluation/default.yaml
  - runtime/remote_cuda.yaml

experiment:
  name: sana_prompt_specific_k4
  method: skew_trained
  prompt_id: golden_retriever
  training_seed: 17
  num_samples: 4
  checkpoint_times: [0.0, 1.0, 2.0, 3.0]

prior:
  mask_groups: 2
  ridges_per_pair: 1
  solver: rk4
  steps_per_interval: 8
  dtype: float32

training:
  updates: 1000
  effective_sets_per_update: 4
  microbatch_sets: 1
  gradient_mode: feature_replay
  lr: 0.0001
  warmup_updates: 50
  ema_decay: 0.995
  validate_every: 100

evaluation:
  calibration_images: 512
  validation_sets: 64
  refinement_validation_sets: 256
  test_sets: 256
  iid_reference_images: 2048
  prior_diagnostic_rows: 8192
  refinement_steps_per_interval: [8, 16, 32]
  raw_slot_metrics: true

execution:
  backend: direct
  available_gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  selected_gpu_id: 0
  device: cuda:0
  gpus_per_run: 1
  max_parallel_runs: 1
  allow_cpu_mathematical_tests: true
  require_cuda_for_model_runs: true
  validate_during_implementation: true
  run_pilot_after_validation: true
  launch_full_sweep_automatically: false
```

Each include must define its own complete module configuration, including model revisions, masks, bounds, encoders, bandwidth files, and paths. The displayed example is a schema illustration, not permission to omit those settings. Runtime metadata must contain resolved immutable checkpoint revisions; placeholders are not acceptable in a completed run.

### Direct launcher contract

The runtime configuration describes the server device pool and per-process device selection. `selected_gpu_id: 0` is an editable starting choice, not a claim that device 0 is idle. The launcher must set and verify the selected GPU visibility before starting the experiment process; `device: cuda:0` refers to the first visible device inside that process. Validate requested IDs against the available and currently permitted device pool. [R16]

Implement and document the following direct interfaces, adapting the underlying Python package names to the repository:

```bash
bash launchers/setup.sh
bash launchers/validate.sh --stage math --device cpu
bash launchers/validate.sh --stage model --gpu 0
bash launchers/pilot.sh --gpu 0
bash launchers/train.sh --gpu 0 --config configs/main.yaml
```

Setup creates and validates the project environment. Every launcher must activate or directly invoke that same environment for its own commands. The mathematical validation entry point must work without CUDA. Model validation and pilot launchers must select a GPU explicitly. Implement matching evaluation and aggregation launchers, and document their exact run/checkpoint arguments in the README.

Provide `launchers/sweep.sh` as a separate opt-in interface with a selected GPU list, maximum concurrency, and an explicit execute option. Its default should print the planned commands, run IDs, GPU assignments, and previously profiled cost estimate without starting the sweep. When execution is requested, track child PIDs and exit statuses, assign distinct GPUs to concurrent runs, and use unique output directories. Do not run environment installation separately in every worker. Keep the experiment's effective batch size, update counts, seeds, and dataset splits unchanged when increasing process concurrency.

### Reproducibility records

Every run stores a unique run ID, code commit and dirty diff, package/CUDA/driver versions, hardware, server hostname, process ID, selected server GPU IDs/UUIDs, `CUDA_VISIBLE_DEVICES` and logical-device mapping, exact launch command, exact configuration, checkpoint revisions, prompt hash, mask/permutation seeds, encoder preprocessing, generator schedule and dtype policy, seed manifests, metric bandwidths, and attention backend.

Checkpoints include prior/baseline weights, EMA, optimizer and LR-scheduler states, global update, RNG states/counters, and best-validation metadata. Resume must reproduce the seed progression; use atomic writes. Store model-weight hashes or revision checks rather than duplicating huge frozen models into every run folder. Keep caches separate from outputs.

Logging defaults to local JSONL/CSV and TensorBoard; external tracking is optional. Never log authentication tokens. Pin dependencies after the direct compatibility check and export an exact environment lock. Re-running an experiment must not run `pip install -U` or change the checkpoint revision.

For learned linear baselines, fixed permutation-conjugated pair rotations (`P^T R P`) are a convenient way to mix channel/spatial pairings while retaining orthogonality and determinant one. Save those permutations. Do not materialize dense image-scale matrices.

## 12. Validation gates and implementation order on the remote server

### Stage A — inspect, set up the environment, and implement with direct checks

Read the repository instructions that exist, inspect the server environment and CUDA devices `0–7`, and prepare a project-specific environment. Implement configuration, interfaces, mathematical field, baselines, generator adapter, training, diagnostics, and direct launchers in that order. Add short docstrings explaining invariants and masks. Run relevant import checks, smoke tests, and the mathematical checks as their components become available; resolve failures before progressing. Finish a README with ordinary remote-terminal commands, environment activation, explicit GPU selection, log locations, and checkpoint-resume instructions. No external cluster file or scheduler integration is required.

### Stage B — direct mathematical validation

Run these checks directly on CPU by default, or on a selected CUDA device when appropriate. Use tiny float64 cases to check:

1. Implicit `M` produces skew `K`; explicit dense `Div K-Kz` matches the analytic pair formula.
2. Directly computed `div w-z^T w` is near floating-point error at random low-dimensional points.
3. The coefficient network has zero derivative with respect to all excluded active inputs; deliberately leaking an active input makes this check fail.
4. Constant `c` heads with `alpha=0` reproduce rotations; the tanh pair example has the expected nonzero radial motion. Constant coefficients with nonzero active dependence are not generally rotations.
5. Increasing RK4 resolution decreases map/Jacobian and trajectory errors in the small cases.
6. Parameter gradients through a tiny solver and differentiable toy generator agree with a central finite difference; feature replay agrees with monolithic backprop.
7. IID and balanced-Gaussian covariance constructions agree with their analytic values; direct conditional rotations preserve their conditional Gaussian law.

These checks target mathematical and gradient risks. Do not build a large test suite that merely restates implementation details.

### Stage C — direct real-model validation and profiling

- Cache exact model/encoder revisions and prompt embeddings directly on the server in the configured project environment. Run GPU-dependent work on one explicitly selected available device from `0–7`.
- Compare the differentiable one-step adapter's output with the official tensor inference path for fixed seeds, identical dtype/backend, and conditioning. Start with max absolute image error `1e-5` for equivalent operation order; if a reproducible backend discrepancy requires a different tolerance, document it before training, inspect its cause, and compare feature/gradient consequences rather than silently relaxing the check.
- Confirm input gradients exist and are finite through the transformer, decoder, preprocessing, and CLIP. Confirm frozen weights receive no gradients or updates.
- Compare replay and monolithic parameter gradients on the smallest feasible real case; record relative discrepancy and memory usage.
- Check that repeated one-step outputs do not depend on intermediate sampler RNG draws.
- Profile one update, one inference set, and a refinement evaluation. Record peak VRAM, wall time, actual generator/field calls, and projected sweep cost.

If memory is insufficient, first use feature replay, then activation checkpointing or a documented batch reduction with matched accumulation. Changing resolution, decoder mode, or model is a new generator configuration with a new IID reference. Do not claim an old reference remains valid.

### Stage D — pilot, full comparison, and report

Run the pilot directly on one selected GPU after Stages B and C pass, inspect training and validation diagnostics, and lock the recipe. Implement every mandatory baseline. When the full comparison is requested, evaluate all mandatory baselines and run the declared main comparison using the direct launchers and configured concurrency. Select checkpoints on validation CLIP objective among numerically eligible checkpoints; use the same selection rule and opportunities for learned baselines. Freeze the selection before final test generation. If no checkpoint is eligible, report that outcome.

Aggregate raw results into:

1. A primary method comparison table with matched frozen-generator settings.
2. Per-prompt diversity and alignment plots.
3. Per-slot prior and output marginal diagnostic panels.
4. A prior-solver refinement table.
5. Inference/training cost and parameter-count tables.
6. Predetermined complete image grids and a concise failure analysis. Produce anonymized grids for manual review; do not invent human ratings or infer human preference solely from an automated metric.

Store every executed command, device assignment, process ID, exit status, checkpoint path, and diagnostic summary. Do not fabricate successful tests, completed runs, or results when only source files were written.

## 13. Completion checklist

- [ ] Setup and validation run directly on the remote server; GPU runs use explicitly selected devices from `0–7`, with recorded visibility mappings and controlled concurrency.
- [ ] The generator is frozen, pinned, deterministic in the primary experiment, and receives standardized transformed Gaussian noise.
- [ ] `K=4` checkpoints come from one seed per set; different set rows remain independent.
- [ ] The velocity has the correct `Div K-Kz` sign and an implicit skew `M` construction.
- [ ] Masks exclude active inputs before learned operations; analytic divergence matches a dense tiny reference.
- [ ] Initialization avoids identical checkpoints and a dead nonlinear branch.
- [ ] The default residual conditioner can be replaced by MLP/transformer through configuration.
- [ ] Differentiable generator/encoder paths and replay gradients have recorded validation evidence.
- [ ] IID, fixed rotation, balanced Gaussian, learned orthogonal, learned conditional rotation, initialized field, and trained field are all evaluated.
- [ ] Every method is scored with the exact training CLIP objective and independent diversity/alignment diagnostics.
- [ ] Calibration, training, validation, test, reference, and visualization randomness are separated and recorded.
- [ ] Marginals are evaluated by raw checkpoint and prompt, with IID null calibration.
- [ ] Diversity survives prior-solver refinement; numerical drift is reported rather than normalized away.
- [ ] Checkpoint selection is independent of final test seeds; all generated set members are retained.
- [ ] Timing, memory, model calls, offline training cost, revisions, configurations, and resume state are saved.
- [ ] The final report distinguishes mathematical invariance, numerical evidence, and empirical diversity gains.

## 14. Primary references and implementation sources

References [R1–R15] are retained from the original plan, which records source checks on 18 November 2026. This remote-server adaptation changes execution guidance and adds the official CUDA device-visibility reference [R16]; it does not revalidate the original model choices or claim new experimental results. Pin the actual versions used by the implementation. Links are for provenance and API details; all method-defining mathematics is included above.

- **[R1]** Sana-Sprint paper: https://arxiv.org/html/2503.09641v2
- **[R2]** Official 0.6B checkpoint and model card: https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
- **[R3]** Official checkpoint component configurations: [transformer](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/transformer/config.json), [scheduler](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/scheduler/scheduler_config.json), [decoder](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/vae/config.json).
- **[R4]** Official [Sana-Sprint pipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/sana/pipeline_sana_sprint.py); [SCM scheduler source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_scm.py). An inspected immutable NVIDIA-hosted copy of SCM is [here](https://huggingface.co/spaces/nvidia/Cosmos-Predict2/blob/94f9c0e9351008ebaa58454f3fef04ab30754edf/diffusers_repo/src/diffusers/schedulers/scheduling_scm.py). Verify the installed tagged version rather than assuming `main` is unchanged.
- **[R5]** Versioned Sana-Sprint API: https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/sana_sprint
- **[R6]** Official [FLUX.1-schnell model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [sampling implementation](https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py), and [CLI configuration](https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py).
- **[R7]** Undistilled Sana option: https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers
- **[R8]** Ma, Chen, and Fox, *A Complete Recipe for Stochastic Gradient MCMC*, Eq. (3), deterministic specialization: https://arxiv.org/abs/1506.04696
- **[R9]** Official CLIP checkpoint: https://huggingface.co/openai/clip-vit-large-patch14
- **[R10]** Kulesza and Taskar, *Determinantal Point Processes for Machine Learning*: https://arxiv.org/abs/1207.6083
- **[R11]** DINOv2 [author repository](https://github.com/facebookresearch/dinov2) and [base checkpoint](https://huggingface.co/facebook/dinov2-base).
- **[R12]** SigLIP checkpoint: https://huggingface.co/google/siglip-base-patch16-224
- **[R13]** Corso et al., *Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models*: https://arxiv.org/abs/2310.13102 ; code: https://github.com/gcorso/particle-guidance
- **[R14]** Vinograd, Achituve, and Fetaya, *Diverse Sampling in Diffusion Models with Divergence-Free Particle Guidance*, revised v2: https://arxiv.org/html/2605.06553v2
- **[R15]** PyTorch non-reentrant activation checkpointing: https://docs.pytorch.org/docs/stable/checkpoint.html
- **[R16]** NVIDIA CUDA environment variables, including device visibility, enumeration, and GPU UUID selection: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html


