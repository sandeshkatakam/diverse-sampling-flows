# Diverse Sampling Flows

Research implementation of the Gaussian-preserving prior-flow experiment in
[`experiment_method_remote_cuda.md`](experiment_method_remote_cuda.md).

The implementation uses a frozen deterministic Sana-Sprint generator and
learns only Gaussian-preserving prior couplings. It reports negative results,
solver drift, unavailable optional models, and failed validation gates rather
than treating the proposed method as successful by construction.

## Environment

The project uses an isolated Python 3.11 environment at `.venv`. Setup,
validation, pilot, training, evaluation, aggregation, and opt-in sweep commands
will be documented as their launchers are implemented. Model caches are kept in
`cache/`; run outputs and resumable checkpoints are kept in `outputs/`.

Tensor sets are `[B,K,C,H,W]`. The prior field operates independently on each
row `[B,C,H,W]`; only the set-slot axis is coupled through the shared-seed
construction.
