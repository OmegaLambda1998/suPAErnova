# Posterior step — performance roadmap

The `posterior` step runs, serially, for every `(subset, seed)` pair: a MAP phase
(`TFPosteriorModel.train_model` → many `tfp.optimizer.lbfgs_minimize` solves) followed by an
HMC phase (`train_hmc` → `_sample_hmc` → NUTS via `tfp.mcmc.sample_chain`), then a
post-sampling recompute of per-sample diagnostics.

Two rounds of low-/medium-risk speedups have already landed (see the `perf/posterior-lowrisk`
and `perf/posterior-medrisk` history):

- post-sampling diagnostics recompute is batched (`_recompute_sample_diagnostics`) instead of
  a sequential `tf.map_fn` over `n_run_steps`;
- `clear_session()` no longer runs per MAP chain, so the traced `vals_and_grads` / sampler
  graphs survive across chains and phases;
- a dead `value_and_gradient` in `train_map` was removed;
- HMC run-state checkpoints are append-only per-chunk shards (was O(n_chunks²));
- each HMC phase runs in one `sample_chain` call when not checkpointing/profiling;
- burn-in reuses the run kernel when tree depths match;
- `SNPAE_FAST_MATH=1` opts out of op-determinism + TF32-disabled.

This note records the larger changes that are **not yet done** because they need the ability
to run the parity / smoke suites while iterating. In rough order of value-for-effort:

## M3 — batch the MAP chains of a stage into one `lbfgs_minimize`

`train_model` runs `stage.n_chains` sequential L-BFGS solves (each up to
`max_iterations=2500`) that differ only by their init strategy / seed. `lbfgs_minimize`
already batches over the leading dims of `initial_position` (today it is called with
`[1, sn, n_params]` and optimises each SN independently), so a `[n_chains, sn, n_params]`
initial position runs every chain of a stage in one solve — roughly a chain-count speedup,
minus a "slowest chain in the batch" tax on the shared stopping criterion.

Approach:

1. Keep the per-chain `for c in range(stage.n_chains)` loop, but only to call
   `self.map.setup(stage, c)` and collect `self.map.unconstrain(self.map.position.current)`
   into a list. `tf.stack` → `[n_chains, sn, n_params]`.
2. One `self.lbfgs(stacked)`. `vals_and_grads` already returns a per-`[..., sn]` objective, so
   no change there; `results.*` gain a leading `[n_chains]` axis.
3. Replace the `train_map` record-keeping (the long `tf.where(improved, ...)` chain over
   `self.map.improved` / `chain_min` / `converged` / `failed` / `negative_log_*` /
   per-parameter `.initial` / `.best`) with a **single cross-chain reduction**:
   - `improved_c = (objective_value_c < self.map.negative_log_prob[None]) & converged_c`
     (compare every chain against the running best from prior stages);
   - pick, per SN, the chain with the smallest `objective_value` among `improved_c`;
   - `chain_min` for those SNe = `chain_total_base + argmin_c`; update the other Variables
     from that chain's slice.

Semantic change to be aware of: the current loop is greedy-incremental (chain 2 compares
against chain 1's result if chain 1 improved); the batched version compares every chain in a
stage against the pre-stage best and then takes the best of the batch. Equivalent in intent,
not bit-identical.

Files: `train_model` / `train_map` / `lbfgs` in `steps/posterior/tf/tf.py`; the Variable
containers in `steps/posterior/tf/map.py`.

## H1 — fold the `seed` loop into a batch axis

`Posterior._run` / `_result` / `_save` / `_load` / `_analyse` (`steps/posterior/posterior.py`)
all iterate `for subset in self.subsets: for seed in self.seeds:`, rebuilding a fresh
`TFPosteriorModel` per seed and running a full MAP + HMC serially with no graph reuse across
seeds. Seeds are independent draws used for the R-hat convergence check.

Approach: make `seed` a leading batch dimension.

- HMC: seeds become additional walkers — one `sample_chain` over `n_walkers * n_seeds`
  chains, seeded per-seed via `tfp.random`. `_restricted_to_valid_sn` / `_repacked_to_valid_spec`
  are untouched (they act on `sn` / `spec`).
- MAP: seeds stack onto the chain batch from M3.
- `PosteriorHMCValue` and the `tf.train.Checkpoint` layout (`models[subset][seed]`) gain a
  seed axis; the `_save` / `_load` / `_was_analysed` loops in `posterior.py` collapse.
- `_result` reshaping (`posterior.py:_result`) and `potential_scale_reduction` in
  `load_checkpoint` need the seed axis threaded through.

This is the biggest single win (removes the entire outer serial dimension) and the biggest
refactor — it touches the step lifecycle, the checkpoint format, and the result schema.

## H2 — fold the `subset` loop

`test` / `train` differ only in which SNe are included. Either concatenate them into one `sn`
batch with a subset index and split at `_result` time (reuses one traced graph for both), or
— lower effort — run the two subset iterations in separate processes.

## Config levers (no code change)

- `n_walkers` defaults to `1.0` → `int(NPROC * 1.0)` walkers. On a many-core box this makes
  the HMC batch `NPROC × sn`, and NUTS's shared `while_loop` runs until the slowest of all of
  them U-turns. Confirm the walker count is actually needed for R-hat.
- `n_leapfrog_run` is passed straight to `max_tree_depth` (so `5` ⇒ up to 2⁵−1 = 31 leapfrog
  steps). The run-phase diagnostics already warn on `max_tree_depth` saturation; if they're
  quiet, a smaller `n_leapfrog_run` is exponentially cheaper.
- `checkpoint_hmc = false` skips the per-chunk resume checkpoint and enables the single
  `sample_chain` call per phase.
- `n_chunk_steps` only matters when `checkpoint_hmc` is on; make it divide the phase step
  counts evenly to avoid an extra retrace for the remainder chunk.
