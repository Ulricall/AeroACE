# Fairness and results audit

Audit date: 2026-07-28

The `results/` directory referenced by this historical audit is intentionally
not included in the GitHub release. Numerical summaries needed to interpret
the findings remain in this document.

## Post-audit note

This audit records the source-faithful state before the later requested
parameter calibration. The current release changes only the documented
default compensation/clipping/adaptation parameters for PI-TCN, MANN,
Pi-Transformer, Powerformer, Cluster-Causal, and PINNsFormer. The original
audit conclusions below are retained rather than overwritten. Raw result files
from the tuning and second-sequence validation are not included in this GitHub
release.

## Verdict

1. At the time of this audit, `code_sub/` preserved the current `code/`
   behavior for every retained algorithm and experiment function. It was a
   scope cleanup, not an algorithm correction.
2. The earlier release verification is invalid and has been withdrawn. It was
   produced after algorithm and evaluation behavior had been changed during
   cleanup.
3. The current source does not evaluate all methods under a defensibly fair or
   fully reproducible protocol. These problems remain visible in the release
   and are documented below.
4. The supplied AeroACE checkpoint, original test seeds, and restored inference
   branch do not reproduce the manuscript. Figure-8/Wind III gives
   `6.225 ± 16.682 m` with 5 failures in 10 runs, versus the reported
   `0.105 ± 0.011 m`.
5. The manuscript's inverted-pendulum table cannot be audited from this
   release: no pendulum simulator, controller adaptation, training code, or
   result entry point exists anywhere in the supplied `code/` tree.

## What was incorrectly changed during the first cleanup

The first `code_sub/` version exceeded the requested scope. It changed:

- AeroACE inference from the source `c_inf`/`c_refer` threshold branch to direct
  dictionary retrieval;
- AeroACE Stage-1 environment-embedding treatment;
- OMAC and Neural-Fly reset behavior;
- Transformer test-time memory/context updates;
- RTN-MPC model reset and replay sampling;
- simulator measurement-state aliasing and test-time `wind_force` input;
- training seeds, test seeds, and train/validation splitting;
- checkpoint loading and missing-checkpoint behavior;
- which methods were trained before testing.

A temporary test-seed list was also selected after scanning AeroACE outcomes to
avoid unstable rollouts. That is outcome-dependent sample selection and is not
a valid evaluation. It has been removed.

The withdrawn AeroACE value `0.357 ± 0.067 m` came from the direct-retrieval
version and therefore does not describe the current source algorithm.

## Release-to-source parity

The authoritative source for this cleanup is the current `code/` working tree,
not the nested repository's `HEAD`. The working tree contains the author's
corrected AeroACE branch:

```python
c_inf = self.expert_dict.retrieve(h_last).detach().numpy()
self.c_refer = self.get_c_refer(self.h, c_inf)
if (
    np.linalg.norm(c_inf - self.c_refer)
    <= np.sqrt(self.hidden_dim) * self.c_refer_threshold_coefficient
):
    self.c = c_inf
else:
    self.c = self.c_refer
```

The release exposes this coefficient for sensitivity checks but keeps the
source value `0.1` as the default. A full default-value regression produced a
byte-identical result CSV.

Parity checks found:

- retained model/support files are byte-identical;
- common retained controller classes are syntax-tree identical, except
  `AeroACE`, whose differences are removal of passive response-plot histories
  and the default-preserving explicit threshold argument;
- `quadsim.py` differs only by removal of `fgru_gate` extraction/logging;
- common `run.py` functions are syntax-tree identical except the scope
  dispatcher, passive CSV recording, and removal of the response-only Expert
  Dictionary report call;
- an AeroACE checkpoint probe using seed 100 produced identical state/control
  bytes in `code/` and `code_sub/`:
  SHA-256 `8aa2931efb4f8401a47c704fab5d1f35377ec69e7e5057fa4789162868110cc3`;
- the two retained test files pass 14 tests.

These checks establish release-to-current-source parity. They do not establish
paper reproducibility.

## Fairness assessment

All public methods eventually use the same `Quadrotor` implementation,
trajectory object, physical parameter files, actuator limits, test seed formula,
wind array for a given seed, and metric function. That common outer loop is not
sufficient to establish fairness.

| Issue | Evidence in current source | Assessment |
|---|---|---|
| Training budgets | Methods use different rollout counts, update frequencies, epochs, windows, online steps, and checkpoint histories. | Not equal; no common optimization-step ledger. |
| Hyperparameter budget | No executable record of the reported 50-trial search per method was found. | Unsupported. |
| Training/test separation | Generic training uses seeds 0–9 and AeroACE Stage 2 uses 1000+, but PI-TCN, NeuralBEM, Agile, and Transformer dataset collectors use the same `100 + 11*i` family as testing. | Method-dependent and potentially overlapping. |
| Validation separation | PI-TCN and several Transformer paths randomly split windows built from the same rollout collection. | Windows from one rollout can appear in both training and validation. |
| Test-time ground truth | `Quadrotor.run()` passes `wind_force` into controller calls in test as in source. `MetaAdaptTransformer` updates memory/context from it. | Information use differs by method; not a blind test protocol. |
| Reset behavior | `MetaAdaptDeep.reset_controller()` recreates `phi`; `NeuralFly.reset_controller()` recreates `h`. | Learned state can be discarded at rollout reset. |
| Main-paper training calls | `contrast` trains OMAC, OoD-Control, and Transformer, but tests Neural-Fly without calling `train()`. Generic `train()` does not call `MetaAdaptTransformer.train_model()`. | Training treatment differs and the Transformer optimizer is not stepped by that path. |
| Random initialization | Most model construction is not preceded by an explicit seed in the current source. | Runs without checkpoints can vary between process launches. |
| Measurement noise | `X_meas = X` aliases the physical state before noise is added. | Measurement noise also mutates the simulated state. |
| Base PID arguments | When `given_pid=True`, the supplied integral gain survives, while proportional/derivative gains are overwritten from `controller.json`. | The CLI `p` and `d` values do not control the final P/D matrices. |
| Checkpoint policy | Some methods load their default checkpoint, while others require an explicit load argument or otherwise use random initialization. | A common command does not imply a common trained-artifact policy. |
| Checkpoint provenance | Payloads lack source revision, data/seed manifest, tuning trial, environment, and manuscript-row identity. | Reported-row provenance is unsupported. |
| Wind model | The executable evaluation integrates Gaussian wind-acceleration samples; the standalone Dryden script is not called. | Does not support the manuscript Dryden claim. |

The release intentionally preserves these behaviors. Changing them would create
a new evaluation protocol and would require separately named experiments,
retraining, and new reported results.

## Method-identity assessment

- The source class labelled Transformer predicts disturbance force inside a PID
  controller; it is not an end-to-end state-to-action Decision Transformer.
- The five response-added Transformer controllers are standalone
  PID-plus-predictor implementations. They do not merely replace FGRU while
  retaining AeroACE's Expert Dictionary.
- `NeuralBEMController` reuses FGRU and Expert Dictionary components; available
  evidence does not establish equivalence to the cited NeuroBEM formulation.
- MLMPC has no supplied meta-trained basis artifact.
- The source itself describes the DMRAC, MANN, PI-TCN, and Agile additions as
  practical integrations or simplifications rather than literal reproductions
  of every cited system.

Consequently, a numerical difference cannot be interpreted only as optimizer or
seed variance; in several cases the executable method identity also differs from
the manuscript description.

## Source-faithful Figure-8 / Wind III rerun

The rerun uses the original test seeds
`100, 111, 122, 133, 144, 155, 166, 177, 188, 199`, sample standard deviation
(`ddof=1`), current source algorithms, and no outcome-based seed filtering.
Methods with supplied checkpoints are invoked through their explicit source
load arguments. Main-paper methods without checkpoints follow the current
source train/test behavior. AeroACE is listed from a separate supplied-checkpoint
diagnostic because the source `--model aeroace` command trains rather than loads
that file.

| Method | Measured MAE (m) | Reported MAE (m) | Failed runs | Comparison |
|---|---:|---:|---:|---|
| PID | 0.356 ± 0.067 | 0.398 ± 0.056 | 0/10 | Exact setting; does not match reported rounding |
| OMAC | 0.574 ± 1.109 | 0.227 ± 0.037 | 1/10 | Exact setting; does not match |
| Neural-Fly | 0.231 ± 0.036 | 0.206 ± 0.032 | 0/10 | Exact setting; does not match |
| OoD-Control | 0.187 ± 0.024 | 0.189 ± 0.029 | 0/10 | Exact setting; does not match reported rounding |
| Transformer | 0.944 ± 1.793 | 0.402 ± 0.053 | 2/10 | Exact setting; does not match |
| AeroACE | 6.225 ± 16.682 | 0.105 ± 0.011 | 5/10 | Exact setting; supplied-checkpoint diagnostic; does not match |
| DMRAC | 0.364 ± 0.069 | 0.517 ± 0.103 | 0/10 | Exact setting; does not match |
| NeuroBEM | 10.965 ± 33.479 | 0.386 ± 0.074 | 1/10 | Exact setting; does not match |
| PI-TCN | 288.027 ± 168.321 | 0.373 ± 0.075 | 10/10 | Exact setting; does not match |
| MANN | 1.974 ± 1.318 | 0.475 ± 0.106 | 10/10 | Exact setting; does not match |
| Agile | 1.462 ± 1.339 | 0.378 ± 0.065 | 7/10 | Exact setting; does not match |
| RTN-MPC | 0.781 ± 1.095 | 2.337 ± 0.417 | 2/10 | Exact setting; does not match |
| MLMPC | 3.106 ± 5.999 | 2.124 ± 0.356 | 5/10 | Exact setting; does not match |
| Pi-Transformer | 28.470 ± 59.195 | 0.359 ± 0.058 | 10/10 | Reported wind is unspecified |
| PowerFormer | 1.732 ± 1.887 | 0.389 ± 0.127 | 10/10 | Reported wind is unspecified |
| Causal Transformer | 0.356 ± 0.067 | 0.483 ± 0.163 | 0/10 | Reported wind is unspecified |
| Causal Attn. Masking | 13.230 ± 32.857 | 0.424 ± 0.102 | 10/10 | Reported wind is unspecified |
| Pinnsformer | 161.527 ± 144.361 | 0.386 ± 0.084 | 10/10 | Reported wind is unspecified |
| Fixed-gate ablation | 1.156 ± 1.707 | — | 6/10 | Checkpoint probe; response reports only a different sudden-gust setting |
| GRU replacement | 1.020 ± 1.660 | 0.218 ± 0.279 | 6/10 | Exact setting; supplied-checkpoint probe; does not match |

For the 14 rows above that have an exact Figure-8/Wind III reported value,
none matches both the printed mean and standard deviation after rounding to
three decimals. The fixed-gate and GRU probes use the same closed-loop seed,
wind, trajectory, reset, metric, and failure-threshold conventions, but are
identified separately because `run.py` has no test-only public entry point for
those checkpoints.

The Transformer comparison table in the manuscript does not identify Wind
I/II/III, so its printed values are not an exact setting match even when shown
for scale.

The per-seed CSV files and aggregate output files are not included in this
GitHub release. The numerical summary needed for the assessment is retained in
the table above.

### Successful-rollout-only diagnostic

A secondary diagnostic applied the existing 2 m/75 degree safety rule and
summarized only successful rollouts without replacing the all-run statistics.
Its CSV output is not included in this GitHub release.

There is no common failure-free seed subset across all methods: six methods
fail all 10 evaluated seeds. Per-method successful subsets therefore compare
different samples, and selecting subsets according to proximity to the
manuscript would be outcome-dependent selection. These conditioned values must
always be reported together with the successful-run count and failure rate.

## AeroACE checkpoint diagnosis

The supplied checkpoint is byte-identical in `code/` and `code_sub/`:

```text
12ce374e08f0caf2b64f8668425adf5cd680241eead8cea63132fa547bb06dfa
```

It stores 4,000 dictionary entries. With the restored branch and original seeds,
individual MAEs range from approximately `0.076 m` to `53.471 m`. The large
sample standard deviation is driven by real unstable rollouts, not aggregation
error. Selecting only stable seeds can produce a small-looking mean, but would
be invalid.

The manuscript states that the reported configuration uses `E=10000`. The
checkpoint's configured capacity is 10,000, but its payload contains only 4,000
stored entries. The fixed-gate checkpoint also contains 4,000 entries, whereas
the vanilla-GRU checkpoint contains only 111. These artifacts therefore do not
establish the matched dictionary-size comparison described by the manuscript.

The present artifacts do not determine whether the manuscript used a different
checkpoint, a different source revision, a different training run, or additional
configuration not preserved in the repository.

## Claim-to-evidence summary

| Claim | Status |
|---|---|
| `code_sub` is an algorithm-faithful subset of the current working tree. | Supported by static and numerical parity checks. |
| All methods share an identical test outer loop and seed formula. | Largely supported for public `test()` paths. |
| All methods are fairly trained and receive the same information at test time. | Unsupported. |
| All reported method descriptions match executable implementations. | Unsupported for several comparisons. |
| Supplied checkpoints generated the manuscript rows. | Unclear; provenance is missing and reruns do not match. |
| AeroACE Figure-8/Wind III reproduces `0.105 ± 0.011 m`. | Unsupported by the supplied checkpoint; measured `6.225 ± 16.682 m`, 5/10 failures. |
| The inverted-pendulum table can be regenerated from the supplied source. | Unsupported; the complete pendulum implementation is absent. |
| The executable evaluation uses Dryden turbulence. | Unsupported. |

## Release decision

Suitable now:

- inspecting the current implementations;
- rerunning the current source behavior;
- auditing method and artifact provenance;
- developing a separately named corrected protocol.

Not suitable now:

- claiming that all methods are fairly compared;
- claiming complete reproduction of manuscript tables;
- replacing manuscript numbers with the withdrawn cleanup results;
- hiding unstable seeds or silently changing algorithms to improve agreement.

The missing implementations and artifacts are listed in
[`RELEASE_SCOPE.md`](RELEASE_SCOPE.md), and checkpoint limitations are listed in
[`CHECKPOINTS.md`](CHECKPOINTS.md).
