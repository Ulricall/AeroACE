# AeroACE release code

This repository is a release-oriented subset of the current `code/` working tree.
It retains the paper algorithm, the comparisons reported in the paper, and the
additional comparisons introduced during review. Response-only explanation and
visualization code—hidden-state/t-SNE plots, gate plots, lag/stress plots, and
new-setting mechanism experiments—is excluded.

The initial release cleanup did **not** change an algorithm, training rule,
reset rule, random seed, or simulator behavior. A later requested calibration
changed only the documented default compensation/clipping/adaptation parameters
for six response-added baselines; it did not change their model equations,
checkpoints, common PID gains, or simulator. Issues found during the audit are reported in
[`docs/FAIRNESS_AND_RESULTS_AUDIT.md`](docs/FAIRNESS_AND_RESULTS_AUDIT.md);
they are not silently repaired here.

For a later requested AeroACE sensitivity check, the original `0.1`
`c_refer` threshold coefficient was exposed as an argument while remaining the
default. A 10-run regression confirmed byte-identical default output. Values
`0.2` and `0.3` are experimental overrides, not release defaults.

## Important result status

The `results/` directory is intentionally excluded from this GitHub release.
Result paths shown in commands below are output locations created when the
commands are run. Audit outputs mentioned for provenance were retained locally
but are not distributed in this repository.

This directory retains the source algorithm implementations but now includes
the documented response-baseline default-parameter calibration below. It is
not a verified reproduction package.

- The retained controller/model files were checked against the current `code/`
  working tree. A one-second AeroACE checkpoint rollout produced identical
  state/control bytes in `code/` and `code_sub/`.
- The supplied AeroACE checkpoint was also evaluated for 10 original test
  seeds on Figure-8/Wind III. It produced `6.225 ± 16.682 m` position MAE with
  5/10 failed rollouts, rather than the manuscript value `0.105 ± 0.011 m`.
- The same 10 unfiltered seeds were run for all retained executable
  comparisons. Among the 14 rows with an exact reported Figure-8/Wind III
  value, none reproduces both the printed mean and standard deviation after
  rounding to three decimals.
- The earlier `0.357 ± 0.067 m` release result is withdrawn. It was generated
  after the cleanup had incorrectly removed AeroACE's `c_refer` branch and
  therefore evaluated a different algorithm.
- The original audit keeps its unfiltered seed results locally. A later
  arithmetic seed sequence was screened for AeroACE stability; it must not be
  described as an unbiased seed selection.

The complete assessment is in
[`docs/FAIRNESS_AND_RESULTS_AUDIT.md`](docs/FAIRNESS_AND_RESULTS_AUDIT.md).
Per-seed CSVs and aggregate result files are intentionally not included.

## Post-audit parameter tuning

For the six response-added baselines with more than five failures under the
arithmetic seed sequence, method-specific compensation, clipping, or adaptation
parameters were tuned while holding the common PID gains and checkpoints fixed:

```text
PI-TCN:          compensation_gain=0.05, force_bound=20
MANN:            gamma_adapt=0.15, w_bound=5
Pi-Transformer:  compensation_gain=1, force_bound=0.25
Powerformer:     compensation_gain=1, force_bound=0.5
Cluster-Causal:  compensation_gain=0.1, force_bound=200
PINNsFormer:     compensation_gain=0.01, force_bound=200
```

All six changed from more than five failures to `0/10` on seeds 213–303 and
also produced `0/10` failures on seeds 713–803. Pi-Transformer used the latter
sequence during bound selection; on a third sequence not used for tuning it
still had `1/10` failure. Raw result files are intentionally not included; the
selected parameters and reporting limitations are documented here and in the
audit document.

## Retained methods

Paper method and main-paper comparisons:

```text
aeroace
pid
omac
neural_fly
ood_control
decision_transformer
```

Comparisons added during review:

```text
dmrac
neural_bem
pitcn
mann
agile_full
pi_transformer
powerformer
causal_transformer
cluster_causal
pinnsformer
rtnmpc
mlmpc
```

The reported fixed-gate and vanilla-GRU controller variants are exposed as
`aeroace_fixed_gate` and `aeroace_vanilla_gru`. The exact retained/excluded
inventory is in [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).

## Environment

Python 3.11 was used for the release audit.

```bash
cd AeroACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch wheels are platform-specific. If the pinned wheel is unavailable,
install the appropriate PyTorch 2.1.x build first and then install the remaining
requirements.

## Source command behavior

Run commands from the repository root:

```bash
python run.py --model pid --trace fig8 --wind gale --test_rounds 10
python run.py --model contrast --trace fig8 --wind gale --test_rounds 10
python run.py --model dmrac --trace fig8 --wind gale --test_rounds 10
```

`contrast` preserves the current source order and behavior:

- PID: test only;
- OMAC and OoD-Control: ten training rollouts, then test;
- Neural-Fly: test only;
- the source implementation named Transformer: ten rollouts through `train()`,
  then test.

The original AeroACE entry point trains before testing:

```bash
python run.py \
  --model aeroace \
  --trace fig8 \
  --wind gale \
  --aero_stage1_eps 100 \
  --aero_stage2_eps 10 \
  --test_rounds 10
```

It does **not** load `params/aeroace_trained.pt`. The checkpoint diagnostic used
during the local audit was kept separate so that the existing
`--model aeroace` semantics were not changed.

Use the separate test-only wrapper for the retained checkpoint:

```bash
python run_aeroace_checkpoint_evaluation.py \
  --checkpoint params/aeroace_trained.pt \
  --output results/aeroace_checkpoint_evaluation \
  --rounds 10 \
  --test-seed-a 213 \
  --test-seed-b 10
```

The original threshold coefficient remains the default. For an isolated
sensitivity check, pass `--c-refer-coefficient`. The requested `0.1`, `0.2`,
and `0.3` comparison was recorded during the local audit; its result files are
not included in this repository.

Test-only commands for supplied comparison checkpoints follow the source
interfaces, for example:

```bash
python run.py --model pitcn --pitcn_train 0 --pitcn_test 1 \
  --pitcn_load_ckpt params/pitcn.pt --trace fig8 --wind gale

python run.py --model mann --mann_train 0 --mann_test 1 \
  --mann_load_ckpt params/mann_feature.pt --trace fig8 --wind gale

python run.py --model neural_bem --neuralbem_train 0 --neuralbem_test 1 \
  --neuralbem_load_ckpt params/neuralbem.pt --trace fig8 --wind gale

python run.py --model agile_full --agile_train 0 --agile_test 1 \
  --agile_load_ckpt params/agile_full.pt --trace fig8 --wind gale
```

Checkpoint hashes and limitations are recorded in
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

## Preserved evaluation details

The following details intentionally match the current source, even where the
audit identifies a fairness or reproducibility concern:

- generic `train()` uses seeds `0, ..., 9`;
- AeroACE Stage 1 uses the episode index as seed;
- AeroACE Stage 2 uses `1000 + episode`;
- the original release audit used `100 + 11 * round`; the current `run.py`
  default is `213 + 10 * round`, and `--test_seed_a/--test_seed_b` expose the
  arithmetic sequence explicitly;
- `Quadrotor.run()` retains the source measurement-state alias behavior;
- the simulator passes `wind_force` to controller calls during testing when
  `expose_wind_gt_to_controller` is absent or true;
- OMAC/Neural-Fly reset behavior and Transformer memory/context updates are
  unchanged from `code/`;
- missing-checkpoint handling and per-method train/test defaults are unchanged.

These details must not be interpreted as a fairness endorsement. See the audit
for their method-specific consequences.

## AeroACE inference branch

The release default preserves the current working-tree coefficient:

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

`c_refer_threshold_coefficient` defaults to `0.1`; making it explicit does not
change the default calculation. The `0.1` regression CSV remains
byte-for-byte identical to the pre-parameterization result.

Do not replace this with direct dictionary retrieval when reproducing the
current source.

## Batch execution and result comparison

`run_release_evaluation.py` is an orchestration wrapper. It invokes the existing
`run.py` commands and aggregates their per-seed CSV output; it does not replace
controller or metric code.

```bash
python run_release_evaluation.py \
  --trace fig8 \
  --wind gale \
  --rounds 10 \
  --test-seed-a 213 \
  --test-seed-b 10 \
  --output results/release_evaluation

python compare_reported_results.py \
  --summary results/release_evaluation/summary.csv
```

For AeroACE variants, the wrapper writes newly trained checkpoints inside the
selected output directory, avoiding accidental overwrite of retained
checkpoints. For the response baselines, the wrapper explicitly loads the
retained release checkpoints, records the selected controller parameters in
`execution_status.csv`, and treats a missing checkpoint-load confirmation as a
failed run. It does not make the methods' different training procedures equal.
The complete 17-baseline rerun was recorded during the local audit. Its result
files are not included in this repository; the relevant qualifications remain
documented in the fairness audit.

## Tests

```bash
python -m unittest -v \
  test_aeroace_deployment_signal.py \
  test_aeroace_online_update.py
```

These are the two source tests retained in the release. The previously added
`test_release_protocol.py` was removed because it asserted cleanup-specific
algorithm changes rather than the current source behavior.

## Known release blockers

Before claiming a complete reproducible release:

1. provide the exact result-generating source revision and checkpoint for every
   reported row;
2. provide training/validation seeds, data manifests, hyperparameter-search
   records, package versions, and exact commands;
3. reconcile the executable wind process and method implementations with the
   manuscript descriptions;
4. regenerate the tables from the released artifacts or revise the reported
   numbers;
5. recover the missing inverted-pendulum experiment and the missing manuscript
   ablation implementations from the exact result-generating source revision;
6. add a project license and replace the placeholder ROS package maintainer
   metadata.

Until then, this directory supports source inspection and transparent reruns,
but it should not be described as reproducing all manuscript results.
