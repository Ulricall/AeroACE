# AeroACE release code

This repository contains the AeroACE algorithm, the comparisons reported in the
paper, and the additional comparisons introduced during review.

## Implemented methods

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

The fixed-gate and vanilla-GRU controller variants are exposed as
`aeroace_fixed_gate` and `aeroace_vanilla_gru`.

## Environment

Python 3.11.

```bash
cd AeroACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch wheels are platform-specific. If the pinned wheel is unavailable,
install the appropriate PyTorch 2.1.x build first and then install the remaining
requirements.

## Running the algorithms

Run commands from the repository root.

Baseline models:

```bash
python run.py --model pid --trace fig8 --wind gale --test_rounds 10
python run.py --model contrast --trace fig8 --wind gale --test_rounds 10
python run.py --model dmrac --trace fig8 --wind gale --test_rounds 10
```

AeroACE (trains before testing):

```bash
python run.py \
  --model aeroace \
  --trace fig8 \
  --wind gale \
  --aero_stage1_eps 100 \
  --aero_stage2_eps 10 \
  --test_rounds 10
```

Evaluating the supplied AeroACE checkpoint (test-only):

```bash
python run_aeroace_checkpoint_evaluation.py \
  --checkpoint params/aeroace_trained.pt \
  --output results/aeroace_checkpoint_evaluation \
  --rounds 10
```

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
