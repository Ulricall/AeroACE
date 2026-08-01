# Release scope

Scope was determined from the final `main.tex` and
`supplementary_nmi.tex`, with `revision/response_round2.tex` used for
response-added comparisons. The final manuscript takes precedence when an older
response statement conflicts with it.

## Retained

### Paper method

- AeroACE: FGRU, Expert Dictionary, bilinear residual-force prediction, static
  dictionary retrieval followed by the current-source `c_refer` threshold
  branch, and the optional online Expert Dictionary update.
- Fixed-gate and vanilla-GRU controller variants referenced in the
  response/manuscript.
- PX4/MAVROS deployment adapter and onboard residual-force reconstruction.

### Main-paper comparisons

- PID
- OMAC
- Neural-Fly
- OoD-Control
- the source implementation labelled Decision Transformer/Transformer

### Comparisons added during review

- DMRAC
- NeuroBEM (the class is named `NeuralBEMController`)
- PI-TCN
- MANN
- Agile
- Pi-Transformer
- PowerFormer
- Causal Transformer
- Causal Attention Masking (the run name is `cluster_causal`)
- Pinnsformer
- RTN-MPC
- MLMPC

### Shared support

- common simulator, trajectories, parameters, metrics, per-seed CSV writer, and
  result aggregation/comparison utilities;
- the main-paper Zig-Zag real-world trajectory definition (not included in the
  numerical-table release runner);
- default checkpoints found in the source tree;
- generic saved-log evaluation and the main trajectory plot utility;
- the two source AeroACE unit-test files.

## Deliberately excluded

The following source-tree material is not part of the release candidate:

- the nested `code/.git` repository and its author/remote metadata;
- `__pycache__`, `.DS_Store`, temporary files, internal notes, copied papers,
  revision text, and temporary development logs. The command logs retained
  under the final baseline-rerun result are execution provenance for the
  published CSVs, not development logs;
- response-only bound and theorem-analysis scripts (`r2_2_*`);
- response-only online-update/noise experiment drivers and generated response
  text (`r2_4_*`);
- wind-characterization and gust/lag experiment/plot drivers (`r2_5_*`,
  `r2_6_*`, `r3_3_*`);
- hidden-state, latent-representation, gate-regime, PCA, t-SNE, retrieval-weight,
  and dictionary-mechanism experiment/plot drivers (`r3_1_*`,
  `plot_fgru_gate_wind.py`, `plot_fgru_tsne.py`,
  `plot_tsne_reference.py`);
- response-generated figures, tables, logs, and new-setting projections;
- `dryden.py` as originally present, because it is a standalone demonstration
  script with plotting code and is not called by the paper evaluation path;
- Neural-Lander implementation/checkpoint, because the final manuscript cites it
  only as related work and does not report it as an evaluated comparison;
- LPF+GRU and force-gated non-recurrent MLP response-analysis controller variants
  and the associated checkpoint;
- sequence-length, force-prediction-MAE, and dictionary-report response
  experiment entry points.

The optional online Expert Dictionary update remains because it is part of the
final manuscript appendix and deployment algorithm. Its response-specific
plotting and generated-text utilities are excluded.

## Present in the manuscript but missing from `code/`

No implementation or loadable artifact was found for:

- the inverted-pendulum simulator, controller adaptations, training code, or
  result-generating entry point for the six-method pendulum table;
- the Mamba replacement for FGRU;
- the learnable-MLP replacement for the Expert Dictionary;
- Force-Vector Memory;
- the AeroACE+MANN Expert Dictionary replacement used in the manuscript
  ablation table;
- the manuscript ablations described as Raw-state NN, Dictionary-only, and
  FGRU-only with an MLP head.

The vanilla Transformer replacement described in the manuscript ablation table
also cannot be identified unambiguously. The response-added standalone
Transformer controllers retained here do **not** implement the manuscript claim
that each variant replaces only the FGRU while keeping the Expert Dictionary and
all other components identical.

Because these implementations are absent from the supplied source tree, this
release candidate cannot reproduce every manuscript table row. They should be
added from the exact result-generating source revision rather than re-created
from prose after the fact.

## Entry-point policy

`run.py --model` exposes the paper method and reported comparison methods.
Development-only PI-TCN, DMRAC, and Agile component variants may remain as
internal class/building-block code where the retained full method depends on
them, but they are not included in the default release evaluation.

All public evaluation paths use `test()` in `run.py`; a comparison method should
not introduce a separate metric implementation or test seed loop.

The initial cleanup preserved command behavior from the current `code/` working
tree. The later requested response-baseline calibration changes only the
documented default compensation/clipping/adaptation parameters for six methods;
the original values remain reproducible through explicit command-line flags.
The release does not use the
nested repository's `HEAD` as the algorithm source, because the working tree
contains the author's corrected AeroACE inference branch. Other fairness or
reproducibility problems remain documented rather than being hidden through
algorithm, reset, simulator-input, data-split, or checkpoint changes.

The original AeroACE `c_refer` threshold coefficient is also exposed as an
explicit argument for the requested sensitivity check. Its release default
remains `0.1`, and the default-value regression is byte-identical to the result
from before parameterization. The tested `0.2` and `0.3` values are recorded
only as experimental overrides.
