# Tab Loss / Label-Signal Experiment

Short controlled train/eval experiment on the active `electric_clean/lespaul_clean_both` chunk. Goal: test whether changing the training signal improves string/fret assignment more than simply training one extra epoch with the existing CE loss.

## Setup

- Active chunk: `electric_clean/lespaul_clean_both`.
- Resume checkpoint: `training-state-83608.pt`.
- Train subset: `800` tracks.
- Validation subset in train configs: `200` tracks.
- Training budget: `1` epoch, `100` optimizer steps per variant.
- Diagnostic subset: same `20` validation tracks from `STRING_FRET_EXPERIMENT.md`.
- Outputs: `generated\experiments\tab_loss_ablation_lespaul_*` and `generated\diagnostics\tab_loss_ablation_lespaul_*`, kept ignored.

## Variants

| Variant | Loss mode | Notes |
|---|---|---|
| `control_ce_1ep` | `ce` | Matched control for one extra epoch from `training-state-83608.pt`. |
| `focal_ce_g1p5_1ep` | `focal_ce` | Weighted focal CE with `focal_gamma=1.5`. |
| `position_margin_l005_m050_1ep` | `ce_plus_position_margin` | CE plus pitch-compatible position margin with weight `0.05` and margin `0.5`. |

Acceptance criterion: exact string/fret F1 must improve by at least `+1 pp` versus matched CE control, pitch F1 must not drop by more than `1 pp`, and `extra/pred` must not increase by more than `5%` relative.

## Results

| Variant | Exact string/fret F1 | Pitch F1 | Wrong-position/ref | Missed/ref | Extra/pred | Exact delta vs CE | Pitch delta vs CE | Extra rel delta | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `control_ce_1ep` | 64.56% | 77.15% | 14.43% | 11.60% | 31.56% | 0.00 pp | 0.00 pp | 0.00% | false |
| `focal_ce_g1p5_1ep` | 65.11% | 77.41% | 14.05% | 11.55% | 31.18% | +0.55 pp | +0.26 pp | -1.19% | false |
| `position_margin_l005_m050_1ep` | 64.58% | 77.11% | 14.35% | 11.64% | 31.61% | +0.02 pp | -0.05 pp | +0.17% | false |

## Interpretation

- `focal_ce` is directionally better than matched CE control, but the improvement is too small for acceptance: `+0.55 pp` exact F1, below the `+1 pp` threshold.
- `ce_plus_position_margin` as implemented is effectively neutral and does not justify a longer run.
- All variants remain below the original `training-state-83608.pt` 20-track diagnostic (`65.43%` exact F1 and `79.23%` pitch F1), so one extra short epoch on an 800-track subset is not a reliable quality improvement.
- Loss-only tuning in this simple form does not solve the string/fret bottleneck.

## Decision

- Do not run full validation for these variants.
- Do not promote focal CE or the current position-margin loss to a longer SynthTab run yet.
- Next useful step: plan a representation/head experiment, for example explicit pitch auxiliary supervision, separate string/fret heads, or a label representation that reduces equivalent-position ambiguity.
- Keep `training-state-83608.pt` as active diagnostic checkpoint and `training-state-58184.pt` as best aggregate checkpoint until a better full validation result exists.

## Repro

```powershell
.\.venv\Scripts\python.exe demo_embedding\tabcnn_synthtab_pipeline.py `
  --config generated\experiments\tab_loss_ablation_lespaul_configs\control_ce_1ep.json `
  --mode train

.\.venv\Scripts\python.exe demo_embedding\diagnose_string_fret_frames.py `
  --config demo_embedding\tabcnn_synthtab_full_chunk_electric_clean_lespaul_clean_both_28ep_resume.json `
  --track-source-experiment generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896 `
  --max-tracks 20 `
  --checkpoint generated\experiments\tab_loss_ablation_lespaul_control_ce_1ep_2026-06-09_21-43-50\models\training-state-83708.pt `
  --output-dir generated\diagnostics\tab_loss_ablation_lespaul_control_ce_1ep `
  --device auto
```

Repeat with the generated focal and position-margin configs/checkpoints listed in `generated\experiments\tab_loss_ablation_lespaul_configs`.
