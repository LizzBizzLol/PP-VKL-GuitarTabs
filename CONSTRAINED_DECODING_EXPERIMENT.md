# Top-K Constrained Decoding Experiment

Eval-only experiment without training. Goal: test whether raw `SoftmaxGroups` top-k probabilities can improve string/fret decoding without changing the model head or loss.

## Setup

- Active chunk: `electric_clean/lespaul_clean_both`.
- Checkpoint: `training-state-83608.pt`.
- Track source: `generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896`.
- Tracks: same `20` full validation tracks from `STRING_FRET_EXPERIMENT.md`.
- Output artifacts: `generated\diagnostics\string_fret_constrained_lespaul_20tracks` and kept ignored.
- Baseline: current `SoftmaxGroups.finalize_output()` top-1 behavior.

## Variants

| Variant | Description |
|---|---|
| `baseline_top1` | Existing per-string top-1 decoding. |
| `topk3_smooth_light` | Per-string Viterbi over top-3 classes plus silence, weak transition penalty. |
| `topk5_smooth_light` | Per-string Viterbi over top-5 classes plus silence, weak transition penalty. |
| `topk5_smooth_medium` | Top-5 Viterbi with medium temporal smoothness. |
| `topk5_smooth_strong` | Top-5 Viterbi with stronger temporal smoothness. |
| `topk5_smooth_medium_dup` | Medium top-5 smoothing plus conservative duplicate-pitch suppression. |

## Results

| Variant | Exact string/fret F1 | Pitch F1 | Wrong-position/ref | Missed/ref | Extra/pred | Exact delta | Pitch delta | Extra rel delta | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_top1` | 65.43% | 79.23% | 15.53% | 10.83% | 28.72% | 0.00 pp | 0.00 pp | 0.00% | false |
| `topk3_smooth_light` | 65.46% | 79.31% | 15.58% | 10.79% | 28.61% | +0.03 pp | +0.09 pp | -0.40% | false |
| `topk5_smooth_light` | 65.46% | 79.31% | 15.58% | 10.79% | 28.61% | +0.03 pp | +0.09 pp | -0.40% | false |
| `topk5_smooth_medium` | 65.55% | 79.40% | 15.56% | 10.79% | 28.46% | +0.12 pp | +0.18 pp | -0.91% | false |
| `topk5_smooth_strong` | 65.59% | 79.55% | 15.64% | 10.81% | 28.22% | +0.16 pp | +0.32 pp | -1.75% | false |
| `topk5_smooth_medium_dup` | 65.78% | 80.93% | 16.69% | 10.79% | 25.95% | +0.35 pp | +1.70 pp | -9.65% | false |

Acceptance criterion: exact string/fret F1 must improve by at least `+1 pp`, pitch F1 must not drop by more than `1 pp`, and `extra/pred` must not increase by more than `5%` relative. No variant passed.

## Interpretation

- Constrained decoding helps a little, but the gain is too small: best exact F1 delta is only `+0.35 pp`.
- Duplicate-pitch suppression improves pitch F1 and reduces extra notes, but it also increases wrong-position/ref, so it does not solve string/fret assignment.
- The previous top-k diagnostic showed the correct position is usually available in top-5, but this experiment shows simple temporal smoothing and duplicate suppression are not enough to select it reliably.
- The bottleneck is therefore closer to the learned scoring/training signal for string/fret positions than to a trivial decoder rule.

## Decision

- Do not promote constrained decoding as an inference fix.
- Do not start another similar `electric_clean` chunk as the next step.
- Small tab loss experiment is now recorded in `TAB_LOSS_EXPERIMENT.md`.
- Simple focal CE / position-margin loss tuning did not pass acceptance.
- Next engineering step: plan a tab head / label-representation experiment.
- Keep `training-state-83608.pt` as the active clean-domain diagnostic checkpoint, but best aggregate model is still `training-state-58184.pt` until a better validated run appears.

## Repro

```powershell
.\.venv\Scripts\python.exe demo_embedding\diagnose_string_fret_frames.py `
  --decode-experiment `
  --config demo_embedding\tabcnn_synthtab_full_chunk_electric_clean_lespaul_clean_both_28ep_resume.json `
  --track-source-experiment generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896 `
  --max-tracks 20 `
  --checkpoint generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896\models\training-state-83608.pt `
  --output-dir generated\diagnostics\string_fret_constrained_lespaul_20tracks `
  --device auto
```
