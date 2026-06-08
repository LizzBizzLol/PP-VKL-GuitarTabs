# String/Fret Frame Experiment

Короткий eval-эксперимент без обучения. Цель: проверить, насколько просадка `tablature F1` связана с выбором струны/лада при уже правильно найденном pitch.

## Setup

- Active chunk: `electric_clean/lespaul_clean_both`.
- Track source: `generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896`.
- Tracks: `20` full validation tracks.
- Selection: `10` worst high-gap, `5` typical, `5` best.
- Excluded: tracks with `ref_silence_ratio > 95%`, because they are not useful for string/fret diagnostics.
- Output artifacts: `generated\diagnostics\string_fret_lespaul_20tracks` and kept ignored.

## Results

| Checkpoint | Exact string/fret F1 | Pitch-only F1 | Oracle tab F1 | Canonical F1 | Wrong-position/ref | Missed/ref | Extra/pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| `training-state-58184.pt` | 61.67% | 73.65% | 73.65% | 55.13% | 14.25% | 12.43% | 36.45% |
| `training-state-83608.pt` | 65.43% | 79.23% | 79.23% | 58.71% | 15.53% | 10.83% | 28.72% |

Raw counts for `training-state-83608.pt`:

- reference notes: `335363`
- predicted notes: `419537`
- exact string/fret true positives: `246969`
- pitch true positives: `299041`
- pitch-correct but string/fret-wrong notes: `52072`
- missed notes: `36322`
- extra notes: `120496`

## Interpretation

- Latest checkpoint `training-state-83608.pt` is better than `training-state-58184.pt` on the same lespaul subset.
- The gap between exact F1 and pitch-only/oracle F1 is `13.80 pp` for `training-state-83608.pt`, so string/fret assignment is a real bottleneck.
- The model still has non-position errors too: missed notes and extra notes remain significant.
- Canonical hard post-processing is worse than exact model output: `58.71%` vs `65.43%`.
- A simple deterministic pitch-to-position rule should not be used as an inference fix.

## Decision

- Do not roll back to `training-state-58184.pt` for the active clean-domain subset; `training-state-83608.pt` is better here.
- Do not implement hard canonical post-processing.
- Next useful engineering step is logits-aware or model-aware string/fret work: inspect `SoftmaxGroups` probabilities, add top-k position diagnostics, then decide between smarter decoding and loss/head changes.
- Blindly adding another similar `electric_clean` chunk is still not justified.

## Repro

```powershell
.\.venv\Scripts\python.exe demo_embedding\diagnose_string_fret_frames.py `
  --config demo_embedding\tabcnn_synthtab_full_chunk_electric_clean_lespaul_clean_both_28ep_resume.json `
  --track-source-experiment generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896 `
  --max-tracks 20 `
  --checkpoint generated\experiments\full_chunk_acoustic_luthier_pick_part1_28ep_resume_from_50372\models\training-state-58184.pt `
  --checkpoint generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896\models\training-state-83608.pt `
  --output-dir generated\diagnostics\string_fret_lespaul_20tracks `
  --device auto
```
