# Logits Top-K Experiment

Eval-only диагностика raw `SoftmaxGroups` logits/probabilities. Цель: понять, находится ли правильная струна/лад рядом в вероятностях модели, или модель вообще не дает правильной позиции высокий score.

## Setup

- Active chunk: `electric_clean/lespaul_clean_both`.
- Checkpoint: `training-state-83608.pt`.
- Track source: `generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896`.
- Tracks: те же `20` full validation tracks из `STRING_FRET_EXPERIMENT.md`.
- Output artifacts: `generated\diagnostics\string_fret_topk_lespaul_20tracks` and kept ignored.
- Output layout: `6` string groups x `21` classes; class `20` is silence.

## Results

| Metric | Value |
|---|---:|
| Reference notes | `335363` |
| Exact matched notes | `246969` |
| Pitch-correct but string/fret-wrong notes | `52072` |
| Correct position top-1 | 73.64% |
| Correct position top-3 | 95.41% |
| Correct position top-5 | 98.40% |
| Pitch-compatible position top-1 | 30.43% |
| Pitch-compatible position top-3 | 65.88% |
| Pitch-compatible position top-5 | 87.48% |
| PCW correct position top-5 | 99.78% |
| PCW pitch-compatible position top-5 | 99.19% |
| Avg correct-position rank | `1.46` |
| Avg pitch-compatible rank | `3.49` |
| Avg PCW correct-position rank | `2.07` |
| Avg PCW pitch-compatible rank | `2.07` |
| Avg PCW top1-minus-correct probability gap | 49.12% |
| Avg extra-note margin over silence | 54.74% |
| Extra notes with margin <= 10% | 6.50% |

`PCW` means pitch-correct but string/fret-wrong under top-1 decoding.

## Interpretation

- The correct string/fret is usually available in top-k: `98.40%` top-5 overall and `99.78%` top-5 for pitch-correct/tab-wrong notes.
- This supports a smarter decoding experiment before changing the model head or loss.
- The average PCW probability gap is large: the chosen top-1 class beats the correct class by about `49.12 pp`, so simple threshold nudging is unlikely to be enough.
- Extra notes are usually confident, not weak borderline activations: only `6.50%` have note-vs-silence margin `<= 10%`.
- Silence/activation calibration alone is not the main fix.

## Decision

- Do not start another SynthTab chunk right now.
- Do not implement canonical hard post-processing.
- Next step: top-k constrained decoding experiment using the existing logits.
- If constrained decoding cannot recover measurable F1, then plan tab head/loss/label representation changes.

## Repro

```powershell
.\.venv\Scripts\python.exe demo_embedding\diagnose_string_fret_frames.py `
  --topk `
  --config demo_embedding\tabcnn_synthtab_full_chunk_electric_clean_lespaul_clean_both_28ep_resume.json `
  --track-source-experiment generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896 `
  --max-tracks 20 `
  --checkpoint generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896\models\training-state-83608.pt `
  --output-dir generated\diagnostics\string_fret_topk_lespaul_20tracks `
  --device auto
```
