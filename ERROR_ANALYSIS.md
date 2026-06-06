# Error Analysis

Короткая диагностика после четырех meaningful SynthTab Full chunks.

## Runs

| Run | Chunk | Checkpoint | Multi-pitch F1 | Tablature F1 | Accuracy | Non-silent accuracy | Collapse |
|---|---|---|---:|---:|---:|---:|---|
| 1 | `electric_clean/semihollow_clean_finger` | `training-state-12712.pt` | 70.71% | 48.72% | 80.18% | 65.89% | false |
| 2 | `electric_distortion/semihollow_clean_finger` | `training-state-26544.pt` | 69.02% | 48.11% | 81.18% | 66.04% | false |
| 3 | `electric_muted` | `training-state-50372.pt` | 73.07% | 56.25% | 87.25% | 70.79% | false |
| 4 | `acoustic/luthier_pick/part_1_-_1_to_B_C` | `training-state-58184.pt` | 77.25% | 58.47% | 78.01% | 76.57% | false |

## What Changed

- Второй chunk не дал заметного прироста: `tablature F1` остался около `48%`.
- `multi_pitch F1` тоже не вырос: `70.71% -> 69.02%`.
- Третий contrastive chunk `electric_muted` дал заметный прирост: `tablature F1` вырос до `56.25%`, `multi_pitch F1` до `73.07%`.
- Четвертый acoustic chunk дал дополнительный прирост: `tablature F1` вырос до `58.47%`, `multi_pitch F1` до `77.25%`, `non_silent_accuracy` до `76.57%`.
- `accuracy` на acoustic chunk ниже, чем на muted, потому что validation менее silence-heavy: `ref_silence_ratio=67.64%` против `78.55%` на muted.
- `collapse_to_silence=false` на всех четырех chunks, значит anti-collapse параметры работают.

## Main Failure Mode

Главная проблема сейчас не чистый silence-collapse, а разрыв между pitch detection и переводом pitch в string/fret:

| Run | Median MP F1 | Median Tab F1 | Median MP-Tab Gap | Mean MP-Tab Gap |
|---|---:|---:|---:|---:|
| Clean | 87.16% | 56.15% | 20.80% | 22.00% |
| Distortion | 85.38% | 55.31% | 18.57% | 20.91% |
| Muted | 90.62% | 63.20% | 13.63% | 16.82% |
| Acoustic | 85.99% | 65.31% | 17.40% | 18.77% |

Практический вывод: проблема string/fret assignment остается. `electric_muted` сильнее всего уменьшил MP-Tab gap, а acoustic поднял aggregate F1 и non-silent accuracy, но gap снова стал около `17-19 pp`. Значит правильно выбранные контрастные chunks пока помогают, но после еще одного run стоит снова оценить, не пора ли переходить к tab head/loss/label representation.

## Silence Behavior

| Run | Ref silence | Pred silence | Delta |
|---|---:|---:|---:|
| Clean | 75.09% | 62.25% | -12.84 pp |
| Distortion | 77.62% | 65.21% | -12.41 pp |
| Muted | 78.55% | 69.20% | -9.35 pp |
| Acoustic | 67.64% | 53.24% | -14.40 pp |

Модель не схлопнулась в silence. Наоборот, она системно предсказывает больше non-silent, чем есть в reference. Это не повод включать `balance_by_silence=true` прямо сейчас: проблема не в трусливом молчании, а в лишней активности и string/fret ошибках.

По per-track diagnostics:

- Clean: среди non-silent tracks `530/708` треков имеют pred silence ниже ref silence больше чем на `10 pp`.
- Distortion: среди non-silent tracks `595/807` треков имеют pred silence ниже ref silence больше чем на `10 pp`.
- Muted: pred silence все еще ниже ref silence, но разрыв сократился до `-9.35 pp`.
- Acoustic: pred silence снова заметно ниже ref silence, разрыв `-14.40 pp`.
- Under-prediction notes по-прежнему не выглядит основной проблемой: на первых двух runs было только по `6` tracks с pred silence выше ref silence больше чем на `10 pp`.

## Example Tracks

High pitch-to-tab gap, clean:

| Track | MP F1 | Tab F1 | Gap | Ref/Pred silence | Non-silent acc |
|---|---:|---:|---:|---:|---:|
| `Creed - Bullets (3)` | 94.9% | 0.0% | 94.9 pp | 93.1% / 84.9% | 0.0% |
| `R.E.M. - Bad Day` | 99.7% | 4.9% | 94.8 pp | 92.8% / 85.7% | 86.0% |
| `Dionysos - Feel Can Dali` | 97.2% | 13.5% | 83.7 pp | 45.4% / 33.1% | 16.3% |

High pitch-to-tab gap, distortion:

| Track | MP F1 | Tab F1 | Gap | Ref/Pred silence | Non-silent acc |
|---|---:|---:|---:|---:|---:|
| `Static-X - Wisconsin Death Trip (2)` | 93.9% | 0.0% | 93.9 pp | 94.9% / 63.7% | 62.7% |
| `Pain Of Salvation - Handful Of Nothing` | 92.2% | 0.7% | 91.5 pp | 94.7% / 89.1% | 99.4% |
| `_MaTrioK_ - Kazowwie` | 93.0% | 14.6% | 78.4 pp | 71.8% / 56.0% | 96.7% |

Typical tracks still show a large tab gap:

| Run | Typical Tab F1 | Typical MP F1 | Typical Gap |
|---|---:|---:|---:|
| Clean | about 56% | about 77-98% | about 21-42 pp |
| Distortion | about 55% | about 80-98% | about 24-43 pp |
| Muted | about 63% | about 88-97% | about 25-34 pp |
| Acoustic | about 65% | about 81-96% | about 16-31 pp |

Best tracks prove the model can solve some cases:

- Clean best examples reach `98-99%` tablature F1.
- Distortion best examples reach `95-100%` tablature F1.
- Muted best examples reach `99-100%` tablature F1.
- Acoustic best examples reach `97-99%` tablature F1.
- Failures are not uniform; data shape and tab-position ambiguity matter.

## Decision

- Do not change architecture immediately, but do not blindly run many more similar chunks.
- The third and fourth contrastive chunks helped, so the next default step can remain chunk-based scaling.
- Next resume checkpoint is `training-state-58184.pt`.
- Prefer the next chunk to be different from the already used data: a new electric_clean timbre/group, another acoustic timbre/group, or a new electric_distortion only if it adds clear data contrast.
- Keep current baseline settings for the next smoke and long run: `silence_weight=0.1`, `note_weight=1.0`, `sampler=balanced`, `balance_by_group=true`, `balance_by_silence=false`, `batch_size=8`, `use_amp=false`.
- If the next contrastive chunk stalls or degrades, then shift focus from data scaling to tab head/loss/label representation.
- Do not enable `balance_by_silence=true` unless a future run shows `collapse_to_silence=true` or pred silence moving toward `1.0`.

## Repro

Analyzer:

```powershell
.\.venv\Scripts\python.exe demo_embedding\analyze_synthtab_errors.py `
  generated\experiments\full_chunk_semihollow_clean_finger_28ep_fresh `
  generated\experiments\full_chunk_electric_distortion_semihollow_clean_finger_28ep_resume_from_12712 `
  generated\experiments\full_chunk_electric_muted_28ep_resume_from_26544 `
  generated\experiments\full_chunk_acoustic_luthier_pick_part1_28ep_resume_from_50372 `
  --label "Clean chunk" `
  --label "Distortion chunk" `
  --label "Muted chunk" `
  --label "Acoustic luthier_pick part1"
```
