# Error Analysis

Короткая диагностика после двух meaningful SynthTab Full chunks.

## Runs

| Run | Chunk | Checkpoint | Multi-pitch F1 | Tablature F1 | Accuracy | Non-silent accuracy | Collapse |
|---|---|---|---:|---:|---:|---:|---|
| 1 | `electric_clean/semihollow_clean_finger` | `training-state-12712.pt` | 70.71% | 48.72% | 80.18% | 65.89% | false |
| 2 | `electric_distortion/semihollow_clean_finger` | `training-state-26544.pt` | 69.02% | 48.11% | 81.18% | 66.04% | false |

## What Changed

- Второй chunk не дал заметного прироста: `tablature F1` остался около `48%`.
- `multi_pitch F1` тоже не вырос: `70.71% -> 69.02%`.
- `accuracy` немного выросла, но это слабый сигнал, потому что validation sparse и silence-heavy.
- `collapse_to_silence=false` на обоих chunks, значит anti-collapse параметры работают.

## Main Failure Mode

Главная проблема сейчас не чистый silence-collapse, а разрыв между pitch detection и переводом pitch в string/fret:

| Run | Median MP F1 | Median Tab F1 | Median MP-Tab Gap | Mean MP-Tab Gap |
|---|---:|---:|---:|---:|
| Clean | 87.16% | 56.15% | 20.80% | 22.00% |
| Distortion | 85.38% | 55.31% | 18.57% | 20.91% |

Практический вывод: модель часто понимает, что звучит по pitch, но хуже выбирает табулатурную позицию. Значит простое добавление похожих chunks может быстро выйти на плато.

## Silence Behavior

| Run | Ref silence | Pred silence | Delta |
|---|---:|---:|---:|
| Clean | 75.09% | 62.25% | -12.84 pp |
| Distortion | 77.62% | 65.21% | -12.41 pp |

Модель не схлопнулась в silence. Наоборот, она системно предсказывает больше non-silent, чем есть в reference. Это не повод включать `balance_by_silence=true` прямо сейчас: проблема не в трусливом молчании, а в лишней активности и string/fret ошибках.

По per-track diagnostics:

- Clean: среди non-silent tracks `530/708` треков имеют pred silence ниже ref silence больше чем на `10 pp`.
- Distortion: среди non-silent tracks `595/807` треков имеют pred silence ниже ref silence больше чем на `10 pp`.
- Under-prediction notes почти не встречается: `6` tracks в каждом run по порогу `+10 pp` pred silence.

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

Best tracks prove the model can solve some cases:

- Clean best examples reach `98-99%` tablature F1.
- Distortion best examples reach `95-100%` tablature F1.
- Failures are not uniform; data shape and tab-position ambiguity matter.

## Decision

- Do not change architecture immediately, but do not blindly run many more similar chunks.
- Before the next long run, prefer one contrastive chunk such as `electric_muted` if available.
- Keep current baseline settings for the next smoke: `silence_weight=0.1`, `note_weight=1.0`, `sampler=balanced`, `balance_by_group=true`, `balance_by_silence=false`.
- If the third contrastive chunk also stays near `48%` tablature F1, shift focus from data scaling to tab head/loss/label representation.
- Do not enable `balance_by_silence=true` unless a future run shows `collapse_to_silence=true` or pred silence moving toward `1.0`.

## Repro

Analyzer:

```powershell
.\.venv\Scripts\python.exe demo_embedding\analyze_synthtab_errors.py `
  generated\experiments\full_chunk_semihollow_clean_finger_28ep_fresh `
  generated\experiments\full_chunk_electric_distortion_semihollow_clean_finger_28ep_resume_from_12712 `
  --label "Clean chunk" `
  --label "Distortion chunk"
```
