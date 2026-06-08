# Tab Head Diagnostic

CPU-only диагностика по шести завершенным SynthTab Full runs. Использованы только сохраненные `summary.json` и per-track `.txt` метрики из `generated\experiments`; модель, аудио и GPU не загружались.

## Executive Summary

- Лучший checkpoint по aggregate `tablature F1`: `training-state-58184.pt` из `acoustic/luthier_pick/part_1_-_1_to_B_C`.
- Последний технический checkpoint цепочки: `training-state-83608.pt` из `electric_clean/lespaul_clean_both`.
- `collapse_to_silence=false` на всех шести runs, поэтому silence-collapse сейчас не bottleneck.
- Главный повторяющийся паттерн: много треков с сильным `multi_pitch F1`, но слабым `tablature F1`.
- Похожие `electric_clean` chunks после acoustic peak не улучшили качество и вернули mean MP-Tab gap примерно к `22 pp`.

## Run Comparison

| Run | Chunk | Checkpoint | MP F1 | Tab F1 | Median MP | Median Tab | Mean gap | High MP/low Tab | Large gap | Silence delta | Collapse |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Run 1 clean | `electric_clean/semihollow_clean_finger` | `training-state-12712.pt` | 70.71% | 48.72% | 87.16% | 56.15% | 22.00 pp | 296/908 | 344/908 | -12.84 pp | false |
| Run 2 distortion | `electric_distortion/semihollow_clean_finger` | `training-state-26544.pt` | 69.02% | 48.11% | 85.38% | 55.31% | 20.91 pp | 260/988 | 282/988 | -12.41 pp | false |
| Run 3 muted | `electric_muted` | `training-state-50372.pt` | 73.07% | 56.25% | 90.62% | 63.20% | 16.82 pp | 349/1703 | 403/1703 | -9.35 pp | false |
| Run 4 acoustic | `acoustic/luthier_pick/part_1_-_1_to_B_C` | `training-state-58184.pt` | 77.25% | 58.47% | 85.99% | 65.31% | 18.77 pp | 90/559 | 104/559 | -14.40 pp | false |
| Run 5 clean peregrine | `electric_clean/peregrine_clean_neck` | `training-state-70896.pt` | 72.64% | 50.48% | 89.95% | 56.81% | 22.16 pp | 294/908 | 359/908 | -12.90 pp | false |
| Run 6 clean lespaul | `electric_clean/lespaul_clean_both` | `training-state-83608.pt` | 73.61% | 51.19% | 90.70% | 57.21% | 22.42 pp | 300/908 | 374/908 | -12.37 pp | false |

## Failure Buckets

`High MP/low Tab` значит per-track `multi_pitch F1 >= 80%` и `tablature F1 < 60%`. `Large gap` значит `multi_pitch F1 - tablature F1 >= 30 pp`.

| Run | Tracks | High MP/low Tab | Large gap >=30pp | Extra activity | Under activity | Median non-silent acc |
|---|---:|---:|---:|---:|---:|---:|
| Run 1 clean | 908 | 296 | 344 | 530 | 6 | 83.79% |
| Run 2 distortion | 988 | 260 | 282 | 595 | 6 | 82.19% |
| Run 3 muted | 1703 | 349 | 403 | 747 | 14 | 90.51% |
| Run 4 acoustic | 559 | 90 | 104 | 404 | 2 | 89.70% |
| Run 5 clean peregrine | 908 | 294 | 359 | 524 | 6 | 86.45% |
| Run 6 clean lespaul | 908 | 300 | 374 | 516 | 5 | 85.77% |

Практический смысл: на clean chunks примерно треть validation треков попадает в категорию “pitch уже хороший, tab еще плохой”. Это указывает не на общий провал аудио-распознавания, а на string/fret assignment.

## Representation Check

- Pipeline импортирует `TabCNN` из `amt_tools.models` в `demo_embedding/tabcnn_synthtab_pipeline.py`.
- Используемый `amt_tools.models.TabCNN` построен на `SoftmaxGroups`: независимые string groups, один silence class и fret/note classes на каждую струну.
- Evaluation использует `TablatureWrapper` и `StackedMultiPitchCollapser`, поэтому отдельно видны string/fret tablature и collapsed multi-pitch.
- Текущий weighting в `build_model()` выставляет только `note_weight` и `silence_weight`; он не отличает “не та струна/лад, но pitch правильный” от обычной class error.

## Decision

- Не продолжать blind scaling на похожих `electric_clean` chunks.
- Не включать `balance_by_silence=true`: модель не уходит в silence, а чаще предсказывает лишнюю non-silent активность.
- Best-by-metrics checkpoint для демонстрации и сравнения: `training-state-58184.pt`.
- Latest chronological checkpoint для продолжения цепочки: `training-state-83608.pt`.
- Следующий инженерный шаг: маленький string/fret-focused эксперимент, а не новый long-run.
- Приоритет следующего эксперимента: post-processing или diagnostic eval для pitch-correct/tab-wrong случаев; если это подтвердит ограничение head/loss, планировать изменение tab head/loss/label representation.

## Repro

```powershell
.\.venv\Scripts\python.exe demo_embedding\diagnose_tablature_gap.py
```
