# Цели проекта

Этот файл фиксирует исходные задачи проекта и используется как чек-лист для дальнейших решений.

## Главная цель

Получить рабочий pipeline для распознавания гитарных табулатур из аудио на базе `TabCNN + SynthTab`, обучить его на максимально возможном объеме SynthTab при ограниченном диске, сохранить воспроизводимые метрики и подготовить базу для последующего дообучения на собственных данных.

## Финальная дальняя цель

Использовать обученную на SynthTab или pretrained модель как базу, а затем дообучить ее на собственных данных:

- ручные табулатуры;
- аудио гитариста, играющего по этим табам;
- возможно, легкий fine-tuning через LoRA или другой parameter-efficient подход.

Практический смысл текущего этапа: получить стабильную базовую модель и воспроизводимый workflow, чтобы позже не начинать обучение на собственных данных с нуля.

## Исходные задачи

- [x] Поднять рабочий `TabCNN on SynthTab` pipeline: `inspect`, `train`, `eval`, CUDA, SynthTab Dev, корректные метрики.
- [x] Разобраться с проблемой старого baseline: высокая `accuracy`, но `F1 = 0` из-за `collapse_to_silence`.
- [x] Добавить диагностические метрики: `pred_silence_ratio`, `ref_silence_ratio`, `non_silent_accuracy`, `collapse_to_silence`.
- [x] Подобрать базовые параметры против silence-collapse.
- [x] Реализовать полноценный resume через `training-state-*.pt`, не через legacy `model-*.pt`.
- [x] Добавить balanced sampler по track/timbre group, с опциональным балансом по silence-density.
- [x] Перейти с `SynthTab Dev` на `SynthTab Full` через chunk-based workflow, не скачивая весь датасет целиком.
- [x] Хранить JAMS/MIDI постоянно, а audio chunks подключать и заменять по одному.
- [x] Провести inspect, smoke и resume на первом full chunk.
- [x] Провести хотя бы один meaningful run на части full SynthTab.
- [x] После каждого значимого запуска сохранять метрики, параметры, chunk, checkpoint, дату, длительность и выводы.
- [ ] Подготовить объяснимую историю для преподавателей: какие данные использовались, почему не весь full SynthTab сразу, какие метрики выросли, какие ограничения остались.
- [ ] После обучения на большем объеме данных сравнить результат с другими сервисами/подходами на одинаковых аудиофрагментах.
- [ ] Подготовить базу для будущего дообучения на собственных данных: ручные табы плюс аудио гитариста; возможно через LoRA или другой легкий fine-tuning.

## Текущий статус

Закрыто:

- Dev pipeline работает.
- CUDA работает.
- Старый silence-collapse диагностирован.
- Добавлены anti-collapse метрики.
- Лучший Dev baseline: `silence_weight=0.1`, `note_weight=1.0`.
- Resume через `training-state-*.pt` работает.
- Balanced sampler работает.
- Первый full chunk `electric_clean/semihollow_clean_finger` подключен.
- Full chunk inspect, smoke и resume прошли.
- Первый meaningful full-chunk run на 28 эпох завершен успешно.
- Collapse-to-silence на первом full chunk не произошел.
- Второй full chunk `electric_distortion/semihollow_clean_finger` обучен через resume до `training-state-26544.pt`.
- Метрики второго full chunk сохранены; collapse-to-silence не произошел, но прирост качества почти отсутствует.
- Третий контрастный chunk `electric_muted` скачан, подключен, прошел inspect и resume smoke от `training-state-26544.pt`.
- Третий full chunk `electric_muted` обучен через resume до `training-state-50372.pt`.
- Метрики третьего full chunk сохранены; `tablature F1` вырос до `56.25%`, collapse-to-silence не произошел.
- Четвертый full chunk `acoustic/luthier_pick/part_1` обучен через resume до `training-state-58184.pt`.
- Метрики четвертого full chunk сохранены; `tablature F1` вырос до `58.47%`, collapse-to-silence не произошел.
- Добавлен loader-fix для non-finite audio samples в SynthTab chunks.
- Пятый full chunk `electric_clean/peregrine_clean_neck` обучен через resume до `training-state-70896.pt`.
- Шестой full chunk `electric_clean/lespaul_clean_both` обучен через resume до `training-state-83608.pt`.
- Выполнена оценка после нескольких chunks: похожие `electric_clean` chunks не улучшили качество относительно acoustic peak.
- Диагностика `electric_clean` просадки зафиксирована в `TAB_HEAD_DIAGNOSTIC.md`: основной bottleneck сейчас string/fret assignment, а не silence-collapse.
- Маленький string/fret-focused eval experiment зафиксирован в `STRING_FRET_EXPERIMENT.md`: `training-state-83608.pt` лучше `58184` на lespaul subset, но string/fret oracle gap остается около `13.8 pp`.

Осталось:

- [x] Зафиксировать итог первого full-chunk run в `PROJECT_LOG.md`.
- [x] Зафиксировать итог второго full-chunk run в `PROJECT_LOG.md`.
- [ ] Продолжить chunk-based обучение на следующих частях SynthTab Full через `resume_from` последнего `training-state-*.pt`.
- [ ] После каждого следующего chunk сохранять метрики и выводы.
- [x] Выполнить короткую диагностику ошибок перед третьим long-run: pitch, string/fret assignment, silence/non-silence, worst/best validation tracks.
- [x] Подготовить третий контрастный chunk и проверить его через inspect/resume smoke.
- [x] Обучить третий контрастный chunk и зафиксировать метрики.
- [x] Подготовить четвертый acoustic chunk с контролем диска, проверить через inspect/resume smoke, обучить и зафиксировать метрики.
- [x] Обучить пятый и шестой chunks через resume и зафиксировать метрики.
- [x] После нескольких chunks оценить, нужны ли изменения в балансировке, AMP, batch size или архитектуре.
- [x] Диагностировать просадку `electric_clean` и принять решение по tab head/loss/label representation или post-processing.
- [x] Спланировать и проверить маленький string/fret-focused эксперимент: post-processing diagnostic или изменение tab head/loss/label representation.
- [ ] Спланировать logits-aware/top-k string-fret диагностику или изменение tab head/loss/label representation.
- [ ] Подготовить сравнение с внешними сервисами/подходами.
- [ ] Позже перейти к дообучению на собственных данных.

## Как пользоваться целями

Перед каждым новым этапом сверяться с этим файлом:

1. Если действие помогает закрыть пункт из `Осталось`, оно приоритетное.
2. Если действие не связано с целями, его делать только после явного решения.
3. Если метрики деградируют или появляется `collapse_to_silence=true`, сначала диагностировать данные и баланс, а не менять архитектуру.
4. Если run успешен, следующий шаг по умолчанию — продолжать chunk-based обучение через `training-state-*.pt`.
5. Любой значимый результат фиксировать в `PROJECT_LOG.md`.
6. Если задача из этого файла выполнена, пометить ее чекбоксом `[x]` в `PROJECT_GOALS.md`.
