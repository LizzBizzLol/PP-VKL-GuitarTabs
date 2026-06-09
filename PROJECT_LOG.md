# Журнал проекта

Формат:
- дата и время
- что изменилось
- вывод или следующий шаг

## 2026-05-16 02:12:53 +05:00

- В проект добавлен постоянный контекст по курсу, команде, прошлым результатам и текущим репозиториям.
- Зафиксировано, что сильная часть прошлого подхода Liss это `MIDI -> tablature`, а слабое место находится в `audio -> MIDI`.
- Зафиксировано, что Валя просит сначала поднять рабочий train/preprocess pipeline, подключить `CUDA`, и только потом переходить к экспериментам с `LoRA` или `LSTM`.
- Проверены локальные репозитории:
  - `D:\TSU-Project-Practicum\tab-cnn`
  - `D:\TSU-Project-Practicum\SynthTab`
- Предварительный вывод:
  - `SynthTab` выглядит как основная база для следующих экспериментов
  - `tab-cnn` стоит рассматривать либо как baseline, либо как вторую очередь работ после приведения `SynthTab` к воспроизводимому запуску

## Правило обновления

- При каждом существенном изменении проекта, решения или проверке я обновляю этот файл новой записью с датой и временем.

## 2026-05-16 02:21:50 +05:00

- В `SynthTab/demo_embedding` добавлен новый единый entrypoint для baseline-задачи `TabCNN on SynthTab`:
  - `tabcnn_synthtab_pipeline.py`
- Добавлен JSON-конфиг baseline-запуска:
  - `tabcnn_synthtab_baseline.json`
- Старые скрипты `exp_training_from_scratch.py` и `evaluate.py` упрощены до совместимых оберток над новым pipeline.
- Обновлен `README.md` в `demo_embedding` с новым сценарием запуска.
- Новый pipeline покрывает:
  - конфигурируемые пути к `SynthTab`, `GuitarSet` и cache
  - train / eval / inspect режимы
  - выбор `CUDA` или `CPU`
  - сохранение конфига и итоговых результатов эксперимента
- Ограничения текущей проверки:
  - в рабочем окружении не найден локальный распакованный `SynthTab Dev`
  - во встроенном Python отсутствуют зависимости `torch`, `amt_tools` и связанные библиотеки
  - поэтому выполнена статическая реализация pipeline, а не полный обучающий прогон

## 2026-05-16 03:04:17 +05:00

- Локально найден и распакован `SynthTab_Dev.zip`.
- Набор данных размещен по пути:
  - `C:\Users\Liss\Documents\New project\datasets\SynthTab_Dev\SynthTab_Dev`
- После проверки структуры выяснилось, что это dev-layout, а не исходный `train/val` layout:
  - данные лежат в `acoustic`, `electric_clean`, `electric_distortion_di`, `electric_muted`
  - `JAMS` лежат отдельно в `jams`
- `SynthTab`-обертка доработана так, чтобы поддерживать оба формата:
  - исходный `train/val`
  - локальный `SynthTab Dev`
- Для dev-layout добавлено:
  - построение track list из `partition/guitar/song`
  - поиск `ground_truth.jams` через `jams/song/ground_truth.jams`
  - детерминированный split `train/val` внутри dev-набора по seed
- Baseline-конфиг обновлен на реальный локальный путь к `SynthTab Dev`.

## 2026-05-16 03:19:15 +05:00

- Создано локальное окружение:
  - `C:\Users\Liss\Documents\New project\.venv_synthtab`
- В окружение установлены зависимости для baseline-пайплайна `TabCNN on SynthTab`.
- Исправлены блокеры запуска, связанные с импортами:
  - `tabcnn_synthtab_pipeline.py` теперь стабильно импортирует локальные модули из `demo_embedding`
  - `SynthTab.py` теперь берет `note_tab.json` по абсолютному пути от расположения файла, а не от текущей рабочей директории
- Успешно выполнен `inspect` через новый pipeline.
- Зафиксирован текущий статус среды:
  - `SynthTab Dev` обнаружен и читается
  - путь к `GuitarSet` валиден
  - `torch` установлен как `2.12.0+cpu`
  - `CUDA` сейчас недоступна в этом окружении (`cuda_available = false`, `device_count = 0`)
- Следующий практический шаг:
  - либо переводить окружение на GPU-сборку PyTorch,
  - либо сначала делать sanity-run на CPU для проверки полного `train -> eval` цикла

## 2026-05-16 03:33:48 +05:00

- Локальное окружение `C:\Users\Liss\Documents\New project\.venv_synthtab` переведено с CPU-сборки PyTorch на официальные CUDA wheels.
- Установлены:
  - `torch 2.5.1+cu121`
  - `torchaudio 2.5.1+cu121`
  - `torchvision 0.20.1+cu121`
- Проверка в окружении показала:
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1`
  - устройство: `NVIDIA GeForce GTX 1660 Ti`
- Повторный `inspect` через baseline pipeline успешно подтвердил:
  - `SynthTab Dev` доступен
  - `GuitarSet` доступен
  - `CUDA` доступна из training environment
- На этом этапе окружение готово для первого GPU sanity-run.

## 2026-05-16 03:40:31 +05:00

- Первый GPU sanity-run `TabCNN on SynthTab Dev` успешно завершен.
- Запуск выполнялся через:
  - `tabcnn_synthtab_pipeline.py`
  - конфиг `tabcnn_synthtab_sanity.json`
  - устройство `cuda:0`
- Параметры sanity-run:
  - `train_tracks = 8`
  - `val_tracks = 2`
  - `epochs = 2`
  - `batch_size = 4`
- По ходу реального запуска были найдены и исправлены дополнительные несовпадения формата `SynthTab Dev`:
  - сохранение dataset seed в `SynthTab`
  - отключение `Subset` для validation и переход на прямое ограничение `tracks`
  - рекурсивный поиск реальных аудиофайлов вместо попытки открыть директорию как файл
  - поиск `.jams` файла в dev-папке вместо ожидания только `ground_truth.jams`
- Итог sanity-run:
  - модель обучилась и провалидацировалась без падения
  - сохранены чекпоинты, optimizer state, tensorboard events и `results/summary.json`
- Метрики sanity-run на `SynthTab Dev val`:
  - `loss_total = 13.871084213256836`
  - `tablature accuracy = 0.6181666666666666`
  - `tablature f1 = 0.0`
  - `multi_pitch f1 = 0.0`
- Практический вывод:
  - полный цикл `train -> checkpoint -> eval` на GPU уже работает
  - текущие метрики на маленьком sanity-run не показательны по качеству, но инфраструктура baseline теперь подтверждена реальным запуском
## 2026-05-16 03:54:36 +05:00

- В `SynthTab/demo_embedding/tabcnn_synthtab_pipeline.py` добавлена диагностика anti-collapse для оценки `TabCNN`:
  - `ref_silence_ratio`
  - `pred_silence_ratio`
  - `ref_non_silent_ratio`
  - `pred_non_silent_ratio`
  - `non_silent_accuracy`
  - флаг `collapse_to_silence`
- Диагностика теперь автоматически записывается в `results/summary.json`.
- В конфиги `tabcnn_synthtab_baseline.json`, `tabcnn_synthtab_baseline_dev.json`, `tabcnn_synthtab_sanity.json` добавлены явные параметры class weighting:
  - `use_class_weights = true`
  - `silence_weight = 0.1`
  - `note_weight = 1.0`
- Повторная evaluation старого baseline checkpoint `generated/baseline_dev_run/models/model-60.pt` подтвердила полный коллапс в тишину:
  - `pred_silence_ratio = 1.0`
  - `pred_non_silent_ratio = 0.0`
  - `non_silent_accuracy = 0.0`
  - `collapse_to_silence = true`
- Новый короткий GPU `weighted sanity-run` сохранён в `generated/sanity_weighted_rerun`.
- Метрики `weighted sanity-run`:
  - `multi_pitch f1 = 0.1280`
  - `tablature f1 = 0.0295`
  - `tablature accuracy = 0.1825`
  - `pred_silence_ratio = 0.1667`
  - `pred_non_silent_ratio = 0.8333`
  - `collapse_to_silence = false`
- Практический вывод:
  - старая высокая `accuracy` была артефактом класса тишины
  - weighting ломает silence-collapse и заставляет модель предсказывать ноты
  - следующий шаг: прогнать уже не sanity, а baseline dev-run с weighting и сравнить метрики на той же схеме

## 2026-05-16 04:05:40 +05:00

- Выполнен полноценный GPU `baseline_dev_run` с class weighting через:
  - `SynthTab/demo_embedding/tabcnn_synthtab_pipeline.py`
  - конфиг `tabcnn_synthtab_baseline_dev.json`
  - отдельную директорию артефактов `generated/baseline_dev_weighted_run`
- Прогон завершился успешно на `cuda:0` с полным циклом `train -> checkpoint -> eval`.
- Итоговые метрики нового weighted baseline на `SynthTab Dev val`:
  - `loss_total = 4.277589321136475`
  - `multi_pitch f1 = 0.19878118884091506`
  - `tablature f1 = 0.15240222159540975`
  - `tablature accuracy = 0.7762333333333332`
  - `tdr = 0.7161764705882352`
  - `pred_silence_ratio = 0.913`
  - `pred_non_silent_ratio = 0.087`
  - `non_silent_accuracy = 0.09722301923309698`
  - `collapse_to_silence = false`
- Сравнение со старым baseline `generated/baseline_dev_run/results/summary.json`:
  - `multi_pitch f1`: `0.0 -> 0.1988`
  - `tablature f1`: `0.0 -> 0.1524`
  - `tablature accuracy`: `0.7547 -> 0.7762`
- Практический вывод:
  - полный silence-collapse действительно устранен
  - weighting дал сильный прирост по содержательным метрикам, не только на коротком sanity-run
  - при этом модель все еще очень silence-heavy (`pred_silence_ratio = 0.913`), хотя уже не в полном коллапсе
  - следующий разумный шаг: проверить несколько значений `silence_weight` (`0.2`, `0.3`, `0.5`) и сравнить баланс между `tablature f1`, `multi_pitch f1`, `pred_silence_ratio` и `non_silent_accuracy`

## 2026-05-16 04:22:20 +05:00

- Выполнена серия GPU baseline-экспериментов с разными значениями `silence_weight`:
  - `0.2` -> `generated/baseline_dev_weighted_run_sw_02`
  - `0.3` -> `generated/baseline_dev_weighted_run_sw_03`
  - `0.5` -> `generated/baseline_dev_weighted_run_sw_05`
- Для чистоты сравнения базовой рабочей точкой считается предыдущий weighted baseline:
  - `silence_weight = 0.1` -> `generated/baseline_dev_weighted_run`
- Сводка результатов:
  - `silence_weight = 0.1`
    - `multi_pitch f1 = 0.1988`
    - `tablature f1 = 0.1524`
    - `pred_silence_ratio = 0.913`
    - `collapse_to_silence = false`
  - `silence_weight = 0.2`
    - `multi_pitch f1 = 0.0`
    - `tablature f1 = 0.0`
    - `pred_silence_ratio = 1.0`
    - `collapse_to_silence = true`
  - `silence_weight = 0.3`
    - `multi_pitch f1 = 0.0`
    - `tablature f1 = 0.0`
    - `pred_silence_ratio = 1.0`
    - `collapse_to_silence = true`
  - `silence_weight = 0.5`
    - `multi_pitch f1 = 0.0`
    - `tablature f1 = 0.0`
    - `pred_silence_ratio = 1.0`
    - `collapse_to_silence = true`
- Практический вывод:
  - переход от `0.1` к `0.2` уже ломает рабочий режим и возвращает модель в полный silence-collapse
  - в текущей постановке `0.1` является лучшей найденной точкой из проверенных
  - если продолжать тюнинг, то следующий диапазон надо искать не выше, а между `0.1` и `0.2`, например `0.12`, `0.15`, `0.18`

## 2026-05-16 11:56:40 +05:00

- Выполнена тонкая серия GPU baseline-экспериментов между `0.1` и `0.2`:
  - `0.12` -> `generated/baseline_dev_weighted_run_sw_012`
  - `0.15` -> `generated/baseline_dev_weighted_run_sw_015`
  - `0.18` -> `generated/baseline_dev_weighted_run_sw_018`
- Сравнение с текущей лучшей точкой `0.1`:
  - `silence_weight = 0.10`
    - `multi_pitch f1 = 0.1988`
    - `tablature f1 = 0.1524`
    - `pred_silence_ratio = 0.913`
    - `collapse_to_silence = false`
  - `silence_weight = 0.12`
    - `multi_pitch f1 = 0.00026`
    - `tablature f1 = 0.00027`
    - `pred_silence_ratio = 0.99993`
    - `collapse_to_silence = true`
  - `silence_weight = 0.15`
    - `multi_pitch f1 = 0.0`
    - `tablature f1 = 0.0`
    - `pred_silence_ratio = 1.0`
    - `collapse_to_silence = true`
  - `silence_weight = 0.18`
    - `multi_pitch f1 = 0.0`
    - `tablature f1 = 0.0`
    - `pred_silence_ratio = 1.0`
    - `collapse_to_silence = true`
- Практический вывод:
  - рабочее окно у `silence_weight` очень узкое
  - уже переход `0.10 -> 0.12` практически уничтожает полезные предсказания
  - на текущей конфигурации `0.10` остается лучшей и единственной устойчивой точкой из всех проверенных
  - если продолжать тюнинг именно этого параметра, то имеет смысл смотреть только ниже `0.12`, например `0.105` или `0.11`, но ожидаемый выигрыш уже выглядит сомнительным по сравнению с риском снова попасть в collapse

## 2026-05-16 12:24:10 +05:00

- Выполнена сверхузкая серия GPU baseline-экспериментов рядом с рабочей точкой `0.10`:
  - `0.102` -> `generated/baseline_dev_weighted_run_sw_0102`
  - `0.104` -> `generated/baseline_dev_weighted_run_sw_0104`
  - `0.106` -> `generated/baseline_dev_weighted_run_sw_0106`
  - `0.108` -> `generated/baseline_dev_weighted_run_sw_0108`
  - `0.110` -> `generated/baseline_dev_weighted_run_sw_0110`
- Сводка относительно базовой точки `0.100`:
  - `0.100`
    - `multi_pitch f1 = 0.1988`
    - `tablature f1 = 0.1524`
    - `accuracy = 0.7762`
    - `pred_silence_ratio = 0.9130`
  - `0.102`
    - `multi_pitch f1 = 0.1836`
    - `tablature f1 = 0.1409`
    - `accuracy = 0.7759`
    - `pred_silence_ratio = 0.9246`
  - `0.104`
    - `multi_pitch f1 = 0.1886`
    - `tablature f1 = 0.1455`
    - `accuracy = 0.7766`
    - `pred_silence_ratio = 0.9161`
  - `0.106`
    - `multi_pitch f1 = 0.1659`
    - `tablature f1 = 0.1255`
    - `accuracy = 0.7725`
    - `pred_silence_ratio = 0.9168`
  - `0.108`
    - `multi_pitch f1 = 0.1798`
    - `tablature f1 = 0.1391`
    - `accuracy = 0.7755`
    - `pred_silence_ratio = 0.9220`
  - `0.110`
    - `multi_pitch f1 = 0.1308`
    - `tablature f1 = 0.0966`
    - `accuracy = 0.7683`
    - `pred_silence_ratio = 0.9300`
- Практический вывод:
  - ни одно из пяти новых значений не улучшило `tablature f1` или `multi_pitch f1` относительно `0.100`
  - ближайшая альтернатива это `0.104`, но она все равно слабее по обеим основным `f1`
  - на текущем baseline лучшей точкой остается `silence_weight = 0.100`

## 2026-05-16 12:40:40 +05:00

- После стабилизации `silence_weight = 0.100` следующим рычагом выбран размер обучающей выборки, а не дальнейший тюнинг весов.
- Проверено, что в `SynthTab Dev` доступно существенно больше данных, чем использовалось в baseline:
  - всего валидных `train` треков порядка `137`
  - всего валидных `val` треков порядка `34`
- При попытке расширить baseline были найдены и исправлены два dataset-level блокера в `SynthTab/demo_embedding/SynthTab.py`:
  - dev-track без соответствующего `jams` каталога теперь отфильтровывается заранее
  - dev-track с аннотацией, несовместимой с 6-струнным `GuitarProfile`, теперь тоже отфильтровывается заранее
- Это сделало большие train/val прогоны устойчивыми и воспроизводимыми, а не зависимыми от случайного ограничения `40` треков.
- Выполнен новый расширенный GPU baseline:
  - конфиг `SynthTab/demo_embedding/tabcnn_synthtab_baseline_dev_more_tracks.json`
  - артефакты `generated/baseline_dev_more_tracks_run`
  - параметры:
    - `train_tracks = 80`
    - `val_tracks = 34`
    - `epochs = 12`
    - `silence_weight = 0.100`
- Сравнение с предыдущим лучшим baseline `40/10`:
  - `multi_pitch f1`: `0.1988 -> 0.3395`
  - `tablature f1`: `0.1524 -> 0.2481`
  - `non_silent_accuracy`: `0.0972 -> 0.2576`
  - `pred_silence_ratio`: `0.9130 -> 0.7838`
  - `collapse_to_silence`: `false -> false`
  - `accuracy`: `0.7762 -> 0.7115`
- Практический вывод:
  - рост охвата train/val данных дал значительно больший выигрыш, чем тонкий тюнинг `silence_weight`
  - падение общей `accuracy` не выглядит проблемой, так как содержательные метрики (`tablature f1`, `multi_pitch f1`, `non_silent_accuracy`) заметно выросли
  - новый baseline `80/34` с `silence_weight = 0.100` является лучшей найденной конфигурацией на текущий момент

## 2026-05-16 13:00:50 +05:00

- Выполнен следующий baseline на почти полном доступном train-pool:
  - конфиг `SynthTab/demo_embedding/tabcnn_synthtab_baseline_dev_full_train.json`
  - артефакты `generated/baseline_dev_full_train_run`
  - фактические размеры после фильтрации:
    - `train_tracks = 134`
    - `val_tracks = 34`
  - остальные ключевые параметры сохранены:
    - `epochs = 12`
    - `silence_weight = 0.100`
- Итоговые метрики полного train-run:
  - `loss_total = 3.0632`
  - `multi_pitch f1 = 0.5048`
  - `tablature f1 = 0.3307`
  - `tablature accuracy = 0.6998`
  - `tdr = 0.6580`
  - `pred_silence_ratio = 0.6863`
  - `pred_non_silent_ratio = 0.3137`
  - `non_silent_accuracy = 0.4117`
  - `collapse_to_silence = false`
- Сравнение с baseline `80/34`:
  - `multi_pitch f1`: `0.3395 -> 0.5048`
  - `tablature f1`: `0.2481 -> 0.3307`
  - `non_silent_accuracy`: `0.2576 -> 0.4117`
  - `pred_silence_ratio`: `0.7838 -> 0.6863`
  - `accuracy`: `0.7115 -> 0.6998`
- Практический вывод:
  - увеличение train-pool до почти полного состава снова дало сильный прирост по содержательным метрикам
  - модель продолжает уходить от silence-heavy режима без скатывания в over-prediction
  - текущее лучшее состояние проекта: `134/34`, `12` эпох, `silence_weight = 0.100`

## 2026-05-16 13:39:30 +05:00

- Выполнен прогон на той же лучшей конфигурации `134/34`, но с увеличением длительности обучения до `20` эпох:
  - конфиг `SynthTab/demo_embedding/tabcnn_synthtab_baseline_dev_full_train_20ep.json`
  - артефакты `generated/baseline_dev_full_train_20ep_run`
- Итоговые метрики `20` эпох:
  - `loss_total = 2.5653`
  - `multi_pitch f1 = 0.5860`
  - `tablature f1 = 0.3919`
  - `tablature accuracy = 0.7117`
  - `tdr = 0.7032`
  - `pred_silence_ratio = 0.6669`
  - `pred_non_silent_ratio = 0.3331`
  - `non_silent_accuracy = 0.4674`
  - `collapse_to_silence = false`
- Сравнение с baseline `134/34`, `12` эпох:
  - `multi_pitch f1`: `0.5048 -> 0.5860`
  - `tablature f1`: `0.3307 -> 0.3919`
  - `accuracy`: `0.6998 -> 0.7117`
  - `pred_silence_ratio`: `0.6863 -> 0.6669`
  - `non_silent_accuracy`: `0.4117 -> 0.4674`
- Практический вывод:
  - увеличение числа эпох на полной обучающей выборке снова дало реальный выигрыш, а не переобучение
  - текущий лучший baseline проекта: `134/34`, `20` эпох, `silence_weight = 0.100`

## 2026-05-16 14:22:50 +05:00

- Выполнен следующий прогон на той же конфигурации, но уже с `24` эпохами:
  - конфиг `SynthTab/demo_embedding/tabcnn_synthtab_baseline_dev_full_train_24ep.json`
  - рабочие артефакты `generated/baseline_dev_full_train_24ep_run_retry`
  - первый запуск оборвался рано без значимых артефактов, повторный запуск завершился успешно
- Итоговые метрики `24` эпох:
  - `loss_total = 2.3779`
  - `multi_pitch f1 = 0.6130`
  - `tablature f1 = 0.4032`
  - `tablature accuracy = 0.7131`
  - `tdr = 0.7133`
  - `pred_silence_ratio = 0.6602`
  - `pred_non_silent_ratio = 0.3398`
  - `non_silent_accuracy = 0.4968`
  - `collapse_to_silence = false`
- Сравнение с `20` эпохами:
  - `multi_pitch f1`: `0.5860 -> 0.6130`
  - `tablature f1`: `0.3919 -> 0.4032`
  - `accuracy`: `0.7117 -> 0.7131`
  - `pred_silence_ratio`: `0.6669 -> 0.6602`
  - `non_silent_accuracy`: `0.4674 -> 0.4968`
- Практический вывод:
  - рост продолжается и после `20` эпох, но уже более умеренно
  - новый лучший baseline проекта: `134/34`, `24` эпохи, `silence_weight = 0.100`
  - по характеру прироста похоже, что мы приближаемся к зоне замедления выигрыша, но явного переобучения пока не видно

## 2026-05-19 23:54 ? Laptop engineering pass: resumable chunk training prep

Implemented/updated on the laptop side for the next full-SynthTab stage:

- Added resumable training-state checkpoints to `workspace/SynthTab/demo_embedding/train.py`.
  - New files are saved as `training-state-<iter>.pt` alongside legacy `model-<iter>.pt` and optimizer state files.
  - Full state includes model weights, optimizer, scheduler, epoch/next_epoch, `model.iter`, Python/NumPy/Torch/CUDA RNG states, and config snapshot.
- Added resume support in `workspace/SynthTab/demo_embedding/tabcnn_synthtab_pipeline.py` via `train.resume_from`.
  - Resume expects `training-state-*.pt` checkpoints.
  - Legacy model-only checkpoints remain intended for eval mode, not safe resume.
- Added sampler configuration:
  - `train.sampler = "shuffle"` keeps old behavior.
  - `train.sampler = "balanced"` enables weighted balanced sampling.
  - Current v1 balances by track group/timbre path; optional `balance_by_silence` can estimate note-density buckets from JAMS without loading audio.
- Added smoke config: `workspace/SynthTab/demo_embedding/tabcnn_synthtab_resume_balanced_smoke.json` and mirrored it to `demo_embedding/`.
- Added full/chunk template config: `workspace/SynthTab/demo_embedding/tabcnn_synthtab_full_chunk_template.json` and mirrored it to `demo_embedding/`.

Mirrored the updated pipeline and training loop into root `demo_embedding/` as well, because `README.md` uses that entrypoint.

Laptop constraints remain: no visible NVIDIA/CUDA through `nvidia-smi`, and full SynthTab is not downloaded here. Use SynthTab Dev for smoke checks; run heavy chunk-based training on the desktop.

## 2026-05-20 00:25 ? Laptop smoke verification on SynthTab Dev

Verified the notebook-side pipeline with a local `.venv` on CPU only:

- Created `.venv` with Python 3.10 and installed `workspace/SynthTab/requirements.txt`.
- Installed CPU PyTorch stack; `torch.cuda.is_available()` is `False` on this laptop.
- `inspect` passed on SynthTab Dev using the workspace entrypoint and the root `demo_embedding` entrypoint.
- Fresh smoke training passed with `tabcnn_synthtab_resume_balanced_smoke.json`:
  - `sanity_steps = 2`
  - generated legacy `model-*.pt`
  - generated legacy optimizer state
  - generated new `training-state-*.pt`
  - final fresh smoke iter: `2`
- Resume smoke training passed from `generated/experiments/laptop_train_smoke_fresh/models/training-state-2.pt`:
  - `run_mode = resume`
  - `start_iter = 2`
  - final resume iter: `4`
- Balanced sampler metadata is written to `run_config.json` and `results/summary.json`.
- Optional `balance_by_silence = true` smoke run passed on 6 tracks and produced JAMS-derived density buckets.
- Added a `soundfile` fallback in `SynthTab.py` because modern `torchaudio.load()` can require TorchCodec/FFmpeg DLLs on Windows.
- Mirrored missing root entrypoint support files so README-style `demo_embedding/tabcnn_synthtab_pipeline.py` inspect works.

Generated smoke outputs live under `generated/experiments/` and remain ignored by git. Do not commit `.venv`, generated cache, model checkpoints, optimizer states, TensorBoard events, or smoke outputs.

## 2026-05-30 15:00 +05:00 — Desktop migration to D: and smoke verification

- Перенесена рабочая точка проекта на диск `D:` через fresh clone:
  - новый основной путь: `D:\PP-VKL-GuitarTabs`
  - remote: `https://github.com/LizzBizzLol/PP-VKL-GuitarTabs.git`
- Созданы директории для будущей работы с full SynthTab вне Git:
  - `D:\DATA\SynthTab_Full\jams_midi`
  - `D:\DATA\SynthTab_Full\current_chunk`
  - `D:\DATA\SynthTab_Full\archive`
- При первом clone Windows уперся в ограничение длины путей на SynthTab Dev; исправлено через `git config --global core.longpaths true` и повторный clone.
- Git LFS подтянут: `1831` LFS-файл, checkout завершился успешно.
- Создано новое окружение `D:\PP-VKL-GuitarTabs\.venv` на Python `3.11.9`.
- Установлен CUDA-stack:
  - `torch 2.5.1+cu121`
  - `torchaudio 2.5.1+cu121`
  - `torchvision 0.20.1+cu121`
- Проверка CUDA из нового окружения:
  - `torch.cuda.is_available() == True`
  - GPU: `NVIDIA GeForce GTX 1660 Ti`
- Проверен `inspect` на SynthTab Dev из нового пути:
  - `D:\PP-VKL-GuitarTabs\workspace\datasets\SynthTab_Dev\SynthTab_Dev`
  - датасет найден
  - CUDA видна
- Fresh smoke train на `D:` успешно завершился:
  - experiment: `generated\experiments\desktop_d_train_smoke_fresh`
  - `sanity_steps = 2`
  - созданы `model-*.pt`, `opt-state-*.pt`, `training-state-*.pt`
  - final fresh iter: `2`
- Resume smoke train успешно завершился:
  - resume checkpoint: `generated\experiments\desktop_d_train_smoke_fresh\models\training-state-2.pt`
  - experiment: `generated\experiments\desktop_d_train_smoke_resume`
  - `run_mode = resume`
  - `start_iter = 2`
  - final resume iter: `4`
- Проверено, что `generated/` остается ignored и не попадает в Git.
- Full SynthTab пока не скачивался; следующий шаг — подключать JAMS/MIDI и первый маленький audio chunk в `D:\DATA\SynthTab_Full`.

## 2026-05-30 15:14 +05:00 — Old C: repo cleanup after D: migration

- Уникальная старая папка экспериментов перенесена со старой копии на `C:` в новый рабочий репозиторий:
  - source: `C:\Users\Liss\Documents\New project\SynthTab-TabCNN-Baseline\workspace\generated`
  - destination: `D:\PP-VKL-GuitarTabs\workspace\generated`
- Проверка после копирования:
  - `530` папок
  - `1562` файла
  - `6.535 GB`
- Подтверждено, что `D:\PP-VKL-GuitarTabs\workspace\generated` игнорируется Git через правило `generated/`.
- Старая копия репозитория `C:\Users\Liss\Documents\New project\SynthTab-TabCNN-Baseline` удалена после проверки переноса.
- Родительский указатель `C:\Users\Liss\Documents\New project\PROJECT_LOG.md` обновлен на актуальный путь `D:\PP-VKL-GuitarTabs\PROJECT_LOG.md`.

## 2026-05-30 15:32 +05:00 — Full SynthTab chunk setup started

- Проверена структура официального Box-архива SynthTab Full:
  - root folder: `SynthTab_Full`
  - JAMS/MIDI archive: `all_jams_midi_V2_60000_tracks.zip`
  - выбран первый маленький audio chunk: `electric_clean/semihollow_clean_finger.zip`
- Скачан JAMS/MIDI archive в `D:\DATA\SynthTab_Full\archive\all_jams_midi_V2_60000_tracks.zip`.
- JAMS/MIDI распакованы в `D:\DATA\SynthTab_Full\jams_midi`:
  - `60634` папки
  - `392681` файл
  - примерно `14.014 GB`
- Начата загрузка первого audio chunk:
  - target: `D:\DATA\SynthTab_Full\archive\semihollow_clean_finger.zip`
  - ожидаемый размер: `26.78 GiB`
  - curl запущен с resume через `--continue-at -`
- Pipeline подготовлен к full-chunk layout:
  - добавлен опциональный `paths.jams_dir`
  - `tabcnn_synthtab_full_chunk_template.json` теперь указывает на `D:\DATA\SynthTab_Full\jams_midi\outall`
  - loader допускает частичный dev/full layout: достаточно одного audio partition и JAMS root
- Regression check после правки:
  - `py_compile` для `SynthTab.py` и `tabcnn_synthtab_pipeline.py` прошел
  - `inspect` на SynthTab Dev прошел из нового D:-пути
  - CUDA по-прежнему видна: `NVIDIA GeForce GTX 1660 Ti`

## 2026-05-30 17:23 +05:00 — First SynthTab Full chunk smoke/resume passed

- Первый audio chunk успешно скачан и распакован:
  - archive: `D:\DATA\SynthTab_Full\archive\semihollow_clean_finger.zip`
  - downloaded size: `26.785 GiB`
  - extracted path: `D:\DATA\SynthTab_Full\current_chunk\electric_clean\semihollow_clean_finger`
  - extracted content: `4561` song dirs, `27295` files, примерно `27.362 GiB`
- `inspect` на full chunk прошел:
  - `paths.synthtab = D:\DATA\SynthTab_Full\current_chunk`
  - `paths.jams_dir = D:\DATA\SynthTab_Full\jams_midi\outall`
  - audio partition: `electric_clean`
  - CUDA: `NVIDIA GeForce GTX 1660 Ti`
- Loader доработан под реальный full-chunk layout:
  - `paths.jams_dir` поддерживает JAMS/MIDI отдельно от audio chunk
  - partial dev/full layout теперь разрешает один audio partition вместо обязательных четырех
  - dev/full track index кешируется в ignored `generated/cache_full_chunks`
  - cache key учитывает base path, JAMS root, profile, seed и список candidate tracks
  - valid track теперь требует не только подходящий JAMS, но и наличие audio file
- Реальный track split после фильтрации:
  - train: `3632`
  - val: `908`
  - отфильтровано `7` JAMS-valid tracks без audio
- Fresh full-chunk smoke прошел:
  - config: `generated\experiments\full_chunk_smoke_config.json`
  - experiment: `generated\experiments\full_chunk_smoke_fresh_v2`
  - `device = cuda:0`
  - `batch_size = 8`
  - `sanity_steps = 2`
  - `run_mode = fresh`
  - `start_iter = 0`
  - `final_iter = 2`
  - созданы `model-2.pt`, `opt-state-2.pt`, `training-state-2.pt`
- Resume full-chunk smoke прошел:
  - config: `generated\experiments\full_chunk_smoke_resume_config.json`
  - experiment: `generated\experiments\full_chunk_smoke_resume_v3`
  - resume checkpoint: `generated\experiments\full_chunk_smoke_fresh_v2\models\training-state-2.pt`
  - `run_mode = resume`
  - `start_iter = 2`
  - `final_iter = 4`
  - созданы `model-4.pt`, `opt-state-4.pt`, `training-state-4.pt`
- По ходу smoke найдены и исправлены два full-run бага:
  - checkpoint validation падал на track без audio; исправлено фильтром `has_audio`
  - resume на CUDA падал при восстановлении CPU RNG state после `map_location=cuda`; исправлено переносом RNG state обратно на CPU перед `torch.random.set_rng_state`
- Meaningful 28-epoch full-chunk run пока не запущен:
  - chunk имеет `454` batch/epoch при `batch_size = 8`
  - smoke показывает, что полный запуск займет много часов или больше
  - следующий запуск нужно делать осознанно отдельной длинной сессией из `demo_embedding\tabcnn_synthtab_full_chunk_template.json`

## 2026-05-30 18:40 +05:00 — First meaningful full-chunk run started

- Создан tracked config для первого чистого meaningful run:
  - `demo_embedding\tabcnn_synthtab_full_chunk_semihollow_clean_finger_28ep.json`
- Run запущен fresh, не из smoke checkpoint:
  - experiment: `generated\experiments\full_chunk_semihollow_clean_finger_28ep_fresh`
  - chunk: `electric_clean\semihollow_clean_finger`
  - `epochs = 28`
  - `batch_size = 8`
  - `checkpoints = 0`
  - `silence_weight = 0.1`
  - `note_weight = 1.0`
  - `sampler = balanced`
  - `balance_by_group = true`
  - `balance_by_silence = false`
  - `run_synthtab_val = true`
- Preflight checks перед запуском:
  - Git был чистый и синхронизирован с `origin/main`
  - CUDA доступна: `NVIDIA GeForce GTX 1660 Ti`
  - `py_compile` для `SynthTab.py` и `tabcnn_synthtab_pipeline.py` прошел
  - `inspect` на SynthTab Dev прошел
  - `inspect` на full chunk прошел
  - свободное место на `D:` перед запуском: примерно `350.86 GB`
- `run_config.json` после старта подтверждает:
  - `run_mode = fresh`
  - `start_iter = 0`
  - `train_tracks = 3632`
  - `val_tracks = 908`
  - `device = cuda:0`
- Первые минуты обучения прошли без CUDA OOM на `batch_size = 8`.
- Оценка по фактическому старту: около `50-55` минут на эпоху, полный train может занять около суток плюс финальная validation.

## 2026-05-30 21:57 +05:00 — Parallel prep while full-chunk run continues

- Текущий meaningful run не остановлен и не изменен.
- Добавлен легкий CLI-отчетчик:
  - `demo_embedding\summarize_synthtab_run.py`
  - читает только `run_config.json`, `results\summary.json` и список `training-state-*.pt`
  - не импортирует training pipeline и не загружает модель
- Добавлен fallback smoke config на случай collapse-to-silence:
  - `demo_embedding\tabcnn_synthtab_full_chunk_semihollow_clean_finger_balance_by_silence_smoke.json`
  - `balance_by_silence = true`
  - `sanity_steps = 2`
  - `run_synthtab_val = false`
- Подготовлен opt-in AMP API:
  - `train.use_amp = false` по умолчанию
  - при `true` CUDA-run использует autocast и GradScaler
  - scaler state сохраняется в full `training-state-*.pt` и восстанавливается при resume
  - текущий running experiment это не меняет, потому что он уже запущен со старым loaded code и `use_amp = false`
- Проверки после parallel prep:
  - `py_compile` прошел для `summarize_synthtab_run.py`, `train.py`, `tabcnn_synthtab_pipeline.py`
  - JSON validation прошла для новых и обновленных configs
  - analyzer dry-run прошел на `artifacts\baseline_dev_full_train_28ep_run`
  - analyzer корректно сообщает `summary: missing` для текущего незавершенного full run
  - `inspect` на SynthTab Dev прошел
  - `inspect` на current full chunk прошел

## 2026-05-31 13:08 +05:00 — First meaningful full-chunk run completed

- Первый meaningful run на SynthTab Full chunk успешно завершен:
  - experiment: `generated\experiments\full_chunk_semihollow_clean_finger_28ep_fresh`
  - chunk: `electric_clean\semihollow_clean_finger`
  - run mode: `fresh`
  - train tracks: `3632`
  - val tracks: `908`
  - epochs: `28`
  - batch size: `8`
  - final iter: `12712`
  - runtime по `summary.json`: `18:30:15`
  - final checkpoint: `generated\experiments\full_chunk_semihollow_clean_finger_28ep_fresh\models\training-state-12712.pt`
- Итоговые метрики на SynthTab Full validation:
  - `multi_pitch precision = 0.7020`
  - `multi_pitch recall = 0.7233`
  - `multi_pitch f1 = 0.7071`
  - `tablature precision = 0.4004`
  - `tablature recall = 0.6643`
  - `tablature f1 = 0.4872`
  - `tablature tdr = 0.7496`
  - `tablature accuracy = 0.8018`
  - `pred_silence_ratio = 0.6225`
  - `ref_silence_ratio = 0.7509`
  - `non_silent_accuracy = 0.6589`
  - `collapse_to_silence = false`
- Сравнение с лучшим Dev baseline:
  - `multi_pitch f1`: `0.6321 -> 0.7071`
  - `tablature f1`: `0.4151 -> 0.4872`
  - `tablature accuracy`: `0.7178 -> 0.8018`
  - `non_silent_accuracy`: `0.5261 -> 0.6589`
- Практический вывод:
  - первый full-chunk workflow подтвержден end-to-end
  - модель не схлопнулась в silence
  - архитектуру и базовые параметры пока не менять
  - следующий training запуск делать через `resume_from` от `training-state-12712.pt`
  - следующий chunk выбирать для расширения тембров: сначала небольшой `electric_distortion`, затем `electric_muted`, затем другой `electric_clean`, затем `acoustic`

## 2026-06-02 01:31 +05:00 — Second full-chunk run completed

- Второй meaningful SynthTab Full chunk завершен через resume от первого full-run:
  - experiment: `generated\experiments\full_chunk_electric_distortion_semihollow_clean_finger_28ep_resume_from_12712`
  - chunk: `electric_distortion\semihollow_clean_finger`
  - run mode: `resume`
  - train tracks: `3953`
  - val tracks: `988`
  - epochs target: `56`
  - batch size: `8`
  - start iter последнего continuation: `25062`
  - final iter: `26544`
  - runtime последнего continuation по `summary.json`: `01:39:26`
  - final checkpoint: `generated\experiments\full_chunk_electric_distortion_semihollow_clean_finger_28ep_resume_from_12712\models\training-state-26544.pt`
- Итоговые метрики на SynthTab Full validation:
  - `multi_pitch precision = 0.6773`
  - `multi_pitch recall = 0.7155`
  - `multi_pitch f1 = 0.6902`
  - `tablature precision = 0.3983`
  - `tablature recall = 0.6464`
  - `tablature f1 = 0.4811`
  - `tablature tdr = 0.7406`
  - `tablature accuracy = 0.8118`
  - `pred_silence_ratio = 0.6521`
  - `ref_silence_ratio = 0.7762`
  - `non_silent_accuracy = 0.6604`
  - `collapse_to_silence = false`
- Сравнение с первым full chunk:
  - `multi_pitch f1`: `0.7071 -> 0.6902`
  - `tablature f1`: `0.4872 -> 0.4811`
  - `tablature accuracy`: `0.8018 -> 0.8118`
  - `non_silent_accuracy`: `0.6589 -> 0.6604`
  - `pred_silence_ratio`: `0.6225 -> 0.6521`
- Практический вывод:
  - второй chunk не дал заметного прироста качества, но и не сломал модель
  - `collapse_to_silence` не появился, значит базовые веса и balanced sampler остаются рабочими
  - качество tablature остается слабым для готового продукта: около `48% F1`
  - перед следующим long-run целесообразна короткая диагностика ошибок, а не слепое масштабирование на много chunks
  - если продолжать обучение, следующий meaningful run должен resume от `training-state-26544.pt`
  - архитектуру, `balance_by_silence`, AMP и batch size пока не менять без отдельной диагностики

## 2026-06-02 20:08 +05:00 — Third chunk `electric_muted` prepared and smoke-tested

- Выбран контрастный третий chunk:
  - Box item: `electric_muted.zip`
  - archive: `D:\DATA\SynthTab_Full\archive\electric_muted.zip`
  - expected/downloaded size: `45755744178` bytes
  - downloaded size: `42.61 GiB`
- Active chunk переключен:
  - старый active `electric_distortion` перенесен в `D:\DATA\SynthTab_Full\archive\extracted_backups\electric_distortion_20260602_042059`
  - новый active path: `D:\DATA\SynthTab_Full\current_chunk\electric_muted`
  - в active chunk найдено `17` timbre/group directories
- Inspect по tracked config прошел:
  - config: `demo_embedding\tabcnn_synthtab_full_chunk_electric_muted_28ep_resume.json`
  - `synthtab_audio_partitions = ["electric_muted"]`
  - `jams_dir_exists = true`
  - CUDA доступна: `NVIDIA GeForce GTX 1660 Ti`
- Resume smoke прошел:
  - smoke config: `generated\experiments\full_chunk_electric_muted_resume_smoke_config.json`
  - smoke experiment: `generated\experiments\full_chunk_electric_muted_resume_smoke_v2`
  - resume checkpoint: `generated\experiments\full_chunk_electric_distortion_semihollow_clean_finger_28ep_resume_from_12712\models\training-state-26544.pt`
  - `run_mode = resume`
  - `start_epoch = 56`
  - `start_iter = 26544`
  - train tracks: `6814`
  - val tracks: `1703`
  - `sanity_steps = 2`
  - создан checkpoint: `training-state-26546.pt`
- Практический вывод:
  - `electric_muted` chunk подключен и pipeline подтвержден без long-run
  - следующий meaningful run можно запускать от `training-state-26544.pt`
  - tracked config уже задает абсолютную цель `epochs = 84`, то есть еще `28` дополнительных эпох после старта
  - базовые параметры оставлены прежними: `batch_size = 8`, `silence_weight = 0.1`, `note_weight = 1.0`, `sampler = balanced`, `balance_by_group = true`, `balance_by_silence = false`, `use_amp = false`
  - если третий meaningful run снова останется около `48%` tablature F1, дальше нужно переходить к tab head/loss/label representation, а не просто добавлять похожие chunks

## 2026-06-06 09:02 +05:00 — Third full-chunk run completed

- Третий meaningful SynthTab Full chunk завершен через resume от второго full-run:
  - experiment: `generated\experiments\full_chunk_electric_muted_28ep_resume_from_26544`
  - chunk: `electric_muted`
  - run mode: `resume`
  - train tracks: `6814`
  - val tracks: `1703`
  - epochs target: `84`
  - batch size: `8`
  - start iter последнего continuation: `36756`
  - final iter: `50372`
  - runtime последнего continuation по `summary.json`: `12:32:05`
  - final checkpoint: `generated\experiments\full_chunk_electric_muted_28ep_resume_from_26544\models\training-state-50372.pt`
- Итоговые метрики на SynthTab Full validation:
  - `multi_pitch precision = 0.7175`
  - `multi_pitch recall = 0.7525`
  - `multi_pitch f1 = 0.7307`
  - `tablature precision = 0.4794`
  - `tablature recall = 0.7227`
  - `tablature f1 = 0.5625`
  - `tablature tdr = 0.7681`
  - `tablature accuracy = 0.8725`
  - `pred_silence_ratio = 0.6920`
  - `ref_silence_ratio = 0.7855`
  - `non_silent_accuracy = 0.7079`
  - `collapse_to_silence = false`
- Сравнение со вторым full chunk:
  - `multi_pitch f1`: `0.6902 -> 0.7307`
  - `tablature f1`: `0.4811 -> 0.5625`
  - `tablature accuracy`: `0.8118 -> 0.8725`
  - `non_silent_accuracy`: `0.6604 -> 0.7079`
  - `pred_silence_ratio`: `0.6521 -> 0.6920`
- Практический вывод:
  - контрастный `electric_muted` chunk дал заметный прирост после стагнации на втором chunk
  - `collapse_to_silence` не появился, поэтому `balance_by_silence=true` включать не нужно
  - текущие параметры остаются рабочими: `batch_size = 8`, `silence_weight = 0.1`, `note_weight = 1.0`, `sampler = balanced`, `balance_by_group = true`, `balance_by_silence = false`, `use_amp = false`
  - следующий meaningful run должен resume от `training-state-50372.pt`
  - следующий chunk лучше выбирать контрастный к уже пройденным: сначала `acoustic`, затем новый `electric_clean` timbre/group, затем новый `electric_distortion`

## 2026-06-06 21:58 +05:00 — Fourth full-chunk run completed

- Четвертый meaningful SynthTab Full chunk завершен через resume от третьего full-run:
  - experiment: `generated\experiments\full_chunk_acoustic_luthier_pick_part1_28ep_resume_from_50372`
  - chunk: `acoustic\luthier_pick\part_1_-_1_to_B_C`
  - run mode: `resume`
  - train tracks: `2237`
  - val tracks: `559`
  - epochs target: `112`
  - batch size: `8`
  - start iter: `50372`
  - final iter: `58184`
  - runtime по `summary.json`: `07:09:23`
  - final checkpoint: `generated\experiments\full_chunk_acoustic_luthier_pick_part1_28ep_resume_from_50372\models\training-state-58184.pt`
- Перед запуском была выполнена безопасная очистка диска:
  - удален rebuildable full-chunk cache
  - удалены старые full chunk zip/extracted backups
  - сохранены JAMS/MIDI, финальные summaries и checkpoints
  - свободное место выросло примерно с `147 GiB` до `633 GiB`
- Для acoustic chunk добавлен loader-fix:
  - `demo_embedding\SynthTab.py` теперь заменяет `NaN/Inf` audio samples на нули перед feature extraction
  - причина: некоторые acoustic renders содержат non-finite samples, из-за чего `librosa` падала с `Audio buffer is not finite everywhere`
- Итоговые метрики на SynthTab Full validation:
  - `multi_pitch precision = 0.7394`
  - `multi_pitch recall = 0.8194`
  - `multi_pitch f1 = 0.7725`
  - `tablature precision = 0.4868`
  - `tablature recall = 0.7599`
  - `tablature f1 = 0.5847`
  - `tablature tdr = 0.8462`
  - `tablature accuracy = 0.7801`
  - `pred_silence_ratio = 0.5324`
  - `ref_silence_ratio = 0.6764`
  - `non_silent_accuracy = 0.7657`
  - `collapse_to_silence = false`
- Сравнение с третьим full chunk:
  - `multi_pitch f1`: `0.7307 -> 0.7725`
  - `tablature f1`: `0.5625 -> 0.5847`
  - `non_silent_accuracy`: `0.7079 -> 0.7657`
  - `pred_silence_ratio`: `0.6920 -> 0.5324`
- Практический вывод:
  - acoustic chunk дал дополнительный прирост, особенно по multi-pitch и non-silent accuracy
  - tablature F1 растет медленнее, но тренд положительный: `48.11% -> 56.25% -> 58.47%` на последних трех chunks
  - collapse-to-silence не появился, поэтому базовые параметры и `balance_by_silence=false` оставляем
  - следующий meaningful run должен resume от `training-state-58184.pt`
  - после еще одного контрастного chunk стоит переоценить, продолжать ли scaling или перейти к tab head/loss/label representation

## 2026-06-08 20:25 +05:00 — Fifth and sixth full-chunk runs completed

- Пятый и шестой meaningful SynthTab Full chunks завершены через automated handoff:
  - 5th chunk: `electric_clean\peregrine_clean_neck`
  - 6th chunk: `electric_clean\lespaul_clean_both`
  - оба прошли download/verify, extract, inspect, resume smoke и long train
  - BOM-проблемы в runtime configs не повторились: smoke configs писались через UTF-8 without BOM
- Пятый chunk `electric_clean\peregrine_clean_neck`:
  - experiment: `generated\experiments\full_chunk_electric_clean_peregrine_clean_neck_28ep_resume_from_58184`
  - train tracks: `3632`
  - val tracks: `908`
  - epochs target: `140`
  - overall chunk iter range: `58184 -> 70896`
  - final continuation start iter по итоговому `summary.json`: `69988`
  - final iter: `70896`
  - final continuation runtime по `summary.json`: `01:03:33`
  - final checkpoint: `generated\experiments\full_chunk_electric_clean_peregrine_clean_neck_28ep_resume_from_58184\models\training-state-70896.pt`
  - `multi_pitch f1 = 0.7264`
  - `tablature f1 = 0.5048`
  - `tablature accuracy = 0.8142`
  - `pred_silence_ratio = 0.6219`
  - `ref_silence_ratio = 0.7509`
  - `non_silent_accuracy = 0.6793`
  - `collapse_to_silence = false`
- Шестой chunk `electric_clean\lespaul_clean_both`:
  - experiment: `generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896`
  - train tracks: `3632`
  - val tracks: `908`
  - epochs target: `168`
  - start iter: `70896`
  - final iter: `83608`
  - runtime по `summary.json`: `11:57:34`
  - final checkpoint: `generated\experiments\full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896\models\training-state-83608.pt`
  - `multi_pitch f1 = 0.7361`
  - `tablature f1 = 0.5119`
  - `tablature accuracy = 0.8187`
  - `pred_silence_ratio = 0.6272`
  - `ref_silence_ratio = 0.7509`
  - `non_silent_accuracy = 0.6777`
  - `collapse_to_silence = false`
- Сравнение с acoustic peak:
  - `tablature f1`: `0.5847 -> 0.5048 -> 0.5119`
  - `multi_pitch f1`: `0.7725 -> 0.7264 -> 0.7361`
  - `non_silent_accuracy`: `0.7657 -> 0.6793 -> 0.6777`
  - `collapse_to_silence` не появился
- Практический вывод:
  - технически chunk-based workflow и automated handoff сработали
  - два похожих `electric_clean` chunks после acoustic не улучшили качество
  - лучший checkpoint по метрикам пока остается `training-state-58184.pt` от acoustic run
  - последний технический checkpoint цепочки: `training-state-83608.pt`
  - следующий шаг: не запускать еще один похожий clean chunk вслепую, а диагностировать string/fret bottleneck и распределение ошибок

## 2026-06-08 20:41 +05:00 — Tab/string-fret bottleneck diagnostic added

- Добавлен CPU-only диагностический скрипт:
  - `demo_embedding\diagnose_tablature_gap.py`
  - читает только `summary.json` и per-track `.txt` метрики
  - не импортирует training pipeline
  - не загружает аудио, модель или checkpoint
  - не использует GPU
- Добавлен tracked отчет:
  - `TAB_HEAD_DIAGNOSTIC.md`
- Диагностика по шести completed runs:
  - best-by-metrics checkpoint: `training-state-58184.pt` от acoustic run
  - latest chronological checkpoint: `training-state-83608.pt`
  - `collapse_to_silence=false` на всех шести runs
  - на clean chunks примерно треть validation tracks имеет высокий `multi_pitch F1` при слабом `tablature F1`
  - mean MP-Tab gap на двух последних clean chunks вернулся к `~22 pp`
- Практический вывод:
  - текущий bottleneck ближе к string/fret assignment, чем к pitch detection или silence-collapse
  - `balance_by_silence=true` не включать
  - следующий похожий `electric_clean` long-run вслепую не запускать
  - следующий полезный шаг: маленький string/fret-focused эксперимент, post-processing diagnostic или план изменения tab head/loss/label representation

## 2026-06-08 21:42 +05:00 — String/fret frame experiment completed

- Добавлен eval-only диагностический скрипт:
  - `demo_embedding\diagnose_string_fret_frames.py`
  - запускает inference на выбранных validation tracks
  - не обучает модель
  - пишет outputs в ignored `generated\diagnostics`
- Эксперимент:
  - active chunk: `electric_clean\lespaul_clean_both`
  - tracks: `20` full validation tracks
  - selection: `10` worst high-gap, `5` typical, `5` best
  - excluded: tracks with `ref_silence_ratio > 95%`
  - checkpoints compared: `training-state-58184.pt` and `training-state-83608.pt`
- Итоговые frame-level метрики:
  - `training-state-58184.pt`: exact string/fret F1 `61.67%`, pitch-only F1 `73.65%`, canonical F1 `55.13%`
  - `training-state-83608.pt`: exact string/fret F1 `65.43%`, pitch-only F1 `79.23%`, canonical F1 `58.71%`
  - oracle upper bound for `83608`: `79.23%`, то есть potential string/fret gain about `13.80 pp`
  - canonical hard post-processing хуже exact output: `58.71%` vs `65.43%`
- Практический вывод:
  - на active clean-domain subset latest checkpoint `training-state-83608.pt` лучше `training-state-58184.pt`
  - string/fret assignment остается реальным bottleneck, но missed/extra notes тоже значимы
  - простой deterministic pitch-to-position post-processing не использовать
  - следующий шаг: logits-aware/top-k string/fret diagnostics или изменение tab head/loss/label representation

## 2026-06-08 22:11 +05:00 — Logits-aware top-k diagnostic completed

- Расширен eval-only скрипт:
  - `demo_embedding\diagnose_string_fret_frames.py --topk`
  - снимает raw `SoftmaxGroups` logits/probabilities до `finalize_output`
  - не меняет `amt_tools`
  - не запускает training
- Эксперимент:
  - active chunk: `electric_clean\lespaul_clean_both`
  - checkpoint: `training-state-83608.pt`
  - tracks: same `20` full validation tracks from `STRING_FRET_EXPERIMENT.md`
  - output layout: `6` string groups x `21` classes, class `20` is silence
  - generated outputs: `generated\diagnostics\string_fret_topk_lespaul_20tracks`
- Итоговые top-k метрики:
  - correct string/fret top-1: `73.64%`
  - correct string/fret top-3: `95.41%`
  - correct string/fret top-5: `98.40%`
  - pitch-compatible top-5: `87.48%`
  - pitch-correct/tab-wrong correct-position top-5: `99.78%`
  - pitch-correct/tab-wrong pitch-compatible top-5: `99.19%`
  - average PCW top1-minus-correct probability gap: `49.12%`
  - average extra-note margin over silence: `54.74%`
  - extra notes with margin `<= 10%`: `6.50%`
- Практический вывод:
  - модель часто содержит правильную string/fret позицию в top-k, но top-1 decoding выбирает другую
  - простой threshold/silence calibration не выглядит главным решением: extra notes обычно уверенные
  - следующий шаг: top-k constrained decoding experiment
  - если constrained decoding не даст прироста, переходить к tab head/loss/label representation

## 2026-06-08 22:41 +05:00 — Top-k constrained decoding experiment completed

- Расширен eval-only скрипт:
  - `demo_embedding\diagnose_string_fret_frames.py --decode-experiment`
  - использует raw `SoftmaxGroups` probabilities до `finalize_output`
  - проверяет top-k constrained decoding variants без обучения
  - пишет outputs в ignored `generated\diagnostics`
- Эксперимент:
  - active chunk: `electric_clean\lespaul_clean_both`
  - checkpoint: `training-state-83608.pt`
  - tracks: same `20` full validation tracks from `STRING_FRET_EXPERIMENT.md`
  - generated outputs: `generated\diagnostics\string_fret_constrained_lespaul_20tracks`
  - tracked report: `CONSTRAINED_DECODING_EXPERIMENT.md`
- Варианты:
  - `baseline_top1`
  - `topk3_smooth_light`
  - `topk5_smooth_light`
  - `topk5_smooth_medium`
  - `topk5_smooth_strong`
  - `topk5_smooth_medium_dup`
- Итоговые метрики:
  - baseline exact string/fret F1: `65.43%`
  - baseline pitch F1: `79.23%`
  - best exact variant: `topk5_smooth_medium_dup`
  - best exact F1: `65.78%`
  - best exact delta: `+0.35 pp`
  - best pitch F1: `80.93%`
  - best pitch delta: `+1.70 pp`
  - best extra/pred: `25.95%`
  - best extra relative delta: `-9.65%`
  - accepted variants: none
- Практический вывод:
  - constrained decoding немного улучшает pitch F1 и снижает extra notes
  - exact string/fret F1 почти не растет и не проходит заранее заданный порог `+1 pp`
  - простой decoder не решает bottleneck
  - следующий шаг: не новый похожий clean chunk, а маленький эксперимент с tab head/loss/label representation

## 2026-06-09 22:15 +05:00 — Small tab loss experiment completed

- Добавлен opt-in API для tab loss modes:
  - default `train.tab_loss_mode = "ce"` сохраняет старое поведение
  - `train.tab_loss_mode = "focal_ce"`
  - `train.tab_loss_mode = "ce_plus_position_margin"`
  - `train.focal_gamma = 1.5`
  - `train.position_margin_weight = 0.05`
  - `train.position_margin = 0.5`
- Эксперимент:
  - active chunk: `electric_clean\lespaul_clean_both`
  - resume checkpoint: `training-state-83608.pt`
  - train subset: `800` tracks
  - budget: `1` epoch, `100` optimizer steps per variant
  - diagnostic subset: same `20` tracks from `STRING_FRET_EXPERIMENT.md`
  - outputs kept ignored under `generated\experiments\tab_loss_ablation_lespaul_*` and `generated\diagnostics\tab_loss_ablation_lespaul_*`
  - tracked report: `TAB_LOSS_EXPERIMENT.md`
- Результаты на 20-track diagnostic:
  - `control_ce_1ep`: exact F1 `64.56%`, pitch F1 `77.15%`, extra/pred `31.56%`
  - `focal_ce_g1p5_1ep`: exact F1 `65.11%`, pitch F1 `77.41%`, extra/pred `31.18%`
  - `position_margin_l005_m050_1ep`: exact F1 `64.58%`, pitch F1 `77.11%`, extra/pred `31.61%`
  - best delta vs matched CE control: focal CE `+0.55 pp` exact F1
  - accepted variants: none
- Практический вывод:
  - focal CE слегка лучше matched control, но не проходит порог `+1 pp`
  - текущий position-margin loss почти нейтрален
  - full validation для этих variants не запускался по decision rules
  - следующий шаг: plan tab head / label-representation experiment, а не продолжать simple loss-only tuning
