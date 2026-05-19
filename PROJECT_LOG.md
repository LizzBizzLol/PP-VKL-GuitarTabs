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
