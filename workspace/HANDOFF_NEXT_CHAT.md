# Handoff For Next Chat

## Краткий контекст проекта

- Проект магистратуры ТюмГУ по генерации гитарных табулатур из аудио.
- В команде: я, Валя и Катя.
- Исторически сравнивались два подхода:
  - прямой `audio -> tablature` (`TabCNN`-ветка)
  - мой pipeline `audio -> MIDI -> tablature` с `BasicPitch` и отдельной моделью `pitch -> string/fret`
- В 1 семестре прямой подход выглядел слабее, но сейчас задача Вали: поднять и проверить обучение `TabCNN` на `SynthTab`, с нормальным pipeline и запуском через `CUDA`.

## Рабочие директории

- Основной рабочий проект:
  - `C:\Users\Liss\Documents\New project\SynthTab`
- Основной pipeline:
  - `C:\Users\Liss\Documents\New project\SynthTab\demo_embedding\tabcnn_synthtab_pipeline.py`
- Справочный репозиторий `tab-cnn`:
  - `D:\TSU-Project-Practicum\tab-cnn`
- Старый проект для контекста:
  - `D:\EJIu3@BetKo\Пары\Магистратура\tabs_ptoject`
- Датасет `SynthTab Dev`:
  - `C:\Users\Liss\Documents\New project\datasets\SynthTab_Dev\SynthTab_Dev`
- Виртуальное окружение:
  - `C:\Users\Liss\Documents\New project\.venv_synthtab`

## Что уже сделано

### 1. Поднят единый pipeline `TabCNN on SynthTab`

- Сделан единый entrypoint:
  - `tabcnn_synthtab_pipeline.py`
- Поддержаны режимы:
  - `inspect`
  - `train`
  - `eval`
- Старые скрипты `exp_training_from_scratch.py` и `evaluate.py` переведены на этот сценарий.
- Есть baseline-конфиги:
  - `tabcnn_synthtab_baseline.json`
  - `tabcnn_synthtab_baseline_dev.json`
  - `tabcnn_synthtab_sanity.json`

### 2. Локальный `SynthTab Dev` подключён

- Архив был распакован локально.
- Реальная структура dev-набора оказалась не `train/val`, а:
  - `acoustic`
  - `electric_clean`
  - `electric_distortion_di`
  - `electric_muted`
  - `jams`
- Поэтому адаптирован:
  - `C:\Users\Liss\Documents\New project\SynthTab\demo_embedding\SynthTab.py`
- Что в нём уже исправлено:
  - поддержка dev-layout
  - детерминированный split для `train/val`
  - рекурсивный поиск аудио
  - поиск `.jams` внутри dev-структуры
  - фиксация `dataset seed`

### 3. Поднято GPU-окружение

- Создано и настроено окружение:
  - `.venv_synthtab`
- Подтверждено:
  - `torch.cuda.is_available() == True`
  - GPU: `NVIDIA GeForce GTX 1660 Ti`
- Используются CUDA-сборки:
  - `torch 2.5.1+cu121`
  - `torchaudio 2.5.1+cu121`
  - `torchvision 0.20.1+cu121`

### 4. Базовый pipeline реально запускается

- Прошёл `inspect`
- Прошёл `sanity-run`
- Прошёл baseline dev-run
- Есть артефакты:
  - `C:\Users\Liss\Documents\New project\generated\sanity_run_direct`
  - `C:\Users\Liss\Documents\New project\generated\baseline_dev_run`

## Что было не так с метриками

- Изначально казалось, что `accuracy` неплохая, но `f1 = 0`.
- Проверка показала: модель коллапсировала в класс тишины.
- В `amt_tools` последний softmax-класс интерпретируется как silence (`-1`).
- На старом baseline checkpoint модель предсказывала почти только silence.
- Поэтому старая `accuracy` почти совпадала с долей silent-фреймов в валидации и вводила в заблуждение.

## Что я сделала для исправления

### 1. Добавила честные диагностические метрики

В `tabcnn_synthtab_pipeline.py` добавлены:

- `ref_silence_ratio`
- `pred_silence_ratio`
- `ref_non_silent_ratio`
- `pred_non_silent_ratio`
- `non_silent_accuracy`
- `collapse_to_silence`

Теперь они пишутся в `results/summary.json`.

### 2. Добавила class weighting против silence-collapse

В train-конфиги и model build добавлены параметры:

- `use_class_weights = true`
- `silence_weight = 0.1`
- `note_weight = 1.0`

Вес класса тишины уменьшен, чтобы модель не минимизировала loss простым предсказанием silence везде.

## Что уже подтверждено экспериментами

### Старый baseline checkpoint

Переоценка старого чекпоинта показала:

- `pred_silence_ratio = 1.0`
- `pred_non_silent_ratio = 0.0`
- `non_silent_accuracy = 0.0`
- `collapse_to_silence = true`

Это прямое подтверждение проблемы.

### Новый weighted sanity-run

Артефакты:

- `C:\Users\Liss\Documents\New project\generated\sanity_weighted_rerun\results\summary.json`

Результат:

- `multi_pitch f1 = 0.1280`
- `tablature f1 = 0.0295`
- `accuracy = 0.1825`
- `pred_silence_ratio = 0.1667`
- `pred_non_silent_ratio = 0.8333`
- `collapse_to_silence = false`

Интерпретация:

- `accuracy` упала, но это хорошо, потому что она перестала быть артефактом тишины
- `f1` ожил
- модель начала реально предсказывать ноты, а не только silence

## Текущее состояние

- Pipeline рабочий
- `CUDA` рабочая
- `SynthTab Dev` подключён
- baseline можно запускать локально
- основная методическая проблема baseline уже найдена: silence-collapse
- первый антиколлапсный фикс уже дал положительный эффект

## Что делать дальше

### Главный следующий шаг

Нужно прогнать уже не короткий `sanity`, а полноценный `baseline dev-run` с class weighting и сравнить его со старым baseline.

Цель:

- проверить, улучшаются ли:
  - `tablature f1`
  - `multi_pitch f1`
  - `non_silent_accuracy`
  - `pred_silence_ratio`
- и не остаётся ли модель в другом плохом режиме, например в переизбытке non-silent предсказаний

### Практический порядок

1. Запустить `baseline_dev_run` с текущим weighting-конфигом.
2. Сравнить новый `summary.json` со старым `generated/baseline_dev_run/results/summary.json`.
3. Если модель переходит в over-prediction:
   - подобрать `silence_weight`, например проверить:
     - `0.2`
     - `0.3`
     - `0.5`
4. После стабилизации baseline:
   - увеличивать число эпох или треков
   - потом уже думать о `fine-tuning`, `LSTM/recurrent`, `LoRA`

## Что изучить в коде перед продолжением

### В первую очередь

1. `C:\Users\Liss\Documents\New project\SynthTab\demo_embedding\tabcnn_synthtab_pipeline.py`
   - основной orchestration-код
   - конфиги
   - train/eval/inspect
   - диагностика метрик

2. `C:\Users\Liss\Documents\New project\SynthTab\demo_embedding\SynthTab.py`
   - как сейчас подключается `SynthTab Dev`
   - как формируются треки и split'ы

3. `C:\Users\Liss\Documents\New project\SynthTab\demo_embedding\train.py`
   - loop обучения
   - checkpoints
   - validation

### Потом

4. `C:\Users\Liss\Documents\New project\.venv_synthtab\Lib\site-packages\amt_tools\models\tabcnn.py`
   - как устроен `TabCNN`

5. `C:\Users\Liss\Documents\New project\.venv_synthtab\Lib\site-packages\amt_tools\models\common.py`
   - `SoftmaxGroups`
   - как считается weighted cross entropy
   - как последний класс превращается в `-1`

6. `C:\Users\Liss\Documents\New project\.venv_synthtab\Lib\site-packages\amt_tools\evaluate.py`
   - `SoftmaxAccuracy`
   - `TablatureEvaluator`
   - `MultipitchEvaluator`

## Что изучить в репозиториях и доках

### Репозитории

1. `SynthTab`
   - понять исходную идею обучения на синтетическом табовом датасете
   - найти, как авторы ожидали проводить train/eval/fine-tuning
   - посмотреть, как у них используются `JAMS`, `MIDI`, splitting, timbre groups

2. `tab-cnn`
   - посмотреть оригинальный baseline-подход
   - сравнить preprocessing, feature setup и режим обучения
   - понять, были ли у оригинального подхода меры против class imbalance

### Что особенно искать

- ожидали ли авторы balancing классов
- как они интерпретируют silence
- какие метрики считают основными
- есть ли рекомендации по train schedule
- как они делят `SynthTab` на `train/val/test`
- как они сравнивают `SynthTab` и `GuitarSet`

## Готовая инструкция для нового чата

Ниже текст, который можно вставить в новый чат почти как есть:

---

Мы продолжаем локальный проект `TabCNN on SynthTab` в `C:\Users\Liss\Documents\New project`.

Текущий основной файл pipeline:
`C:\Users\Liss\Documents\New project\SynthTab\demo_embedding\tabcnn_synthtab_pipeline.py`

Текущий датасет:
`C:\Users\Liss\Documents\New project\datasets\SynthTab_Dev\SynthTab_Dev`

Окружение:
`C:\Users\Liss\Documents\New project\.venv_synthtab`

Что уже сделано:
- подключён `SynthTab Dev`
- адаптирован `SynthTab.py` под dev-layout
- сделан единый `train/eval/inspect` pipeline
- настроен `CUDA`
- baseline и sanity-run уже запускались
- найдена проблема метрик: модель коллапсировала в silence
- в pipeline уже добавлены anti-collapse diagnostics
- добавлен class weighting против класса тишины

Что важно проверить в первую очередь:
- `tabcnn_synthtab_pipeline.py`
- `SynthTab.py`
- `train.py`
- `amt_tools/models/tabcnn.py`
- `amt_tools/models/common.py`
- `amt_tools/evaluate.py`

Что уже подтверждено:
- старый baseline checkpoint падал в `collapse_to_silence = true`
- новый weighted sanity-run дал:
  - `multi_pitch f1 = 0.1280`
  - `tablature f1 = 0.0295`
  - `pred_silence_ratio = 0.1667`
  - `collapse_to_silence = false`

Следующий шаг:
- прогнать полноценный `baseline_dev_run` с class weighting
- сравнить новый `summary.json` со старым baseline
- при необходимости подобрать `silence_weight`

Дополнительно прошу:
- изучить локальные репозитории `SynthTab` и `tab-cnn`
- изучить, как в них задуманы train/eval, метрики и работа с silence
- после этого продолжить baseline-эксперименты, не ломая текущий pipeline

Все проектные заметки уже есть в:
- `C:\Users\Liss\Documents\New project\PROJECT_CONTEXT.md`
- `C:\Users\Liss\Documents\New project\PROJECT_LOG.md`
- `C:\Users\Liss\Documents\New project\HANDOFF_NEXT_CHAT.md`

---
