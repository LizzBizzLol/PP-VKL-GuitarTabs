# Люма, начни отсюда

Это стартовый файл для продолжения проекта Guitar Tabs на другом компьютере.

Основной репозиторий:

`D:\PP-VKL-GuitarTabs`

GitHub:

`https://github.com/LizzBizzLol/PP-VKL-GuitarTabs`

## 1. Что читать в первую очередь

Прочитай файлы в таком порядке:

1. `LUMA_START_HERE.md` — этот файл.
2. `LUMA_NEXT_TASK_FULL_SYNTHTAB.md` — полное ТЗ следующего этапа.
3. `README.md` — структура проекта, команды запуска, лучшие метрики.
4. `PROJECT_LOG.md` — хронологический лог работы.
5. `workspace/HANDOFF_NEXT_CHAT.md` — подробный handoff по предыдущим этапам.

Эти документы уже содержат основную память проекта. Если чат недоступен, ориентируйся на них.

## 2. Текущее состояние

Проект посвящён распознаванию гитарных табулатур по аудио.

Уже сделано:

- адаптирован код TabCNN и SynthTab;
- реализован pipeline обучения/eval/inspect на SynthTab Dev;
- подобраны базовые параметры против collapse-to-silence;
- лучшая dev-метрика: примерно `multi-pitch F1 = 0.632`, `tablature F1 = 0.415`, `non-silent accuracy = 0.526`;
- добавлен полноценный resumable training через `training-state-*.pt`;
- добавлен balanced sampler;
- добавлены smoke-конфиги и full/chunk template;
- ноутбучный smoke workflow проверен на SynthTab Dev.

Последние важные коммиты:

- `98ddcd5 docs: add next full SynthTab training task`
- `1de8b18 feat: add resumable balanced SynthTab training prep`
- `a7b9fdb test: verify laptop SynthTab smoke workflow`

## 3. Что уже проверено на ноуте

На ноуте создано `.venv`, установлены зависимости и проверены короткие CPU-запуски.

Проверено:

- `inspect` на SynthTab Dev проходит;
- fresh smoke-train проходит;
- создаются `model-*.pt`, `opt-state-*.pt`, `training-state-*.pt`;
- resume из `training-state-2.pt` проходит;
- `run_config.json` фиксирует `run_mode = "resume"`, `start_iter = 2`, `final_iter = 4`;
- balanced sampler работает;
- `balance_by_silence = true` работает и создаёт JAMS-derived density buckets;
- добавлен fallback через `soundfile`, потому что на Windows `torchaudio.load()` может требовать TorchCodec/FFmpeg DLL.

Сгенерированные результаты smoke-run лежали в `generated/`, но эта папка ignored и не должна коммититься.

## 4. Важное про SynthTab

Текущий `SynthTab_Dev.zip` — это не полный датасет, а development subset.

Dev-набор:

https://rochester.app.box.com/v/SynthTab-Dev

Полный датасет:

https://rochester.app.box.com/v/SynthTab-Full

Full SynthTab занимает почти `2 TB`, поэтому его нельзя просто скачать и распаковать целиком на текущие устройства.

Основная стратегия для компьютера:

1. скачать/подключить часть full SynthTab;
2. обучить модель на этой части;
3. сохранить `training-state-*.pt`;
4. заменить chunk датасета;
5. продолжить через `resume_from`;
6. повторять по следующим частям.

## 5. Что делать дальше на компьютере

Главная задача — перейти от проверенного Dev-pipeline к chunk-based обучению на full SynthTab.

Рекомендуемый порядок:

1. Прочитать документы из раздела 1.
2. Подтянуть свежий `main` из GitHub.
3. Собрать окружение по `README.md`.
4. Повторить `inspect` и smoke-train на SynthTab Dev.
5. Проверить resume из `training-state-*.pt`.
6. Скачать/подключить маленький chunk full SynthTab.
7. Адаптировать путь в `tabcnn_synthtab_full_chunk_template.json`.
8. Запустить smoke-train на chunk.
9. Если всё стабильно — запускать последовательное обучение по chunk-ам.
10. После каждого chunk сохранять метрики, checkpoint и описание данных.

## 6. Что не делать случайно

Не коммить:

- `.venv/`
- `generated/`
- датасеты;
- `.zip` архивы датасетов;
- `model-*.pt`;
- `opt-state-*.pt`;
- `training-state-*.pt`;
- TensorBoard events.

Не скачивать full SynthTab целиком без проверки свободного места.

Не продолжать обучение из legacy `model-*.pt`: для resume использовать только `training-state-*.pt`.

## 7. Минимальная команда проверки

Пример inspect-команды из корня репозитория:

```powershell
./.venv/Scripts/python.exe demo_embedding/tabcnn_synthtab_pipeline.py --mode inspect --config demo_embedding/tabcnn_synthtab_resume_balanced_smoke.json --experiment-dir generated/experiments/laptop_inspect_smoke_root
```

Для resume в конфиге нужно указать:

```json
"resume_from": "path/to/training-state-*.pt"
```

## 8. Ближайший инженерный фокус

Приоритет не в новом коде, а в аккуратном запуске на большем объёме данных:

- проверить full SynthTab structure;
- выбрать первый маленький chunk;
- подтвердить, что loader видит данные;
- запустить короткий train;
- проверить checkpoint/resume;
- только потом запускать долгий этап.

Если что-то ломается, сначала обнови `PROJECT_LOG.md` и зафиксируй факт, команду, ошибку и решение.
