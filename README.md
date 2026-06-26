# Multilabel Image Classification with PyTorch

> Репозиторий для [Junior ML Contest 2026](https://ai.itmo.ru/junior_ml_contest).  
> Лёгкий multilabel-классификатор изображений: YAML-конфиг, обучение, eval, ONNX export, batch inference.  
> **Полное описание задачи, реализации и вклада — в PDF заявки** (3 стр.) и черновике [`docs/jmlc_project_description.md`](docs/jmlc_project_description.md).

## Правовой статус и NDA

Проект разработан в рамках работы в компании **Sber / Sber AI** и защищён **NDA**. Публикуемая версия подготовлена для [Junior ML Contest 2026](https://ai.itmo.ru/junior_ml_contest) и содержит **обезличенное ядро ML**: multilabel-классификация, DB loss, EVA02-backbone, обучение, метрики, ONNX export и batch inference.

Исходный код и методология разработаны в рамках рабочей деятельности и **принадлежат компании ПАО «Сбербанк» / экосистеме Sber AI**. Материалы проекта, включая архитектуру пайплайна, описание данных, таксономию классов, конфигурации prod-среды и интеграции с внутренними системами, **подпадают под режим NDA** и не могут быть опубликованы в полном объёме.

Из публикации намеренно исключены:

- внутренние хранилища данных и пути к файловым системам;
- интеграции с корпоративными платформами (batch/map-reduce jobs, внутренние таблицы);
- autolabel-пайплайны, промпты и агрегация разметки на prod-данных;
- полная продуктовая таксономия классов и метрики на реальных пользовательских данных;
- веса моделей, обученные на закрытых датасетах;
- любые идентификаторы, позволяющие восстановить prod-инфраструктуру.

**Сохранено и воспроизводимо в demo-режиме:** ядро ML — multilabel-классификация изображений, обучение (PyTorch), DB loss для дисбаланса классов, backbone EVA02, eval-метрики, экспорт в ONNX, batch-inference API на синтетических данных.

Demo-данные синтетические (`class_0` … `class_N`). **Права на оригинальный код остаются у правообладателя**; публикация не означает transfer прав на код или модели третьим лицам. Использование вне рамок конкурса — только с разрешения правообладателя.

На защите можно сказать:

> «В prod решение обрабатывает большие объёмы данных; здесь показываю инженерное ядро, воспроизводимое локально через `python scripts/smoke_demo.py`.»

Расширенное описание задачи и реализации для PDF: [`docs/jmlc_project_description.md`](docs/jmlc_project_description.md).

## Быстрый старт

```bash
pip install -r requirements.txt
python scripts/smoke_demo.py
```

Smoke test на CPU (~15 с): 1 эпоха обучения + forward pass на demo-данных. Конфиг: `config/smoke.yaml`.

## Установка и обучение

```bash
pip install -r requirements.txt

# обучение (основной конфиг — EVA02, нужен GPU + timm для pretrained)
python -m src.main --config config.yaml

# multi-GPU
torchrun --nproc_per_node=2 src/main.py --config config.yaml

# экспорт в ONNX
python -m src.export_onnx \
    --checkpoint_path checkpoints/final_model.pth \
    --config config.yaml \
    --output checkpoints/model.onnx
```

Wandb опционален (`wandb.enabled: false` по умолчанию). Для включения: `WANDB_API_KEY` + `enabled: true` в `config.yaml`.

## Формат данных

CSV: первая колонка — путь к изображению, остальные — бинарные метки (0/1) для каждого класса.

```csv
image_path,class_0,class_1,class_2,class_3
data/demo/images/sample_00.png,1,0,0,0
```

Demo: `data/demo/`. Перед своим обучением замените пути и `num_classes` в `config.yaml`.

## Структура проекта

```
jmlc/
├── config.yaml          # основной конфиг (EVA02)
├── config/smoke.yaml    # быстрый CPU-тест
├── data/demo/           # синтетические данные
├── docs/                # черновик текста для PDF
├── scripts/smoke_demo.py
└── src/
    ├── main.py          # точка входа
    ├── trainer.py       # обучение (DDP, bf16, DB loss)
    ├── dataset.py       # CSV + augmentations
    ├── model.py         # backbone + FC head
    ├── eva.py           # EVA02 (timm-compatible)
    ├── losses.py        # DB loss
    ├── evaluator.py     # multilabel-метрики
    ├── inferencer.py    # batch inference
    └── export_onnx.py   # деплой
```

## Автор

Весь код в репозитории реализован **мной лично** (участником конкурса). Подробный перечень вклада — в PDF-описании проекта.
