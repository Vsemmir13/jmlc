# JMLC: Multi-Label Image Classification

Публичная воспроизводимая версия проекта для [**Junior ML Contest 2026**](https://ai.itmo.ru/junior_ml_contest): пайплайн
multi-label классификации изображений на PyTorch с YAML-конфигурацией, обучением,
валидацией, экспортом в ONNX и batch inference.

Проект разработан в рамках работы в компании Sber и защищён NDA, поэтому публикуемая версия подготовлена с обезличенными данными и пропусками NDA кода: внутренние пути, интеграции и данные заменены на локальный demo-режим, но сохранена основная структура ML-пайплайна и код, достаточный для запуска.

## Что внутри

- обучение multi-label классификатора изображений;
- поддержка CSV-датасета с несколькими бинарными метками на изображение;
- backbone на базе EVA02 для примера;
- DB loss для работы с дисбалансом классов;
- расчет multilabel-метрик на валидации;
- экспорт модели в ONNX;
- batch inference API;
- синтетический demo-датасет и smoke test для быстрой проверки.

## Быстрый старт

```bash
pip install -r requirements.txt
python scripts/smoke_demo.py
```

Smoke test запускается на CPU и выполняет короткое обучение на синтетических
demo-данных из `data/demo/`. Конфиг для проверки: `config/smoke.yaml`; в нем
используется маленький локальный CNN, чтобы быстрый старт не скачивал веса.

## Обучение

```bash
pip install -r requirements.txt

# обучение по основному конфигу
python -m src.main --config config.yaml

# multi-GPU запуск
torchrun --nproc_per_node=2 src/main.py --config config.yaml
```

Основной конфиг `config.yaml` рассчитан на обучение с GPU. Быстрый локальный
запуск без GPU удобнее проверять через `scripts/smoke_demo.py`.

WandB опционален и по умолчанию отключен. Для логирования нужно задать
`WANDB_API_KEY` и включить `wandb.enabled: true` в конфиге.

## Экспорт в ONNX

```bash
python -m src.export_onnx \
    --checkpoint_path checkpoints/final_model.pth \
    --config config.yaml \
    --output checkpoints/model.onnx
```

## Формат данных

На вход используется CSV-файл: первая колонка содержит путь к изображению,
остальные колонки — бинарные метки классов.

```csv
image_path,class_0,class_1,class_2,class_3
data/demo/images/sample_00.png,1,0,0,0
data/demo/images/sample_01.png,0,1,0,0
```

Для запуска на своих данных нужно заменить CSV-файлы, пути к изображениям и
значение `num_classes` в конфиге.

## Структура проекта

```text
jmlc/
├── config.yaml
├── config/
│   └── smoke.yaml
├── data/
│   └── demo/
├── scripts/
│   └── smoke_demo.py
└── src/
    ├── config.py
    ├── dataset.py
    ├── eva.py
    ├── evaluator.py
    ├── export_onnx.py
    ├── inferencer.py
    ├── losses.py
    ├── main.py
    ├── model.py
    └── trainer.py
```

## Основные компоненты

- `src/main.py` — точка входа для обучения;
- `src/trainer.py` — training loop, DDP, mixed precision, сохранение checkpoint;
- `src/dataset.py` — чтение CSV, загрузка изображений и аугментации;
- `src/model.py` — сборка backbone и классификационной головы;
- `src/losses.py` — DB loss;
- `src/evaluator.py` — расчет метрик для multi-label задачи;
- `src/inferencer.py` — batch inference;
- `src/export_onnx.py` — экспорт checkpoint в ONNX.

## Публичная версия

В репозитории нет закрытых данных, production-конфигураций, внутренних путей,
ключей, сервисных интеграций и весов моделей, обученных на закрытых датасетах.
Demo-данные синтетические и нужны только для проверки работоспособности кода.

## Автор

Код публичной версии подготовлен для конкурсной подачи на Junior ML Contest 2026.
