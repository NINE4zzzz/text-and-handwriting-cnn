<div align="center">

# 🧠 AI vs Human Text & Handwritten Math Symbols

**Промежуточная аттестация №3 · Инженер-архитектор в сфере ИИ · Университет Иннополис**

Две независимые задачи в одном воспроизводимом Jupyter Notebook:<br>
**NLP**: отличаем тексты ИИ от человеческих &nbsp;·&nbsp; **CV**: распознаём цифры и знаки «+», «−»

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-BiGRU-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LeNet--5-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![CPU](https://img.shields.io/badge/runs%20on-CPU-success)

</div>

---

## 📌 О проекте

Ноутбук [`Attestation_3.ipynb`](Attestation_3.ipynb) запускается сверху вниз, не требует GPU и состоит из двух частей и общих выводов.

| Часть | Задача | Данные | Модели |
|:--|:--|:--|:--|
| **1 · NLP** | Бинарная классификация: текст написан человеком или ИИ | [Kaggle: AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) | TF-IDF + LogisticRegression (baseline) и **BiGRU** на PyTorch |
| **2 · CV** | Распознавание 12 классов: цифры `0–9`, `+`, `−` | MNIST (Keras) + [HASYv2](https://www.kaggle.com/datasets/guru001/hasyv2) | **LeNet-5** на Keras / TensorFlow |
| **Выводы** | Сводная таблица, ограничения, идеи улучшений | — | — |

## ✨ Что внутри

### Часть 1. NLP: AI vs Human

- **Автоопределение колонок**: ноутбук сам находит колонку с текстом и колонку с меткой, поэтому подходит под разные версии CSV.
- **EDA**: баланс классов (barplot + pie), пропуски и пустые тексты, статистика длин (min / max / mean / median / квантили 25-50-75-95), гистограммы по классам, дубликаты и конфликты разметки.
- **Предобработка**: удаление пустых и коротких (< 10 симв.) текстов, точных дубликатов и текстов с конфликтующими метками; очистка текста; стратифицированное разбиение **70 / 15 / 15**.
- **Baseline**: TF-IDF (1–2-граммы, `sublinear_tf`) + LogisticRegression, `C` подбирается по validation macro F1.
- **Нейросеть**: двунаправленная GRU (`pack_padded_sequence`, mean + max pooling), `CrossEntropyLoss` с весами классов, `AdamW`, **early stopping по val macro F1**.
- **Метрики и графики**: accuracy, macro precision / recall / F1, ROC-AUC, classification report, confusion matrix, ROC-кривые, кривые loss и F1 по эпохам, сравнительный barplot, топ-n-граммы по классам и стилевые признаки.

### Часть 2. CV: MNIST + «+» и «−»

- **Единый 12-классовый датасет** 28×28: MNIST (цифры) и подмножество HASYv2 (только `+` и `-`), нормализация в `[0, 1]`, стратифицированное разбиение.
- **LeNet-5**: `Conv → Pool → Conv → Pool → Flatten → Dense → Dropout → Dense → Softmax`, `sparse_categorical_crossentropy`, `Adam`, `EarlyStopping` + `ReduceLROnPlateau` по `val_loss`, веса классов, macro F1 по эпохам.
- **Оценка**: кривые accuracy / loss / F1, confusion matrix 12×12, classification report, примеры верных и ошибочных предсказаний, **отдельная accuracy по цифрам и по символам**.
- **Мини-ассистент**: функции `recognize(img)` и `evaluate_expression(images)` распознают картинки и считают простые выражения вроде `7 + 2 − 4`.

## 🛡️ Fallback: ноутбук не падает без данных

| Чего нет | Что делает ноутбук |
|:--|:--|
| MNIST недоступен (нет интернета) | Берёт `sklearn.datasets.load_digits` (8×8 → 28×28) и **явно помечает это** в выводах |
| HASYv2 не найден | Генерирует синтетические «+» и «−» (дрожащие штрихи, разная толщина, наклон, шум) и помечает это в выводах |
| CSV для Части 1 не найден | Пробует `kagglehub`, иначе останавливается с понятным сообщением |

> ⚠️ Метрики на fallback-данных демонстрируют работоспособность пайплайна, но не заменяют эксперимент на оригинальных датасетах.

## 🚀 Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Создать окружение и поставить зависимости
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Положить данные (см. ниже) и запустить
jupyter notebook Attestation_3.ipynb
```

### 📂 Данные

**Часть 1.** Скачайте CSV с [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) и положите рядом с ноутбуком или в `data/`. Подойдёт `AI_Human.csv` или `ai_vs_human.csv`. Папку можно задать переменной `AI_HUMAN_DIR`.

**Часть 2.** MNIST загружается автоматически. Для HASYv2 распакуйте [архив с Kaggle](https://www.kaggle.com/datasets/guru001/hasyv2) так, чтобы получилось:

```text
hasyv2/
├── hasy-data-labels.csv
└── hasy-data/
    ├── v2-00000.png
    └── ...
```

Другой путь задаётся так: `os.environ["HASY_DIR"] = "/путь/к/hasyv2"` до запуска ячеек Части 2.

### 🔑 Автоскачивание через Kaggle (опционально)

```bash
pip install kagglehub
```

Затем настройте [учётные данные Kaggle](https://github.com/Kaggle/kagglehub#authenticate), и ноутбук скачает данные сам.

## ⚙️ Зафиксированные гиперпараметры

| Компонент | Параметры |
|:--|:--|
| Общее | `SEED = 42` (`random`, `numpy`, `tensorflow`, `torch`) |
| Часть 1, данные | подвыборка `NLP_SAMPLE_SIZE = 50 000` (`None` = весь датасет), разбиение 70 / 15 / 15 |
| TF-IDF | `ngram_range=(1, 2)`, `min_df=3`, `max_df=0.95`, `max_features=300 000`, `sublinear_tf=True` |
| LogReg | `solver="liblinear"`, `class_weight="balanced"`, `C ∈ {0.5, 2, 8, 32}` (выбор по val macro F1) |
| BiGRU | `max_len=256`, `max_vocab=30 000`, `emb_dim=128`, `hidden_size=64`, `num_layers=1`, `dropout=0.3`, `lr=2e-3`, `weight_decay=1e-2`, `batch_size=128`, `epochs=8`, `patience=2` |
| LeNet-5 | `batch_size=128`, `epochs=30`, `lr=1e-3`, `dropout=0.3`, EarlyStopping `patience=5`, ReduceLROnPlateau `factor=0.5`, `patience=2` |

Все значения собраны в словарях `GRU_CFG`, `CV_CFG`, `TFIDF_PARAMS` в начале соответствующих секций.

## 📊 Результаты

Итоговая таблица строится автоматически в разделе **«Выводы»** ноутбука. Заполните её своими цифрами после запуска:

| Часть | Модель | Accuracy | Macro F1 | ROC-AUC |
|:--|:--|:--:|:--:|:--:|
| 1 · NLP | TF-IDF + LogReg | — | — | — |
| 1 · NLP | BiGRU (PyTorch) | — | — | — |
| 2 · CV (12 классов) | LeNet-5 | — | — | — |
| 2 · CV (только цифры) | LeNet-5 | — | — | — |
| 2 · CV (только «+», «−») | LeNet-5 | — | — | — |

## ⚠️ Ограничения

- **Часть 1.** Работа идёт на подвыборке ради скорости на CPU. Высокие метрики на этом датасете могут отражать его артефакты (однотипные темы и промпты, разница в оформлении), а не «стиль ИИ». Использован один seed и одно разбиение. BERT / DistilBERT не применялся.
- **Часть 2.** Цифры (MNIST) и символы (HASYv2 или синтетика) приходят из разных источников, поэтому возможен domain shift. Символов заметно меньше, чем цифр, и это компенсируется весами классов.

## 🔭 Идеи для развития

- **NLP:** проверка на текстах других LLM и из других доменов, char-n-граммы, перплексия языковой модели как признак, DistilBERT на подвыборке 3–5 тыс. примеров, калибровка вероятностей, кросс-валидация и несколько seed.
- **CV:** аугментация (повороты, сдвиги, масштаб, толщина штриха), больше классов HASYv2 (`×`, `÷`, `=`, скобки), более глубокая CNN с BatchNorm, единая предобработка штриха, сегментация целых рукописных выражений.

## 🗂️ Структура репозитория

```text
.
├── Attestation_3.ipynb        # основной ноутбук (запускать сверху вниз)
├── Attestation_3_cells.py     # тот же код в формате ячеек (# %% [markdown] / # %% [code])
├── requirements.txt           # зависимости
├── README.md
├── data/                      # (не в git) AI_Human.csv
└── hasyv2/                    # (не в git) hasy-data-labels.csv + hasy-data/
```

## 💡 Полезно знать

- **GPU не нужен.** PyTorch использует `cuda` только если он доступен, TensorFlow выбирает устройство сам.
- **Время на CPU:** Часть 1 с подвыборкой 50 000 текстов занимает порядка десятков минут (в основном BiGRU), Часть 2 считается за несколько минут. Для ускорения уменьшите `NLP_SAMPLE_SIZE`.
- **Порядок импортов важен.** PyTorch импортируется раньше TensorFlow, а `torch._dynamo` подгружается заранее: иначе в окружении, где стоят оба фреймворка, ядро может упасть с segfault. Не меняйте этот блок.

## 📄 Лицензия

Добавьте файл `LICENSE` (например, MIT) на свой выбор. Датасеты распространяются на условиях их авторов на Kaggle.

---

<div align="center">

Сделано в рамках промежуточной аттестации №3 · Университет Иннополис

</div>
