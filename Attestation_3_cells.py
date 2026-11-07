# %% [markdown]
# # Промежуточная аттестация №3
# **Инженер-архитектор в сфере искусственного интеллекта · Университет Иннополис**
#
# Ноутбук состоит из двух независимых частей и общих выводов:
#
# | Раздел | Задача | Данные | Модели |
# |---|---|---|---|
# | **Часть 1** | NLP: бинарная классификация «текст написан человеком или ИИ» | Kaggle `shanegerami/ai-vs-human-text` | TF-IDF + LogisticRegression (baseline) и BiGRU на PyTorch |
# | **Часть 2** | CV: распознавание рукописных цифр и символов «+», «−» | MNIST (Keras) + HASYv2 (Kaggle `guru001/hasyv2`) | LeNet-5 на Keras/TensorFlow |
# | **Выводы** | Сводка результатов, ограничения, идеи улучшений | — | — |
#
# ## Что нужно подготовить перед запуском
# 1. **Часть 1.** Положить CSV с Kaggle (`AI_Human.csv` или `ai_vs_human.csv`) рядом с ноутбуком (или в папку `data/`). Если файла нет, ноутбук попробует скачать его через `kagglehub`, а если и это не выйдет, остановится с понятным сообщением.
# 2. **Часть 2.** MNIST загружается автоматически. HASYv2 ищется локально (папка с файлом `hasy-data-labels.csv`; можно указать путь в переменной окружения `HASY_DIR`), затем пробуется `kagglehub`. Если датасета нет, ноутбук **не падает**, а использует синтетические «+» и «−» (явно помечено в выводах).
#
# Зависимости: `pip install numpy pandas scikit-learn matplotlib seaborn torch tensorflow pillow` (опционально `kagglehub`).
# Всё рассчитано на **CPU**; GPU используется автоматически, если он есть.

# %% [markdown]
# ## 0. Общая подготовка: импорты, seed, устройство

# %% [code]
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # меньше шума от TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import re, glob, random, time, copy, warnings
from collections import Counter
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# PyTorch импортируем РАНЬШЕ TensorFlow; torch._dynamo подгружаем заранее: при совместном использовании
# двух фреймворков ленивая загрузка dynamo/triton после TensorFlow может привести к падению ядра (segfault).
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
try:
    import torch._dynamo  # noqa: F401
except Exception:
    pass
import tensorflow as tf

# ---- Воспроизводимость: фиксируем seed во всех библиотеках ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)     # python + numpy + tensorflow
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- Устройство для PyTorch (TensorFlow выбирает устройство сам) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
COLORS = {"Human": "#4C78A8", "AI": "#E45756"}

print("Python-библиотеки загружены")
print(f"TensorFlow {tf.__version__} | PyTorch {torch.__version__} | torch device: {device}")
print("TF-устройства:", [d.device_type for d in tf.config.list_physical_devices()])

# %% [markdown]
# ---
# # Часть 1. NLP: «AI vs Human text» (бинарная классификация)
#
# **Цель:** по тексту определить, написан он человеком (класс `0`) или сгенерирован ИИ (класс `1`).
#
# **План:** загрузка и адаптация под реальные колонки → EDA → предобработка и разбиение 70/15/15 → две модели (TF-IDF + LogReg и BiGRU) → метрики и графики → выводы.

# %% [markdown]
# ## 1.1. Загрузка данных
#
# Ищем файл в нескольких типичных местах. Структура колонок заранее неизвестна, поэтому сначала выводим `head()`, `columns`, `info()`, а затем **автоматически определяем** колонку с текстом (самые длинные строки) и колонку с меткой (малое число уникальных значений).

# %% [code]
def find_nlp_csv():
    names = ["ai_vs_human.csv", "AI_Human.csv", "ai_human.csv", "AI_vs_Human.csv"]
    dirs = [".", "data", "input", "../input", os.environ.get("AI_HUMAN_DIR")]
    for d in dirs:
        if not d:
            continue
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    for pattern in ["*.csv", "*/*.csv", "*/*/*.csv"]:        # любой csv с 'human' в имени
        for p in glob.glob(pattern):
            if "human" in os.path.basename(p).lower():
                return p
    return None


csv_path = find_nlp_csv()
if csv_path is None:                                   # попытка скачать через kagglehub
    try:
        import kagglehub
        d = kagglehub.dataset_download("shanegerami/ai-vs-human-text")
        found = glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
        csv_path = found[0] if found else None
    except Exception as e:
        print("kagglehub не сработал:", type(e).__name__)

if csv_path is None:
    raise FileNotFoundError(
        "Не найден CSV с датасетом AI vs Human text.\n"
        "Скачайте https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text и положите файл "
        "(AI_Human.csv или ai_vs_human.csv) рядом с ноутбуком, либо задайте переменную окружения AI_HUMAN_DIR."
    )

print("Файл с данными:", csv_path)
df_raw = pd.read_csv(csv_path)
print("\nРазмер:", df_raw.shape)
display(df_raw.head())
print("\nКолонки:", list(df_raw.columns))
df_raw.info()

# %% [code]
def detect_columns(df):
    """Колонка с текстом = строковая колонка с наибольшей средней длиной; метка = колонка с <=10 уникальных значений."""
    text_col, best_len = None, -1
    for c in df.columns:
        s = df[c].dropna().head(2000)
        if len(s) and s.map(lambda v: isinstance(v, str)).mean() > 0.9:
            m = s.str.len().mean()
            if m > best_len:
                text_col, best_len = c, m
    label_col = None
    for c in df.columns:
        if c != text_col and df[c].nunique(dropna=True) <= 10:
            label_col = c
            break
    if text_col is None or label_col is None:
        raise ValueError(f"Не удалось определить колонки текста/метки. Колонки: {list(df.columns)}")
    return text_col, label_col


def encode_labels(s):
    """Приводит метку к 0 (human) / 1 (AI) для числовых и строковых значений."""
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0.5).astype(int), {"<=0.5": 0, ">0.5": 1}
    ai_markers = {"ai", "generated", "machine", "llm", "chatgpt", "gpt", "1", "true", "ai-generated", "ai generated"}
    low = s.astype(str).str.strip().str.lower()
    mapping = {v: int(v in ai_markers) for v in low.unique()}
    return low.map(mapping).astype(int), mapping


text_col, label_col = detect_columns(df_raw)
print(f"Колонка текста: '{text_col}' | колонка метки: '{label_col}'")

df = df_raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_raw"})
n_label_nan = int(df["label_raw"].isna().sum())
df = df.dropna(subset=["label_raw"]).copy()
df["label"], label_mapping = encode_labels(df["label_raw"])
print("Кодирование меток:", label_mapping, "| строк без метки удалено:", n_label_nan)
df = df[["text", "label"]].reset_index(drop=True)
df["class"] = df["label"].map({0: "Human", 1: "AI"})
display(df.head(3))

# %% [markdown]
# ## 1.2. EDA (разведочный анализ)
#
# Смотрим: размер, баланс классов, пропуски/пустые тексты, статистику длин (символы и слова), распределения длин по классам, дубликаты и признаки возможной утечки.

# %% [code]
print("Размер датасета:", df.shape)
counts = df["label"].value_counts().sort_index()
share = (counts / counts.sum() * 100).round(2)
print(pd.DataFrame({"count": counts, "share, %": share}).rename(index={0: "Human (0)", 1: "AI (1)"}))

labs = ["Human", "AI"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=labs, y=counts.values, hue=labs, palette=COLORS, legend=False, ax=axes[0])
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")
axes[0].set_title("Распределение классов (barplot)")
axes[0].set_xlabel("Класс")
axes[0].set_ylabel("Число текстов")
axes[1].pie(counts.values, labels=labs, autopct="%1.1f%%", startangle=90,
            colors=[COLORS[l] for l in labs], wedgeprops={"edgecolor": "white"})
axes[1].set_title("Доля классов (pie chart)")
plt.tight_layout()
plt.show()

# %% [code]
txt = df["text"].fillna("")
n_missing_text = int(df["text"].isna().sum())
n_empty = int((txt.str.strip() == "").sum())
n_short = int((txt.str.strip().str.len() < 10).sum())
print("Пропуски в тексте:      ", n_missing_text)
print("Пустые тексты (после strip):", n_empty)
print("Слишком короткие (<10 символов):", n_short)
print("Пропуски в метках (удалены при загрузке):", n_label_nan)

df["n_chars"] = txt.str.len()
df["n_words"] = txt.str.split().str.len()


def length_stats(s):
    return pd.Series({"min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(),
                      "q25": s.quantile(0.25), "q50": s.quantile(0.50),
                      "q75": s.quantile(0.75), "q95": s.quantile(0.95)})


stats_all = pd.DataFrame({"символы": length_stats(df["n_chars"]), "слова": length_stats(df["n_words"])}).round(1)
print("\nСтатистика длин текстов (весь датасет):")
display(stats_all)

rows = []
for cls in ["Human", "AI"]:
    for col, name in [("n_chars", "символы"), ("n_words", "слова")]:
        st = length_stats(df.loc[df["class"] == cls, col]).round(1)
        st.name = (cls, name)
        rows.append(st)
print("Статистика длин по классам:")
display(pd.DataFrame(rows))

# %% [code]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, col, title in [(axes[0], "n_chars", "символах"), (axes[1], "n_words", "словах")]:
    cap = df[col].quantile(0.99)             # обрезаем длинный хвост для читаемости
    sns.histplot(data=df[df[col] <= cap], x=col, hue="class", hue_order=["Human", "AI"], palette=COLORS,
                 bins=60, stat="density", common_norm=False, element="step", ax=ax)
    ax.set_title(f"Распределение длины текстов в {title} (до 99-го перцентиля)")
    ax.set_xlabel(f"Длина, {title}")
    ax.set_ylabel("Плотность")
plt.tight_layout()
plt.show()

# %% [code]
# --- Дубликаты и возможные утечки ---
df["_h"] = pd.util.hash_pandas_object(df["text"], index=False)     # хэш текста: быстрее, чем сравнивать строки
n_dup_rows = int(df.duplicated("_h").sum())
n_conflict = int((df.groupby("_h")["label"].nunique() > 1).sum())
print(f"Полных дубликатов текста: {n_dup_rows} ({n_dup_rows / len(df) * 100:.2f}%)")
print(f"Текстов, встречающихся с РАЗНЫМИ метками (конфликт разметки): {n_conflict}")

# простые «стилевые» признаки на подвыборке (чтобы не считать регулярки по всему большому датасету)
eda_s = df.sample(min(len(df), 30_000), random_state=SEED).copy()
t_s = eda_s["text"].fillna("")
eda_s["has_newline"] = t_s.str.contains("\n").astype(int)
eda_s["punct_ratio"] = t_s.str.count(r"[^\w\s]") / eda_s["n_chars"].clip(lower=1)
eda_s["upper_ratio"] = t_s.str.count(r"[A-ZА-Я]") / eda_s["n_chars"].clip(lower=1)
print("\nСредние значения простых признаков по классам (подвыборка до 30 000 текстов):")
display(eda_s.groupby("class")[["n_chars", "n_words", "has_newline", "punct_ratio", "upper_ratio"]].mean().round(4))

# %% [code]
imb = counts.max() / counts.min()
print("ВЫВОД ПО EDA (формируется автоматически по данным):")
print(f"  • Соотношение классов major/minor = {imb:.2f}; миноритарный класс: "
      f"{'AI' if counts[1] < counts[0] else 'Human'} ({share.min():.1f}% данных).")
print("  • Дисбаланс " + ("умеренный/заметный → используем class_weight и macro F1 как основную метрику."
                          if imb > 1.2 else "незначительный."))
print(f"  • Дубликатов: {n_dup_rows}, конфликтов разметки: {n_conflict} → удаляем при предобработке (иначе возможна утечка train→test).")
print(f"  • Пустых/коротких текстов: {n_empty}/{n_short}; пропусков в метках: {n_label_nan}.")
print("  • Если средние длины/доля переносов строк заметно различаются между классами, модель может "
      "опираться на артефакты оформления, а не на «стиль ИИ» (см. таблицу выше).")

# %% [markdown]
# **Вывод по EDA.** Баланс классов, количество дубликатов и пустых текстов зависят от конкретной версии файла и печатаются в ячейке выше. Важные моменты:
# * при **дисбалансе** accuracy вводит в заблуждение, поэтому основной метрикой берём **macro F1** (плюс ROC-AUC) и используем веса классов;
# * **дубликаты** между train и test приводят к завышенным метрикам (утечка), поэтому удаляем их до разбиения; тексты с конфликтующими метками удаляем полностью;
# * различия в **длине** и «оформлении» (переносы строк, пунктуация) между классами — потенциальный источник утечки: модель может научиться отличать источник текстов, а не авторство.

# %% [markdown]
# ## 1.3. Предобработка и разбиение
#
# 1. Удаляем пропуски и тексты короче 10 символов.
# 2. Удаляем дубликаты (и тексты с конфликтующими метками).
# 3. Очищаем текст: нижний регистр, удаление ссылок, замена «лишних» символов пробелом, схлопывание пробелов. **Базовую пунктуацию оставляем** — она несёт стилевой сигнал.
# 4. Для ускорения на CPU берём стратифицированную подвыборку `NLP_SAMPLE_SIZE` (исходный датасет очень большой). `None` — использовать всё.
# 5. Разбиение train/val/test = 70/15/15 со `stratify`.

# %% [code]
NLP_SAMPLE_SIZE = 50_000      # None → использовать весь датасет (долго на CPU)
MIN_CHARS = 10

log = [("исходно", len(df))]
d = df[["text", "label", "class"]].copy()

d = d.dropna(subset=["text"])
d["text"] = d["text"].astype(str)
d = d[d["text"].str.strip().str.len() >= MIN_CHARS]
log.append(("после удаления пустых/коротких (<10 симв.)", len(d)))

d["_h"] = pd.util.hash_pandas_object(d["text"], index=False)
d = d.drop_duplicates(subset=["_h", "label"], keep="first")           # одинаковый текст и метка → оставляем один
log.append(("после удаления точных дубликатов", len(d)))
d = d.drop_duplicates(subset="_h", keep=False)                         # одинаковый текст, разные метки → удаляем оба
log.append(("после удаления конфликтов разметки", len(d)))


def clean_text(s: pd.Series) -> pd.Series:
    s = s.str.lower()
    s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)       # ссылки
    s = s.str.replace(r"[^\w\s.,!?;:'\"()\-]", " ", regex=True)        # лишние символы → пробел
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()             # лишние пробелы/переносы
    return s


d["text"] = clean_text(d["text"])
d = d[d["text"].str.len() >= MIN_CHARS]
log.append(("после очистки текста", len(d)))
d = d.drop(columns=["_h"]).reset_index(drop=True)

if NLP_SAMPLE_SIZE and len(d) > NLP_SAMPLE_SIZE:
    d, _ = train_test_split(d, train_size=NLP_SAMPLE_SIZE, stratify=d["label"], random_state=SEED)
    d = d.reset_index(drop=True)
    log.append((f"стратифицированная подвыборка {NLP_SAMPLE_SIZE:,}", len(d)))

print(pd.DataFrame(log, columns=["шаг", "строк"]).to_string(index=False))

train_df, tmp_df = train_test_split(d, test_size=0.30, stratify=d["label"], random_state=SEED)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=SEED)
train_df, val_df, test_df = [x.reset_index(drop=True) for x in (train_df, val_df, test_df)]

split_info = pd.DataFrame({
    "размер": [len(train_df), len(val_df), len(test_df)],
    "доля AI, %": [round(x["label"].mean() * 100, 2) for x in (train_df, val_df, test_df)],
}, index=["train (70%)", "val (15%)", "test (15%)"])
display(split_info)

y_train, y_val, y_test = train_df["label"].values, val_df["label"].values, test_df["label"].values
print("\nПример очищенного текста:\n", train_df["text"].iloc[0][:300], "...")

# %% [markdown]
# ## 1.4. Вспомогательные функции для метрик и графиков
# Единые функции, чтобы обе модели оценивались одинаково: accuracy, **macro** precision/recall/F1 и ROC-AUC (по вероятности класса «AI»).

# %% [code]
CLASS_LABELS = ["Human", "AI"]


def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision (macro)": precision_score(y_true, y_pred, average="macro"),
        "recall (macro)": recall_score(y_true, y_pred, average="macro"),
        "macro F1": f1_score(y_true, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }


def plot_confusion(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    ax.set_title(title)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинный класс")


nlp_results, nlp_preds, nlp_scores = {}, {}, {}

# %% [markdown]
# ## 1.5. Модель A — baseline: TF-IDF + LogisticRegression
#
# TF-IDF (униграммы и биграммы, sublinear tf) превращает текст в разреженный вектор, логистическая регрессия — быстрая и хорошо интерпретируемая линейная модель.
# Подбираем коэффициент регуляризации `C` **по macro F1 на validation** (test не трогаем), `class_weight="balanced"` компенсирует дисбаланс.

# %% [code]
TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=300_000,
                    sublinear_tf=True, dtype=np.float32)
C_GRID = [0.5, 2, 8, 32]

t0 = time.time()
tfidf = TfidfVectorizer(**TFIDF_PARAMS)
X_train_t = tfidf.fit_transform(train_df["text"])
X_val_t = tfidf.transform(val_df["text"])
X_test_t = tfidf.transform(test_df["text"])
print(f"TF-IDF: матрица train {X_train_t.shape}, время {time.time() - t0:.0f} c")

c_rows, best_lr, best_f1_lr, best_C = [], None, -1, None
for C in C_GRID:
    t1 = time.time()
    clf = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(X_train_t, y_train)
    f1v = f1_score(y_val, clf.predict(X_val_t), average="macro")
    c_rows.append({"C": C, "val macro F1": round(f1v, 4), "время, c": round(time.time() - t1, 1)})
    if f1v > best_f1_lr:
        best_lr, best_f1_lr, best_C = clf, f1v, C
display(pd.DataFrame(c_rows))
print(f"Лучшее C = {best_C} (val macro F1 = {best_f1_lr:.4f})")

lr_proba = best_lr.predict_proba(X_test_t)[:, 1]
lr_pred = best_lr.predict(X_test_t)
nlp_results["TF-IDF + LogReg"] = compute_metrics(y_test, lr_pred, lr_proba)
nlp_preds["TF-IDF + LogReg"], nlp_scores["TF-IDF + LogReg"] = lr_pred, lr_proba
print("\nTEST, TF-IDF + LogReg:")
print(classification_report(y_test, lr_pred, target_names=CLASS_LABELS, digits=4))

# %% [markdown]
# ## 1.6. Модель B — BiGRU на PyTorch
#
# Двунаправленная GRU с `Embedding`, упаковкой последовательностей (`pack_padded_sequence`, чтобы паддинг не искажал состояние) и пулингом mean+max по выходам.
#
# **Зафиксированные гиперпараметры** собраны в словаре `GRU_CFG`. Функция потерь — `CrossEntropyLoss` с весами классов, оптимизатор — `AdamW`, **early stopping по validation macro F1** (лучшие веса восстанавливаются).
# BERT не используем: на CPU его дообучение слишком долгое; GRU позволяет работать на всей подвыборке.

# %% [code]
GRU_CFG = dict(max_len=256, max_vocab=30_000, emb_dim=128, hidden_size=64, num_layers=1,
               dropout=0.3, lr=2e-3, weight_decay=1e-2, batch_size=128, epochs=8, patience=2)
print("Гиперпараметры BiGRU:", GRU_CFG)

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(t):
    return TOKEN_RE.findall(t)[:GRU_CFG["max_len"]]


t0 = time.time()
tok_train = [tokenize(t) for t in train_df["text"]]
tok_val = [tokenize(t) for t in val_df["text"]]
tok_test = [tokenize(t) for t in test_df["text"]]

cnt = Counter()
for toks in tok_train:                       # словарь строим ТОЛЬКО по train (без утечки)
    cnt.update(toks)
PAD, UNK = 0, 1
itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common(GRU_CFG["max_vocab"] - 2) if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}
print(f"Размер словаря: {len(itos)} | токенизация: {time.time() - t0:.0f} c")


def encode(tokens_list):
    L = GRU_CFG["max_len"]
    X = np.zeros((len(tokens_list), L), dtype=np.int64)
    lens = np.ones(len(tokens_list), dtype=np.int64)
    for i, toks in enumerate(tokens_list):
        ids = [stoi.get(w, UNK) for w in toks]
        if ids:
            X[i, :len(ids)] = ids
            lens[i] = len(ids)
    return X, lens


Xtr, Ltr = encode(tok_train)
Xva, Lva = encode(tok_val)
Xte, Lte = encode(tok_test)

g = torch.Generator().manual_seed(SEED)
mk = lambda X, L, y: TensorDataset(torch.from_numpy(X), torch.from_numpy(L), torch.from_numpy(y.astype(np.int64)))
train_loader = DataLoader(mk(Xtr, Ltr, y_train), batch_size=GRU_CFG["batch_size"], shuffle=True, generator=g)
val_loader = DataLoader(mk(Xva, Lva, y_val), batch_size=256)
test_loader = DataLoader(mk(Xte, Lte, y_test), batch_size=256)


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, dropout, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 4, n_classes)            # mean + max по двум направлениям

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != PAD).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


gru_model = BiGRUClassifier(len(itos), GRU_CFG["emb_dim"], GRU_CFG["hidden_size"],
                            GRU_CFG["num_layers"], GRU_CFG["dropout"]).to(device)
print(f"Параметров: {sum(p.numel() for p in gru_model.parameters()):,}")

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
print("Веса классов [Human, AI]:", cw.round(3))
criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
optimizer = torch.optim.AdamW(gru_model.parameters(), lr=GRU_CFG["lr"], weight_decay=GRU_CFG["weight_decay"])

# %% [code]
def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    tot_loss, n, ys, ps, prs = 0.0, 0, [], [], []
    with torch.set_grad_enabled(train_mode):
        for xb, lb, yb in loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item() * len(yb)
            n += len(yb)
            ys.append(yb.cpu().numpy())
            ps.append(logits.argmax(1).cpu().numpy())
            prs.append(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
    y_t, y_p, y_s = np.concatenate(ys), np.concatenate(ps), np.concatenate(prs)
    return tot_loss / n, f1_score(y_t, y_p, average="macro"), y_t, y_p, y_s


history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
best_val_f1, best_state, best_epoch, patience_left = -1.0, None, 0, GRU_CFG["patience"]

t_start = time.time()
for epoch in range(1, GRU_CFG["epochs"] + 1):
    t0 = time.time()
    tr_loss, tr_f1, *_ = run_epoch(gru_model, train_loader, optimizer)
    va_loss, va_f1, *_ = run_epoch(gru_model, val_loader)
    for k, v in zip(history, [tr_loss, va_loss, tr_f1, va_f1]):
        history[k].append(v)
    flag = ""
    if va_f1 > best_val_f1 + 1e-4:
        best_val_f1, best_epoch, patience_left = va_f1, epoch, GRU_CFG["patience"]
        best_state = copy.deepcopy(gru_model.state_dict())
        flag = "  ← лучшая"
    else:
        patience_left -= 1
    print(f"Эпоха {epoch:02d} | train loss {tr_loss:.4f} F1 {tr_f1:.4f} | val loss {va_loss:.4f} F1 {va_f1:.4f} | "
          f"{time.time() - t0:.0f} c{flag}")
    if patience_left <= 0:
        print(f"Early stopping: val macro F1 не улучшался {GRU_CFG['patience']} эпох подряд.")
        break

gru_model.load_state_dict(best_state)
print(f"\nЛучшая эпоха: {best_epoch}, val macro F1 = {best_val_f1:.4f}, общее время обучения {time.time() - t_start:.0f} c")

# %% [code]
_, _, yt, gru_pred, gru_proba = run_epoch(gru_model, test_loader)
assert (yt == y_test).all()
nlp_results["BiGRU (PyTorch)"] = compute_metrics(y_test, gru_pred, gru_proba)
nlp_preds["BiGRU (PyTorch)"], nlp_scores["BiGRU (PyTorch)"] = gru_pred, gru_proba
print("TEST, BiGRU:")
print(classification_report(y_test, gru_pred, target_names=CLASS_LABELS, digits=4))

# %% [markdown]
# ## 1.7. Результаты: кривые обучения, confusion matrix, сравнение моделей

# %% [code]
ep = np.arange(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, history["train_loss"], "o-", label="train loss")
axes[0].plot(ep, history["val_loss"], "s-", label="val loss")
axes[0].axvline(best_epoch, color="gray", ls="--", label=f"лучшая эпоха ({best_epoch})")
axes[0].set_title("BiGRU: loss по эпохам")
axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("CrossEntropyLoss (взвешенный)"); axes[0].legend()
axes[1].plot(ep, history["train_f1"], "o-", label="train macro F1")
axes[1].plot(ep, history["val_f1"], "s-", label="val macro F1")
axes[1].axvline(best_epoch, color="gray", ls="--", label=f"лучшая эпоха ({best_epoch})")
axes[1].set_title("BiGRU: macro F1 по эпохам")
axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("macro F1"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% [code]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, nlp_preds):
    plot_confusion(ax, y_test, nlp_preds[name], f"Confusion matrix (test)\n{name}")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 5))
for name, sc in nlp_scores.items():
    fpr, tpr, _ = roc_curve(y_test, sc)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {nlp_results[name]['ROC-AUC']:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="случайный классификатор")
plt.title("ROC-кривые на test"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

# %% [code]
cmp_df = pd.DataFrame(nlp_results).T.round(4)
print("Сравнение моделей на TEST:")
display(cmp_df)

cmp_long = cmp_df.reset_index().melt(id_vars="index", var_name="метрика", value_name="значение")
plt.figure(figsize=(10, 4.5))
ax = sns.barplot(data=cmp_long, x="метрика", y="значение", hue="index")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
ax.set_ylim(max(0, cmp_df.values.min() - 0.05), 1.01)
ax.set_title("Baseline (TF-IDF + LogReg) vs нейросеть (BiGRU): метрики на test")
ax.set_xlabel("Метрика"); ax.set_ylabel("Значение")
ax.legend(title="Модель", loc="upper left", bbox_to_anchor=(1.01, 1.0))
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 1.8. Что отличает тексты ИИ от человеческих
# Смотрим на самые «весомые» n-граммы линейной модели и на простые стилевые признаки.
# Положительный коэффициент — признак в пользу класса **AI**, отрицательный — в пользу **Human**.

# %% [code]
feat = np.array(tfidf.get_feature_names_out())
coef = best_lr.coef_[0]
top_ai, top_h = np.argsort(coef)[-20:], np.argsort(coef)[:20]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].barh(feat[top_h], coef[top_h], color=COLORS["Human"])
axes[0].set_title("Топ-20 n-грамм в пользу Human"); axes[0].set_xlabel("Коэффициент LogReg")
axes[1].barh(feat[top_ai], coef[top_ai], color=COLORS["AI"])
axes[1].set_title("Топ-20 n-грамм в пользу AI"); axes[1].set_xlabel("Коэффициент LogReg")
plt.tight_layout(); plt.show()


def style_features(texts):
    rows = []
    for t in texts:
        words = t.split()
        n = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        rows.append({
            "ср. длина слова": np.mean([len(w) for w in words]) if words else 0.0,
            "слов в предложении": n / max(len(sents), 1),
            "запятых на 100 слов": 100 * t.count(",") / n,
            "лексич. разнообразие (первые 200 слов)": len(set(words[:200])) / max(min(n, 200), 1),
        })
    return pd.DataFrame(rows)


sf_idx = np.random.RandomState(SEED).choice(len(test_df), size=min(3000, len(test_df)), replace=False)
sf = style_features(test_df["text"].iloc[sf_idx])
sf["class"] = test_df["label"].iloc[sf_idx].map({0: "Human", 1: "AI"}).values
print("Средние стилевые признаки по классам (test, подвыборка):")
display(sf.groupby("class").mean().round(3))

# %% [code]
best_nlp = cmp_df["macro F1"].idxmax()
other_nlp = [m for m in cmp_df.index if m != best_nlp][0]
diff = cmp_df.loc[best_nlp, "macro F1"] - cmp_df.loc[other_nlp, "macro F1"]
print("АВТОМАТИЧЕСКИЙ ИТОГ ЧАСТИ 1")
print(f"  • Лучшая модель по macro F1 на test: {best_nlp} ({cmp_df.loc[best_nlp, 'macro F1']:.4f}); "
      f"преимущество над «{other_nlp}»: {diff:+.4f}.")
print(f"  • ROC-AUC: " + ", ".join(f"{m} = {cmp_df.loc[m, 'ROC-AUC']:.4f}" for m in cmp_df.index) + ".")
if cmp_df["macro F1"].min() > 0.97:
    print("  • Обе модели дают очень высокое качество (macro F1 > 0.97). Это может означать заметные лексические/стилевые "
          "различия между классами, но также артефакты датасета или утечку шаблонов — стоит проверить на текстах из другого источника.")
print(f"  • Обучение BiGRU: лучшая эпоха {best_epoch} из {len(ep)} запущенных; "
      f"данные: {len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test.")

# %% [markdown]
# ### Выводы по Части 1
# * **Какая модель лучше.** Победитель определяется автоматически в ячейке выше (по macro F1 на test). Как правило, на таких данных TF-IDF + LogReg — очень сильный baseline: n-граммы напрямую ловят характерные для генеративных моделей обороты и «шаблонную» лексику, а линейная модель на разреженных признаках обучается быстро и почти не переобучается. BiGRU учитывает порядок слов и может выигрывать на более тонких стилевых закономерностях, но обучается дольше, чувствительна к `max_len`, словарю и гиперпараметрам, а длинные тексты обрезаются.
# * **Какие признаки различают AI/human.** Из графика коэффициентов видно, какие слова и биграммы выдают каждый класс (у ИИ — «гладкие» связующие конструкции и типовые формулировки, у людей — разговорные слова, личные обороты, опечатки, неровная пунктуация). Таблица стилевых признаков показывает средние различия по длине слов/предложений, запятым и лексическому разнообразию.
# * **Ограничения.**
#   1. Используется **подвыборка** (`NLP_SAMPLE_SIZE`) ради скорости на CPU; результаты на полном датасете могут чуть отличаться.
#   2. **Возможная утечка/артефакты датасета:** однотипные темы и промпты, схожие шаблоны генерации, различия в оформлении (переносы строк, длина) между классами. Высокие метрики не гарантируют обобщения на тексты других моделей, доменов и языков.
#   3. Очистка убирает регистр и часть символов, то есть часть стилевого сигнала (регистр, спецсимволы) теряется.
#   4. Один фиксированный seed и одно разбиение: нет оценки разброса (кросс-валидация, несколько seed).
#   5. BERT/DistilBERT не использовался (CPU); fine-tuning трансформера на подвыборке 3–5 тыс. примеров — естественное продолжение.
# * **Идеи улучшения:** кросс-доменная проверка (тексты других LLM), char-n-граммы, перплексия языковой модели как признак, DistilBERT, калибровка вероятностей.

# %% [markdown]
# ---
# # Часть 2. CV: рукописные цифры (MNIST) и символы «+», «−» (HASYv2)
#
# **Цель:** научить сверточную сеть **LeNet-5** распознавать 12 классов: цифры `0–9`, плюс (`+`, класс 10) и минус (`-`, класс 11).
#
# **План:** загрузка MNIST и подмножества HASYv2 (с fallback) → приведение к 28×28 и объединение → стратифицированное разбиение → LeNet-5 → оценка и визуализации → мини-«ассистент» распознавания → выводы.

# %% [markdown]
# ## 2.1. Данные
#
# * **MNIST** — штатно через `tensorflow.keras.datasets.mnist` (70 000 изображений 28×28). Если загрузка невозможна (нет интернета), включается аварийный fallback на `sklearn.datasets.load_digits` (8×8 → 28×28) — **это сильно меньше и хуже оригинала, и будет явно отмечено в выводах**.
# * **HASYv2** — берём только классы `+` и `-` (колонка `latex` в `hasy-data-labels.csv`). Изображения 32×32 приводим к 28×28, делаем штрих белым на чёрном фоне (как в MNIST) и нормализуем.
# * **Fallback для HASYv2:** если датасета нет, генерируем синтетические «+» и «−» (дрожащие линии разной длины/толщины/наклона с шумом). Как подключить настоящий датасет, описано ниже.
#
# **Как подключить HASYv2 вручную:** скачайте и распакуйте https://www.kaggle.com/datasets/guru001/hasyv2 в папку `hasyv2/` рядом с ноутбуком (должен быть файл `hasyv2/hasy-data-labels.csv` и папка `hasy-data/` с PNG) или укажите путь: `os.environ["HASY_DIR"] = "/путь/к/hasyv2"` до запуска ячеек ниже.

# %% [code]
CLASS_NAMES = [str(i) for i in range(10)] + ["+", "-"]       # индексы 0..11
N_CLASSES = len(CLASS_NAMES)


def load_mnist():
    try:
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
        return (np.concatenate([xtr, xte]), np.concatenate([ytr, yte]),
                "tf.keras.datasets.mnist (70 000 изображений)")
    except Exception as e:
        print(f"⚠️ MNIST через Keras недоступен ({type(e).__name__}). Включён FALLBACK: sklearn load_digits (8×8 → 28×28).")
        from sklearn.datasets import load_digits
        dg = load_digits()
        imgs = [np.asarray(Image.fromarray((im / 16 * 255).astype(np.uint8)).resize((28, 28), Image.Resampling.BICUBIC))
                for im in dg.images]
        return np.stack(imgs).astype(np.uint8), dg.target.astype(int), \
            f"FALLBACK sklearn load_digits ({len(imgs)} изображений 8×8 → 28×28)"


def _find_labels_csv(base):
    for pat in ["hasy-data-labels.csv", "*/hasy-data-labels.csv", "*/*/hasy-data-labels.csv", "*/*/*/hasy-data-labels.csv"]:
        hits = glob.glob(os.path.join(base, pat))
        if hits:
            return hits[0]
    return None


def find_hasy_root():
    for b in [os.environ.get("HASY_DIR"), ".", "data", "hasyv2", "HASYv2", "archive", "/kaggle/input/hasyv2"]:
        if b and os.path.isdir(b):
            hit = _find_labels_csv(b)
            if hit:
                return os.path.dirname(os.path.abspath(hit))
    try:
        import kagglehub
        p = kagglehub.dataset_download("guru001/hasyv2")
        hit = _find_labels_csv(p)
        if hit:
            return os.path.dirname(os.path.abspath(hit))
    except Exception as e:
        print(f"HASYv2: локально не найден, kagglehub недоступен ({type(e).__name__}).")
    return None


def load_hasy_plus_minus(root):
    lab = pd.read_csv(os.path.join(root, "hasy-data-labels.csv"), dtype=str, keep_default_na=False)
    lab.columns = [c.strip().lower() for c in lab.columns]
    if "latex" not in lab.columns:                              # на случай другой структуры: берём символы из symbols.csv
        sym = pd.read_csv(os.path.join(root, "symbols.csv"), dtype=str, keep_default_na=False)
        sym.columns = [c.strip().lower() for c in sym.columns]
        lab = lab.merge(sym[["symbol_id", "latex"]], on="symbol_id", how="left")
    lab["latex"] = lab["latex"].str.strip()
    sel = lab[lab["latex"].isin(["+", "-"])]
    print("HASYv2: найдено изображений:", sel["latex"].value_counts().to_dict())
    index = None
    xs, ys = [], []
    for p, sym in zip(sel["path"], sel["latex"]):
        fp = os.path.join(root, p)
        if not os.path.exists(fp):
            if index is None:                                   # строим индекс файлов только при необходимости
                index = {os.path.basename(f): f for f in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)}
            fp = index.get(os.path.basename(p))
            if fp is None:
                continue
        arr = np.asarray(Image.open(fp).convert("L"), dtype=np.float32)
        if arr.mean() > 127:                                    # чёрный символ на белом → инвертируем (как MNIST)
            arr = 255 - arr
        im = Image.fromarray(arr.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / arr.max() * 255                         # растягиваем контраст (тонкие штрихи после ресайза бледнеют)
        xs.append(arr.astype(np.uint8))
        ys.append(10 if sym == "+" else 11)
    return np.stack(xs), np.array(ys)


def make_synthetic_symbols(n_per_class, seed=SEED):
    """Синтетические «+» и «−»: дрожащие штрихи, разная длина/толщина/наклон, шум."""
    rng = np.random.default_rng(seed)
    S = 84                                                      # рисуем в 3× разрешении и уменьшаем до 28×28

    def wobble_line(d, p0, p1, w):
        pts = [(p0[0] + (p1[0] - p0[0]) * t + rng.normal(0, 1.2),
                p0[1] + (p1[1] - p0[1]) * t + rng.normal(0, 1.2)) for t in np.linspace(0, 1, 6)]
        d.line(pts, fill=255, width=w, joint="curve")

    xs, ys = [], []
    for cls_id in (10, 11):
        for _ in range(n_per_class):
            img = Image.new("L", (S, S), 0)
            d = ImageDraw.Draw(img)
            cx, cy = S / 2 + rng.uniform(-6, 6), S / 2 + rng.uniform(-6, 6)
            w = int(rng.integers(5, 11))
            half = rng.uniform(16, 30)
            wobble_line(d, (cx - half, cy), (cx + half, cy), w)
            if cls_id == 10:
                half_v = rng.uniform(16, 30)
                wobble_line(d, (cx, cy - half_v), (cx, cy + half_v), w)
            img = img.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BICUBIC, center=(S / 2, S / 2))
            arr = np.asarray(img.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.float32)
            arr = arr / max(arr.max(), 1) * 255 + rng.normal(0, 4, arr.shape)
            xs.append(np.clip(arr, 0, 255).astype(np.uint8))
            ys.append(cls_id)
    return np.stack(xs), np.array(ys)


x_mnist, y_mnist, MNIST_SOURCE = load_mnist()
print("MNIST:", x_mnist.shape, "| источник:", MNIST_SOURCE)

hasy_root = find_hasy_root()
x_sym, y_sym = (None, None)
if hasy_root:
    print("HASYv2 найден:", hasy_root)
    x_sym, y_sym = load_hasy_plus_minus(hasy_root)
    HASY_SOURCE = "HASYv2 (реальные данные)"
if x_sym is None or len(x_sym) == 0 or len(set(y_sym)) < 2:
    print("⚠️ HASYv2 недоступен → FALLBACK: синтетические «+» и «−».")
    x_sym, y_sym = make_synthetic_symbols(n_per_class=1500)
    HASY_SOURCE = "СИНТЕТИКА (fallback; HASYv2 не найден)"
print("Символы +/-:", x_sym.shape, "| источник:", HASY_SOURCE)

# %% [markdown]
# ### Объединение в единый датасет (12 классов)
# Приводим всё к `float32`, нормализуем в `[0, 1]`, добавляем ось каналов → `(N, 28, 28, 1)`. Метки: `0–9` — цифры, `10` — «+», `11` — «−».

# %% [code]
X = np.concatenate([x_mnist, x_sym]).astype("float32") / 255.0
X = X[..., None]
y = np.concatenate([y_mnist, y_sym]).astype(int)
print("Итоговый датасет:", X.shape, "| диапазон:", X.min(), "-", X.max())

cls_counts = pd.Series(y).value_counts().sort_index()
cls_counts.index = CLASS_NAMES
plt.figure(figsize=(9, 3.8))
bars = sns.barplot(x=cls_counts.index, y=cls_counts.values,
                   hue=["цифра"] * 10 + ["символ"] * 2, palette={"цифра": "#4C78A8", "символ": "#F58518"})
for c in bars.containers:
    bars.bar_label(c, fontsize=8, padding=2)
plt.title("Число изображений по классам (до разбиения)")
plt.xlabel("Класс"); plt.ylabel("Количество"); plt.legend(title="Тип")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(N_CLASSES, 8, figsize=(8, N_CLASSES * 0.95))
rng_vis = np.random.RandomState(SEED)
for k in range(N_CLASSES):
    idx = rng_vis.choice(np.where(y == k)[0], 8, replace=False)
    for j, i in enumerate(idx):
        axes[k, j].imshow(X[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[k, j].axis("off")
    axes[k, 0].set_title(f"«{CLASS_NAMES[k]}»", fontsize=8, loc="left")
plt.suptitle("Примеры изображений по классам", y=1.0)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Разбиение train / val / test = 70 / 15 / 15 (стратифицированное)
# Оригинальные train/test MNIST объединены и разбиты заново, чтобы пропорции классов (включая «+» и «−») одинаково сохранялись во всех частях.
# Так как символов намного меньше, чем цифр, используем **веса классов** при обучении.

# %% [code]
X_train, X_tmp, y_train_cv, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val_cv, y_test_cv = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

split_cv = pd.DataFrame({n: np.bincount(part, minlength=N_CLASSES)
                         for n, part in [("train", y_train_cv), ("val", y_val_cv), ("test", y_test_cv)]},
                        index=CLASS_NAMES).T
display(split_cv)

cw_arr = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_train_cv)
class_weight_cv = {i: float(w) for i, w in enumerate(cw_arr)}
print("Веса классов:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight_cv.items()})

# %% [markdown]
# ## 2.2. Модель LeNet-5 (Keras / TensorFlow)
#
# Архитектура: `Conv(6, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(120) → Dropout → Dense(84) → Dense(12, softmax)`.
# * функция потерь: `sparse_categorical_crossentropy`; оптимизатор: `Adam`;
# * метрики: accuracy (в процессе) и **macro F1** (считаем после каждой эпохи на val и на подвыборке train);
# * `EarlyStopping` и `ReduceLROnPlateau` по `val_loss`;
# * гиперпараметры фиксируем в `CV_CFG`.

# %% [code]
CV_CFG = dict(batch_size=128, epochs=30, lr=1e-3, dropout=0.3,
              es_patience=5, rlrop_factor=0.5, rlrop_patience=2, min_lr=1e-5)
print("Гиперпараметры LeNet-5:", CV_CFG)


def build_lenet5(num_classes=N_CLASSES, dropout=CV_CFG["dropout"], lr=CV_CFG["lr"]):
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=(28, 28, 1)),
        L.Conv2D(6, 5, padding="same", activation="relu"),
        L.MaxPooling2D(2),
        L.Conv2D(16, 5, activation="relu"),
        L.MaxPooling2D(2),
        L.Flatten(),
        L.Dense(120, activation="relu"),
        L.Dropout(dropout),
        L.Dense(84, activation="relu"),
        L.Dense(num_classes, activation="softmax"),
    ], name="LeNet5")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Считает macro F1 на val и на подвыборке train после каждой эпохи."""
    def __init__(self, x_tr, y_tr, x_va, y_va):
        super().__init__()
        self.x_tr, self.y_tr, self.x_va, self.y_va = x_tr, y_tr, x_va, y_va
        self.train_f1, self.val_f1 = [], []

    def on_epoch_end(self, epoch, logs=None):
        p_tr = self.model.predict(self.x_tr, batch_size=512, verbose=0).argmax(1)
        p_va = self.model.predict(self.x_va, batch_size=512, verbose=0).argmax(1)
        self.train_f1.append(f1_score(self.y_tr, p_tr, average="macro"))
        self.val_f1.append(f1_score(self.y_va, p_va, average="macro"))


tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
lenet = build_lenet5()
lenet.summary()

sub = np.random.RandomState(SEED).choice(len(X_train), size=min(10_000, len(X_train)), replace=False)
f1_cb = MacroF1Callback(X_train[sub], y_train_cv[sub], X_val, y_val_cv)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CV_CFG["es_patience"], restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=CV_CFG["rlrop_factor"],
                                         patience=CV_CFG["rlrop_patience"], min_lr=CV_CFG["min_lr"], verbose=1),
    f1_cb,
]

t0 = time.time()
hist = lenet.fit(X_train, y_train_cv, validation_data=(X_val, y_val_cv),
                 epochs=CV_CFG["epochs"], batch_size=CV_CFG["batch_size"],
                 class_weight=class_weight_cv, callbacks=callbacks, verbose=2)
print(f"\nОбучение завершено за {time.time() - t0:.0f} c, эпох: {len(hist.history['loss'])}")

# %% [markdown]
# ## 2.3. Оценка и визуализации

# %% [code]
h = hist.history
ep_cv = np.arange(1, len(h["loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(ep_cv, h["loss"], "o-", label="train loss"); axes[0].plot(ep_cv, h["val_loss"], "s-", label="val loss")
axes[0].set_title("Loss по эпохам"); axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("sparse categorical crossentropy"); axes[0].legend()
axes[1].plot(ep_cv, h["accuracy"], "o-", label="train accuracy"); axes[1].plot(ep_cv, h["val_accuracy"], "s-", label="val accuracy")
axes[1].set_title("Accuracy по эпохам"); axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
axes[2].plot(ep_cv, f1_cb.train_f1, "o-", label="train macro F1 (подвыборка)"); axes[2].plot(ep_cv, f1_cb.val_f1, "s-", label="val macro F1")
axes[2].set_title("Macro F1 по эпохам"); axes[2].set_xlabel("Эпоха"); axes[2].set_ylabel("macro F1"); axes[2].legend()
plt.tight_layout(); plt.show()

# %% [code]
test_loss, test_acc = lenet.evaluate(X_test, y_test_cv, verbose=0)
proba_test = lenet.predict(X_test, batch_size=512, verbose=0)
pred_test = proba_test.argmax(1)
test_macro_f1 = f1_score(y_test_cv, pred_test, average="macro")
print(f"TEST: loss = {test_loss:.4f} | accuracy = {test_acc:.4f} | macro F1 = {test_macro_f1:.4f}\n")
print(classification_report(y_test_cv, pred_test, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, digits=4, zero_division=0))

cm_cv = confusion_matrix(y_test_cv, pred_test, labels=list(range(N_CLASSES)))
plt.figure(figsize=(9, 7))
sns.heatmap(cm_cv, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion matrix LeNet-5 (test, 12 классов)")
plt.xlabel("Предсказано"); plt.ylabel("Истинный класс")
plt.tight_layout(); plt.show()

cm_off = cm_cv.copy(); np.fill_diagonal(cm_off, 0)
pairs = sorted([(CLASS_NAMES[i], CLASS_NAMES[j], int(cm_off[i, j]))
                for i in range(N_CLASSES) for j in range(N_CLASSES) if cm_off[i, j] > 0], key=lambda t: -t[2])[:8]
print("Самые частые ошибки (истина → предсказание: число случаев):")
for a, b, n in pairs:
    print(f"  «{a}» → «{b}»: {n}")
if not pairs:
    print("  ошибок нет")

# %% [code]
# --- Accuracy отдельно по цифрам и по символам ---
is_digit = y_test_cv < 10
acc_digits = float((pred_test[is_digit] == y_test_cv[is_digit]).mean())
acc_symbols = float((pred_test[~is_digit] == y_test_cv[~is_digit]).mean())
acc_plus = float((pred_test[y_test_cv == 10] == 10).mean())
acc_minus = float((pred_test[y_test_cv == 11] == 11).mean())
per_class_acc = cm_cv.diagonal() / cm_cv.sum(1).clip(min=1)

acc_table = pd.DataFrame({"accuracy": [acc_digits, acc_symbols, acc_plus, acc_minus, test_acc],
                          "объектов в test": [int(is_digit.sum()), int((~is_digit).sum()),
                                              int((y_test_cv == 10).sum()), int((y_test_cv == 11).sum()), len(y_test_cv)]},
                         index=["цифры 0–9", "символы (+, −)", "«+»", "«−»", "все классы"])
display(acc_table.round(4))

plt.figure(figsize=(9, 3.8))
ax = sns.barplot(x=CLASS_NAMES, y=per_class_acc, hue=["цифра"] * 10 + ["символ"] * 2,
                 palette={"цифра": "#4C78A8", "символ": "#F58518"})
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
ax.set_ylim(max(0, per_class_acc.min() - 0.05), 1.01)
ax.set_title("Accuracy (recall) по классам на test")
ax.set_xlabel("Класс"); ax.set_ylabel("Accuracy"); ax.legend(title="Тип")
plt.tight_layout(); plt.show()

# %% [code]
def show_examples(idx, title, n=10):
    idx = list(idx)[:n]
    if not idx:
        print(f"{title}: примеров нет")
        return
    cols = min(5, len(idx)); rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X_test[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ok = pred_test[i] == y_test_cv[i]
        ax.set_title(f"истина «{CLASS_NAMES[y_test_cv[i]]}» → «{CLASS_NAMES[pred_test[i]]}»\np = {proba_test[i].max():.2f}",
                     fontsize=8, color="green" if ok else "red")
    plt.suptitle(title); plt.tight_layout(); plt.show()


rng_ex = np.random.RandomState(SEED)
correct_idx = rng_ex.permutation(np.where(pred_test == y_test_cv)[0])
wrong_idx = np.where(pred_test != y_test_cv)[0]
wrong_idx = rng_ex.permutation(wrong_idx)
print(f"Верно: {len(correct_idx)} | ошибок: {len(wrong_idx)} из {len(y_test_cv)}")
show_examples(correct_idx, "Правильные предсказания (случайные примеры)")
show_examples(wrong_idx, "Ошибочные предсказания (случайные примеры)")

# %% [markdown]
# ## 2.4. Мини-«ассистент»: распознавание введённых символов
# Задание говорит о функционале распознавания в «ассистенте». Ниже — функции, которые принимают картинку (путь, `PIL.Image` или массив) и возвращают класс; плюс пример разбора простого выражения из последовательности распознанных цифр и знаков `+` / `−`.

# %% [code]
def preprocess_for_model(img):
    """Путь / PIL / ndarray → float32 (28, 28, 1), белый штрих на чёрном фоне, значения в [0, 1]."""
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(-1)
        if arr.max() <= 1.0:
            arr = arr * 255
    if arr.mean() > 127:                       # чёрный штрих на белом → инвертируем
        arr = 255 - arr
    arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS), dtype=np.float32)
    return (arr / max(arr.max(), 1.0))[..., None]


def recognize(img):
    p = lenet.predict(preprocess_for_model(img)[None], verbose=0)[0]
    k = int(p.argmax())
    return CLASS_NAMES[k], float(p[k])


def evaluate_expression(images):
    tokens = [recognize(im)[0] for im in images]
    expr = "".join(tokens)
    if re.fullmatch(r"-?\d+([+-]\d+)*", expr):
        return expr, sum(int(t) for t in re.findall(r"[+-]?\d+", expr))
    return expr, None


demo_classes = [7, 10, 2, 11, 4]                                  # «7 + 2 - 4»
demo_idx = [int(np.where(y_test_cv == c)[0][0]) for c in demo_classes]
demo_imgs = [X_test[i, :, :, 0] for i in demo_idx]
expr, value = evaluate_expression(demo_imgs)
fig, axes = plt.subplots(1, len(demo_imgs), figsize=(len(demo_imgs) * 1.6, 2))
for ax, im in zip(axes, demo_imgs):
    ax.imshow(im, cmap="gray"); ax.axis("off")
plt.suptitle("Входные изображения для «ассистента»"); plt.tight_layout(); plt.show()
print(f"Распознано: {expr}  →  результат: {value if value is not None else 'не удалось разобрать выражение'} (ожидается 5)")
print("Для своих картинок: recognize('путь/к/файлу.png') → (класс, уверенность)")

# %% [code]
digit_cls = acc_digits >= acc_symbols
print("АВТОМАТИЧЕСКИЙ ИТОГ ЧАСТИ 2")
print(f"  • Источники данных: MNIST — {MNIST_SOURCE}; «+/−» — {HASY_SOURCE}.")
print(f"  • TEST: accuracy = {test_acc:.4f}, macro F1 = {test_macro_f1:.4f}; "
      f"цифры: {acc_digits:.4f}, символы: {acc_symbols:.4f} («+» {acc_plus:.4f}, «−» {acc_minus:.4f}).")
print(f"  • Сеть справляется {'лучше' if digit_cls else 'не хуже'} с {'цифрами' if digit_cls else 'символами'}"
      f"{' (символов существенно меньше, а их штрихи проще/иной природы)' if digit_cls else ''}.")
if pairs:
    print(f"  • Чаще всего путает: «{pairs[0][0]}» → «{pairs[0][1]}» ({pairs[0][2]} шт. в test).")
if "FALLBACK" in MNIST_SOURCE or "СИНТЕТИКА" in HASY_SOURCE:
    print("  • ВНИМАНИЕ: использован fallback (см. источники выше) — числа не сопоставимы с «настоящими» MNIST/HASYv2.")

# %% [markdown]
# ### Выводы по Части 2
# * **Цифры и символы.** LeNet-5 — маленькая сеть, но на MNIST такая архитектура обычно уверенно достигает высокой точности. Точные числа по цифрам и по «+»/«−» вынесены в таблицу и автоматический итог выше.
# * **Где путается.** Типичные ошибки на цифрах — похожие начертания (4↔9, 3↔5↔8, 7↔1); для «+» и «−» возможна путаница с «1», «7» и между собой, если штрих «+» слабо выражен. Конкретные пары ошибок — в списке «самые частые ошибки» и на confusion matrix.
# * **Важная оговорка (domain shift).** Цифры (MNIST) и символы (HASYv2 или синтетика) приходят из **разных источников** с разным стилем (толщина штриха, масштаб, предобработка). Поэтому высокая accuracy на «+»/«−» частично может объясняться тем, что сеть распознаёт «происхождение» картинки, а не только форму знака. На реальных рукописных выражениях качество может быть ниже.
# * **Дисбаланс:** символов намного меньше, чем цифр; поэтому использованы веса классов и macro F1.
# * **Fallback.** Если в ячейке загрузки сработал fallback (MNIST через `load_digits` и/или синтетические «+/−»), это явно напечатано в «Источниках данных»; такие результаты демонстрируют работоспособность пайплайна, но **не** заменяют эксперимент на оригинальных MNIST и HASYv2.
# * **Как улучшить:** аугментация (повороты, сдвиги, масштаб, изменение толщины штриха) через `RandomRotation`/`RandomTranslation`/`RandomZoom`; использовать больше данных HASYv2 (расширить набор знаков: `×`, `÷`, `=`, скобки); более глубокая CNN с BatchNorm; предобработка «как в MNIST» (центрирование по центру масс, единый масштаб штриха); сегментация реальных рукописных выражений на отдельные символы.

# %% [markdown]
# ---
# # Выводы
#
# ## Сводка результатов (формируется автоматически)

# %% [code]
summary = pd.DataFrame([
    {"Часть": "1 · NLP", "Модель": m, "accuracy": r["accuracy"], "macro F1": r["macro F1"], "ROC-AUC": r["ROC-AUC"]}
    for m, r in nlp_results.items()
] + [
    {"Часть": "2 · CV (12 классов)", "Модель": "LeNet-5", "accuracy": test_acc, "macro F1": test_macro_f1, "ROC-AUC": np.nan},
    {"Часть": "2 · CV (только цифры)", "Модель": "LeNet-5", "accuracy": acc_digits, "macro F1": np.nan, "ROC-AUC": np.nan},
    {"Часть": "2 · CV (только «+», «−»)", "Модель": "LeNet-5", "accuracy": acc_symbols, "macro F1": np.nan, "ROC-AUC": np.nan},
])
display(summary.round(4).fillna("—"))
print("Данные Части 2:", MNIST_SOURCE, "|", HASY_SOURCE)
print(f"Данные Части 1: {len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test "
      f"(подвыборка {'включена' if NLP_SAMPLE_SIZE else 'отключена'})")

# %% [markdown]
# ## Итоговые выводы
#
# **Что получилось**
# * **Часть 1.** Построен полный пайплайн: EDA → очистка и дедупликация → стратифицированное разбиение 70/15/15 → baseline TF-IDF + LogisticRegression и нейросеть BiGRU (PyTorch, взвешенный CrossEntropyLoss, AdamW, early stopping по val macro F1). Модели сравнены по accuracy, macro precision/recall/F1 и ROC-AUC, построены confusion matrix, ROC-кривые, кривые обучения и анализ значимых n-грамм.
# * **Часть 2.** Собран единый 12-классовый датасет (цифры 0–9, «+», «−»), обучена LeNet-5 с EarlyStopping и ReduceLROnPlateau; показаны кривые обучения, confusion matrix, classification report, примеры верных и ошибочных предсказаний, раздельная accuracy по цифрам и символам; добавлен мини-«ассистент» (`recognize`, `evaluate_expression`).
#
# **Ограничения**
# * Часть 1: подвыборка датасета ради CPU; возможные артефакты датасета (одинаковые темы/промпты, различия в оформлении) и утечка шаблонов генерации; один seed и одно разбиение; BERT не использовался.
# * Часть 2: цифры и символы из разных источников (domain shift); символов мало по сравнению с цифрами; при отсутствии данных сработали fallback-ветки (источники данных напечатаны выше) — это нужно учитывать при интерпретации чисел.
#
# **Идеи для улучшения**
# * NLP: кросс-доменная проверка на текстах других LLM и из других источников, char-n-граммы, DistilBERT на подвыборке 3–5 тыс. примеров, калибровка вероятностей, кросс-валидация и несколько seed.
# * CV: аугментация, больше классов и данных HASYv2, более глубокая CNN с BatchNorm, единая предобработка штриха, распознавание целых рукописных выражений (сегментация + классификация).
