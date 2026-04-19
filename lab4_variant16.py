"""
Лабораторная работа 4 — НЗБ-стегоанализ изображений
Вариант 16:
  p  = 1           (LSB)
  Встраивание: ±1  (LSB matching, псевдослучайные позиции — чётный вариант)
  Метод стегоанализа: WS (Weighted Stego, структурный метод — доп. балл)
  Доп. задание 5: сравнение 4 развёрток (построчная, серпантинная,
                  Гильберта–Пеано, зигзаг)

Запуск:
    python lab4_variant16.py <путь_к_датасету> [K]
    python lab4_variant16.py E:/BOWS2/cover 10000
    python lab4_variant16.py E:/BOWS2/cover 1000   # для краткой версии TIFF

Зависимости:
    pip install numpy pillow scikit-learn scipy matplotlib tqdm
"""

import os
import sys
import random
import warnings
from itertools import product as iproduct

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")          # без GUI — сохраняем в файл
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
#  КОНСТАНТЫ И ПАРАМЕТРЫ
# ════════════════════════════════════════════════════════════

SEED   = 42
QS     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # доли заполнения
SCANS  = ["row", "serpentine", "hilbert", "zigzag"]              # 4 развёртки (доп. №5)

np.random.seed(SEED)
random.seed(SEED)


# ════════════════════════════════════════════════════════════
#  РАЗВЁРТКИ ДВУМЕРНОЙ ОБЛАСТИ  (доп. задание 5)
# ════════════════════════════════════════════════════════════

def scan_row(h: int, w: int) -> list:
    """Построчная развёртка."""
    return [(i, j) for i in range(h) for j in range(w)]


def scan_serpentine(h: int, w: int) -> list:
    """Серпантинная (змейка): чётные строки — слева направо, нечётные — справа налево."""
    order = []
    for i in range(h):
        row = [(i, j) for j in range(w)]
        if i % 2 == 1:
            row = row[::-1]
        order.extend(row)
    return order


def _hilbert_d2xy(n: int, d: int):
    """
    Преобразование d → (x, y) по кривой Гильберта для квадрата n×n (n — степень 2).
    Классический алгоритм из Википедии / Hacker's Delight.
    """
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if (d & 1) ^ rx else 0
        # поворот
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def scan_hilbert(h: int, w: int) -> list:
    n = 1
    while n < max(h, w):
        n <<= 1

    # Генерируем все d сразу через numpy
    d_all = np.arange(n * n, dtype=np.int32)
    x = np.zeros(n * n, dtype=np.int32)
    y = np.zeros(n * n, dtype=np.int32)
    d = d_all.copy()
    s = 1
    while s < n:
        rx = ((d & 2) > 0).astype(np.int32)
        ry = ((d & 1) ^ rx).astype(np.int32)
        # поворот
        mask = ry == 0
        flip = mask & (rx == 1)
        x[flip] = s - 1 - x[flip]
        y[flip] = s - 1 - y[flip]
        swap = mask  # ry==0 → swap x,y
        x[swap], y[swap] = y[swap].copy(), x[swap].copy()
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1

    # Фильтрация в прямоугольник h×w
    valid = (x < h) & (y < w)
    return list(zip(x[valid].tolist(), y[valid].tolist()))


def scan_zigzag(h: int, w: int) -> list:
    """
    Зигзагообразная развёртка (как в JPEG / описании СВИ-14 Cox et al.).
    Диагонали чередуют направление.
    """
    order = []
    for s in range(h + w - 1):
        diag = []
        for i in range(max(0, s - w + 1), min(s + 1, h)):
            j = s - i
            if 0 <= j < w:
                diag.append((i, j))
        if s % 2 == 0:
            diag = diag[::-1]
        order.extend(diag)
    return order


# Кэш развёрток (вычисляем один раз на размер изображения)
_SCAN_CACHE: dict = {}

def get_scan_order(h: int, w: int, scan_type: str) -> list:
    key = (h, w, scan_type)
    if key not in _SCAN_CACHE:
        fn = {"row": scan_row, "serpentine": scan_serpentine,
              "hilbert": scan_hilbert, "zigzag": scan_zigzag}[scan_type]
        _SCAN_CACHE[key] = fn(h, w)
    return _SCAN_CACHE[key]


# ════════════════════════════════════════════════════════════
#  ±1-ВСТРАИВАНИЕ  (вариант 16 — чётный → псевдослучайные позиции)
# ════════════════════════════════════════════════════════════

def embed_pm1(img: np.ndarray, q: float, seed: int) -> np.ndarray:
    """
    ±1 LSB Matching.
    Чётный вариант → позиции выбираются псевдослучайно.
    Доля q от всех пикселей изменяется.
    """
    rng  = np.random.default_rng(seed)
    flat = img.astype(np.int16).ravel()
    n    = len(flat)
    k    = int(q * n)

    # псевдослучайные позиции без повторений
    positions = rng.choice(n, size=k, replace=False)
    message   = rng.integers(0, 2, size=k)

    for pos, bit in zip(positions, message):
        if flat[pos] % 2 != bit:
            delta = 1 if rng.random() < 0.5 else -1
            if flat[pos] == 0:
                delta = 1
            elif flat[pos] == 255:
                delta = -1
            flat[pos] += delta

    return np.clip(flat, 0, 255).astype(np.uint8).reshape(img.shape)


# ════════════════════════════════════════════════════════════
#  WS-СТЕГОАНАЛИЗ
#
#  Weighted Stego (WS) — структурный метод.
#  Идея: для cover-изображения средневзвешенная разность между
#  пикселем и его предсказанием (усреднение соседей) близка к нулю.
#  ±1-встраивание нарушает эту симметрию.
#
#  Мы вычисляем WS-оценку по развёртке (для доп. задания 5),
#  а затем формируем многомерный вектор признаков для классификатора.
# ════════════════════════════════════════════════════════════

def ws_feature_vector(img: np.ndarray, scan_type: str = "row") -> np.ndarray:
    img_f = img.astype(np.float64)
    h, w  = img_f.shape

    # Предиктор: среднее 4 соседей
    kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=np.float64) / 4.0
    F = convolve2d(img_f, kernel, mode="same", boundary="symm")

    lsb    = img_f % 2
    delta  = 1.0 - 2.0 * lsb
    diff   = img_f - F
    abs_diff = np.abs(diff)

    # Веса (локальная дисперсия)
    vk       = np.ones((3, 3), dtype=np.float64) / 9.0
    loc_mean = convolve2d(img_f, vk, mode="same", boundary="symm")
    loc_var  = convolve2d((img_f - loc_mean) ** 2, vk, mode="same", boundary="symm")
    weights  = 1.0 / (1.0 + loc_var + 1e-8)
    weights /= weights.sum()

    # A. Глобальные WS-оценки
    ws_w  = float(2.0 * np.sum(weights * diff * delta))
    ws_uw = float(2.0 * np.mean(diff * delta))

    # B. Статистики невязки
    feat_B = [
        float(np.mean(abs_diff)),
        float(np.std(diff)),
        float(np.percentile(abs_diff, 25)),
        float(np.percentile(abs_diff, 75)),
        float(np.max(abs_diff)),
        float(np.sum(loc_var < 1.0)) / (h * w),
        float(np.mean(weights)),
    ]

    # C. Признаки по развёртке — ВЕКТОРИЗОВАННО
    order   = get_scan_order(h, w, scan_type)
    # Строим индексный массив один раз и берём сразу всё
    rows    = np.array([r for r, c in order], dtype=np.int32)
    cols    = np.array([c for r, c in order], dtype=np.int32)
    lsb_seq = img[rows, cols] % 2          # векторная индексация — мгновенно

    transitions = float(np.sum(lsb_seq[:-1] != lsb_seq[1:]))
    trans_rate  = transitions / max(len(lsb_seq) - 1, 1)

    # Длины серий через np.diff — без Python-цикла
    changes   = np.flatnonzero(np.diff(lsb_seq.astype(np.int8))) + 1
    run_ends  = np.concatenate([[0], changes, [len(lsb_seq)]])
    rl_arr    = np.diff(run_ends).astype(np.float64)
    mean_run  = float(rl_arr.mean())
    std_run   = float(rl_arr.std())

    feat_C = [trans_rate, mean_run, std_run, transitions / len(lsb_seq)]

    return np.array([ws_w, ws_uw] + feat_B + feat_C, dtype=np.float64)

# ════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ДАТАСЕТА
# ════════════════════════════════════════════════════════════

def load_dataset(dataset_path: str, K: int) -> list:
    """
    Загружает до K grayscale-изображений из папки.
    Поддерживает .pgm, .tif, .tiff, .png, .bmp.
    """
    exts  = {".pgm", ".tif", ".tiff", ".png", ".bmp"}
    files = sorted([
        f for f in os.listdir(dataset_path)
        if os.path.splitext(f)[1].lower() in exts
    ])[:K]

    if len(files) == 0:
        raise FileNotFoundError(f"В папке '{dataset_path}' не найдено изображений.")

    images = []
    for fname in files:
        path = os.path.join(dataset_path, fname)
        img  = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        images.append(img)

    print(f"  Загружено {len(images)} изображений, размер: {images[0].shape}")
    return images


# ════════════════════════════════════════════════════════════
#  ЭКСПЕРИМЕНТ: один прогон (q, scan_type)
# ════════════════════════════════════════════════════════════

def run_experiment(images: list, q: float, scan_type: str) -> dict:
    """
    Формирует выборку cover/stego, считает признаки,
    обучает RandomForest, возвращает метрики.
    """
    K     = len(images)
    half  = K // 2

    # Первые half — stego, вторые half — cover
    X_list, y_list = [], []

    for idx, img in enumerate(images):
        if idx < half:
            stego = embed_pm1(img, q, seed=idx)          # ±1-встраивание
            feat  = ws_feature_vector(stego, scan_type)
            label = 1                                     # stego
        else:
            feat  = ws_feature_vector(img, scan_type)
            label = 0                                     # cover

        X_list.append(feat)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # train 70 % / test 30 %
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        random_state=SEED, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, zero_division=0)
    return {"acc": acc, "f1": f1}


# ════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ════════════════════════════════════════════════════════════

DARK_BG  = "#0a0a14"
PANEL_BG = "#13132a"
COLORS   = {
    "row":        "#4fc3f7",
    "serpentine": "#a5d6a7",
    "hilbert":    "#ce93d8",
    "zigzag":     "#ff8a65",
}
SCAN_LABELS = {
    "row":        "Построчная",
    "serpentine": "Серпантинная",
    "hilbert":    "Гильберта–Пеано",
    "zigzag":     "Зигзаг",
}


def _style(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="#dde0ff", fontsize=10, pad=6)
    ax.tick_params(colors="#888", labelsize=8)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#ccc")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(alpha=0.15, color="#555", linewidth=0.7)


def save_plots(results: dict, qs: list, output_dir: str):
    """
    results[scan_type][q] = {"acc": float, "f1": float}
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Accuracy по всем развёрткам ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for sc in SCANS:
        accs = [results[sc][q]["acc"] for q in qs]
        ax1.plot(qs, accs, color=COLORS[sc], lw=2,
                 marker="o", ms=5, label=SCAN_LABELS[sc])
    ax1.axhline(0.5, color="#555", ls="--", lw=1)
    ax1.set_xlabel("q (заполненность)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.45, 1.02)
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor="white")
    _style(ax1, "Accuracy(q) — сравнение развёрток")

    # ── 2. F1 по всем развёрткам ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for sc in SCANS:
        f1s = [results[sc][q]["f1"] for q in qs]
        ax2.plot(qs, f1s, color=COLORS[sc], lw=2,
                 marker="s", ms=5, ls="--", label=SCAN_LABELS[sc])
    ax2.axhline(0.5, color="#555", ls="--", lw=1)
    ax2.set_xlabel("q (заполненность)")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor="white")
    _style(ax2, "F1(q) — сравнение развёрток")

    # ── 3. Тепловая карта Accuracy ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    heat_acc = np.array([[results[sc][q]["acc"] for q in qs] for sc in SCANS])
    im3 = ax3.imshow(heat_acc, aspect="auto", cmap="plasma",
                     vmin=0.5, vmax=1.0)
    ax3.set_xticks(range(len(qs)))
    ax3.set_xticklabels([f"{q:.1f}" for q in qs], fontsize=7, color="#888")
    ax3.set_yticks(range(len(SCANS)))
    ax3.set_yticklabels([SCAN_LABELS[sc] for sc in SCANS], fontsize=8, color="#ccc")
    for i, sc in enumerate(SCANS):
        for j, q in enumerate(qs):
            val = results[sc][q]["acc"]
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if val < 0.85 else "black")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#888")
    ax3.set_xlabel("q")
    _style(ax3, "Тепловая карта Accuracy")

    # ── 4. Тепловая карта F1 ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    heat_f1 = np.array([[results[sc][q]["f1"] for q in qs] for sc in SCANS])
    im4 = ax4.imshow(heat_f1, aspect="auto", cmap="plasma",
                     vmin=0.5, vmax=1.0)
    ax4.set_xticks(range(len(qs)))
    ax4.set_xticklabels([f"{q:.1f}" for q in qs], fontsize=7, color="#888")
    ax4.set_yticks(range(len(SCANS)))
    ax4.set_yticklabels([SCAN_LABELS[sc] for sc in SCANS], fontsize=8, color="#ccc")
    for i, sc in enumerate(SCANS):
        for j, q in enumerate(qs):
            val = results[sc][q]["f1"]
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if val < 0.85 else "black")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#888")
    ax4.set_xlabel("q")
    _style(ax4, "Тепловая карта F1")

    fig.suptitle(
        "ЛР 4 — Вариант 16: WS-стегоанализ  ·  ±1-встраивание  ·  LSB (p=1)\n"
        "Доп. задание 5: сравнение развёрток (построчная / серпантин / Гильберт / зигзаг)",
        color="#dde0ff", fontsize=11, fontweight="bold", y=0.995
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "lab4_results.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[График] Сохранён: {path}")


# ════════════════════════════════════════════════════════════
#  ИТОГОВАЯ ТАБЛИЦА В КОНСОЛЬ
# ════════════════════════════════════════════════════════════

def print_table(results: dict, qs: list):
    print("\n" + "=" * 72)
    print("  ИТОГОВАЯ ТАБЛИЦА  (Accuracy / F1)")
    print("=" * 72)
    header = f"  {'Развёртка':20s}" + "".join(f"  q={q:.1f}" for q in qs)
    print(header)
    print("-" * 72)
    for sc in SCANS:
        row_acc = f"  {SCAN_LABELS[sc]:20s}" + \
                  "".join(f"  {results[sc][q]['acc']:.3f}" for q in qs)
        row_f1  = f"  {'':20s}" + \
                  "".join(f"  {results[sc][q]['f1']:.3f}" for q in qs)
        print(row_acc)
        print(row_f1 + "  ← F1")
        print()
    print("=" * 72)


# ════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Использование: python lab4_variant16.py <датасет> [K] [output_dir]")
        print("Пример:         python lab4_variant16.py E:/BOWS2/cover 10000 lab4_out")
        sys.exit(1)

    dataset_path = sys.argv[1]
    K            = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    output_dir   = sys.argv[3] if len(sys.argv) > 3 else "lab4_output"

    print("=" * 65)
    print("  ЛР 4 — Вариант 16: WS-стегоанализ")
    print(f"  Датасет : {dataset_path}")
    print(f"  K       : {K}")
    print(f"  Развёртки: {', '.join(SCANS)}")
    print(f"  q       : {QS}")
    print("=" * 65)

    print("\n[1] Загрузка изображений...")
    images = load_dataset(dataset_path, K)
    K_real = len(images)
    if K_real < 10:
        print("  ОШИБКА: слишком мало изображений (нужно хотя бы 10).")
        sys.exit(1)

    # ── Основной эксперимент ─────────────────────────────
    results = {sc: {} for sc in SCANS}

    for sc in SCANS:
        print(f"\n[2] Развёртка: {SCAN_LABELS[sc]}")
        for q in tqdm(QS, desc=f"  q-sweep [{sc}]"):
            metrics = run_experiment(images, q, sc)
            results[sc][q] = metrics
            tqdm.write(f"    q={q:.1f}  Acc={metrics['acc']:.4f}  F1={metrics['f1']:.4f}")

    # ── Вывод результатов ────────────────────────────────
    print_table(results, QS)
    save_plots(results, QS, output_dir)

    print(f"\n✓ Готово. Результаты в: {output_dir}/")


if __name__ == "__main__":
    main()