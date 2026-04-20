"""
Лабораторная работа 4 — НЗБ-стегоанализ изображений
Вариант 16:
  p  = 1           (LSB)
  Встраивание: ±1  (LSB matching, псевдослучайные позиции — чётный вариант)
  Метод стегоанализа: WS (Weighted Stego, структурный метод — доп. балл)
  Доп. задание 5: сравнение 4 развёрток (построчная, серпантинная,
                  Гильберта–Пеано, зигзаг)

Запуск:
    python lab4_variant16.py E:/BOWS2/cover/cover 10000 lab4_output

Зависимости:
    pip install numpy pillow scikit-learn scipy matplotlib tqdm joblib
"""

import os
import sys
import warnings

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
#  КОНСТАНТЫ
# ════════════════════════════════════════════════════════════

SEED  = 42
QS    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SCANS = ["row", "serpentine", "hilbert", "zigzag"]

np.random.seed(SEED)

# Предиктор WS: среднее 4 соседей (без центра)
_WS_KERNEL = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=np.float64) / 4.0


# ════════════════════════════════════════════════════════════
#  КЭШ РАЗВЁРТОК
#  Индексы строятся один раз на размер изображения и хранятся
#  как numpy-массивы — векторная индексация без Python-цикла.
# ════════════════════════════════════════════════════════════

_SCAN_CACHE: dict = {}


def _build_row(h, w):
    rows = np.repeat(np.arange(h, dtype=np.int32), w)
    cols = np.tile(np.arange(w, dtype=np.int32), h)
    return rows, cols


def _build_serpentine(h, w):
    rows_list, cols_list = [], []
    for i in range(h):
        r = np.full(w, i, dtype=np.int32)
        c = np.arange(w, dtype=np.int32)
        if i % 2 == 1:
            c = c[::-1]
        rows_list.append(r)
        cols_list.append(c)
    return np.concatenate(rows_list), np.concatenate(cols_list)


def _build_hilbert(h, w):
    """Кривая Гильберта–Пеано, полностью векторизованная."""
    n = 1
    while n < max(h, w):
        n <<= 1

    total = n * n
    x     = np.zeros(total, dtype=np.int64)
    y     = np.zeros(total, dtype=np.int64)
    tmp_d = np.arange(total, dtype=np.int64)
    s = 1
    while s < n:
        rx = ((tmp_d & 2) > 0).astype(np.int64)
        ry = ((tmp_d & 1) ^ rx).astype(np.int64)
        mask_swap     = ry == 0
        flip          = mask_swap & (rx == 1)
        x[flip]       = s - 1 - x[flip]
        y[flip]       = s - 1 - y[flip]
        x[mask_swap], y[mask_swap] = y[mask_swap].copy(), x[mask_swap].copy()
        x += s * rx
        y += s * ry
        tmp_d >>= 2
        s <<= 1

    valid = (x < h) & (y < w)
    return x[valid].astype(np.int32), y[valid].astype(np.int32)


def _build_zigzag(h, w):
    rows_list, cols_list = [], []
    for s in range(h + w - 1):
        i_min  = max(0, s - w + 1)
        i_max  = min(s + 1, h)
        diag_r = np.arange(i_min, i_max, dtype=np.int32)
        diag_c = (s - diag_r).astype(np.int32)
        if s % 2 == 0:
            diag_r = diag_r[::-1]
            diag_c = diag_c[::-1]
        rows_list.append(diag_r)
        cols_list.append(diag_c)
    return np.concatenate(rows_list), np.concatenate(cols_list)


_BUILDERS = {
    "row":        _build_row,
    "serpentine": _build_serpentine,
    "hilbert":    _build_hilbert,
    "zigzag":     _build_zigzag,
}


def get_scan_indices(h: int, w: int, scan_type: str):
    """
    Возвращает (rows, cols) — numpy int32-массивы порядка обхода.
    Результат кэшируется: для 10000 изображений одного размера
    строится ровно один раз.
    """
    key = (h, w, scan_type)
    if key not in _SCAN_CACHE:
        _SCAN_CACHE[key] = _BUILDERS[scan_type](h, w)
    return _SCAN_CACHE[key]


# ════════════════════════════════════════════════════════════
#  ±1-ВСТРАИВАНИЕ
# ════════════════════════════════════════════════════════════

def embed_pm1(img: np.ndarray, q: float, seed: int) -> np.ndarray:
    """
    ±1 LSB Matching. Чётный вариант → псевдослучайные позиции.
    Каждый вызов изолирован через default_rng(seed) — thread-safe.
    """
    rng  = np.random.default_rng(seed)
    flat = img.astype(np.int16).ravel()
    n    = len(flat)
    k    = int(q * n)

    positions = rng.choice(n, size=k, replace=False)
    message   = rng.integers(0, 2, size=k)
    deltas    = rng.integers(0, 2, size=k) * 2 - 1   # случайные ±1

    for pos, bit, delta in zip(positions, message, deltas):
        if flat[pos] % 2 != bit:
            if flat[pos] == 0:
                delta = 1
            elif flat[pos] == 255:
                delta = -1
            flat[pos] += delta

    return np.clip(flat, 0, 255).astype(np.uint8).reshape(img.shape)


# ════════════════════════════════════════════════════════════
#  WS-ПРИЗНАКИ  (вертикальная оптимизация)
#
#  Что ускорено по сравнению с наивной версией:
#  1. convolve2d → scipy.ndimage.convolve  (быстрее для малых ядер)
#  2. convolve2d для скользящего среднего → uniform_filter
#     (сепарабельный: O(N·M) вместо O(N·M²))
#  3. rows/cols берутся из кэша — не пересоздаются на каждом изображении
#  4. LSB-последовательность — векторная индексация img[rows, cols]
#  5. Длины серий — через np.diff, без Python-цикла
# ════════════════════════════════════════════════════════════

def ws_feature_vector(img: np.ndarray, scan_type: str) -> np.ndarray:
    """
    13-мерный вектор WS-признаков.

    A (2): взвешенная и невзвешенная WS-оценка
    B (7): статистики невязки (img - предиктор)
    C (4): переходы и серии LSB вдоль развёртки scan_type
    """
    img_f = img.astype(np.float64)
    h, w  = img_f.shape

    # ── Предиктор F: среднее 4 соседей ────────────────────
    F = convolve(img_f, _WS_KERNEL, mode="mirror")

    lsb      = img_f % 2
    delta_pm = 1.0 - 2.0 * lsb        # +1 если LSB=0, -1 если LSB=1
    diff     = img_f - F
    abs_diff = np.abs(diff)

    # ── Локальная дисперсия → веса ─────────────────────────
    # uniform_filter реализован как два 1D-прохода (сепарабельный),
    # что в 5-10x быстрее полной 2D-свёртки
    loc_mean = uniform_filter(img_f, size=3, mode="mirror")
    loc_var  = uniform_filter((img_f - loc_mean) ** 2, size=3, mode="mirror")
    weights  = 1.0 / (1.0 + loc_var + 1e-8)
    weights /= weights.sum()

    # ── A. WS-оценки ───────────────────────────────────────
    ws_w  = float(2.0 * np.sum(weights * diff * delta_pm))
    ws_uw = float(2.0 * np.mean(diff * delta_pm))

    # ── B. Статистики невязки ──────────────────────────────
    feat_B = [
        float(np.mean(abs_diff)),
        float(np.std(diff)),
        float(np.percentile(abs_diff, 25)),
        float(np.percentile(abs_diff, 75)),
        float(np.max(abs_diff)),
        float(np.sum(loc_var < 1.0)) / (h * w),
        float(np.mean(weights)),
    ]

    # ── C. Признаки по развёртке ───────────────────────────
    rows, cols = get_scan_indices(h, w, scan_type)  # из кэша
    lsb_seq    = img[rows, cols] % 2                # векторная индексация

    n_seq       = len(lsb_seq)
    transitions = float(np.sum(lsb_seq[:-1] != lsb_seq[1:]))
    trans_rate  = transitions / max(n_seq - 1, 1)

    # Длины серий через np.diff — O(N), без Python-цикла
    changes  = np.flatnonzero(np.diff(lsb_seq.astype(np.int8))) + 1
    run_ends = np.concatenate([[0], changes, [n_seq]])
    rl_arr   = np.diff(run_ends).astype(np.float64)

    feat_C = [
        trans_rate,
        float(rl_arr.mean()),
        float(rl_arr.std()),
        transitions / n_seq,
    ]

    return np.array([ws_w, ws_uw] + feat_B + feat_C, dtype=np.float64)


# ════════════════════════════════════════════════════════════
#  ОБРАБОТКА ОДНОГО ИЗОБРАЖЕНИЯ  (вызывается параллельно)
# ════════════════════════════════════════════════════════════

def process_image(img: np.ndarray, idx: int, half: int,
                  q: float, scan_type: str) -> tuple:
    """
    Встраивание (если stego) + вычисление признаков.
    Функция stateless и thread-safe:
      - embed_pm1 использует изолированный default_rng(seed=idx)
      - ws_feature_vector читает только кэш (без записи)
    """
    if idx < half:
        img   = embed_pm1(img, q, seed=idx)
        label = 1
    else:
        label = 0
    return ws_feature_vector(img, scan_type), label


# ════════════════════════════════════════════════════════════
#  ЭКСПЕРИМЕНТ: один прогон (q, scan_type)
# ════════════════════════════════════════════════════════════

def run_experiment(images: list, q: float, scan_type: str) -> dict:
    """
    Горизонтальная оптимизация: все изображения обрабатываются
    параллельно через joblib Threads.

    prefer="threads":
      numpy и scipy отпускают GIL во время вычислений,
      поэтому треды эффективнее процессов — нет накладных расходов
      на сериализацию и копирование данных между процессами.
    """
    half = len(images) // 2

    out = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_image)(img, idx, half, q, scan_type)
        for idx, img in enumerate(images)
    )

    X = np.vstack([f for f, _ in out])
    y = np.array([lb for _, lb in out])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        random_state=SEED, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    return {
        "acc": float(accuracy_score(y_te, y_pred)),
        "f1":  float(f1_score(y_te, y_pred, zero_division=0)),
    }


# ════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ДАТАСЕТА
# ════════════════════════════════════════════════════════════

def load_dataset(dataset_path: str, K: int) -> list:
    exts  = {".pgm", ".tif", ".tiff", ".png", ".bmp"}
    files = sorted([
        f for f in os.listdir(dataset_path)
        if os.path.splitext(f)[1].lower() in exts
    ])[:K]

    if not files:
        raise FileNotFoundError(f"В папке '{dataset_path}' не найдено изображений.")

    images = []
    for fname in tqdm(files, desc="  Загрузка"):
        path = os.path.join(dataset_path, fname)
        img  = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        images.append(img)

    print(f"  Загружено: {len(images)} изображений, размер: {images[0].shape}")
    return images


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
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Accuracy(q)
    ax1 = fig.add_subplot(gs[0, 0])
    for sc in SCANS:
        accs = [results[sc][q]["acc"] for q in qs]
        ax1.plot(qs, accs, color=COLORS[sc], lw=2,
                 marker="o", ms=5, label=SCAN_LABELS[sc])
    ax1.axhline(0.5, color="#555", ls="--", lw=1, label="случайный классификатор")
    ax1.set_xlabel("q (заполненность)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.45, 1.02)
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor="white")
    _style(ax1, "Accuracy(q) — сравнение развёрток")

    # 2. F1(q)
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

    # 3. Тепловая карта Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    heat_acc = np.array([[results[sc][q]["acc"] for q in qs] for sc in SCANS])
    im3 = ax3.imshow(heat_acc, aspect="auto", cmap="plasma", vmin=0.5, vmax=1.0)
    ax3.set_xticks(range(len(qs)))
    ax3.set_xticklabels([f"{q:.1f}" for q in qs], fontsize=7, color="#888")
    ax3.set_yticks(range(len(SCANS)))
    ax3.set_yticklabels([SCAN_LABELS[sc] for sc in SCANS], fontsize=8, color="#ccc")
    for i, sc in enumerate(SCANS):
        for j, q in enumerate(qs):
            val = results[sc][q]["acc"]
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if val < 0.85 else "black")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xlabel("q")
    _style(ax3, "Тепловая карта Accuracy")

    # 4. Тепловая карта F1
    ax4 = fig.add_subplot(gs[1, 1])
    heat_f1 = np.array([[results[sc][q]["f1"] for q in qs] for sc in SCANS])
    im4 = ax4.imshow(heat_f1, aspect="auto", cmap="plasma", vmin=0.5, vmax=1.0)
    ax4.set_xticks(range(len(qs)))
    ax4.set_xticklabels([f"{q:.1f}" for q in qs], fontsize=7, color="#888")
    ax4.set_yticks(range(len(SCANS)))
    ax4.set_yticklabels([SCAN_LABELS[sc] for sc in SCANS], fontsize=8, color="#ccc")
    for i, sc in enumerate(SCANS):
        for j, q in enumerate(qs):
            val = results[sc][q]["f1"]
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if val < 0.85 else "black")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
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
#  ТАБЛИЦА В КОНСОЛЬ
# ════════════════════════════════════════════════════════════

def print_table(results: dict, qs: list):
    col_w = 7
    print("\n" + "=" * (24 + col_w * len(qs)))
    print("  ИТОГОВАЯ ТАБЛИЦА  (Accuracy / F1)")
    print("=" * (24 + col_w * len(qs)))
    header = f"  {'Развёртка':22s}" + "".join(f" q={q:.1f}" for q in qs)
    print(header)
    print("-" * (24 + col_w * len(qs)))
    for sc in SCANS:
        row_acc = f"  {SCAN_LABELS[sc] + ' Acc':22s}" + \
                  "".join(f"  {results[sc][q]['acc']:.3f}" for q in qs)
        row_f1  = f"  {SCAN_LABELS[sc] + ' F1':22s}" + \
                  "".join(f"  {results[sc][q]['f1']:.3f}" for q in qs)
        print(row_acc)
        print(row_f1)
        print()
    print("=" * (24 + col_w * len(qs)))


# ════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Использование: python lab4_variant16.py <датасет> [K] [output_dir]")
        print("Пример:  python lab4_variant16.py E:/BOWS2/cover/cover 10000 lab4_output")
        sys.exit(1)

    dataset_path = sys.argv[1]
    K            = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    output_dir   = sys.argv[3] if len(sys.argv) > 3 else "lab4_output"

    print("=" * 65)
    print("  ЛР 4 — Вариант 16: WS-стегоанализ (оптимизированная версия)")
    print(f"  Датасет  : {dataset_path}")
    print(f"  K        : {K}")
    print(f"  Развёртки: {', '.join(SCANS)}")
    print(f"  q        : {QS}")
    print("=" * 65)

    # ── Прогрев кэша развёрток ───────────────────────────
    # Строим индексы до старта параллельных потоков,
    # чтобы исключить гонку при первом обращении из разных тредов.
    print("\n[0] Прогрев кэша развёрток (256×256)...")
    for sc in SCANS:
        get_scan_indices(256, 256, sc)
        print(f"  {SCAN_LABELS[sc]:22s} — готово")

    # ── Загрузка ─────────────────────────────────────────
    print("\n[1] Загрузка изображений...")
    images = load_dataset(dataset_path, K)

    # Если размер изображений отличается от 256×256 — перестраиваем кэш
    h0, w0 = images[0].shape
    if (h0, w0) != (256, 256):
        print(f"  Размер {h0}×{w0} — перестраиваем кэш...")
        _SCAN_CACHE.clear()
        for sc in SCANS:
            get_scan_indices(h0, w0, sc)

    # ── Основной эксперимент ─────────────────────────────
    results = {sc: {} for sc in SCANS}

    for sc in SCANS:
        print(f"\n[2] Развёртка: {SCAN_LABELS[sc]}")
        for q in tqdm(QS, desc="  q-sweep"):
            metrics        = run_experiment(images, q, sc)
            results[sc][q] = metrics
            tqdm.write(
                f"    q={q:.1f}  Acc={metrics['acc']:.4f}  F1={metrics['f1']:.4f}"
            )

    print_table(results, QS)
    save_plots(results, QS, output_dir)
    print(f"\n✓ Готово. Результаты в: {output_dir}/")


if __name__ == "__main__":
    main()
