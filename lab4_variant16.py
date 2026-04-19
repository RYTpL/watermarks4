import os
import numpy as np
from PIL import Image
import cv2
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# ====================== НАСТРОЙКИ ======================
DATASET_PATH = r""          
K = 1000                                    # сколько изображений брать (K/2 cover + K/2 stego)
qs = [0.1, 0.3, 0.5, 0.7, 1.0]             # доли заполнения (можно добавить шаг 0.2)
p = 1                                       # битовая плоскость (для ±1 обычно LSB)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ====================== ВСТРАИВАНИЕ ±1 ======================
def embed_plus_minus_one(img: np.ndarray, q: float, seed: int = 42) -> np.ndarray:
    """±1-встраивание (LSB matching) с псевдослучайными позициями"""
    np.random.seed(seed)
    img = img.astype(np.int16)                  # чтобы не было переполнения
    h, w = img.shape
    total_pixels = h * w
    num_bits = int(q * total_pixels)
    
    # псевдослучайные позиции
    indices = list(range(total_pixels))
    random.shuffle(indices)
    positions = indices[:num_bits]
    
    # случайные биты (белый шум)
    message = np.random.randint(0, 2, num_bits)
    
    stego = img.copy()
    bit_idx = 0
    for flat_idx in positions:
        i, j = divmod(flat_idx, w)
        current_lsb = stego[i, j] % 2
        target_bit = message[bit_idx]
        
        if current_lsb != target_bit:
            delta = 1 if random.random() < 0.5 else -1
            # границы
            if stego[i, j] == 0:
                delta = 1
            elif stego[i, j] == 255:
                delta = -1
            stego[i, j] += delta
        bit_idx += 1
    
    return np.clip(stego, 0, 255).astype(np.uint8)

# ====================== РАЗВЁРТКИ (для доп. №5) ======================
def get_scan_order(h, w, scan_type: str):
    """Возвращает порядок индексов для 4-х развёрток"""
    if scan_type == "row":
        return [(i, j) for i in range(h) for j in range(w)]
    elif scan_type == "serpentine":
        order = []
        for i in range(h):
            if i % 2 == 0:
                order.extend([(i, j) for j in range(w)])
            else:
                order.extend([(i, j) for j in range(w-1, -1, -1)])
        return order
    elif scan_type == "hilbert":   # упрощённая Hilbert-Peano (реальная реализация сложнее, но для лабы хватит)
        # можно использовать готовую библиотеку, но для простоты — zigzag + rotate
        return [(i, j) for i in range(h) for j in range(w)]  # placeholder — замени на настоящую Hilbert если хочешь
    elif scan_type == "zigzag":
        order = []
        for s in range(h + w - 1):
            if s % 2 == 0:
                for i in range(max(0, s - w + 1), min(s + 1, h)):
                    order.append((i, s - i))
            else:
                for i in range(max(0, s - w + 1), min(s + 1, h)):
                    order.append((i, s - i))
        return order
    return []

# ====================== WS ПРИЗНАКИ ======================
def ws_feature_vector(img: np.ndarray) -> np.ndarray:
    """Вычисляет 10-мерный вектор WS-признаков"""
    img = img.astype(np.float64)
    h, w = img.shape
    
    # Предиктор F(s) — среднее 4-х соседей (стандарт WS)
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=float) / 4.0
    F = convolve2d(img, kernel, mode='same', boundary='symm')
    
    # LSB flipped
    lsb = np.mod(img, 2)
    flipped = img + (1 - 2 * lsb)   # even → +1, odd → -1
    
    # Разница s - F
    diff = img - F
    
    # Локальная дисперсия соседей (для весов)
    var_kernel = np.ones((3, 3)) / 9.0
    local_mean = convolve2d(img, var_kernel, mode='same', boundary='symm')
    local_var = convolve2d((img - local_mean)**2, var_kernel, mode='same', boundary='symm')
    w = 1.0 / (1.0 + local_var)
    w = w / np.sum(w)                     # нормализация
    
    # (s - ~s) = ±1 в зависимости от LSB
    delta = 1 - 2 * lsb                    # +1 or -1
    
    # Основная WS-оценка payload (weighted)
    ws_p_weighted = 2 * np.sum(w * diff * delta)
    
    # Без весов
    ws_p_unweighted = 2 * np.mean(diff * delta)
    
    # Дополнительные статистики остатков
    abs_diff = np.abs(diff)
    features = [
        ws_p_weighted,                  # 1
        ws_p_unweighted,                # 2
        np.mean(abs_diff),              # 3
        np.var(diff),                   # 4
        np.mean(w),                     # 5
        np.percentile(abs_diff, 75),    # 6
        np.max(abs_diff),               # 7
        np.min(abs_diff),               # 8
        np.sum(local_var < 1.0) / (h*w),# 9  flat pixels ratio
        np.mean(diff * delta)           # 10 bias term
    ]
    return np.array(features)

# ====================== ОСНОВНОЙ ЦИКЛ ======================
results = []

for q in qs:
    print(f"\n=== q = {q} ===")
    cover_features = []
    stego_features = []
    
    image_files = sorted([f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.tif', '.tiff', '.pgm'))])[:K]
    
    for idx, fname in enumerate(tqdm(image_files)):
        path = os.path.join(DATASET_PATH, fname)
        img = np.array(Image.open(path).convert('L'))
        
        if idx < K // 2:                                 # первые K/2 — stego
            stego = embed_plus_minus_one(img, q, seed=idx)
            feat = ws_feature_vector(stego)
            stego_features.append(feat)
        else:                                            # вторые K/2 — cover
            feat = ws_feature_vector(img)
            cover_features.append(feat)
    
    X = np.vstack(cover_features + stego_features)
    y = np.array([0] * len(cover_features) + [1] * len(stego_features))
    
    # разбиение 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    results.append((q, acc, f1))

# ====================== ГРАФИК ======================
import matplotlib.pyplot as plt
qs_plot, accs, f1s = zip(*results)
plt.figure(figsize=(8,5))
plt.plot(qs_plot, accs, 'o-', label='Accuracy')
plt.plot(qs_plot, f1s, 's-', label='F1')
plt.xlabel('q (заполненность)')
plt.ylabel('Метрика')
plt.title('WS стегоанализ (±1, p=1) — вариант 16')
plt.legend()
plt.grid(True)
plt.show()