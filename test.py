import os
from PIL import Image

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
PATH1 = r"E:\wm4\watermarks4\BOWS2"
PATH2 = r"E:\wm4\watermarks4\BOWS2\cover"
PATH3 = r"E:\wm4\watermarks4\BOWS2\cover\cover"
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

for path in [PATH1, PATH2, PATH3]:
    print(f"\nПроверяем путь: {path}")
    if not os.path.exists(path):
        print("   → Папка НЕ существует!")
        continue
    
    files = [f for f in os.listdir(path) if f.lower().endswith(('.pgm', '.tif', '.tiff'))]
    print(f"   → Существует. Найдено .pgm/.tif файлов: {len(files)}")
    if files:
        print(f"   → Первые 5 файлов: {files[:5]}")
        
        # Пробуем открыть первое изображение
        try:
            img_path = os.path.join(path, files[0])
            img = Image.open(img_path).convert('L')
            print(f"   → Успешно открыто первое изображение: {files[0]}  ({img.size})")
        except Exception as e:
            print(f"   → Ошибка при открытии изображения: {e}")