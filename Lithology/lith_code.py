import os
import pandas as pd

# Указываем пути
excel_file = 'Lithology_Patterns_Presentation.xlsx'  # Путь к Excel-файлу
png_folder = 'patterns'  # Папка с PNG-файлами

# Загружаем таблицу
df = pd.read_excel(excel_file)

# Создаем словарь {название литологии : код}
name_to_code = {}
for _, row in df.iterrows():
    lithology_name = row['Lithology name']
    code = str(row['Code']).zfill(4)  # Делаем код 4-значным (0001, 0002, ...)
    name_to_code[lithology_name] = code

# Переименовываем файлы
for filename in os.listdir(png_folder):
    if filename.endswith('.png'):
        # Удаляем расширение .png
        name_without_ext = os.path.splitext(filename)[0]
        
        # Проверяем, есть ли название в словаре
        if name_without_ext in name_to_code:
            new_filename = f"{name_to_code[name_without_ext]}.png"
            old_path = os.path.join(png_folder, filename)
            new_path = os.path.join(png_folder, new_filename)
            
            # Переименовываем
            os.rename(old_path, new_path)
            print(f"Переименован: {filename} → {new_filename}")
        else:
            print(f"Название '{name_without_ext}' не найдено в таблице. Файл не переименован.")

print("Готово! Все файлы переименованы в формат XXXX.png.")