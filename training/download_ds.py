import kagglehub
import shutil
import os

# 1. Descargar dataset
source_path = kagglehub.dataset_download("aishwaryatechie/usd-bill-classification-dataset")
print("Dataset descargado en:", source_path)

# 2. Carpeta destino (CAMBIA ESTA RUTA)
dest_path = r"C:\9no-Semestre\UX\datasets\usd"

# 3. Si existe, no duplicamos archivos
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

# 4. Copiar todo el dataset
shutil.copytree(source_path, dest_path, dirs_exist_ok=True)

print("✔ Dataset copiado a:", dest_path)
