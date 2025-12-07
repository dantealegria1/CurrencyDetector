# evaluate_model.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# --- CONFIGURACIÓN ---
IMAGE_SIZE = 224
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = r"C:\9no-Semestre\UX\datasets\usd\USA currency"
MODEL_PATH = os.path.join(BASE_DIR, "usd_model.h5")

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente.")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False
)

print("\nGenerando predicciones...")
predictions = model.predict(val_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
labels = list(val_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
print("\n=== MATRIZ DE CONFUSIÓN ===")
print(cm)

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_true, y_pred, target_names=labels))

print("\nEvaluación completada.")
