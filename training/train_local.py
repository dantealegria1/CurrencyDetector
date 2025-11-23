import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Ruta absoluta al dataset USD ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NUEVA RUTA (CAMBIADA)
base_dir = r"C:\9no-Semestre\UX\datasets\usd\USA currency"

print("Usando dataset en:", base_dir)
print("Existe?", os.path.isdir(base_dir))
print("Subcarpetas:", os.listdir(base_dir))

# --- Configuración ---
IMAGE_SIZE = 224
BATCH_SIZE = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# --- Guardar etiquetas ---
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
labels_path = os.path.join(BASE_DIR, 'labels.txt')

with open(labels_path, 'w') as f:
    f.write(labels)

print("Etiquetas:", train_generator.class_indices)

# --- Modelo base ---
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# --- Ajustar número de clases DINÁMICO ---
num_classes = train_generator.num_classes

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Entrenar ---
history = model.fit(
    train_generator,
    epochs=4,
    validation_data=val_generator
)

# --- Fine tuning ---
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# --- Guardar modelo .h5 ---
output_path = os.path.join(BASE_DIR, "usd_model.h5")
model.save(output_path)

print("Modelo generado en:", output_path)
print("Etiquetas generadas en:", labels_path)
