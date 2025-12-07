import tensorflow as tf
import os

# --- Ruta del dataset ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = r"C:\9no-Semestre\UX\datasets\usd\USA currency"

# --- Configuración ---
IMAGE_SIZE = 224
BATCH_SIZE = 32

# --- Data Augmentation ---
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.3, 1.5],
    shear_range=0.2,
    fill_mode="nearest"
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

labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open(os.path.join(BASE_DIR, 'labels.txt'), 'w') as f:
    f.write(labels)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

num_classes = train_generator.num_classes

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=val_generator
)

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

model.save(os.path.join(BASE_DIR, "usd_model.h5"))
