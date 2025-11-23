# 🧾 Clasificador de Billetes USD con TensorFlow

Este proyecto entrena un modelo de **clasificación de imágenes** para identificar diferentes denominaciones de **billetes de dólar estadounidense (USD)** utilizando **TensorFlow** y un modelo preentrenado **MobileNetV2**.

---

## 📂 1. Estructura del Dataset

El script utiliza un dataset ubicado en:

```

C:\9no-Semestre\UX\datasets\usd\USA currency

```

El dataset debe tener subcarpetas, cada una representando una clase:

```

USA currency/
├── 1-dollar/
├── 5-dollar/
├── 10-dollar/
├── 20-dollar/
├── 50-dollar/
└── 100-dollar/

````

Cada carpeta contiene imágenes de la denominación correspondiente.

El script detecta automáticamente estas clases y las guarda en `labels.txt`.

---

## ⚙️ 2. Preprocesamiento de Datos

Se utiliza `ImageDataGenerator` para:

- Reescalar imágenes *(0–1)*
- Separar datos en:
  - **80% entrenamiento**
  - **20% validación**
- Cambiar el tamaño de las imágenes a **224 × 224 píxeles**

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
````

---

## 🧠 3. Modelo Utilizado

### ✔ MobileNetV2 (preentrenado)

* Entrenado originalmente en ImageNet
* Usado como **feature extractor**
* `include_top=False` → se elimina la capa de clasificación original
* Inicialmente se congela (`trainable=False`)

### ✔ Capas añadidas por el proyecto

Las capas personalizadas permiten la clasificación final:

* `Conv2D(32, 3, activation='relu')`
* `Dropout(0.2)`
* `GlobalAveragePooling2D()`
* `Dense(num_classes, activation='softmax')`

Estas capas construyen un clasificador adaptado a las clases detectadas dinámicamente en el dataset.

---

## 🚀 4. Entrenamiento

El proceso tiene dos fases:

### 🔹 **Fase 1 — Entrenamiento del Clasificador**

* MobileNetV2 congelado
* 4 épocas
* Solo se entrenan las capas nuevas (clasificador)

### 🔹 **Fase 2 — Fine Tuning**

* Se descongela MobileNetV2 parcialmente
* Se re-entrena desde la capa 100 en adelante
* 5 épocas adicionales
* Learning rate muy bajo (`1e-5`)

Este ajuste fino mejora la exactitud al adaptar el modelo a las características visuales reales de los billetes.

---

## 📝 5. Archivo de Etiquetas

Se genera automáticamente un archivo:

```
labels.txt
```

El cual contiene la lista de clases, por ejemplo:

```
1-dollar
5-dollar
10-dollar
20-dollar
50-dollar
100-dollar
```

---

## 💾 6. Guardado del Modelo

El modelo final entrenado se guarda como:

```
usd_model.h5
```

Este archivo puede ser utilizado para:

* Clasificación en Python
* TensorFlow Lite
* Aplicaciones móviles
* APIs de clasificación
* Integración en apps web

---

## 📌 7. Resumen del Flujo Completo

1. Detecta clases automáticamente desde carpetas.
2. Preprocesa imágenes (rescale + resize).
3. Construye un modelo basado en MobileNetV2.
4. Entrena el clasificador.
5. Ajusta finamente el modelo base.
6. Guarda el modelo `.h5` y las etiquetas.

---

## 📚 Requisitos

* Python 3.8+
* TensorFlow 2.10+
* NumPy
* Matplotlib (opcional)

Instalación recomendada:

```bash
pip install tensorflow numpy matplotlib
```

---

## 🏁 Estado

✔ Modelo funcional
✔ Entrenamiento completo
✔ Guardado en formato `.h5`
✔ Etiquetas generadas

---

## 🖼️ Ejemplo de Uso (pronto)

*(Puedes agregar aquí ejemplos de inferencia una vez implementes la fase de predicción.)*

---

Si quieres, puedo generarte:

* Ejemplo de inferencia
* Un script separado para predicción
* La conversión a TensorFlow Lite
* Un README en inglés

```
