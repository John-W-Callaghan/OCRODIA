import os
import json
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, Input, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = os.path.expanduser('~/Documents/GitHub/OCRODIA')
ODIA_DIR = os.path.join(BASE_DIR, 'odiaData', 'characters')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

BEST_ID = 44
BENGALI_MODEL_PATH = os.path.join(MODELS_DIR, f'model({BEST_ID}).h5')
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42
AUG_FACTOR = 10

# Utility functions
def load_image_paths(data_dir):
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))], key=lambda x: int(x))
    paths, labels = [], []
    for idx, cls in enumerate(subdirs):
        folder = os.path.join(data_dir, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(folder, fn))
                labels.append(idx)
    return np.array(paths), np.array(labels), subdirs

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return img[..., None]

# Load and split dataset
paths, labels, class_names = load_image_paths(ODIA_DIR)
idx = np.arange(len(paths))
np.random.seed(RANDOM_STATE)
np.random.shuffle(idx)
paths, labels = paths[idx], labels[idx]
train_p, val_p, train_l, val_l = train_test_split(paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)

X_train = np.stack([preprocess_image(p) for p in train_p])
y_train = train_l
X_val = np.stack([preprocess_image(p) for p in val_p])
y_val = val_l

# Shared callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    fill_mode='reflect'
)
datagen.fit(X_train)
steps = max(1, (len(X_train) * AUG_FACTOR) // BATCH_SIZE)

# Load Bengali model
base_model = load_model(BENGALI_MODEL_PATH)

# Define a function to build and train a model
def train_model(trainable: bool, output_name: str):
    for layer in base_model.layers:
        layer.trainable = trainable

    inp = Input(shape=(*IMG_SIZE, 1), name="odia_input")
    x = inp
    for layer in base_model.layers[:-3]:
        x = layer(x)

    x = layers.Flatten()(x)
    if trainable:
        x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="odia_transfer_model")
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(os.path.join(MODELS_DIR, output_name), monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=0
    )

    metrics = {
        'trainable': trainable,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }

    json_path = os.path.join(MODELS_DIR, f'metrics_odia_{BEST_ID}_{"finetuned" if trainable else "frozen"}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

# Train both versions
metrics_frozen = train_model(False, f'model({BEST_ID})_frozen.h5')
metrics_finetuned = train_model(True, f'model({BEST_ID})_finetuned.h5')

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.plot(metrics_frozen['history']['val_accuracy'], label='Frozen Base', linestyle='--')
plt.plot(metrics_finetuned['history']['val_accuracy'], label='Fine-Tuned Base', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Comparison of Frozen vs Fine-Tuned Transfer Learning')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'transfer_comparison_accuracy.png'))
plt.show()
