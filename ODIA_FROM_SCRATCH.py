import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import random

# ─── SET SEEDS FOR STABILITY ─────────────────────────────────────────────
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.getcwd())
ODIA_DIR       = os.path.join(BASE_DIR, 'odiaData', 'characters')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
OUTPUT_MODEL   = os.path.join(MODELS_DIR, 'odia_model.h5')

IMG_SIZE       = (32, 32)
BATCH_SIZE     = 64
EPOCHS         = 30
TEST_SIZE      = 0.2
VAL_SIZE       = 0.25  # of train set after test split
RANDOM_STATE   = 42
AUG_FACTOR     = 20

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────

def load_image_paths(data_dir):
    subdirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: int(x)
    )
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

def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, cmap='viridis', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    paths, labels, class_names = load_image_paths(ODIA_DIR)
    idx = np.arange(len(paths))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    # Train/Val/Test split (60/20/20)
    paths_train, paths_test, labels_train, labels_test = train_test_split(
        paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_train, labels_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=labels_train)

    print("Train:", Counter(labels_train))
    print("Val  :", Counter(labels_val))
    print("Test :", Counter(labels_test))

    X_train = np.stack([preprocess_image(p) for p in paths_train])
    X_val   = np.stack([preprocess_image(p) for p in paths_val])
    X_test  = np.stack([preprocess_image(p) for p in paths_test])
    y_train, y_val, y_test = labels_train, labels_val, labels_test

    # Build model from scratch
    inp = Input(shape=(*IMG_SIZE, 1), name="odia_input")
    x = layers.Conv2D(32, (3, 3), activation='relu')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="odia_scratch_model")
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', save_best_only=True, verbose=1)

    # Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.15,
        fill_mode='reflect'
    )
    datagen.fit(X_train, augment=True, seed=RANDOM_STATE)

    steps = max(1, (len(X_train) * AUG_FACTOR) // BATCH_SIZE)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=2
    )

    # Evaluate best model
    best = tf.keras.models.load_model(OUTPUT_MODEL)
    val_loss, val_acc = best.evaluate(X_val, y_val, verbose=2)
    test_loss, test_acc = best.evaluate(X_test, y_test, verbose=2)
    print(f"Val  → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"Test → Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Confusion matrix
    preds = best.predict(X_test)
    top_preds = np.argmax(preds, axis=1)
    plot_confusion(y_test, top_preds, class_names, os.path.join(MODELS_DIR, "confusion_matrix.png"))

    # Save metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'classes': class_names
    }
    with open(os.path.join(MODELS_DIR, 'metrics_odia_scratch.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved model & metrics to:", OUTPUT_MODEL)

if __name__ == "__main__":
    main()
