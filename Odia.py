import os
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model, Input, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.getcwd())
ODIA_DIR       = os.path.join(BASE_DIR, 'odiaData', 'characters')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
BEST_ID        = 44
BENGALI_MODEL  = os.path.join(MODELS_DIR, f'model({BEST_ID}).h5')
OUTPUT_MODEL   = os.path.join(MODELS_DIR, f'model({BEST_ID})_odia.h5')

IMG_SIZE       = (32, 32)
BATCH_SIZE     = 64
EPOCHS         = 20
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

def load_image_paths(data_dir):
    # list only directories, sorted numerically
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
    class_names = subdirs
    return np.array(paths), np.array(labels), class_names

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.equalizeHist(img)
    img = img.astype('float32') / 255.0
    return img[..., None]

def main():
     # ─── LOAD & FREEZE BASE MODEL ────────────────────────────────────────────
    base = load_model(BENGALI_MODEL)
    for layer in base.layers:
        layer.trainable = False

    # ─── BUILD A TRUE FEATURE EXTRACTOR ──────────────────────────────────────
    # We manually apply all layers up to (but excluding) the last 2 layers
    # (which are Dropout and the final softmax Dense).
    from tensorflow.keras import Input, Model

    inp = Input(shape=(*IMG_SIZE, 1))
    x = inp
    for layer in base.layers[:-2]:
        x = layer(x)
    feat_extractor = Model(inputs=inp, outputs=x, name="bengali_feature_extractor")

    # ─── ATTACH YOUR NEW ODIA CLASSIFIER HEAD ────────────────────────────────
    features = feat_extractor(inp, training=False)
    x = layers.Dense(64, activation='relu')(features)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="odia_transfer_model")
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ─── CALLBACKS ──────────────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    checkpoint = ModelCheckpoint(
        OUTPUT_MODEL, monitor='val_loss', save_best_only=True, verbose=1
    )

    # ─── DATA AUGMENTATION ──────────────────────────────────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train)

    # ─── TRAIN ──────────────────────────────────────────────────────────────────
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=2
    )

    # ─── FINAL EVAL ─────────────────────────────────────────────────────────────
    best = load_model(OUTPUT_MODEL)
    loss, acc = best.evaluate(X_val, y_val, verbose=2)
    print(f"Best Odia model → Loss: {loss:.4f}, Acc: {acc:.4f}")

    # ─── SAVE METRICS ──────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'val_loss': float(loss),
        'val_acc': float(acc),
        'classes': class_names
    }
    with open(os.path.join(MODELS_DIR, f'metrics_odia_{BEST_ID}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Saved transfer-learned model and metrics to:", OUTPUT_MODEL)

if __name__ == "__main__":
    main()
