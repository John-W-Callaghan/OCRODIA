import os
import numpy as np
import cv2
import json
from collections import Counter
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
AUG_FACTOR     = 10  # how many augmented samples per real sample

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
    img = cv2.equalizeHist(img)
    img = img.astype('float32') / 255.0
    return img[..., None]

def main():
    # ─── 1) LOAD & PREP DATA ────────────────────────────────────────────────
    paths, labels, class_names = load_image_paths(ODIA_DIR)
    idx = np.arange(len(paths))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    # ─── STRATIFIED SPLIT ──────────────────────────────────────────────────
    train_p, val_p, train_l, val_l = train_test_split(
        paths, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # ─── PRINT CLASS DISTRIBUTIONS ─────────────────────────────────────────
    print("Train class counts:", Counter(train_l))
    print("Val   class counts:", Counter(val_l))

    X_train = np.stack([preprocess_image(p) for p in train_p])
    y_train = train_l
    X_val   = np.stack([preprocess_image(p) for p in val_p])
    y_val   = val_l

    # ─── OPTIONAL: SANITY-CHECK AUGMENTATIONS ──────────────────────────────
    # import matplotlib.pyplot as plt
    # datagen = ImageDataGenerator(rotation_range=360, width_shift_range=0.2,
    #     height_shift_range=0.2, shear_range=0.2, zoom_range=[0.8,1.2],
    #     brightness_range=[0.5,1.5], fill_mode='reflect')
    # fig, axs = plt.subplots(2, 5, figsize=(10,4))
    # for i in range(5):
    #     orig = X_train[y_train==i][0].squeeze()
    #     axs[0,i].imshow(orig, cmap='gray'); axs[0,i].axis('off')
    #     aug_im = datagen.flow(
    #         X_train[y_train==i:i+1], batch_size=1
    #     ).next()[0].squeeze()
    #     axs[1,i].imshow(aug_im, cmap='gray'); axs[1,i].axis('off')
    # plt.show()

    # ─── 2) LOAD & FREEZE BASE MODEL ────────────────────────────────────────
    print("Loading Bengali model:", BENGALI_MODEL)
    base = load_model(BENGALI_MODEL)
    for layer in base.layers:
        layer.trainable = False

    # ─── 3) BUILD FEATURE EXTRACTOR ─────────────────────────────────────────
    inp = Input(shape=(*IMG_SIZE, 1))
    x = inp
    for layer in base.layers[:-2]:
        x = layer(x)
    feat_extractor = Model(inputs=inp, outputs=x, name="bengali_feat_ext")

    # ─── 4) ATTACH ODIA HEAD ────────────────────────────────────────────────
    features = feat_extractor(inp, training=False)
    x = layers.Dense(64, activation='relu')(features)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="odia_transfer")
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ─── 5) CALLBACKS ───────────────────────────────────────────────────────
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', save_best_only=True, verbose=1)

    # ─── 6) HEAVY AUGMENTATION & TRAIN ─────────────────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=[0.8, 1.2],
        brightness_range=[0.5, 1.5],
        fill_mode='reflect'
    )
    datagen.fit(X_train)

    steps = max(1, (len(X_train) * AUG_FACTOR) // BATCH_SIZE)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=2
    )

    # ─── 7) FINAL EVALUATION & SAVE METRICS ────────────────────────────────
    best = load_model(OUTPUT_MODEL)
    loss, acc = best.evaluate(X_val, y_val, verbose=2)
    print(f"Best Odia model → Loss: {loss:.4f}, Acc: {acc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'val_loss': float(loss),
        'val_acc': float(acc),
        'classes': class_names
    }
    with open(os.path.join(MODELS_DIR, f'metrics_odia_{BEST_ID}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved model & metrics to:", OUTPUT_MODEL)

if __name__ == "__main__":
    main()
