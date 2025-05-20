import os
import numpy as np
import cv2
from keras import load_model
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import json

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.getcwd())           # e.g. /Users/you/OCRODIA
ODIA_DIR       = os.path.join(BASE_DIR, 'odiaData', 'characters')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
BEST_ID        = 44                                     # our best Bengali model
BENGALI_MODEL  = os.path.join(MODELS_DIR, f'model({BEST_ID}).h5')
OUTPUT_MODEL   = os.path.join(MODELS_DIR, f'model({BEST_ID})_odia.h5')

IMG_SIZE       = (32, 32)
BATCH_SIZE     = 64
EPOCHS         = 20
TEST_SIZE      = 0.2
RANDOM_STATE   = 42


def main():
    # ─── DATA LOADING & LABELING ──────────────────────────────────────────────────
    def load_image_paths(data_dir):
        # if data_dir contains subfolders per class, use them
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        paths, labels = [], []
        if subdirs:
            subdirs = sorted(subdirs)
            for idx, cls in enumerate(subdirs):
                folder = os.path.join(data_dir, cls)
                for fn in os.listdir(folder):
                    if fn.lower().endswith(('.png','.jpg','.jpeg')):
                        paths.append(os.path.join(folder, fn))
                        labels.append(idx)
            class_names = subdirs
        else:
            # flat folder: assume filename is the class index, e.g. "0.jpg", "12.png"
            for fn in os.listdir(data_dir):
                if not fn.lower().endswith(('.png','.jpg','.jpeg')):
                    continue
                label = int(os.path.splitext(fn)[0])  # parse "0" from "0.jpg"
                paths.append(os.path.join(data_dir, fn))
                labels.append(label)
            class_names = [str(i) for i in sorted(set(labels))]
        return np.array(paths), np.array(labels), class_names

    def preprocess_image(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.equalizeHist(img)
        img = img.astype('float32') / 255.0
        return img[..., None]

    # load & shuffle
    paths, labels, class_names = load_image_paths(ODIA_DIR)
    idx = np.arange(len(paths))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    # train/val split
    train_p, val_p, train_l, val_l = train_test_split(
        paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # build arrays
    X_train = np.stack([preprocess_image(p) for p in train_p])
    y_train = train_l
    X_val   = np.stack([preprocess_image(p) for p in val_p])
    y_val   = val_l

    print(f"Odia data: {len(class_names)} classes, {len(paths)} total images")
    print("Train:", X_train.shape, y_train.shape, "Val:", X_val.shape, y_val.shape)

    # ─── TRANSFER LEARNING ─────────────────────────────────────────────────────────
    print("Loading Bengali model:", BENGALI_MODEL)
    base = load_model(BENGALI_MODEL)

    # freeze all but last 3 layers
    for layer in base.layers[:-3]:
        layer.trainable = False

    # attach new head
    x = base.layers[-3].output
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(len(class_names), activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ─── TRAIN & EVALUATE ──────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Final Odia model → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # ─── SAVE MODEL & METRICS ─────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(OUTPUT_MODEL)
    metrics = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
        'classes': class_names
    }
    with open(os.path.join(MODELS_DIR, f'metrics_odia_{BEST_ID}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Saved transfer-learned model to:", OUTPUT_MODEL)

if __name__ == "__main__":
    # everything you’ve written above lives in this block
    # or you wrap your code in a `main()` function and call it here
    main()