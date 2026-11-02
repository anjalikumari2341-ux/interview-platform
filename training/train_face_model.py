"""
train_face_model.py  -  Facial-confidence model (FER2013 + MobileNetV2 features)

Colab setup cell (run first):
    !pip install numpy==1.26.4
    !pip install joblib opencv-python-headless scikit-learn pandas matplotlib seaborn tqdm tensorflow
Upload fer2013.csv to /content, then paste this file into a cell and run it.

What changed compared with the old notebook
  1. 70 : 30 train / test split (stratified). A validation set (15 % of the 70 %)
     is used to pick the top-3 models, so the test set is touched only once.
     Set USE_OFFICIAL_SPLIT = True to use FER2013's own Training / PublicTest /
     PrivateTest split instead (that one is roughly 80 : 20, not 70 : 30).
  2. Augmentation (flip, brighter, darker) is applied AFTER the split and only to
     the training images, so augmented copies of a test image can never leak in.
  3. Memory: the old code kept every 96x96x3 float32 image in RAM (about 3.5 GB
     before augmentation and over 10 GB after, which crashes free Colab). Images are
     now turned into 1280-number features batch by batch and the pixels are dropped.
  4. 'neutral' faces are now labelled 'confident' (INCLUDE_NEUTRAL_AS_CONFIDENT).
     Before, neutral was skipped, so the model never saw a calm, neutral face - the
     normal look of a candidate in an interview - and tended to call it 'not confident'.
  5. Scaler and PCA are fitted on the (balanced) training part only.

Output files (folder OUT_DIR): face_best_model.pkl, face_scaler.pkl, face_pca.pkl, charts
IMPORTANT: the app must preprocess faces exactly like this script (CLAHE first).
           modules/video_analyzer.py in the fixed zip already does.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TEST_SIZE        = 0.30      # 70 : 30 train / test
VAL_FRACTION     = 0.15      # share of the 70 % kept for choosing models
RANDOM_STATE     = 42
OUT_DIR          = Path("fer_data")
CSV_PATH         = "fer2013.csv"

USE_OFFICIAL_SPLIT           = False   # False = pooled stratified 70:30
INCLUDE_NEUTRAL_AS_CONFIDENT = True    # see note 4 above
QUICK_MODE                   = False   # True = 3 fast models, for a quick test run

IMG_SIZE    = 48
TARGET_SIZE = 96
BATCH_SIZE  = 256
N_AUGMENTED = 4                        # augment_image() returns exactly 4 versions

EMOTION_MAP = {0: "angry", 1: "disgust", 2: "fearful", 3: "happy",
               4: "sad", 5: "surprised", 6: "neutral"}
CONFIDENCE_MAP = {"happy": 1, "surprised": 1,
                  "angry": 0, "disgust": 0, "fearful": 0, "sad": 0}
LABEL_NAMES = {0: "not_confident", 1: "confident"}
LABEL_LIST  = [LABEL_NAMES[0], LABEL_NAMES[1]]


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing (must match modules/video_analyzer.py)
# ──────────────────────────────────────────────────────────────────────────────
def apply_clahe(img: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)


def preprocess_for_mobilenet(img: np.ndarray) -> np.ndarray:
    """48x48 gray -> CLAHE -> 96x96 RGB -> MobileNetV2 scaling to [-1, 1]."""
    img_up = cv2.resize(apply_clahe(img), (TARGET_SIZE, TARGET_SIZE),
                        interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(img_up, cv2.COLOR_GRAY2RGB).astype(np.float32)
    return rgb / 127.5 - 1.0      # identical to keras mobilenet_v2.preprocess_input


def augment_image(img: np.ndarray) -> list:
    """Original + horizontal flip + brighter + darker (exactly N_AUGMENTED images)."""
    bright = np.clip(img.astype(np.int32) + 20, 0, 255).astype(np.uint8)
    dark   = np.clip(img.astype(np.int32) - 20, 0, 255).astype(np.uint8)
    return [img, cv2.flip(img, 1), bright, dark]


def load_feature_extractor():
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(TARGET_SIZE, TARGET_SIZE, 3), pooling="avg")
    base.trainable = False
    return base


def extract_features(imgs, extractor, augment=False, batch_size=BATCH_SIZE):
    """Images -> 1280-number CNN features, one batch at a time (low memory)."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it            # noqa: E731
    feats, buf = [], []

    def flush():
        if buf:
            feats.append(extractor.predict(np.stack(buf), batch_size=batch_size, verbose=0))
            buf.clear()

    for img in tqdm(imgs, desc="CNN features"):
        for variant in (augment_image(img) if augment else [img]):
            buf.append(preprocess_for_mobilenet(variant))
            if len(buf) >= batch_size:
                flush()
    flush()
    return np.vstack(feats).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data + split
# ──────────────────────────────────────────────────────────────────────────────
def load_fer(csv_path=CSV_PATH, include_neutral=INCLUDE_NEUTRAL_AS_CONFIDENT):
    df = pd.read_csv(csv_path)
    cmap = dict(CONFIDENCE_MAP)
    if include_neutral:
        cmap["neutral"] = 1
    usage_col = df["Usage"] if "Usage" in df.columns else ["Training"] * len(df)
    imgs, labels, usage, skipped = [], [], [], 0
    for emo_id, pix, use in zip(df["emotion"], df["pixels"], usage_col):
        label = cmap.get(EMOTION_MAP[int(emo_id)])
        if label is None:
            continue
        arr = np.array(pix.split(), dtype=np.uint8)
        if arr.size != IMG_SIZE * IMG_SIZE:
            skipped += 1
            continue
        imgs.append(arr.reshape(IMG_SIZE, IMG_SIZE))
        labels.append(label)
        usage.append(use)
    print(f"Loaded {len(imgs)} images (skipped {skipped} bad rows)")
    return np.stack(imgs), np.array(labels), np.array(usage)


def split_indices(y, usage, use_official=USE_OFFICIAL_SPLIT, test_size=TEST_SIZE,
                  val_fraction=VAL_FRACTION, seed=RANDOM_STATE):
    idx = np.arange(len(y))
    if use_official:
        pool = idx[usage == "Training"]
        test_idx = idx[np.isin(usage, ["PublicTest", "PrivateTest"])]
    else:
        pool, test_idx = train_test_split(idx, test_size=test_size, random_state=seed,
                                          stratify=y)
    train_idx, val_idx = train_test_split(pool, test_size=val_fraction, random_state=seed,
                                          stratify=y[pool])
    return train_idx, val_idx, test_idx


def balance(X, y, seed=RANDOM_STATE):
    """Under-sample the bigger class so both classes have the same size."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    keep = np.concatenate([rng.choice(np.where(y == c)[0], counts.min(), replace=False)
                           for c in classes])
    rng.shuffle(keep)
    return X[keep], y[keep]


# ──────────────────────────────────────────────────────────────────────────────
# Models + evaluation
# ──────────────────────────────────────────────────────────────────────────────
def build_models(quick=False):
    if quick:
        return {
            "Logistic Regression": LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        }
    return {
        "SVM (RBF)": SVC(kernel="rbf", C=50, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel="linear", C=5, probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                                                        subsample=0.8, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(1024, 512, 256, 128), activation="relu", max_iter=500,
                             learning_rate_init=0.001, early_stopping=True, validation_fraction=0.1,
                             random_state=RANDOM_STATE),
    }


def score_model(clf, X_tr, y_tr, X_te, y_te, dev_acc, average="weighted"):
    pred = clf.predict(X_te)
    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_acc = accuracy_score(y_te, pred)
    return {
        "dev_acc": dev_acc, "test_acc": test_acc, "train_acc": train_acc,
        "precision": precision_score(y_te, pred, average=average, zero_division=0),
        "recall":    recall_score(y_te, pred, average=average, zero_division=0),
        "f1":        f1_score(y_te, pred, average=average, zero_division=0),
        "overfit_gap": train_acc - test_acc,
    }


def save_charts(results, trained, best_name, X_test, y_test, label_names, title,
                out_dir, show=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results)
    x = np.arange(len(names))

    def finish(fig, ax, fname, ylabel, ylim):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, [results[n]["dev_acc"] * 100 for n in names], 0.4,
           label="Validation / CV (inside the 70%)", color="steelblue", edgecolor="black")
    ax.bar(x + 0.2, [results[n]["test_acc"] * 100 for n in names], 0.4,
           label="Test (30%)", color="seagreen", edgecolor="black")
    ax.axhline(50, linestyle="--", color="red", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Accuracy - {title}")
    finish(fig, ax, "accuracy_comparison.png", "Accuracy (%)", (0, 110))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (key, color) in enumerate([("precision", "steelblue"),
                                      ("recall", "seagreen"), ("f1", "tomato")]):
        ax.bar(x + (i - 1) * 0.25, [results[n][key] * 100 for n in names], 0.25,
               label=key.capitalize(), color=color, edgecolor="black")
    ax.set_title(f"Precision / Recall / F1 (test set) - {title}")
    finish(fig, ax, "precision_recall_f1.png", "Score (%)", (0, 115))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, [results[n]["train_acc"] * 100 for n in names], 0.4,
           label="Train", color="steelblue", edgecolor="black")
    ax.bar(x + 0.2, [results[n]["test_acc"] * 100 for n in names], 0.4,
           label="Test", color="seagreen", edgecolor="black")
    for i, n in enumerate(names):
        gap = results[n]["overfit_gap"] * 100
        top = max(results[n]["train_acc"], results[n]["test_acc"]) * 100 + 0.5
        ax.annotate(f"gap={gap:.1f}%", (i, top), ha="center", fontsize=8,
                    color="red" if gap > 5 else "green")
    ax.set_title(f"Train vs Test accuracy (overfit gap) - {title}")
    finish(fig, ax, "overfit_gap.png", "Accuracy (%)", (0, 115))

    cm = confusion_matrix(y_test, trained[best_name].predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_title(f"Confusion matrix - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def train_and_evaluate(F_train, y_train, F_val, y_val, F_test, y_test,
                       out_dir=OUT_DIR, quick=QUICK_MODE, make_plots=True, show=True):
    """F_* are raw 1280-number CNN features (before scaler / PCA)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    F_train, y_train = balance(F_train, y_train)
    print(f"Train (balanced): {len(y_train)}   Val: {len(y_val)}   Test: {len(y_test)}")

    # Scaler + PCA are fitted on the TRAIN part only
    scaler = StandardScaler().fit(F_train)
    pca = PCA(n_components=0.99, random_state=RANDOM_STATE).fit(scaler.transform(F_train))
    X_train = pca.transform(scaler.transform(F_train))
    X_val   = pca.transform(scaler.transform(F_val))
    X_test  = pca.transform(scaler.transform(F_test))
    print(f"PCA keeps {X_train.shape[1]} components (99% variance)")
    joblib.dump(scaler, out_dir / "face_scaler.pkl")
    joblib.dump(pca, out_dir / "face_pca.pkl")

    results, trained = {}, {}
    print(f"\n{'Model':<22}{'Val':>8}{'Test':>8}{'F1':>8}{'Gap':>9}{'Time':>8}")
    for name, clf in build_models(quick).items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        results[name] = score_model(clf, X_train, y_train, X_test, y_test, val_acc)
        trained[name] = clf
        r = results[name]
        print(f"{name:<22}{val_acc * 100:>7.1f}%{r['test_acc'] * 100:>7.1f}%"
              f"{r['f1'] * 100:>7.1f}%{r['overfit_gap'] * 100:>+8.1f}%{time.time() - t0:>7.0f}s")

    # Top-3 models by VALIDATION accuracy form the ensemble (test set not used to choose)
    top3 = sorted(results, key=lambda n: results[n]["dev_acc"], reverse=True)[:3]
    print(f"\nEnsemble of: {top3}")
    ensemble = VotingClassifier(estimators=[(n, trained[n]) for n in top3], voting="soft")
    ensemble.fit(X_train, y_train)
    ens_val = accuracy_score(y_val, ensemble.predict(X_val))
    results["Ensemble (top3)"] = score_model(ensemble, X_train, y_train, X_test, y_test, ens_val)
    trained["Ensemble (top3)"] = ensemble

    print("\n== Ensemble on the held-out 30% ==")
    print(classification_report(y_test, ensemble.predict(X_test),
                                target_names=LABEL_LIST, zero_division=0))

    joblib.dump(ensemble, out_dir / "face_best_model.pkl")
    # Verify exactly the way the app uses the files: raw features -> scaler -> PCA -> model
    m = joblib.load(out_dir / "face_best_model.pkl")
    s = joblib.load(out_dir / "face_scaler.pkl")
    p = joblib.load(out_dir / "face_pca.pkl")
    probs = m.predict_proba(p.transform(s.transform(F_test[:3])))
    assert probs.shape[1] == 2, "model must output 2 classes"
    print("Saved + verified: face_best_model.pkl, face_scaler.pkl, face_pca.pkl")

    if make_plots:
        save_charts(results, trained, "Ensemble (top3)", X_test, y_test, LABEL_LIST,
                    "Facial confidence (FER2013)", out_dir, show=show)
    return results


def download_if_colab(paths):
    try:
        from google.colab import files
    except ImportError:
        return
    for p in paths:
        if Path(p).exists():
            files.download(str(p))


def main():
    imgs, y, usage = load_fer()
    train_idx, val_idx, test_idx = split_indices(y, usage)
    n = len(y)
    print(f"Split of original images -> train {len(train_idx) / n:.1%}, "
          f"val {len(val_idx) / n:.1%}, test {len(test_idx) / n:.1%}")
    print(f"Classes (all images): confident={int((y == 1).sum())}  not_confident={int((y == 0).sum())}")

    extractor = load_feature_extractor()
    F_train = extract_features(imgs[train_idx], extractor, augment=True)   # augmented
    y_train = np.repeat(y[train_idx], N_AUGMENTED)                          # same order
    F_val   = extract_features(imgs[val_idx], extractor)
    F_test  = extract_features(imgs[test_idx], extractor)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_and_evaluate(F_train, y_train, F_val, y[val_idx], F_test, y[test_idx])
    download_if_colab([OUT_DIR / f for f in
                       ["face_best_model.pkl", "face_scaler.pkl", "face_pca.pkl",
                        "accuracy_comparison.png", "precision_recall_f1.png",
                        "overfit_gap.png", "confusion_matrix.png"]])
    print("\nAll done. Copy face_best_model.pkl, face_scaler.pkl and face_pca.pkl "
          "into your project's model/ folder.")


if __name__ == "__main__":
    main()
