"""
train_voice_model.py  -  Voice-confidence model (RAVDESS)

Colab setup cell (run first):
    !pip install numpy==1.26.4
    !pip install joblib librosa tqdm pandas scikit-learn matplotlib seaborn
Upload the RAVDESS zip to /content, then paste this file into a cell and run it.

What changed compared with the old notebook
  1. 70 : 30 train / test split, done BY ACTOR.
     RAVDESS has 24 actors who each say the same two sentences in every emotion.
     A random split puts the same person in both train and test, so the model
     "recognises the speaker" and the accuracy looks far better than it really is.
     Here ~7 actors are held out completely for testing.
  2. The scaler is fitted on the training part only.
  3. Cross-validation uses only the 70 % training part (also split by actor) and
     scales inside each fold, so nothing from the test set leaks in.
  4. class_weight="balanced" because the 'neutral' class is much smaller.
  5. The test set is used once, at the end.

Output files (folder OUT_DIR): voice_best_model.pkl, voice_scaler.pkl, charts, metadata.csv
"""
from __future__ import annotations

import time
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TEST_SIZE    = 0.30          # 70 : 30 train / test
RANDOM_STATE = 42
OUT_DIR      = Path("ravdess_data")

# Audio settings - must match modules/audio_processor.py in the app
N_MFCC      = 40
SAMPLE_RATE = 22050
DURATION    = 3.0

EMOTION_MAP = {
    "01": "neutral", "02": "calm",  "03": "happy",
    "04": "sad",     "05": "angry", "06": "fearful",
    "07": "disgust", "08": "surprised",
}
CONFIDENCE_MAP = {
    "calm": 2, "happy": 2,        # confident
    "neutral": 1,                 # neutral
    "sad": 0, "fearful": 0,       # not_confident
}
LABEL_NAMES = {0: "not_confident", 1: "neutral", 2: "confident"}
LABEL_LIST  = [LABEL_NAMES[i] for i in range(3)]


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def parse_filename(fname: str):
    """RAVDESS name: modality-channel-EMOTION-intensity-statement-repetition-ACTOR.wav"""
    parts = Path(fname).stem.split("-")
    if len(parts) < 7:
        return None
    emotion = EMOTION_MAP.get(parts[2])
    if emotion not in CONFIDENCE_MAP:
        return None
    return emotion, CONFIDENCE_MAP[emotion], int(parts[6])


def load_audio(path: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    target = int(SAMPLE_RATE * DURATION)
    return np.pad(y, (0, max(0, target - len(y))))[:target]


def extract_features(y: np.ndarray) -> np.ndarray:
    import librosa
    mfcc     = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta    = librosa.feature.delta(mfcc)
    delta2   = librosa.feature.delta(mfcc, order=2)
    chroma   = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(y=y)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    return np.concatenate([
        np.mean(mfcc,     axis=1), np.std(mfcc,     axis=1),
        np.mean(delta,    axis=1), np.std(delta,    axis=1),
        np.mean(delta2,   axis=1), np.std(delta2,   axis=1),
        np.mean(chroma,   axis=1), np.std(chroma,   axis=1),
        [np.mean(zcr),      np.std(zcr)],
        [np.mean(rms),      np.std(rms)],
        [np.mean(rolloff),  np.std(rolloff)],
        [np.mean(centroid), np.std(centroid)],
    ]).astype(np.float32)


def build_dataset(wav_files):
    try:
        from tqdm import tqdm
    except ImportError:                       # tqdm is optional
        tqdm = lambda it, **kw: it            # noqa: E731
    feats, labels, actors, rows = [], [], [], []
    for wav in tqdm(wav_files, desc="Extracting features"):
        parsed = parse_filename(wav.name)
        if parsed is None:
            continue
        emotion, label, actor = parsed
        try:
            feat = extract_features(load_audio(str(wav)))
        except Exception as exc:
            print(f"  skipped {wav.name}: {exc}")
            continue
        feats.append(feat)
        labels.append(label)
        actors.append(actor)
        rows.append({"filename": wav.name, "emotion": emotion, "label": label,
                     "label_name": LABEL_NAMES[label], "actor": actor})
    return np.array(feats), np.array(labels), np.array(actors), pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 70 : 30 split by actor
# ──────────────────────────────────────────────────────────────────────────────
def split_by_actor(actors, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """Hold out ~test_size of the ACTORS (every actor has every emotion)."""
    unique = np.unique(actors)
    shuffled = np.random.RandomState(seed).permutation(unique)
    n_test = max(1, int(round(test_size * len(unique))))
    test_actors = sorted(int(a) for a in shuffled[:n_test])
    is_test = np.isin(actors, test_actors)
    return np.where(~is_test)[0], np.where(is_test)[0], test_actors


def build_models():
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                   class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                        max_depth=4, random_state=RANDOM_STATE),
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

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, [results[n]["dev_acc"] * 100 for n in names], 0.4,
           label="Validation / CV (inside the 70%)", color="steelblue", edgecolor="black")
    ax.bar(x + 0.2, [results[n]["test_acc"] * 100 for n in names], 0.4,
           label="Test (30%)", color="seagreen", edgecolor="black")
    ax.set_title(f"Accuracy - {title}")
    finish(fig, ax, "accuracy_comparison.png", "Accuracy (%)", (0, 110))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (key, color) in enumerate([("precision", "steelblue"),
                                      ("recall", "seagreen"), ("f1", "tomato")]):
        ax.bar(x + (i - 1) * 0.25, [results[n][key] * 100 for n in names], 0.25,
               label=key.capitalize(), color=color, edgecolor="black")
    ax.set_title(f"Precision / Recall / F1 (test set) - {title}")
    finish(fig, ax, "precision_recall_f1.png", "Score (%)", (0, 115))

    fig, ax = plt.subplots(figsize=(11, 5))
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


# ──────────────────────────────────────────────────────────────────────────────
# Train + evaluate
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(X, y, actors, out_dir=OUT_DIR, make_plots=True, show=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, te, test_actors = split_by_actor(actors)
    X_tr_raw, X_te_raw, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    print(f"Train: {len(tr)} clips ({len(tr) / len(y):.0%})   "
          f"Test: {len(te)} clips ({len(te) / len(y):.0%})")
    print(f"Test actors (never seen in training): {test_actors}")

    # Scaler is fitted on the TRAIN part only
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    joblib.dump(scaler, out_dir / "voice_scaler.pkl")

    results, trained = {}, {}
    cv = GroupKFold(n_splits=5)
    print("\n-- 5-fold cross-validation on the training part (split by actor) --")
    for name, clf in build_models().items():
        cv_scores = cross_val_score(make_pipeline(StandardScaler(), clone(clf)),
                                    X_tr_raw, y_tr, groups=actors[tr], cv=cv,
                                    scoring="accuracy")
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        results[name] = score_model(clf, X_tr, y_tr, X_te, y_te, cv_scores.mean())
        trained[name] = clf
        print(f"  {name:<18} CV {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}   "
              f"test {results[name]['test_acc']:.3f}   ({time.time() - t0:.1f}s)")

    ensemble = VotingClassifier(estimators=list(build_models().items()), voting="soft")
    ensemble.fit(X_tr, y_tr)
    results["Ensemble"] = score_model(ensemble, X_tr, y_tr, X_te, y_te,
                                      float(np.mean([r["dev_acc"] for r in results.values()])))
    trained["Ensemble"] = ensemble

    pred = ensemble.predict(X_te)
    print("\n== Ensemble on the held-out 30% (unseen actors) ==")
    print(classification_report(y_te, pred, target_names=LABEL_LIST, zero_division=0))

    joblib.dump(ensemble, out_dir / "voice_best_model.pkl")
    check_model = joblib.load(out_dir / "voice_best_model.pkl")
    check_scaler = joblib.load(out_dir / "voice_scaler.pkl")
    probs = check_model.predict_proba(check_scaler.transform(X_te_raw[:5]))
    assert probs.shape[1] == 3, "model must output 3 classes"
    print("Saved + verified: voice_best_model.pkl, voice_scaler.pkl")

    if make_plots:
        save_charts(results, trained, "Ensemble", X_te, y_te, LABEL_LIST,
                    "Voice confidence", out_dir, show=show)
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
    zips = list(Path("/content").glob("*.zip")) + list(Path(".").glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No .zip found - upload your RAVDESS zip first.")
    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(audio_dir)
    wav_files = sorted(audio_dir.rglob("*.wav"))
    print(f"{len(wav_files)} .wav files found")

    X, y, actors, df = build_dataset(wav_files)
    np.save(OUT_DIR / "features.npy", X)
    np.save(OUT_DIR / "labels.npy", y)
    df.to_csv(OUT_DIR / "metadata.csv", index=False)
    print(f"Feature matrix: {X.shape}   classes: "
          f"{ {LABEL_NAMES[k]: v for k, v in sorted(Counter(y.tolist()).items())} }")

    train_and_evaluate(X, y, actors)
    download_if_colab([OUT_DIR / f for f in
                       ["voice_best_model.pkl", "voice_scaler.pkl", "accuracy_comparison.png",
                        "precision_recall_f1.png", "overfit_gap.png", "confusion_matrix.png",
                        "metadata.csv"]])
    print("\nAll done. Copy voice_best_model.pkl and voice_scaler.pkl into your project's model/ folder.")


if __name__ == "__main__":
    main()
