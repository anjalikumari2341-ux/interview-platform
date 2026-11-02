"""
train_filler_model.py  -  Filler-word classifier (transcripts.csv)

Colab setup cell (run first):
    !pip install numpy==1.26.4
    !pip install joblib pandas scikit-learn matplotlib seaborn
Upload transcripts.csv to /content, then paste this file into a cell and run it.

What changed compared with the old notebook
  1. 70 : 30 train / test split (stratified).
  2. Cross-validation uses only the 70 % training part (it used all the data before).
  3. The filler list and the feature functions are now IDENTICAL to
     modules/speech_analyzer.py in the app (the old list had 23 words, the app uses 30,
     so the model was trained on different numbers than it receives in the app).
  4. The hand-written "clean" sentences are only kept if they really contain no filler
     by this counting; before, "...a different route home" was labelled clean even
     though 'er' inside 'different' counted as a filler.
  5. Colab-only download code no longer crashes outside Colab.

!! READ THIS - what this model can and cannot show !!
  The label is  filler_count >= 1  and filler_count is also an input feature, so the
  model only re-learns that rule. Accuracy near 100 % is guaranteed and proves nothing;
  the script prints the accuracy of the trivial rule so you can see it. Splitting the
  data differently cannot fix that, because the problem is how the label is defined.
  The app does not use this model's prediction for the score (the score comes from
  counting filler words), so nothing breaks - but in your report describe the filler
  detector as RULE-BASED (lexicon + counting), not as a trained ML model. A truly
  learned detector would need transcripts where a human has marked the fillers.

Output files (folder OUT_DIR): best_filler_model.pkl, filler_words.pkl, charts
"""
from __future__ import annotations

import string
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

TEST_SIZE    = 0.30          # 70 : 30 train / test
RANDOM_STATE = 42
OUT_DIR      = Path("filler_out")
CSV_PATH     = "transcripts.csv"

# Must match FILLER_WORDS in modules/speech_analyzer.py
FILLER_WORDS = [
    "you know", "i mean", "kind of", "sort of", "you see", "i guess", "i think",
    "um", "uh", "er", "ah", "hmm", "erm", "uhh", "umm",
    "like", "basically", "literally", "actually", "honestly",
    "obviously", "essentially", "totally", "clearly",
    "right", "okay", "so", "well", "just",
]
FEATURE_COLS = ["filler_count", "word_count", "filler_ratio", "unique_fillers"]

EXTRA_CLEAN = [
    "she went to the market early in the morning",
    "the children played outside until sunset",
    "he finished reading the book in two days",
    "the train arrived five minutes late today",
    "they decided to take a different route home",
    "the meeting was rescheduled to next thursday",
    "she submitted the assignment before midnight",
    "the restaurant was fully booked on saturday",
    "he repaired the bicycle in under an hour",
    "the flowers bloomed after the spring rain",
    "they watched a documentary about ocean life",
    "the package was delivered to the wrong address",
    "she learned three new recipes this weekend",
    "the laptop battery drains faster in cold weather",
    "he scored the highest marks in the final exam",
    "the library closes at nine on weekdays",
    "they planted vegetables in the backyard garden",
    "the flight was delayed due to heavy fog",
    "she designed the logo using simple shapes",
    "the dog followed him everywhere around the house",
    "the electricity went out during the thunderstorm",
    "he called the helpdesk and resolved the issue quickly",
    "the bridge was under construction for six months",
    "she saved the file and closed the application",
    "the professor explained the concept with a diagram",
    "they finished the project two days before deadline",
    "the supermarket ran out of stock on wednesday",
    "he fixed the bug by reviewing the error logs",
    "the students submitted their feedback after class",
    "the road was blocked due to a fallen tree",
]


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions (copies of the ones in modules/speech_analyzer.py)
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def count_fillers(text: str) -> int:
    return sum(text.count(w) for w in FILLER_WORDS)


def count_words(text: str) -> int:
    return len(text.split())


def filler_ratio(text: str) -> float:
    words = count_words(text)
    return count_fillers(text) / words if words > 0 else 0.0


def count_unique_fillers(text: str) -> int:
    return sum(1 for w in FILLER_WORDS if w in text)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["filler_count"]   = df["transcript"].apply(count_fillers)
    df["word_count"]     = df["transcript"].apply(count_words)
    df["filler_ratio"]   = df["transcript"].apply(filler_ratio)
    df["unique_fillers"] = df["transcript"].apply(count_unique_fillers)
    df["label"]          = (df["filler_count"] >= 1).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(csv_path=CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path).drop(columns=["url"], errors="ignore")
    df["transcript"] = df["transcript"].astype(str).apply(clean_text)
    df = add_features(df)
    print("Label counts before balancing:\n", df["label"].value_counts().to_string())

    extra = add_features(pd.DataFrame({"transcript": EXTRA_CLEAN}))
    extra = extra[extra["label"] == 0]            # keep only truly clean sentences
    print(f"Hand-written clean sentences kept: {len(extra)} of {len(EXTRA_CLEAN)}")
    df = pd.concat([df[["transcript"] + FEATURE_COLS + ["label"]], extra], ignore_index=True)

    majority_label = df["label"].value_counts().idxmax()
    minority_label = 1 - majority_label
    n_min = int((df["label"] == minority_label).sum())
    if n_min == 0:
        raise ValueError("Only one class present - cannot train a classifier.")
    major = resample(df[df["label"] == majority_label], replace=False,
                     n_samples=n_min, random_state=RANDOM_STATE)
    df = pd.concat([major, df[df["label"] == minority_label]], ignore_index=True)
    print("Label counts after balancing:\n", df["label"].value_counts().to_string())
    if len(df) < 200:
        print(f"WARNING: only {len(df)} samples after balancing - too few for reliable "
              "results. This is why the accuracy is not trustworthy.")
    return df


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4,
                                                min_samples_leaf=3, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=0.8, probability=True, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                        learning_rate=0.05, subsample=0.8,
                                                        random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    }


def score_model(clf, X_tr, y_tr, X_te, y_te, dev_acc):
    pred = clf.predict(X_te)
    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_acc = accuracy_score(y_te, pred)
    return {
        "dev_acc": dev_acc, "test_acc": test_acc, "train_acc": train_acc,
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall":    recall_score(y_te, pred, zero_division=0),
        "f1":        f1_score(y_te, pred, zero_division=0),
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
def train_and_evaluate(df, out_dir=OUT_DIR, make_plots=True, show=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_names = ["No Filler", "Filler Detected"]

    X, y = df[FEATURE_COLS], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"\nTrain: {len(X_train)} ({len(X_train) / len(X):.0%})   "
          f"Test: {len(X_test)} ({len(X_test) / len(X):.0%})")

    trivial = accuracy_score(y_test, (X_test["filler_count"] >= 1).astype(int))
    print(f"Accuracy of the trivial rule 'filler_count >= 1' on the test set: {trivial:.1%}")
    print("  -> the label is defined by that rule, so any model reaching about this "
          "number has only re-learned the rule.")

    # Cross-validation on the TRAINING part only
    n_splits = int(min(5, y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=RANDOM_STATE)

    results, trained = {}, {}
    for name, clf in build_models().items():
        cv_acc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy").mean()
        clf.fit(X_train, y_train)
        results[name] = score_model(clf, X_train, y_train, X_test, y_test, cv_acc)
        trained[name] = clf
        r = results[name]
        print(f"  {name:<20} CV {cv_acc:.3f}   test {r['test_acc']:.3f}   F1 {r['f1']:.3f}")

    best_name = max(results, key=lambda n: results[n]["dev_acc"])   # chosen on CV, not on test
    best_model = trained[best_name]
    print(f"\nBest model (by cross-validation): {best_name}")
    print(classification_report(y_test, best_model.predict(X_test),
                                target_names=label_names, zero_division=0))

    joblib.dump(best_model, out_dir / "best_filler_model.pkl")
    joblib.dump(FILLER_WORDS, out_dir / "filler_words.pkl")
    reloaded = joblib.load(out_dir / "best_filler_model.pkl")
    reloaded.predict(X_test.iloc[:3].values)
    print("Saved + verified: best_filler_model.pkl, filler_words.pkl")

    if make_plots:
        save_charts(results, trained, best_name, X_test, y_test, label_names,
                    "Filler detection", out_dir, show=show)
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
    df = build_dataset()
    train_and_evaluate(df)
    download_if_colab([OUT_DIR / f for f in
                       ["best_filler_model.pkl", "filler_words.pkl", "accuracy_comparison.png",
                        "precision_recall_f1.png", "overfit_gap.png", "confusion_matrix.png"]])
    print("\nAll done. Copy best_filler_model.pkl into your project's model/ folder.")


if __name__ == "__main__":
    main()
