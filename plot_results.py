from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ("NB", "LR", "SVM", "MLP")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rank(r):
    return -r["val_accuracy"], -r["val_macro_f1"], r["run_id"]


def finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def comparison(rows, destination: Path):
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.bar(x - .18, [r["val_accuracy"] for r in rows], width=.36, label="Validation accuracy")
    ax.bar(x + .18, [r["val_macro_f1"] for r in rows], width=.36, label="Validation macro-F1")
    ax.set_xticks(x, [r["model"] for r in rows])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Score")
    ax.set_title("Best configuration per model | same main split")
    for i, r in enumerate(rows):
        ax.text(i, max(r["val_accuracy"], r["val_macro_f1"]) + .025,
                f"{r['val_accuracy']:.3f}", ha="center", fontsize=9)
    ax.legend(loc="lower right")
    finish(fig, destination)


def plot_all(out: Path):
    """只读 outputs 下的结果文件绘制全部图表；数字来自真实运行记录，不做任何拟合。"""
    figures = out / "figures"
    figures.mkdir(exist_ok=True, parents=True)
    src = out / "tuning_results.json"
    if not src.exists():
        src = out / "baseline_results.json"
    if not src.exists():
        raise FileNotFoundError("先运行 draft_main.py baseline 或 all，再画图。")
    rows = read_json(src)
    best = [min([r for r in rows if r["model"] == m], key=rank)
            for m in MODELS if any(r["model"] == m for r in rows)]
    comparison(best, figures / "02_model_comparison")

    distribution = out / "class_distribution.csv"
    if distribution.exists():
        d = pd.read_csv(distribution)
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.bar(x-.2, d.train, .4, label="Training subset")
        ax.bar(x+.2, d.validation, .4, label="Validation subset")
        ax.set_xticks(x, d.target)
        ax.set_xlabel("Class label (class-name mapping not provided)")
        ax.set_ylabel("Documents")
        ax.set_title("Stratified main split | training file only")
        ax.legend()
        finish(fig, figures / "01_class_distribution")

    for name, number, key in [("NB", "03", "alpha"), ("LR", "04", "C"), ("SVM", "05", "C")]:
        selected = sorted([r for r in rows if r["model"] == name], key=lambda r: r["params"][key])
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(6.9, 4.4))
        x = [r["params"][key] for r in selected]
        ax.plot(x, [r["train_accuracy"] for r in selected], marker="o", label="Training accuracy")
        ax.plot(x, [r["val_accuracy"] for r in selected], marker="s", label="Validation accuracy")
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.set_ylim(0, 1.04)
        ax.set_xlabel(key + " (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title(name + " parameter comparison | raw unigram TF-IDF")
        ax.legend()
        finish(fig, figures / f"{number}_{name.lower()}_parameter")

    mlp_rows = [r for r in rows if r["model"] == "MLP"]
    if mlp_rows:
        labels = [f"{tuple(r['params']['hidden_layer_sizes'])}\na={r['params']['alpha']:g}" for r in mlp_rows]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.bar(np.arange(len(mlp_rows)), [r["val_accuracy"] for r in mlp_rows])
        ax.set_xticks(np.arange(len(mlp_rows)), labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Best-checkpoint validation accuracy")
        ax.set_title("MLP structure and regularization comparison")
        finish(fig, figures / "06_mlp_configurations")
        mlp = min(mlp_rows, key=rank)
        log = pd.read_csv(out / "runs" / mlp["run_id"] / "epochs.csv")
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.plot(log.epoch, log.train_ce, label="Training cross-entropy")
        ax.plot(log.epoch, log.val_ce, label="Validation cross-entropy")
        ax.axvline(mlp["best_epoch"], linestyle="--", label=f"Saved epoch: {mlp['best_epoch']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy (same definition, no regularization term)")
        ax.set_title("MLP loss history | best raw-unigram configuration")
        ax.legend()
        finish(fig, figures / "07_mlp_loss")
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.plot(log.epoch, log.train_accuracy, label="Training accuracy")
        ax.plot(log.epoch, log.val_accuracy, label="Validation accuracy")
        ax.axvline(mlp["best_epoch"], linestyle="--", label="Saved checkpoint")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.03)
        ax.set_title("MLP accuracy history")
        ax.legend()
        finish(fig, figures / "07b_mlp_accuracy")

    ablation = out / "ablation_results.json"
    if ablation.exists():
        ar = read_json(ablation)
        x = np.arange(len(ar))
        fig, ax = plt.subplots(figsize=(8.0, 4.7))
        ax.bar(x-.18, [r["val_accuracy"] for r in ar], .36, label="Validation accuracy")
        ax.bar(x+.18, [r["val_macro_f1"] for r in ar], .36, label="Validation macro-F1")
        ax.set_xticks(x, [r["feature"].replace("_", "\n", 1) for r in ar])
        ax.set_ylim(0, 1.06)
        ax.set_ylabel("Score")
        ax.set_title(f"Header cleanup x n-grams | fixed {ar[0]['model']} and feature budget")
        for i, r in enumerate(ar):
            ax.text(i, max(r["val_accuracy"], r["val_macro_f1"])+.025,
                    f"{r['val_accuracy']:.3f}", ha="center", fontsize=9)
        ax.legend(loc="lower right")
        finish(fig, figures / "08_ablation")

    stability = out / "stability_summary.csv"
    if stability.exists():
        d = pd.read_csv(stability)
        fig, ax = plt.subplots(figsize=(8.2, 4.9))
        x = np.arange(len(d))
        ax.errorbar(x, d.accuracy_mean, yerr=d.accuracy_std, fmt="o", capsize=5,
                    label="Mean +/- sample SD across 3 splits")
        ax.set_xticks(x, d.fixed_config_name, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Validation accuracy")
        ax.set_ylim(0, 1.03)
        ax.set_title("Split sensitivity (not an independent-test confidence interval)")
        ax.legend(loc="lower right", fontsize=8)
        finish(fig, figures / "09_split_stability")

    confusion = out / "selected_confusion_matrix.csv"
    if confusion.exists():
        cm = pd.read_csv(confusion)
        counts = cm.to_numpy()
        normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.9, 5.7))
        image = ax.imshow(normalized, vmin=0, vmax=1)
        fig.colorbar(image, ax=ax, label="Fraction within true class")
        ax.set_xticks(range(len(cm.columns)), cm.columns)
        ax.set_yticks(range(len(cm.columns)), cm.columns)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Selected model | main validation confusion matrix")
        finish(fig, figures / "10_confusion_matrix")

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.scatter([r["fit_seconds"] for r in best], [r["val_accuracy"] for r in best], s=65)
    for r in best:
        ax.annotate(r["model"], (r["fit_seconds"], r["val_accuracy"]),
                    textcoords="offset points", xytext=(5, 6))
    ax.set_xscale("log")
    ax.set_xlabel("Model fitting time in seconds (log scale; excludes evaluation/TF-IDF)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation performance and fitting cost")
    ax.set_ylim(0, 1.03)
    finish(fig, figures / "11_performance_vs_fit_time")

    files = sorted(p.name for p in figures.glob("*.png"))
    print(f"Generated {len(files)} PNG figures and matching SVG files in: {figures}")
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    plot_all(args.output.resolve())


if __name__ == "__main__":
    main()
