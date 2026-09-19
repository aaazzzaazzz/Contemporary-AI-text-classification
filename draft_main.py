# -*- coding: utf-8 -*-
"""实验一：新闻文本分类。

常用命令（在本文件所在文件夹运行）：
    python draft_main.py audit       # 检查数据与主划分，无拟合
    python draft_main.py selftest    # 小型自动测试
    python draft_main.py smoke       # 1200 条样本试跑，仅检查程序
    python draft_main.py baseline    # 80/20 划分，4 个基础模型
    python draft_main.py all         # 16 组调参 + 4 组消融 + 3 个划分种子；不读取测试集
    python plot_results.py           # 将真实 CSV 结果绘制成图
    python draft_main.py predict     # 完整实验结束后，加载已锁定模型

损失口径：NB/LR 拟合后输出交叉熵；SVC 输出单独命名的间隔评价损失；
MLP 每轮输出内部优化损失及同一定义的训练/验证交叉熵。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, hinge_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("NB", "LR", "SVM", "MLP")
BASE_FEATURE = "raw_uni"
FEATURES = {
    "raw_uni": {"clean": False, "ngram_range": (1, 1)},
    "clean_uni": {"clean": True, "ngram_range": (1, 1)},
    "raw_unibi": {"clean": False, "ngram_range": (1, 2)},
    "clean_unibi": {"clean": True, "ngram_range": (1, 2)},
}
BASE_PARAMS = {
    "NB": {"alpha": 1.0},
    "LR": {"C": 1.0},
    "SVM": {"C": 1.0},
    "MLP": {"hidden_layer_sizes": [100], "alpha": 0.0001},
}
# 16 组起步搜索；未穷尽 TUNING.md 中的全部候选值。
GRID = {
    "NB": [{"alpha": x} for x in (0.1, 0.5, 1.0, 2.0)],
    "LR": [{"C": x} for x in (0.1, 1.0, 10.0)],
    "SVM": [{"C": x} for x in (0.1, 1.0, 10.0)],
    "MLP": [{"hidden_layer_sizes": h, "alpha": a}
            for h in ([100], [200], [100, 50]) for a in (0.0001, 0.001)],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Cannot serialize {type(x)}")


def canonical(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=json_default)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default,
                              allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, frame: pd.DataFrame, *, header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, header=header, encoding="utf-8-sig")
    tmp.replace(path)


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "matplotlib"):
        result[name] = importlib.metadata.version(name)
    return result


def read_training(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train_data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"找不到训练文件：{path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["text", "target"]:
        raise ValueError(f"本数据应有 text、target 两列，实际为 {list(df.columns)}")
    if df["target"].isna().any() or not pd.api.types.is_integer_dtype(df["target"]):
        raise ValueError("target 必须是无缺失的整数标签。请检查原始文件，不要猜测或补造标签。")
    if df["text"].isna().any() or df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("训练文本存在缺失/空文本；请先记录并确定处理规则，程序不会悄悄删行。")
    if set(df["target"].unique()) != set(range(10)):
        raise ValueError("此程序针对已提供的 0..9 十分类数据；标签集合发生变化。")
    df["text"] = df["text"].astype(str)
    df["row_id"] = np.arange(len(df), dtype=int)  # 原训练CSV中的数据行号，从0开始
    return df


def split_rows(df: pd.DataFrame, seed: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """只在训练文件的行号内分层划分，测试行不会进入这里。"""
    tr, va = train_test_split(df["row_id"].to_numpy(), test_size=ratio,
                              random_state=seed, stratify=df["target"].to_numpy())
    assert len(set(tr) & set(va)) == 0
    assert set(tr) | set(va) == set(df["row_id"])
    return np.asarray(tr), np.asarray(va)


def subject_and_body(text: str) -> str:
    """确定性清洗：仅去掉起始邮件头的其他字段，保留 Subject 和原正文。"""
    normal = text.replace("\r\n", "\n").replace("\r", "\n")
    head, sep, body = normal.partition("\n\n")
    looks_like_headers = re.search(
        r"(?mi)^(From|Subject|Organization|Lines|Reply-To|NNTP-Posting-Host):", head)
    if not sep or not looks_like_headers:
        return normal
    msg = Parser().parsestr(head + "\n\n", headersonly=True)
    subject = str(msg.get("Subject", ""))
    return subject + "\n\n" + body


def prepare_texts(texts: list[str], feature: str) -> list[str]:
    return [subject_and_body(t) for t in texts] if FEATURES[feature]["clean"] else texts


def ranking(row: dict[str, Any]) -> tuple[float, float, str]:
    # 在运行前固定：Accuracy 优先；同分看 Macro-F1；再同分按确定性的 run_id。
    return (-float(row["val_accuracy"]), -float(row["val_macro_f1"]), row["run_id"])


def make_model(name: str, params: dict[str, Any], seed: int) -> Any:
    """按名称创建分类器。除搜索超参外统一固定：LR 用 lbfgs（稀疏文本上收敛快）；
    SVM 用线性核 + OVR（高维稀疏文本上线性核通常足够）；MLP 每次只训练 1 轮，
    由 train_mlp 手动控制早停与最佳轮恢复。"""
    if name == "NB":
        return MultinomialNB(**params)
    if name == "LR":
        return LogisticRegression(**params, solver="lbfgs", max_iter=2000,
                                  tol=1e-4, random_state=seed)
    if name == "SVM":
        return SVC(**params, kernel="linear", decision_function_shape="ovr",
                   probability=False, cache_size=512, random_state=seed)
    if name == "MLP":
        p = dict(params)
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        # 同一次训练中 RandomState 随 epoch 前进；不同运行从同一 seed 独立初始化。
        return MLPClassifier(**p, activation="relu", solver="adam", batch_size=200,
                             learning_rate_init=0.001, max_iter=1, shuffle=True,
                             early_stopping=False, random_state=np.random.RandomState(seed))
    raise ValueError(name)


def evaluate(model: Any, name: str, X: Any, y: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """全部是评价操作，不调用 fit/partial_fit。SVM 的损失与交叉熵分开命名。"""
    tic = time.perf_counter()
    pred = model.predict(X)
    predict_seconds = time.perf_counter() - tic
    if name == "SVM":
        # sklearn 的多分类 Crammer-Singer 评价形式，输入 SVC 的 OVR 分数。
        # SVC 内部采用 OVO 训练；本指标不冒称为其实际优化目标，不与交叉熵横向比值。
        loss = hinge_loss(y, model.decision_function(X), labels=model.classes_)
        kind = "cs_hinge_on_ovr_scores_eval"
    else:
        loss = log_loss(y, model.predict_proba(X), labels=model.classes_)
        kind = "cross_entropy_eval"
    if not np.isfinite(loss):
        raise FloatingPointError(f"{name} 出现非有限损失")
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=model.classes_, average="macro", zero_division=0)),
        "eval_loss": float(loss), "loss_kind": kind,
        "predict_seconds": float(predict_seconds),
    }
    return metrics, pred


def train_mlp(model: Any, Xt: Any, yt: np.ndarray, Xv: Any, yv: np.ndarray,
              run_id: str, log_path: Path, epochs: int, patience: int) -> tuple[Any, dict[str, Any]]:
    """每次 partial_fit 只喂训练子集；每轮分别评价并保存真正的最佳模型副本。"""
    best_model = None
    best_key = (-math.inf, -math.inf)
    best_accuracy = -math.inf
    best_epoch = 0
    stale = 0
    fit_seconds = 0.0
    eval_seconds = 0.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        model.partial_fit(Xt, yt, classes=np.unique(yt))
        elapsed_fit = time.perf_counter() - tic
        fit_seconds += elapsed_fit
        tic = time.perf_counter()
        mt, _ = evaluate(model, "MLP", Xt, yt)
        mv, _ = evaluate(model, "MLP", Xv, yv)
        elapsed_eval = time.perf_counter() - tic
        eval_seconds += elapsed_eval
        key = (mv["accuracy"], -mv["eval_loss"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_model = copy.deepcopy(model) 
        if mv["accuracy"] > best_accuracy + 1e-12:
            best_accuracy = mv["accuracy"]
            stale = 0
        else:
            stale += 1
        row = {
            "run_id": run_id, "epoch": epoch,
            "optimizer_loss": float(model.loss_),
            "train_ce": mt["eval_loss"], "val_ce": mv["eval_loss"],
            "train_accuracy": mt["accuracy"], "val_accuracy": mv["accuracy"],
            "val_macro_f1": mv["macro_f1"],
            "fit_seconds_this_epoch": elapsed_fit, "eval_seconds_this_epoch": elapsed_eval,
            "best_epoch_so_far": best_epoch, "epochs_without_accuracy_improvement": stale,
        }
        history.append(row)
        save_csv(log_path, pd.DataFrame(history))
        print(f"[{run_id}] epoch={epoch:03d} | optimizer_loss={model.loss_:.6f} | "
              f"train_ce={mt['eval_loss']:.6f} | val_ce={mv['eval_loss']:.6f} | "
              f"val_acc={mv['accuracy']:.4f}", flush=True)
        if stale >= patience:
            print(f"[早停] 连续 {patience} 轮验证 Accuracy 未提升；恢复第 {best_epoch} 轮。", flush=True)
            break
    if best_model is None:
        raise RuntimeError("MLP 未产生可用检查点")
    return best_model, {"fit_seconds": fit_seconds, "epoch_eval_seconds": eval_seconds,
                        "trained_epochs": len(history), "best_epoch": best_epoch,
                        "stop_reason": "patience" if stale >= patience else "epoch_limit"}


class Study:
    def __init__(self, args: argparse.Namespace, *, smoke: bool = False):
        self.args = args
        self.data_dir = args.data_dir.resolve()
        self.out = args.output.resolve() / "smoke" if smoke else args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.out / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self.smoke = smoke
        full = read_training(self.data_dir)
        if smoke and len(full) > 1200:
            selected, _ = train_test_split(full.row_id.to_numpy(), train_size=1200,
                                          random_state=9876, stratify=full.target.to_numpy())
            full = full.iloc[np.sort(selected)].copy()
        self.df = full.set_index("row_id", drop=False)
        self.data_hash = sha256(self.data_dir / "train_data.csv")
        self.code_hash = sha256(Path(__file__))
        self.versions = versions()
        self.classes = np.sort(self.df.target.unique())
        self.epochs = 3 if smoke else args.epochs
        self.patience = 3 if smoke else args.patience
        self.max_features = 1000 if smoke else args.max_features
        self.splits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.feature_cache: dict[tuple[int, str], tuple[Any, Any, Any, Any, Any, dict[str, Any]]] = {}
        self.identity = {
            "code_sha256": self.code_hash, "train_sha256": self.data_hash,
            "sample_mode": "smoke_1200" if smoke else "full_training_file",
            "split_seed": args.split_seed, "model_seed": args.model_seed,
            "val_ratio": args.val_ratio, "max_features": self.max_features,
            "max_epochs": self.epochs, "patience": self.patience,
            "threads": args.threads, "versions": self.versions,
            "selection": "main_accuracy_then_macro_f1_then_run_id",
            "mlp_checkpoint": "val_accuracy_then_lower_val_ce; patience_on_accuracy_only",
        }
        # 防止在一个目录混入不同数据/软件/超参数/代码产生的旧结果。
        protocol = self.out / "protocol.json"
        if protocol.exists() and load_json(protocol)["identity"] != self.identity:
            raise RuntimeError("输出目录已有不同配置/代码/版本的结果。请用 --output outputs_new，"
                               "不要混用旧结果。原结果不会被删除。")
        if not protocol.exists():
            save_json(protocol, {"created_utc": now(), "identity": self.identity,
                                "grid": GRID, "features": FEATURES,
                                "additional_split_seeds": [2024, 2025],
                                "test_access_during_training": False,
                                "test_example_is_ground_truth": False,
                                "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                                             "logical_cpus": os.cpu_count(), "threadpools": threadpool_info()}})

    def split(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """按种子分层划分并缓存；若发现跨划分完全相同的文本则直接报错，
        避免把本应分组的数据静默泄漏进验证集。"""
        if seed not in self.splits:
            tr, va = split_rows(self.df, seed, self.args.val_ratio)
            ttext, vtext = set(self.df.loc[tr, "text"]), set(self.df.loc[va, "text"])
            if ttext & vtext:
                raise ValueError("发现训练/验证间完全相同的原始文本。请设计分组划分后重跑，不能静默泄漏。")
            role = pd.DataFrame({"row_id": self.df.row_id.to_numpy(), "target": self.df.target.to_numpy()})
            role["role"] = np.where(role.row_id.isin(tr), "train", "validation")
            role["split_seed"] = seed
            save_csv(self.out / f"split_seed{seed}.csv", role)
            self.splits[seed] = tr, va
        return self.splits[seed]

    def audit(self) -> dict[str, Any]:
        """不训练，只核对数据与主划分并保存审计结果，用于训练前确认数据完整。"""
        tr, va = self.split(self.args.split_seed)
        training_texts = self.df.loc[tr, "text"].tolist()
        cleaned = prepare_texts(training_texts, "clean_uni")
        stats = {
            "generated_utc": now(), "sample_mode": self.identity["sample_mode"],
            "train_csv_rows_used": len(self.df), "columns": ["text", "target"],
            "class_labels": self.classes.tolist(),
            "class_names": "源文件未提供类别名称映射，报告使用 0..9，不能自行猜测。",
            "class_counts_all_labeled": self.df.target.value_counts().sort_index().to_dict(),
            "training_subset_rows": len(tr), "validation_rows": len(va),
            "training_class_counts": self.df.loc[tr, "target"].value_counts().sort_index().to_dict(),
            "validation_class_counts": self.df.loc[va, "target"].value_counts().sort_index().to_dict(),
            "missing_values": self.df[["text", "target"]].isna().sum().to_dict(),
            "raw_duplicate_texts": int(self.df.text.duplicated().sum()),
            "raw_overlap_between_train_and_validation": 0,
            "training_rows_changed_by_header_cleanup": sum(a != b for a, b in zip(training_texts, cleaned)),
            "training_empty_after_cleanup": sum(not t.strip() for t in cleaned),
            "test_file_opened_by_this_audit": False,
            "train_sha256": self.data_hash,
        }
        save_json(self.out / "data_audit.json", stats)
        save_csv(self.out / "class_distribution.csv", pd.DataFrame({
            "target": self.classes,
            "train": [int((self.df.loc[tr, "target"] == c).sum()) for c in self.classes],
            "validation": [int((self.df.loc[va, "target"] == c).sum()) for c in self.classes]}))
        print(f"[数据核对] 已标注 {len(self.df)} 条；训练子集 {len(tr)} 条；验证 {len(va)} 条；"
              "test_data_unlabeled.csv 未读取。", flush=True)
        return stats

    def features(self, seed: int, feature: str) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        """生成指定特征方案下的训练/验证矩阵。词表与 IDF 只在训练子集上拟合，
        验证集仅做 transform——防止验证信息提前泄漏。结果按 (seed, feature) 缓存。"""
        key = (seed, feature)
        if key not in self.feature_cache:
            tr, va = self.split(seed)
            tic = time.perf_counter()
            ts = prepare_texts(self.df.loc[tr, "text"].tolist(), feature)
            vs = prepare_texts(self.df.loc[va, "text"].tolist(), feature)
            prep_seconds = time.perf_counter() - tic
            vectorizer = TfidfVectorizer(max_features=self.max_features,
                                         ngram_range=FEATURES[feature]["ngram_range"],
                                         lowercase=True, stop_words=None, min_df=1,
                                         max_df=1.0, sublinear_tf=False, norm="l2", dtype=np.float64)
            tic = time.perf_counter()
            Xt = vectorizer.fit_transform(ts)  # 唯一拟合词表/IDF的位置：仅训练子集
            fit_seconds = time.perf_counter() - tic
            tic = time.perf_counter()
            Xv = vectorizer.transform(vs)
            val_transform_seconds = time.perf_counter() - tic
            overlap = set(t for t in ts if t.strip()) & set(t for t in vs if t.strip())
            stats = {"n_features": Xt.shape[1], "tfidf_fit_seconds": fit_seconds,
                     "text_prepare_seconds": prep_seconds, "val_transform_seconds": val_transform_seconds,
                     "train_zero_vectors": int((Xt.getnnz(axis=1) == 0).sum()),
                     "val_zero_vectors": int((Xv.getnnz(axis=1) == 0).sum()),
                     "clean_nonempty_overlap_count": len(overlap),
                     "tfidf_fit_rows": len(tr), "tfidf_fit_role": "training_subset_only"}
            if overlap:
                print(f"[提示] 特征方案 {feature} 清洗后有 {len(overlap)} 个非空文本跨划分相同；"
                      "已记录，不自动删样本或重划分。", flush=True)
            self.feature_cache[key] = (vectorizer, Xt, Xv,
                                      self.df.loc[tr, "target"].to_numpy(),
                                      self.df.loc[va, "target"].to_numpy(), stats)
        return self.feature_cache[key]

    def run_one(self, name: str, params: dict[str, Any], feature: str = BASE_FEATURE,
                seed: int | None = None) -> dict[str, Any]:
        """训练一个配置并保存全部产物。result.json 最后写作为"成功完成"标志；
        已完成的配置直接复用，中断的配置下次从头重训，不伪称续训。"""
        seed = self.args.split_seed if seed is None else seed
        spec = {"model": name, "params": params, "feature": feature, "split_seed": seed}
        digest = hashlib.sha256(canonical({"identity": self.identity, "spec": spec}).encode()).hexdigest()[:12]
        run_id = f"{name}_{feature}_s{seed}_{digest}"
        rd = self.runs_dir / run_id
        rd.mkdir(exist_ok=True)
        result_path, model_path = rd / "result.json", rd / "bundle.joblib"
        if result_path.exists():
            previous = load_json(result_path)
            if (previous.get("status") == "ok" and model_path.exists()
                    and (rd / "validation_predictions.csv").exists()):
                print(f"[复用已完成配置] {run_id} | val_acc={previous['val_accuracy']:.4f}", flush=True)
                return previous
        save_json(rd / "config.json", {"spec": spec, "identity": self.identity})
        print(f"\n[开始] {run_id} | params={canonical(params)}", flush=True)
        tic_run = time.perf_counter()
        captured: list[Any] = []
        try:
            vectorizer, Xt, Xv, yt, yv, feature_stats = self.features(seed, feature)
            model = make_model(name, params, self.args.model_seed)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                if name == "MLP":
                    model, fit_info = train_mlp(model, Xt, yt, Xv, yv, run_id, rd / "epochs.csv",
                                               self.epochs, self.patience)
                else:
                    tic = time.perf_counter()
                    model.fit(Xt, yt)
                    fit_info = {"fit_seconds": time.perf_counter() - tic,
                                "epoch_eval_seconds": 0.0, "trained_epochs": None,
                                "best_epoch": None, "stop_reason": "estimator_fit_returned"}
                print(f"[拟合完成] {name}，正在计算训练/验证评价损失。", flush=True)
                tic = time.perf_counter()
                mt, _ = evaluate(model, name, Xt, yt)
                mv, pv = evaluate(model, name, Xv, yv)
                final_eval_seconds = time.perf_counter() - tic
            warning_text = "\n".join(f"{w.category.__name__}: {w.message}" for w in captured)
            (rd / "warnings.txt").write_text(warning_text or "No warnings.\n", encoding="utf-8")
            if warning_text:
                print(f"[警告已保存] {warning_text}", flush=True)
            tr, va = self.split(seed)
            bundle = {"model": model, "vectorizer": vectorizer, "feature": feature,
                      "run_id": run_id, "spec": spec, "identity": self.identity,
                      "train_row_ids": tr, "validation_row_ids": va,
                      "classes": self.classes, "created_utc": now()}
            save_bundle(model_path, bundle)
            save_csv(rd / "validation_predictions.csv", pd.DataFrame({
                "row_id": va, "true_target": yv, "predicted_target": pv,
                "correct": pv == yv}))
            save_json(rd / "classification_report.json", classification_report(
                yv, pv, labels=self.classes, output_dict=True, zero_division=0))
            save_csv(rd / "confusion_matrix.csv", pd.DataFrame(confusion_matrix(yv, pv, labels=self.classes),
                                                             columns=[str(x) for x in self.classes]))
            row: dict[str, Any] = {
                "run_id": run_id, "status": "ok", "created_utc": now(), **spec,
                "params_json": canonical(params), "n_train": len(yt), "n_validation": len(yv),
                "train_accuracy": mt["accuracy"], "val_accuracy": mv["accuracy"],
                "train_macro_f1": mt["macro_f1"], "val_macro_f1": mv["macro_f1"],
                "train_eval_loss": mt["eval_loss"], "val_eval_loss": mv["eval_loss"],
                "loss_kind": mv["loss_kind"], "val_predict_seconds": mv["predict_seconds"],
                "val_predict_ms_per_sample": mv["predict_seconds"] * 1000 / len(yv),
                "final_eval_seconds": final_eval_seconds, **fit_info, **feature_stats,
                "bundle_size_mb": model_path.stat().st_size / (1024 ** 2),
                "warning_count": len(captured), "warnings": warning_text,
                "total_run_seconds": time.perf_counter() - tic_run,
                "test_used": False,
            }
            if hasattr(model, "n_iter_"):
                row["estimator_n_iter"] = np.asarray(model.n_iter_).tolist()
            if name == "SVM":
                row["n_support_vectors"] = int(model.support_vectors_.shape[0])
            if name == "MLP":
                row["n_trainable_parameters"] = sum(a.size for a in model.coefs_ + model.intercepts_)
            save_json(result_path, row)  # 最后写成功标志；中断时不会把未完成训练当成结果
            print(f"[结果] {name} | {row['loss_kind']} | train_loss={row['train_eval_loss']:.6f} | "
                  f"val_loss={row['val_eval_loss']:.6f} | val_acc={row['val_accuracy']:.4f} | "
                  f"val_macro_f1={row['val_macro_f1']:.4f}", flush=True)
            return row
        except Exception as exc:
            save_json(result_path, {"run_id": run_id, "status": "failed", "spec": spec,
                                   "error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
            raise

    def table(self, name: str, rows: list[dict[str, Any]]) -> None:
        flat = [{k: v for k, v in r.items() if k != "params"} for r in rows]
        save_csv(self.out / f"{name}.csv", pd.DataFrame(flat))
        save_json(self.out / f"{name}.json", rows)

    def baseline(self) -> list[dict[str, Any]]:
        """先用四个模型的默认参数跑通流程，确认环境与管线正常，再进入调参。"""
        self.audit()
        rows = [self.run_one(m, BASE_PARAMS[m]) for m in MODEL_NAMES]
        self.table("baseline_results", rows)
        return rows

    def tune(self) -> list[dict[str, Any]]:
        """按 GRID 跑 16 组参数配置。组间比较规则 ranking 在实验前固定：
        验证 Accuracy 优先，同分看 Macro-F1，再同分按 run_id 保证确定性。"""
        self.audit()
        rows = [self.run_one(m, p) for m in MODEL_NAMES for p in GRID[m]]
        self.table("tuning_results", rows)
        best = [min([r for r in rows if r["model"] == m], key=ranking) for m in MODEL_NAMES]
        self.table("best_by_model", best)
        baseline = [next(r for r in rows if r["model"] == m and r["params"] == BASE_PARAMS[m])
                    for m in MODEL_NAMES]
        self.table("baseline_results", baseline)
        return rows

    def ablation(self) -> list[dict[str, Any]]:
        """固定调参最优的分类器与参数，只改变文本清洗与 n-gram 条件；
        一次只研究一个因素——特征方案。"""
        source = self.out / "tuning_results.json"
        if not source.exists():
            raise RuntimeError("请先完成 tune 或直接运行 all。消融需要先确定固定分类器。")
        tuning = load_json(source)
        if len(tuning) != 16:
            raise RuntimeError("主参数实验未完整完成，不能开始选择消融分类器。")
        base = min(tuning, key=ranking)
        rows = [base]
        for feature in ("clean_uni", "raw_unibi", "clean_unibi"):
            rows.append(self.run_one(base["model"], base["params"], feature))
        self.table("ablation_results", rows)
        return rows

    def stability(self) -> list[dict[str, Any]]:
        """把各模型最优配置在 3 个划分种子（主 42 + 2024/2025）上重跑，
        检验结论对数据划分的敏感性。三个划分的样本有重叠，标准差仅代表划分波动。"""
        bp, ap = self.out / "best_by_model.json", self.out / "ablation_results.json"
        if not bp.exists() or not ap.exists():
            raise RuntimeError("先完成 tune 和 ablation，或直接运行 all。")
        configs = load_json(bp)
        selected_ablation = min(load_json(ap), key=ranking)
        if selected_ablation["run_id"] not in {r["run_id"] for r in configs}:
            configs.append(selected_ablation)
        rows = []
        seeds = list(dict.fromkeys([self.args.split_seed, 2024, 2025]))
        if len(seeds) != 3:
            raise ValueError("主划分种子请勿设置为 2024/2025，二者用于补充稳定性检查。")
        for config in configs:
            config_name = config["model"] + "_" + config["feature"]
            for seed in seeds:
                result = self.run_one(config["model"], config["params"], config["feature"], seed)
                rows.append({**result, "fixed_config_name": config_name})
        self.table("stability_results", rows)
        summaries = []
        for name in sorted({r["fixed_config_name"] for r in rows}):
            selected = [r for r in rows if r["fixed_config_name"] == name]
            summaries.append({"fixed_config_name": name, "n_splits": len(selected),
                              "accuracy_mean": np.mean([r["val_accuracy"] for r in selected]),
                              "accuracy_std": np.std([r["val_accuracy"] for r in selected], ddof=1),
                              "macro_f1_mean": np.mean([r["val_macro_f1"] for r in selected]),
                              "macro_f1_std": np.std([r["val_macro_f1"] for r in selected], ddof=1)})
        save_csv(self.out / "stability_summary.csv", pd.DataFrame(summaries))
        save_json(self.out / "stability_summary.json", summaries)
        return rows

    def lock_and_analyze(self) -> dict[str, Any]:
        """在主验证集上按预先固定的规则锁定最终模型，复制其产物到 outputs 顶层"""
        if self.smoke:
            raise RuntimeError("smoke 只能查程序，禁止锁定提交模型。")
        tp, ap, sp = [self.out / f"{x}.json" for x in
                      ("tuning_results", "ablation_results", "stability_results")]
        if not all(p.exists() for p in (tp, ap, sp)):
            raise RuntimeError("必须完成主参数实验、消融和稳定性检查；请运行 all。")
        tuning, ablation, stability = load_json(tp), load_json(ap), load_json(sp)
        if len(tuning) != 16 or len(ablation) != 4 or len(stability) not in (12, 15):
            raise RuntimeError("结果数量不完整。请检查失败记录，不能跳过缺失配置直接提交。")
        candidates = {r["run_id"]: r for r in tuning + ablation}
        selected = min(candidates.values(), key=ranking)
        source = self.runs_dir / selected["run_id"]
        shutil.copy2(source / "bundle.joblib", self.out / "selected_model.joblib")
        manifest = {"locked_utc": now(), "identity": self.identity, "selected": selected,
                    "selection_data": "main_validation_only", "main_split_seed": self.args.split_seed,
                    "validation_refitted": False, "test_read_or_fitted": False,
                    "tuning_configurations": 16, "ablation_conditions": 4,
                    "unique_main_fits": len(candidates), "stability_entries": len(stability),
                    "note": "消融 A 与主调参复用；稳定性主种子复用，不重复统计为新增拟合。"}
        save_json(self.out / "selected_model.json", manifest)
        for name in ("classification_report.json", "confusion_matrix.csv", "validation_predictions.csv"):
            shutil.copy2(source / name, self.out / ("selected_" + name))
        pred = pd.read_csv(source / "validation_predictions.csv")
        errors = pred[pred.true_target != pred.predicted_target].copy()
        errors["text"] = self.df.loc[errors.row_id.to_numpy(), "text"].to_numpy()
        save_csv(self.out / "validation_errors.csv", errors)
        pairs = (errors.groupby(["true_target", "predicted_target"]).size()
                 .rename("count").reset_index().sort_values(
                     ["count", "true_target", "predicted_target"], ascending=[False, True, True]))
        save_csv(self.out / "confused_pairs.csv", pairs)
        # 每个高频错误类别对先取一个样本；不按“容易解释”程度手挑。
        examples = []
        for _, p in pairs.head(10).iterrows():
            e = errors[(errors.true_target == p.true_target) &
                       (errors.predicted_target == p.predicted_target)].sort_values("row_id").iloc[0]
            examples.append(e.to_dict())
        save_csv(self.out / "error_examples_for_review.csv", pd.DataFrame(examples))
        print(f"[方案已锁定] {selected['run_id']}，尚未读取测试集。", flush=True)
        return selected

    def predict(self) -> None:
        """加载锁定模型生成最终提交 CSV。这是整个程序唯一读取测试 CSV 的位置，
        只做 transform + predict，绝不拟合；输出单列、无表头、无索引。"""
        manifest_path = self.out / "selected_model.json"
        if not manifest_path.exists():
            raise RuntimeError("没有已锁定的完整实验。请先运行 python draft_main.py all。")
        manifest = load_json(manifest_path)
        if manifest["identity"] != self.identity or self.smoke:
            raise RuntimeError("锁定模型与当前配置、数据或软件版本不一致。")
        bundle = joblib.load(self.out / "selected_model.joblib")
        if bundle["identity"] != self.identity:
            raise RuntimeError("模型文件与清单不一致。")
        path = self.data_dir / "test_data_unlabeled.csv"
        test = pd.read_csv(path)  # 整个实验控制器唯一读取真实测试 CSV 的位置
        if list(test.columns) != ["text"]:
            raise ValueError("本实验的测试文件应只有 text 列。不要把标签文件传进来。")
        text = test["text"].fillna("").astype(str).tolist()  # 不删任何测试行
        X = bundle["vectorizer"].transform(prepare_texts(text, bundle["feature"]))
        pred = bundle["model"].predict(X).astype(int)
        if len(pred) != len(test) or not set(pred).issubset(set(self.classes)):
            raise AssertionError("预测行数或标签范围不正确")
        output = self.out / "predictions.csv"
        pd.DataFrame(pred).to_csv(output, index=False, header=False, encoding="utf-8")
        check = pd.read_csv(output, header=None)
        assert check.shape == (len(test), 1)
        assert np.array_equal(check.iloc[:, 0].to_numpy(), pred)
        save_json(self.out / "prediction_check.json", {
            "created_utc": now(), "test_rows": len(test), "prediction_rows": len(pred),
            "prediction_columns": 1, "header": False, "index": False,
            "original_test_row_order_preserved": True,
            "predicted_labels": sorted(set(int(x) for x in pred)),
            "missing_test_texts_filled_with_empty_string": int(test.text.isna().sum()),
            "test_sha256": sha256(path), "prediction_sha256": sha256(output),
            "selected_run_id": bundle["run_id"], "fit_called_during_prediction": False,
            "test_accuracy": "unavailable: test has no ground-truth labels",
            "example_predictions_used_as_labels": False,
        })
        print(f"[完成] {output}：{len(pred)} 行，1 列，无表头、无索引。"
              "测试文件没有真实标签，未计算测试准确率。", flush=True)


def selftest() -> None:
    """针对实现约束的小型自动测试；数据为自建玩具样本，不冒充正式结果。"""
    checks = []
    raw = "From: alice@example.invalid\nSubject: computer memory\nOrganization: Example\n\nbody survives"
    cleaned = subject_and_body(raw)
    assert cleaned == "computer memory\n\nbody survives"
    checks.append("清洗保留 Subject/正文，去除其他起始邮件头")
    assert subject_and_body("plain body without headers") == "plain body without headers"
    checks.append("没有邮件头的文本不会误删正文")
    v = TfidfVectorizer()
    v.fit_transform(["trainingalpha baseword", "trainingbeta baseword"])
    X = v.transform(["validationonlysentinel baseword"])
    assert "validationonlysentinel" not in v.vocabulary_ and X.shape[0] == 1
    checks.append("验证专属词不会进入仅训练拟合的词表")
    toy = pd.DataFrame({"row_id": np.arange(100), "target": np.tile(np.arange(10), 10)})
    a, b = split_rows(toy, 42, 0.2)
    a2, b2 = split_rows(toy, 42, 0.2)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert len(a) == 80 and len(b) == 20 and set(a).isdisjoint(b)
    checks.append("分层划分可重复，训练/验证行号互斥")
    texts = [f"classword{c} classword{c} shared doc{x}" for c in range(10) for x in range(8)]
    y = np.repeat(np.arange(10), 8)
    ti, vi = train_test_split(np.arange(80), test_size=.25, stratify=y, random_state=42)
    v = TfidfVectorizer()
    Xt = v.fit_transform([texts[i] for i in ti])
    Xv = v.transform([texts[i] for i in vi])
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        for name in MODEL_NAMES:
            m = make_model(name, BASE_PARAMS[name], 42)
            if name == "MLP":
                m.set_params(batch_size=min(200, len(ti)))
                m, info = train_mlp(m, Xt, y[ti], Xv, y[vi], "SELFTEST", p / "epochs.csv", 3, 3)
                h = pd.read_csv(p / "epochs.csv")
                val, _ = evaluate(m, name, Xv, y[vi])
                best = h.iloc[info["best_epoch"] - 1]
                assert abs(val["eval_loss"] - best.val_ce) < 1e-10
                assert abs(val["accuracy"] - best.val_accuracy) < 1e-10
                checks.append("MLP 返回的是保存的最佳轮模型，指标与该轮日志一致")
            else:
                m.fit(Xt, y[ti])
            metric, pred = evaluate(m, name, Xv, y[vi])
            assert np.isfinite(metric["eval_loss"]) and len(pred) == len(vi)
            save_bundle(p / "roundtrip.joblib", {"model": m, "vectorizer": v})
            restored = joblib.load(p / "roundtrip.joblib")
            assert np.array_equal(restored["model"].predict(Xv), pred)
            checks.append(f"{name} 的损失有限且保存/加载预测一致")
        pd.DataFrame([1, 2, 0]).to_csv(p / "pred.csv", header=False, index=False)
        assert (p / "pred.csv").read_text().splitlines() == ["1", "2", "0"]
        checks.append("提交 CSV 单列、无表头、无索引")
    print("\nSELFTEST PASSED: " + str(len(checks)) + " checks")
    for s in checks:
        print("PASS: " + s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="baseline", choices=[
        "audit", "selftest", "smoke", "baseline", "tune", "ablation", "stability", "all", "predict"])
    parser.add_argument("--data-dir", type=Path, default=ROOT,
                        help="原始 CSV 所在文件夹，默认脚本所在文件夹")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="实验结果文件夹")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threads", type=int, default=1, help="固定数值计算线程，减少过度并行与计时差异")
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1 or min(args.epochs, args.patience, args.max_features, args.threads) < 1:
        parser.error("比例必须在 (0,1)，epoch/patience/特征数/线程数必须大于零。")
    if args.split_seed in (2024, 2025):
        parser.error("2024、2025 为补充划分种子；主划分请用其他值（默认42）。")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    with threadpool_limits(limits=args.threads):
        if args.action == "selftest":
            selftest()
            return
        study = Study(args, smoke=args.action == "smoke")
        if args.action == "audit":
            study.audit()
        elif args.action in ("baseline", "smoke"):
            study.baseline()
        elif args.action == "tune":
            study.tune()
        elif args.action == "ablation":
            study.ablation()
        elif args.action == "stability":
            study.stability()
            study.lock_and_analyze()
        elif args.action == "all":
            study.tune()
            study.ablation()
            study.stability()
            study.lock_and_analyze()
        elif args.action == "predict":
            study.predict()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n运行已中断。已完成的配置可复用；未完成配置下次从头训练，不伪称逐轮续训。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[失败] {exc}\n检查路径/环境以及 outputs/runs 下的失败记录。", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
