# 实验一：新闻文本分类

基于 TF-IDF 特征（词表上限 5000），对训练集做 8:2 分层划分（seed 42），用朴素贝叶斯、逻辑回归、线性 SVM、MLP 四种模型进行 16 组参数调参、4 组消融与 3 个划分种子的稳定性分析，依据验证集指标锁定最佳模型，并对测试集生成预测。

## 运行环境

Python 3.11 及以上（本机验证 3.12）  
依赖见 `requirements.txt`

## 安装

```bat
conda create -n project1 python=3.11 pip -y
conda activate project1
pip install -r requirements.txt
```

## 数据

`train_data.csv`（7368 条带标签）与 `test_data_unlabeled.csv`（2457 条无标签）与本目录下脚本位于同一文件夹，程序自动读取，无需修改路径。

## 运行

按顺序执行以下命令：

```bat
python -u draft_main.py selftest
python -u draft_main.py audit
python -u draft_main.py all
python plot_results.py
python -u draft_main.py predict
```

`selftest`：程序自检，预期输出 `SELFTEST PASSED: 10 checks`  
`audit`：核对数据，预期输出 7368 条带标签数据，划分为训练 5894 / 1474  
`all`：完整训练（16 组参数调参、4 组消融、3 个划分种子的稳定性分析），结束时输出 `[方案已锁定]`  
`plot_results.py`：绘制图表，保存到 `outputs/figures/`  
`predict`：生成最终预测，必须在 `all` 完成后执行

可选用 `python -u draft_main.py smoke` 先做 1200 条数据的小规模试跑，确认环境正常后再跑 `all`。

## 主要输出

| 文件 | 内容 |
|---|---|
| `outputs/RESULTS_SUMMARY.md` | 结果汇总（选定模型与验证集指标） |
| `outputs/tuning_results.csv` | 16 组参数配置的验证集比较 |
| `outputs/ablation_results.csv` | 文本处理/特征条件消融 |
| `outputs/stability_summary.csv` | 3 个划分种子的稳定性 |
| `outputs/figures/` | 图表（PNG/SVG） |
| `outputs/predictions.csv` | 最终测试集预测（2457 行、1 列、无表头无索引） |

测试集无标签，`predictions.csv` 为选定模型对测试集的预测，本地不计算测试准确率。
