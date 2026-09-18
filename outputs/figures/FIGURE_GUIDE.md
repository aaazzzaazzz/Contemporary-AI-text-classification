# 图表说明

所有数字来自结果文件；图中没有测试准确率。

01：训练/验证类别分布。02：各模型最佳基础特征配置。03—05：参数比较。
06：MLP 结构/正则化。07：逐轮训练与验证交叉熵；虚线为保存轮。
07b：逐轮准确率。08：消融；09：划分敏感性；10：按真实类别归一化的混淆矩阵。
11：性能与模型拟合耗时（不含 TF-IDF 和评价时间）。

主报告最多5页，不必将所有图放入正文；选择能支持实际结论的图。

01_class_distribution.png
02_model_comparison.png
03_nb_parameter.png
04_lr_parameter.png
05_svm_parameter.png
06_mlp_configurations.png
07_mlp_loss.png
07b_mlp_accuracy.png
08_ablation.png
09_split_stability.png
10_confusion_matrix.png
11_performance_vs_fit_time.png