# 模拟COVID-19动态：从确定性拟合到贝叶斯推断的迭代方法

本仓库包含了 **MATH 0290: 微分方程** 的完整项目。该项目旨在通过开发、校准和分析房室模型，来描述COVID-19在德国传播的动态过程。

**GitHub Repository:** [https://github.com/AxelYoung/DE_Project](https://github.com/AxelYoung/DE_Project)

## 项目简介

本项目的核心目标是将数学建模技术应用于真实的流行病学问题。项目遵循一条迭代式的分析路径：从一个基础的 **SIR (易感-感染-康复)** 模型开始，逐步演进到一个更复杂的 **SEIAR (易感-暴露-感染-无症状-康复)** 模型，以更好地解释无症状传播等关键现实因素。

整个分析过程分为两个主要阶段：
1.  **阶段一：确定性拟合 (Deterministic Fitting):** 使用非线性最小二乘法进行初步探索，为简化模型寻找单一的最佳拟合参数。这个阶段突显了使用恒定参数来描述一个动态变化的真实世界系统的局限性。
2.  **阶段二：贝叶斯推断 (Bayesian Inference with MCMC):** 采用完整的贝叶斯方法，通过马尔可夫链蒙特卡洛 (MCMC) 算法来推断时变参数的完整后验分布。这个高级模型引入了分段变化的传播率和报告率，以精确捕捉公共卫生干预措施带来的影响。

最终，校准后的模型被用于进行 **情景分析 (Scenario Analysis)**，展示了其作为评估公共卫生策略有效性的实用工具价值。

## 仓库结构

```
.
├── 1/                            # 存放分析过程中生成的所有图表
│   ├── bayesian_fit_to_data.png
│   ├── bayesian_posterior_plots.png
│   ├── bayesian_trace_plots.png
│   ├── linear_*.png
│   └── scenario_simulation.png
├── bayesian_summary.csv          # 贝叶斯模型输出的参数统计摘要
├── data_processing.py            # 数据预处理脚本
├── plot_bayesian_results.py      # 贝叶斯结果可视化与情景分析脚本
├── script_bayesian.py            # 核心的贝叶斯MCMC分析脚本
├── appendix_scripts.tex          # 附录中引用的代码清单
├── latex.tex                     # LaTeX项目报告源文件
├── germany_covid_processed.csv   # 用于模型拟合的已处理、清洗后的数据
└── README.md                     # 本说明文件
```
*   `owid-covid-data.csv` (原始数据文件) 由于体积过大，已被 `.gitignore` 排除，未上传至本仓库。

## 如何运行分析

### 1. 环境配置

本项目所需的所有 Python 依赖库都已在 `requirements.txt` 文件中列出。

您可以通过 pip 使用以下命令一键安装所有依赖：
```bash
pip install -r requirements.txt
```

### 2. 数据准备

首先，运行数据预处理脚本，以生成用于模型拟合的德国清洁数据集。请确保原始数据文件 `owid-covid-data.csv` 位于项目根目录。

```bash
python data_processing.py
```
该命令将在根目录生成 `germany_covid_processed.csv` 文件。

### 3. 运行贝叶斯模型

接下来，运行核心的贝叶斯分析脚本。此步骤将执行 MCMC 采样，计算量较大，可能需要几分钟时间才能完成。

```bash
python script_bayesian.py
```
成功运行后，该脚本会：
*   在控制台打印参数后验分布的统计摘要。
*   将统计摘要保存到 `bayesian_summary.csv` 文件中。
*   生成所有分析图表（模型拟合图、后验分布图、诊断图等）并保存。

### 4. 可视化与情景分析

最后，您可以运行绘图脚本来复现报告中的模型拟合图和情景分析图。

```bash
python plot_bayesian_results.py
```
此脚本会读取 `bayesian_summary.csv` 的结果，重新运行模型（无采样过程，速度很快）并生成最终图表，保存在 `/1` 文件夹内。

## 主要发现

1.  **简单模型的局限性:** 具有恒定参数的简单确定性模型不足以捕捉受干预措施影响的真实世界疫情动态。
2.  **干预措施的有效性:** 分段式贝叶斯模型成功地量化了德国在封锁期间传播率 ($\beta_t$) 的显著下降，为非药物干预措施 (NPIs) 的有效性提供了数据驱动的证据。
3.  **无症状传播与报告率的作用:** 模型强调了考虑无症状传播和可变报告率 ($\rho_t$) 的重要性，表明官方病例数可能严重低估了疫情的真实规模。
4.  **情景分析的价值:** 模拟显示，将基线传播率降低20%可以显著"压平"疫情曲线并延迟其峰值，这突显了减少社交接触措施的价值。
