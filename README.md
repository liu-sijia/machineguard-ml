# MachineGuard ML

> Reproducible Equipment Fault Classification & Indicator Regression

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MachineGuard ML 是一个面向设备状态数据的可复现机器学习工具包，提供两条独立流程：使用 Logistic Regression 完成故障状态分类，以及使用 NumPy 手写 BP 神经网络完成连续指标回归。项目将原本依赖固定表格和本机路径的分析脚本重构为参数化命令行工程，可读取 CSV/Excel、自动完成数据拆分与标准化，并输出指标报告和预测结果。

**English summary:** A compact machine-learning toolkit for equipment fault classification and continuous indicator regression, featuring a scikit-learn logistic regression pipeline and a NumPy backpropagation neural network.

## 核心能力

- 支持 `.csv`、`.xlsx`、`.xls` 表格输入
- 自定义工作表、特征列、目标列和测试集比例
- 按数据顺序切分训练集与测试集，适配时序采集记录
- Logistic Regression 故障分类与标准化预处理
- NumPy 手写全连接 BP 神经网络回归
- 固定随机种子，保证训练过程可复现
- 分类输出 Accuracy、Precision、Recall、F1 和混淆矩阵
- 回归输出 MAE、MSE、RMSE、R² 和 MAPE
- 统一生成 `metrics.json` 与 `predictions.csv`

## 工程亮点

- **算法实现与任务编排分离**：手写网络、数据读取、训练流程和 CLI 各自独立。
- **动态输入维度**：不再假设固定传感器数量，可通过列名选择任意数值特征。
- **数据泄漏控制**：标准化器只在训练集拟合，再转换测试集。
- **数值稳定性**：Sigmoid 输入裁剪，MAPE 自动跳过零目标值。
- **结果可审查**：指标和逐行预测分别保存，便于复核与二次分析。
- **无数据复现**：合成数据生成器只生成结构化演示数据，不依赖私有数据集。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 开发语言 | Python 3.10+ |
| 数据处理 | pandas、NumPy |
| 分类模型 | scikit-learn Logistic Regression |
| 回归模型 | NumPy 手写 Backpropagation Neural Network |
| 输入格式 | CSV、Excel |
| 测试 | pytest |

## 工作流程

```text
CSV / Excel
     │
     ▼
字段选择与缺失值过滤
     │
     ▼
按顺序切分训练集 / 测试集
     │
     ├──► 标准化 + Logistic Regression ──► 故障分类指标
     │
     └──► 标准化 + 手写 BP 神经网络 ─────► 连续指标回归指标
                                                   │
                                                   ▼
                                  metrics.json + predictions.csv
```

## 目录结构

```text
.
├─ src/machineguard_ml/
│  ├─ cli.py                 # 命令行入口
│  ├─ data.py                # CSV/Excel 加载与字段校验
│  ├─ workflows.py           # 分类、回归与指标计算
│  └─ models/
│     └─ bp_regressor.py     # NumPy 手写 BP 神经网络
├─ examples/
│  └─ generate_demo_data.py  # 合成演示数据生成器
├─ tests/                    # 数据与训练流程测试
├─ docs/                     # 架构、数据格式和公开检查说明
├─ pyproject.toml
└─ requirements.txt
```

## 快速开始

### 1. 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 2. 生成合成演示数据

```powershell
python examples\generate_demo_data.py
```

该命令会在本地生成 `data/demo_equipment.csv`。`data/` 已加入 `.gitignore`，不会进入版本控制。

### 3. 运行故障分类

```powershell
machineguard classify `
  --input data\demo_equipment.csv `
  --target fault `
  --features load speed vibration temperature pressure `
  --output-dir output\classification
```

### 4. 运行指标回归

```powershell
machineguard regress `
  --input data\demo_equipment.csv `
  --target remaining_time `
  --features load speed vibration temperature pressure `
  --epochs 500 `
  --output-dir output\regression
```

如果使用 Excel，可增加 `--sheet Sheet1`。完整参数可通过以下命令查看：

```powershell
machineguard --help
machineguard regress --help
```

## 输出结果

每次运行生成两个文件：

- `metrics.json`：任务类型、特征列、目标列、数据规模和评估指标
- `predictions.csv`：测试集中每条记录的真实值和预测值

实际评估指标由输入数据、特征选择和训练参数共同决定。

## 数据格式

输入表格采用“一行一个观测样本、一列一个字段”的宽表结构。分类目标应为离散标签，回归目标应为连续数值。详细要求见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

## 测试

```powershell
python -m pip install -e .[dev]
pytest
```

测试使用运行时生成的合成数组与临时文件，不包含外部数据集。

## 技术边界

- 当前分类器为线性 Logistic Regression，不包含树模型或深度分类网络。
- BP 回归器采用单隐藏层结构，未实现 GPU 训练、早停或模型持久化。
- 数据拆分采用顺序留出法，未实现时间序列交叉验证。
- 当前数据读取面向二维表格，不包含实时流式采集接口。

## License

本项目基于 [MIT License](LICENSE) 开源。
