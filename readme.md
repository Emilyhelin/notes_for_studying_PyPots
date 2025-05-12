
# 📘 PyPOTS 学习笔记大纲

## 1. 项目简介

- **项目名称**：PyPOTS（Partially-Observed Time Series）
- **项目地址**：[GitHub - WenjieDu/PyPOTS](https://github.com/WenjieDu/PyPOTS)
- **项目官网**：[pypots.com](https://pypots.com/)
- **项目定位**：专注于处理部分观测时间序列数据的机器学习工具箱，支持缺失值填补、预测、分类、聚类和异常检测等任务。

## 2. 背景与动机

- **现实挑战**：在实际应用中，传感器故障、通信错误等因素常导致时间序列数据存在缺失值。
- **现有工具的不足**：缺乏专门处理部分观测时间序列的工具包。
- **PyPOTS 的使命**：提供一个易用且功能全面的工具箱，帮助工程师和研究人员专注于核心问题，而非数据预处理。

## 3. 安装与配置

### 3.1 安装方式

- 使用 pip 安装：
```bash
pip install pypots
```

- 使用 conda 安装：
```bash
conda install -c conda-forge pypots
```

- 使用 Docker 运行：
```bash
docker run -it --name pypots wenjiedu/pypots
```

### 3.2 依赖环境

- **Python 版本**：>= 3.8
- **主要依赖包**：h5py、numpy、scipy、pandas、matplotlib、torch >=1.10.0 等

## 4. 核心功能模块

### 4.1 缺失值填补（Imputation）

- 支持的模型：SAITS、BRITS、CSDI、ImputeFormer 等
- 应用场景：医疗数据、工业传感器数据等存在缺失值的时间序列数据填补。

### 4.2 时间序列预测（Forecasting）

- 支持的模型：Transformer、Informer、ETSformer、PatchTST 等
- 应用场景：金融市场预测、能源消耗预测等。

### 4.3 分类与聚类（Classification & Clustering）

- 支持的模型：TS2Vec、TEFN、TimesNet 等
- 应用场景：用户行为分析、设备状态识别等。

### 4.4 异常检测（Anomaly Detection）

- 支持的模型：SAITS、BRITS、FreTS 等
- 应用场景：工业设备故障检测、网络入侵检测等。

## 5. PyPOTS 生态系统

### 5.1 TSDB（Time Series Data Beans）

- 提供多种公开的时间序列数据集，方便用户加载和使用。

### 5.2 PyGrinder

- 用于模拟真实世界的缺失数据，可在完整数据集上引入不同类型的缺失。

### 5.3 BenchPOTS

- 为公平评估各种 POTS 算法的性能而创建的基准测试套件。

### 5.4 BrewPOTS

- 提供 PyPOTS 的使用教程，帮助用户学习如何处理 POTS 数据集。

## 6. 实战示例

### 6.1 使用 SAITS 进行缺失值填补

```python
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae

saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256)
saits.fit(dataset)
imputation = saits.impute(dataset)
mae = calc_mae(imputation, ground_truth, mask)
```

### 6.2 使用 TimesNet 进行时间序列预测

```python
from pypots.forecasting import TimesNet

model = TimesNet(n_steps=48, n_features=37)
model.fit(train_data)
forecast = model.predict(test_data)
```

## 7. 高级功能与技巧

- **超参数优化**：集成了 Microsoft NNI 框架，支持自动化的超参数调优。
- **模型评估**：提供统一的评估指标计算方法，方便模型性能比较。
- **可视化工具**：内置多种可视化工具，辅助数据分析与结果展示。

## 8. 资源与社区

- **官方文档**：[PyPOTS Documentation](https://docs.pypots.com/)
- **论文资源**：[arXiv 论文](https://arxiv.org/abs/2305.18811)
- **社区交流**：欢迎通过 GitHub Issues 提交问题或建议，参与社区讨论。
