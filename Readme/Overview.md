# MF 仓库概况{#overview}

## 目的与范围

MF (Matched Filter，匹配滤波) 仓库是一个专门用于从高光谱卫星数据中提取甲烷浓度增强信息的代码库。它实现了多种匹配滤波算法变体，特别关注多层匹配滤波 (Multi-Layer Matched Filter, MLMF) 方法，该方法可提高高甲烷浓度区域的提取精度。该系统支持多种高光谱卫星平台，并包含模拟甲烷羽流、生成单位吸收光谱和估算排放速率的工具。

有关特定甲烷提取算法的详细信息，请参阅 [甲烷反演算法](./Methaneretrieval.md)，
有关卫星数据处理的信息，请参阅 [卫星数据预处理]()。

## 系统架构

MF 仓库采用模块化架构组织，包含几个关键组件协同工作以执行甲烷浓度提取。

### 高层架构

下方是 MF 仓库的高层架构图，展示了主要组件及其交互：

### 数据处理流程

MF 仓库实现了从输入数据到甲烷提取结果的完整数据处理流程：

## 核心功能

### 甲烷提取算法

该仓库实现了几种复杂度递增的甲烷提取算法：

```
Algorithm FeaturesCore ComponentsMethane Retrieval AlgorithmsStandard Matched FilterColumnwise Matched FilterMulti-Layer Matched FilterColumnwise Multi-Layer
Matched FilterKalman Filter
(Placeholder)Lognormal
(Placeholder)Unit Absorption SpectrumRadiance Lookup TablesCovariance MatrixTransmittance SpectraIterative RefinementAlbedo AdjustmentSparsity AdjustmentDynamic Threshold
Adjustment
```

####单位吸收光谱生成

单位吸收光谱 (UAS) 是匹配滤波算法的关键组件。它表示甲烷在不同浓度下的光谱特征。系统使用辐射查找表生成 UAS 和透射光谱：

### 卫星数据支持

该系统支持多种高光谱卫星平台，提供专门的数据读取器和处理函数。

| 平台      | 数据格式    | 支持算法    |
| --------- | ----------- | ----------- |
| PRISMA    | HDF5        | MF, MLMF    |
| EnMAP     | GeoTIFF     | MF, MLMF    |
| EMIT      | NetCDF      | MF, MLMF    |
| ZY1       | DAT/HDR     | MF, MLMF    |
| GF5B-AHSI | TIFF        | MF, MLMF    |

每个卫星平台在 `utils/satellites_data` 目录下都有自己的数据读取器。

### 图像模拟

该系统包含模拟带有甲烷羽流的卫星图像的功能，这对于测试和验证提取算法非常有用：

## 仓库结构

MF 仓库按以下主要目录组织：

| 目录                       | 目的                                       |
| -------------------------- | ------------------------------------------ |
| `methane_retrieval_algorithms/` | 核心甲烷提取算法实现                       |
| `utils/`                   | 用于数据处理、UAS 生成和模拟的工具函数       |
| `utils/satellites_data/`   | 卫星数据读取器和处理工具                   |
| `utils/emission_estimate/` | 用于估算甲烷排放速率的工具                 |
| `data/`                    | 输入数据存储（查找表、卫星通道等）         |
| `results/`                 | 提取结果输出存储                           |
| `figures/`                 | 可视化输出存储                             |
| `tasks/`                   | 执行特定任务或工作流程的脚本               |
| `docs/`                    | 文档文件                                   |

## 多层匹配滤波算法 (MLMF)

该仓库的关键创新是多层匹配滤波 (MLMF) 算法，它通过处理高甲烷浓度下的非线性吸收效应，改进了标准匹配滤波。

MLMF 算法的工作原理如下：

1. 使用标准匹配滤波器计算初始甲烷浓度估算值。
2. 对于高浓度像素，动态调整处理过程。
3. 对不同浓度范围使用多层单位吸收光谱和透射光谱。
4. 根据当前估算值迭代优化结果。

MLMF 算法的核心实现在 `methane_retrieval_algorithms/ml_matchedfilter.py` 中。

主要函数 `ml_matched_filter()` 接收以下关键参数：

- `data_cube`: 卫星数据的 3D 数组
- `initial_unit_absorption_spectrum`: 用于第一遍处理的初始 UAS
- `uas_list`: 不同浓度范围的 UAS 数组
- `transmittance_list`: 不同浓度范围的透射光谱数组
- `iterate`: 迭代优化标志
- `albedoadjust`: 反照率调整标志
- `dynamic_adjust`: 动态阈值调整标志

## 工作流程示例

使用 MF 仓库的典型甲烷提取工作流程包括以下步骤：

多层匹配滤波算法的示例参数：

- 波长范围：2150-2500 nm (短波红外)
- 动态调整阈值：5000-50000 ppm·m
- 太阳天顶角：0-90 度
- 地面高度：0-5 km