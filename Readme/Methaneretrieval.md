# 甲烷反演算法

## 目的与范围

本文提供了 MF 仓库中实现的甲烷反演算法的技术文档。这些算法用于从高光谱卫星图像中检测和量化甲烷浓度。有关卫星数据处理的信息，请参阅 [Satellite Data Processing](https://deepwiki.com/Emeric53/MF/3-satellite-data-processing)；有关单位吸收光谱生成的详细信息，请参阅 [Unit Absorption Spectrum Generation](https://deepwiki.com/Emeric53/MF/4-unit-absorption-spectrum-generation)。

## 算法概述

仓库实现了四种主要的反演算法，其复杂性逐步增加：

1. 标准匹配滤波 (Standard Matched Filter, MF)
2. 多层匹配滤波 (Multi-Layer Matched Filter, MLMF)
3. 列式匹配滤波 (Columnwise Matched Filter, CMF)
4. 列式多层匹配滤波 (Columnwise Multi-Layer Matched Filter, CMLMF)

这些算法利用甲烷在短波红外 (SWIR) 区域（通常为 2100-2500 nm）独特的吸收特征。

### 甲烷反演算法层级结构

## 标准匹配滤波 (MF)

标准匹配滤波是系统的基础算法，实现在 `matchedfilter.py` 中。

### 算法流程图

### 实现细节

`matched_filter` 函数接受卫星数据立方体和单位吸收光谱作为输入，并输出一个甲烷浓度值的二维数组：

```python
def matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
) -> np.ndarray
```

算法的关键步骤：

1. **背景计算 (Background Calculation)**：计算所有像素的平均光谱。
2. **目标光谱创建 (Target Spectrum Creation)**：将背景光谱乘以单位吸收光谱。
3. **协方差计算 (Covariance Calculation)**：从辐射度差计算协方差矩阵。
4. **反照率调整 (Albedo Adjustment)**：可选地调整地表反射率的变化。
5. **浓度计算 (Concentration Computation)**：应用匹配滤波方程计算浓度。
6. **迭代优化 (Iterative Refinement)**：可选地通过多次迭代提高精度。

## 多层匹配滤波 (MLMF)

多层匹配滤波在标准方法的基础上进行了增强，以更好地处理高甲烷浓度，实现在 `ml_matchedfilter.py` 中。

### 实现细节

`ml_matched_filter` 函数通过多层处理扩展了标准算法：

```python
def ml_matched_filter(
    data_cube: np.ndarray,
    initial_unit_absorption_spectrum: np.ndarray,
    uas_list: np.ndarray,
    transmittance_list: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
    dynamic_adjust: bool = True,
    threshold: float = 5000,
    threshold_step: float = 5000,
    max_threshold: float = 50000,
) -> np.ndarray
```

主要增强功能：

1. **多层 UAS (Multiple UAS Layers)**：对不同的浓度范围使用不同的单位吸收光谱。
2. **透射率建模 (Transmittance Modeling)**：考虑高浓度下的非线性效应。
3. **动态阈值调整 (Dynamic Threshold Adjustment)**：逐步处理更高浓度的像素。
4. **分层处理 (Layer-by-Layer Processing)**：对每个浓度范围应用适当的 UAS。

函数 `generate_uas_transmittance_list` 生成不同浓度层所需的 UAS 和透射率数组：

``` python
def generate_uas_transmittance_list(satelitetype, sza, altitude)
```

## 逐列处理方法

两种算法的逐列变体旨在通过按列组处理数据，更有效地处理大型图像。

### 处理策略

### 实现细节

#### 逐列匹配滤波 (Columnwise Matched Filter)

```python
def columnwise_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
    group_size: int = 5,
) -> np.ndarray
```

#### 逐列多层匹配滤波 (Columnwise Multi-Layer Matched Filter)

```python
def columnwise_ml_matched_filter(
    data_cube: np.ndarray,
    initial_unit_absorption_spectrum: np.ndarray,
    uas_list: np.ndarray,
    transmittance_list: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
    group_size: int = 5,
    dynamic_adjust: bool = True,
    threshold: float = 5000,
    threshold_step: float = 5000,
    max_threshold: float = 50000,
) -> np.ndarray
```

两个函数都执行以下操作：

1. 将数据立方体分割成大小为 `group_size` 的列组。
2. 使用相应的非逐列原始算法独立处理每个组。
3. 将处理后的组合并成完整的浓度图。

这种方法减少了内存需求，并允许潜在的并行化。

## 单位吸收光谱生成

单位吸收光谱 (UAS) 是一个关键输入，表示甲烷在不同浓度下的光谱特征。

### UAS 生成过程



### 实现细节

`generate_radiance_lut_and_uas.py` 中的 UAS 生成函数包括：

```python
def generate_satellite_uas_for_specific_range_from_lut(
    satellite_name: str,
    start_enhancement: float,
    end_enhancement: float,
    lower_wavelength: float,
    upper_wavelength: float,
    sza: float,
    altitude: float,
)
```

此函数执行以下操作：

1. 创建一个甲烷浓度范围。
2. 从查找表中检索每个浓度的辐射度值。
3. 计算辐射度值的自然对数。
4. 对对数辐射度与浓度进行线性拟合。
5. 将此斜率作为单位吸收光谱返回。

查找表是根据 MODTRAN 辐射传输模型的结果生成的。

## 算法参数和性能特性

### 算法通用参数

| 参数                       | 类型         | 描述                     | 影响                             |
| :------------------------- | :----------- | :----------------------- | :------------------------------- |
| `data_cube`                | np.ndarray   | 3D 高光谱图像数据        | 主要输入数据                     |
| `unit_absorption_spectrum` | np.ndarray   | 甲烷的光谱特征           | 定义检测目标                     |
| `iterate`                  | bool         | 启用迭代优化             | 提高精度，但计算成本更高         |
| `albedoadjust`             | bool         | 考虑地表反射率变化       | 在非均匀地表上获得更好的结果     |
| `sparsity`                 | bool         | 调整稀疏羽流特征         | 增强点源检测能力                 |

### 多层特有参数

| 参数             | 类型       | 描述                       | 影响                         |
| :--------------- | :--------- | :------------------------- | :--------------------------- |
| `uas_list`       | np.ndarray | 不同浓度范围的多层 UAS     | 更好地处理浓度变化           |
| `transmittance_list` | np.ndarray | 不同浓度的透射率数据       | 考虑非线性效应               |
| `dynamic_adjust` | bool       | 启用多层处理               | 提高高浓度检测能力           |
| `threshold`      | float      | 初始浓度阈值               | 分层处理的起始点             |
| `threshold_step` | float      | 阈值增量                   | 控制分层处理的进展           |
| `max_threshold`  | float      | 最大浓度阈值               | 多层处理的上限               |

### 列式处理参数

| 参数         | 类型   | 描述                         | 影响                     |
| :----------- | :----- | :--------------------------- | :----------------------- |
| `group_size` | int    | 一起处理的列数               | 内存使用与组边界空间信息的权衡 |

## 算法选择指南

根据你的具体需求选择合适的算法：

| 算法             | 何时使用                               | 优点                         | 局限性                       |
| :--------------- | :------------------------------------- | :--------------------------- | :--------------------------- |
| 标准 MF          | 初始检测、小图像、均匀低浓度           | 速度更快、实现更简单         | 对高浓度区域精度较低         |
| 多层 MF          | 量化分析、宽浓度范围                   | 在不同浓度范围内精度更高     | 计算需求更高                 |
| 列式 MF          | 大图像、内存受限                       | 内存使用量低                 | 可能在组边界失去一些空间上下文 |
| 列式 MLMF        | 具有不同浓度的大图像                   | 结合了内存效率和精度         | 计算量最大                   |

为获得最佳结果：

1. 使用标准 MF 进行初始检测。
2. 如果存在高浓度，切换到 MLMF。
3. 对于超出内存限制的大图像，使用列式变体。
4. 根据场景特征和精度要求调整参数。

## 算法与支持组件之间的数据流
