# 多层匹配滤波算法

这是一个基于匹配滤波算法和高光谱卫星传感器数据进行点源尺度甲烷浓度增强反演的项目。

## 目录

- [项目概览](#项目概览)
- [主要功能](#主要功能)
- [环境设置](#环境设置)
- [项目结构](#项目结构)
- [数据](#数据)
- [使用方法](#使用方法)
- [结果](#结果)

## 项目概览

该项目为了实现多种基于高光谱卫星成像仪的甲烷浓度增强反演，通过一系列数据读取和处理算法，反演算法和数据导出；核心功能在于不同的卫星传感器数据读取，结合不同的匹配滤波算法处理管道进行浓度增强反演。

## 主要功能

- [功能1，例如：实现基于最优估计理论的甲烷反演算法]
- [功能2，例如：支持处理 Sentinel-5P TROPOMI 卫星数据]
- [功能3，例如：提供反演结果的空间可视化和时序分析工具]
- [功能4，例如：自动化批量处理大量卫星数据]
- ...

## 环境设置

本项目推荐使用 Conda 管理环境。确保你的系统中已经安装了 Conda 。

1. **创建并激活环境:**
使用项目根目录下的 `mf_environment.yml` 文件创建并激活所需的 Conda 环境。

    ```bash
    conda env create -f mf_environment.yml
    conda activate mf_environment
    ```

这会将项目代码安装到当前环境中，方便后续 Python 环境及第三方库能正确运行。

## 项目结构

本项目的主要目录和文件说明如下：

.
├── .vscode/                  # VS Code 编辑器配置文件
├── archive/                  # 存放旧版本代码、不再使用的文件等
├── data/                     # 项目所需的输入数据存放目录
├── docs/                     # 项目相关的文档，可能包含更详细的技术文档、报告等
│   └── README.md             # 此文件或其他文档
├── figures/                  # 项目运行过程中生成的图表、可视化结果图片等输出文件
├── methane_retrieval_algorithms/ # **核心算法代码库**。包含实现各种甲烷反演算法、数据处理逻辑等的核心 Python 模块。
│   └── [模块和文件]          # 例如： retrieval_methods/, data_readers/, etc.
├── other_code/               # 存放一些辅助性、实验性或不属于核心库的脚本或代码 (请按需说明具体内容)
├── results/                  # 项目运行生成的反演结果、评估报告、统计文件等结构化输出文件
├── tasks/                    # 存放用于执行特定任务或工作流的脚本，通常是项目的主要入口点。
│   └── [任务脚本]            # 例如： run_s5p_retrieval.py, evaluate_performance.py
├── utils/                    # 存放项目内部使用的通用工具函数、辅助类或小脚本。
│   └── [工具文件]            # 例如： data_processing_utils.py, plot_helpers.py
├── .gitignore                # Git 版本控制忽略文件列表，指定哪些文件不应提交到仓库
└── mf_environment.yml        # Conda 环境定义文件，列出了项目所需的依赖库

## 数据

- **输入数据:**
  - 请将项目运行所需的原始输入数据文件（例如卫星数据、地表高程模型、先验廓线等）存放在 `data/` 目录及其子目录中。
  - **数据格式:** [明确说明期望的数据文件格式，例如：NetCDF (.nc)、HDF5 (.h5)、CSV (.csv) 等。]
  - **数据组织:** [说明数据在 `data/` 目录下的组织方式，例如：数据按日期分文件夹存放 (`data/YYYY/MM/DD/`)，或按数据类型分文件夹存放 (`data/satellite_data/`, `data/ancillary_data/`)。]
- **数据获取:**

## 使用方法

[详细说明如何运行项目的核心功能。这通常涉及运行 `tasks/` 目录下的脚本，或者说明如何在自己的脚本中导入核心库进行使用。]

1. **激活环境:** 确保你已经激活了项目的 Conda 环境：
  
    ```bash
      conda activate mf_environment
    ```

2. **运行预定义任务:** 你可以直接运行 `tasks/` 目录下的脚本来执行特定的反演或分析任务。

    ```bash
    # 示例：运行一个处理 Sentinel-5P 数据的任务脚本
    python tasks/run_s5p_retrieval.py --input data/path/to/your/s5p_data.nc --output results/path/for/output --param config/params.yaml
    ```

3. **在自定义脚本中使用核心库:** 你可以在自己的 Python 脚本中导入 `methane_retrieval_algorithms` 或 `utils` 中的模块和函数来构建更复杂的流程或进行实验。

    ```python
    # example_custom_process.py
    import os
    from methane_retrieval_algorithms.retrieval_methods import OptimalEstimation
    from methane_retrieval_algorithms.data_readers import S5PDataReader
    from utils.plotting import plot_retrieval_map
    
    # 定义输入和输出路径
    input_file = 'data/specific_s5p_file.nc'
    output_dir = 'results/custom_run/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    reader = S5PDataReader(input_file)
    data = reader.read_data()
    
    # 执行反演 (假设 OptimalEstimation 需要数据作为输入)
    retriever = OptimalEstimation()
    retrieval_result = retriever.apply(data)
    
    # 保存或处理结果
    retrieval_result.to_csv(os.path.join(output_dir, 'retrieval_output.csv'))
    
    # 绘制结果
    plot_retrieval_map(retrieval_result, os.path.join(output_dir, 'retrieval_map.png'))
    
    print("Custom retrieval process completed.")
    ```

## 甲烷反演算法

本项目所编写的所有甲烷反演算法存放在 `methane_retrieval_algorithms` 目录中

- `columnwise_matchedfilter.py`
  该函数基于原始匹配滤波算法进行浓度增强反演，以影像中的列为计算单元进行计算
- `columnwise_ml_matchedfilter.py`
  该函数基于**多层匹配滤波算法**进行浓度增强反演，以影像中的列为计算单元进行计算

  ```python
  def columnwise_ml_matched_filter(
      data_cube: np.ndarray, # 卫星数据 cube
      initial_unit_absorption_spectrum: np.ndarray, # 初始单位吸收谱 numpy 数组
      uas_list: np.ndarray, # 单位吸收谱 多维数组
      transmittance_list: np.ndarray, # 透射率多维数组
      iterate: bool = False, # 是否迭代运算，默认为否
      albedoadjust: bool = False, # 是否校正 albedo，默认为否
      sparsity: bool = False, # 是否考虑系数分布，默认为否
      group_size: int = 5, # 行计算单元尺寸，默认以五列为一组
      dynamic_adjust: bool = True,  # 新增动态调整标志
      threshold: float = 5000,  # 初始浓度增强阈值
      threshold_step: float = 5000,  # 阈值调整步长
      max_threshold: float = 50000,  # 最大浓度增强阈值
  ) -> np.ndarray:
      """Calculate the methane enhancement of the image data based on the original matched filter method.

      Args:
          data_cube (np.ndarray): 3D array representing the image data cube.
          unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
          iterate (bool): Flag indicating whether to perform iterative computation.
          albedoadjust (bool): Flag indicating whether to adjust for albedo.
          sparsity (bool): Flag for sparsity adjustment, not used here but can be implemented.
          group_size (int): The number of columns in each group to process together.

      Returns:
          np.ndarray: 2D array representing the concentration of methane.
      """
  ```
  
- `columnwise_kalmanfilter_matchedfilter.py`
  该函数基于卡曼滤波匹配滤波算法进行浓度增强反演，以影像中的列为计算单元进行计算
- `columnwise_lognormal_matchedfilter.py`
  该函数基于对数正态分布滤波算法进行浓度增强反演，以影像中的列为计算单元进行计算
- `matchedfilter.py`
  该函数基于原始匹配滤波算法进行浓度增强反演，以整副影像为计算单元进行计算
- `ml_matchedfilter.py`
  该函数基于**多层滤波匹配滤波算法**进行浓度增强反演，以整副影像为计算单元进行计算
- `kalmanfilter_matchedfilter.py`
  该函数基于卡曼滤波匹配滤波算法进行浓度增强反演，以整副影像为计算单元进行计算
- `lognormal_matchedfilter.py`
  该函数基于对数正态匹配滤波算法进行浓度增强反演，以整副影像为计算单元进行计算

## 结果

项目运行生成的输出文件将存放在 `results` 和 `figures` 目录中。

- `results/`: 存放结构化的结果数据，例如：
  - 反演得到的甲烷浓度值 (CSV, NetCDF, etc.)
  - 算法性能评估指标 (JSON, CSV)
  - 详细的反演报告或日志文件 (TXT)
  - [请具体说明这里包含的文件类型和含义]
- `figures/`: 存放可视化输出，例如：
  - 反演结果的空间分布图
