# 多层匹配滤波算法

这是一个基于匹配滤波算法和高光谱卫星传感器数据进行点源尺度甲烷浓度增强反演的项目。

## 目录

- [项目概览](#项目概览)
- [环境设置](#环境设置)
- [项目结构](#项目结构)
- [内部数据](#内部数据)
- 主要功能
  - [卫星数据预处理](#卫星数据预处理)
  - [单位吸收谱构建](#单位吸收谱构建)
  - [甲烷浓度增强反演算法](#甲烷浓度增强反演算法)
  - [烟羽排放速率估算](#烟羽排放速率估算)

- 使用示例
  - [甲烷浓度增强反演流程示例](#甲烷浓度增强反演流程示例)
  - [点源排放速率估算流程示例](#点源排放速率估算流程示例)

## 项目概览

该项目为了实现多种基于高光谱卫星成像仪的甲烷浓度增强反演，通过一系列数据读取和处理算法，反演算法和数据导出；核心功能在于不同的卫星传感器数据读取，结合不同的匹配滤波算法处理管道进行浓度增强反演。同时，项目中提供了基于分割提取后的烟羽数据进行排放速率估算的算法。

## 环境设置

本项目推荐使用 Conda 管理环境。确保你的系统中已经安装了 Conda ，推荐使用 miniforge3 作为 Conda 环境和包管理器。

 **创建并激活环境:**
使用项目根目录下的 `mf_environment.yml` 文件创建并激活所需的 Conda 环境。

  ```bash
    conda env create -f mf_environment.yml
    conda activate mf_environment
  ```

这样能设置一份与本人在代码编写与调试时完全一致的 Python 环境，方便后续正确运行。

## 项目结构

本项目的主要目录和文件说明如下：

```
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
```

## 内部数据

在`data`目录下存放了项目运行时所需的内部数据文件，或者一些中间结果文件，包括辐亮度查找表和卫星传感器光谱通道信息。

- **辐亮度查找表**
`\data\looluptables\` 目录下存储了对应不同高光谱卫星传感器的辐亮度查找表，存储格式为 npz。
- **卫星传感器光谱通道信息**
`\data\satellite_channels\` 目录下存储了对应不同高光谱卫星传感器的短波红外通道信息，包括中心波长和 FWHM（Full Width at Half Maximum），存储格式为 npz。

## 卫星数据预处理

卫星数据预处理是在建立在卫星原始数据和甲烷浓度增强算法之间一座重要的桥梁。本项目能实现对不同高光谱成像仪卫星数据的读取和预处理，为甲烷浓度增强反演做准备。

### 支持的卫星平台与数据格式

| 平台 (Platform) | 数据格式 (Data Format) | 支持算法 (Supported by) |
|:---|:---|:---|
| PRISMA | HDF5 | MF, MLMF |
| EnMAP | GeoTIFF | MF, MLMF |
| EMIT | NetCDF | MF, MLMF |
| ZY1 | DAT/HDR | MF, MLMF |
| GF5B-AHSI | TIFF | MF, MLMF |

### 支持的卫星数据预处理函数

在 `\utils\satellites_data\` 目录下存放了用于对各种高光谱成像仪卫星数据进行预处理的代码文件，包含一些通用的函数以及对应不同特定传感器的专用处理函数。

- `general_functions.py`
通用函数，以下列举该文件中部分函数的作用和输入及输出参数解释。

  ```python
    def read_tiff_in_numpy(filepath:str) -> np.ndarray:
      """
      读取一个 TIFF 文件，并返回所有通道的 numpy 数组。

      :param filepath: TIFF 文件的路径
      :return: 一个 3D numpy array，形状为 (bands, height, width)
      """
    
    def save_ndarray_to_tiff(data_array:np.ndarray, output_path:str, reference_filepath:str = None):
      """
      将 numpy 数组导出为 TIFF 文件, 可选项：基于参考 TIFF 文件设置同样的地理参考。

      :param data_array: 将要导出的 numpy 数组
      :param outputpath: TIFF 文件的导出路径
      :param reference_filepath: (Optional) 地理参考 TIFF 文件的路径
      """

    def gaussian_response_weights(center: float, fwhm: float, coaser_wavelengths: np.ndarray) -> np.ndarray:
      """
      基于高斯分布构建光谱响应函数的权重，用于进行光谱卷积。

      :param center: 当前通道的中心波长信息
      :param fwhm: 对应当前通道的 FWHM 信息
      :param coaser_wavelengths: 波长的 numpy 列表
      :return: 对应原始波长的光谱响应权重 numpy 数组
      """

    def convolute_into_higher_spectral_res(center_wavelengths: np.ndarray,fwhms: np.ndarray,raw_wvls: np.ndarray,raw_data: np.ndarray,) -> np.ndarray:
      """
      基于特定的中心波长和 FWHM 数组，对原始光谱数据进行光谱卷积
      Perform convolution of raw data with Gaussian response functions for specific center wavelengths and FWHMs.

      :param center_wavelengths: 中心波长 numpy 数组
      :param fwhms: FWHMs numpy 数组
      :param raw_wvls: 原始 raw 波长 numpy 数组
      :param raw_data: 原始数据 numpy 数组
      :return: 光谱卷积后的数据 numpy 数组
      """
    
    def load_satellite_channels(channel_path: str, lower_wavelength: float = 1000, upper_wavelength: float = 2500) -> tuple[np.ndarray, np.ndarray]:
      """
      基于光谱通道信息文件的路径，加载特定波长范围内的光谱通道中心波长和 FWHMs 数组，默认范围为 1000 nm - 2500 nm。

      :param path: 卫星光谱信息文件路径
      :param lower_wavelength: 波长范围的下限，默认为 1000。
      :param upper_wavelength: 波长范围的上限，默认为 2500。
      :return: Tuple of central wavelengths and FWHMs within the specified range
      """

  ```

## 单位吸收谱构建

这部分是单位吸收谱的构建流程和相关函数。

## 甲烷浓度增强反演算法

本项目所编写的所有甲烷反演算法存放在 `methane_retrieval_algorithms` 目录中，其中**原始匹配滤波算法**和**多层匹配滤波算法**编写和运行经过了完整验证,剩余两个匹配滤波算法变体后续看情况进行完善。

- **匹配滤波算法**
  匹配滤波算法是广泛使用的高光谱成像仪辐亮度甲烷排放反演算法。
  - `methane_retrieval_algorithms\matchedfilter.py`
    `methane_retrieval_algorithms\columnwise_matchedfilter.py`
    两个文件分别对应整幅影像进行计算和将影像以列组合为计算单位进行计算

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
          """
          基于多层匹配滤波算法和短波红外波段的卫星大气层顶辐亮度观测进行甲烷浓度增强的反演。

          Args:
              data_cube (np.ndarray): 短波红外波段卫星观测影像的 3 维数组
              initial_unit_absorption_spectrum (np.ndarray): 用于初始计算的单位吸收谱 1 维数组，要求维度与 data_cube 的光谱维度一致
              iterate (bool): Flag indicating whether to perform iterative computation.
              albedoadjust (bool): Flag indicating whether to adjust for albedo.
              sparsity (bool): Flag for sparsity adjustment, not used here but can be implemented.
              group_size (int): The number of columns in each group to process together.
          Returns:
              np.ndarray: 甲烷浓度增强结果，格式为 2 维 numpy 数组。
          """
      ```

- **多层匹配滤波算法**
  多层匹配滤波算法基于多次线性拟合逼近指数衰减的思想，对传统匹配滤波算法进行改进得到更精确的反演结果。
  - `methane_retrieval_algorithms\ml_matchedfilter.py`
    `methane_retrieval_algorithms\columnwise_ml_matchedfilter.py`
    两个文件分别对应整幅影像为计算单位和以影像中的列组合为计算单位进行计算的多层匹配滤波算法

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
          """
          基于多层匹配滤波算法和短波红外波段的卫星大气层顶辐亮度观测进行甲烷浓度增强的反演。

          Args:
              data_cube (np.ndarray): 短波红外波段卫星观测影像的 3 维数组
              initial_unit_absorption_spectrum (np.ndarray): 用于初始计算的单位吸收谱 1 维数组，要求维度与 data_cube 的光谱维度一致
              iterate (bool): Flag indicating whether to perform iterative computation.
              albedoadjust (bool): Flag indicating whether to adjust for albedo.
              sparsity (bool): Flag for sparsity adjustment, not used here but can be implemented.
              group_size (int): The number of columns in each group to process together.
          Returns:
              np.ndarray: 甲烷浓度增强结果，格式为 2 维 numpy 数组。
          """
      ```

以下是其他两项研究提出的匹配滤波算法的变体，具体函数体还未补充完整。

- kalman-filter 匹配滤波算法
  - `methane_retrieval_algorithms\kalmanfilter_matchedfilter.py`
  - `methane_retrieval_algorithms\columnwise_kalmanfilter_matchedfilter.py`

- lognormal 匹配滤波算法
  - `methane_retrieval_algorithms\lognormal_matchedfilter.py`
  - `methane_retrieval_algorithms\columnwise_lognormal_matchedfilter.py`

## 烟羽排放速率估算

在`utils\emission_estimate`目录下，存放了用于对甲烷排放烟羽 TIFF 文件进行排放速率估算的各种算法代码。本研究使用的方法主要是 Integrated Mass Enhancement 方法。

- `Integrated_mass_enhancement.py`

## 甲烷浓度增强反演流程示例

## 点源排放速率估算流程示例

## 结果

项目运行生成的输出文件将存放在 `results` 和 `figures` 目录中。

- `results/`: 存放结构化的结果数据，例如：
  - 反演得到的甲烷浓度值 (CSV, NetCDF, etc.)
  - 算法性能评估指标 (JSON, CSV)
  - 详细的反演报告或日志文件 (TXT)
- `figures/`: 存放可视化输出，例如：
  - 反演结果的空间分布图
