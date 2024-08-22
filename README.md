# 项目文档
该项目为甲烷点源尺度浓度反演和排放量量化的研究项目\
使用方法为匹配滤波算法,使用的数据为高光谱传感器辐亮度数据。 

## 项目分支
1.[版块介绍](#板块介绍)
- 1.1 [数据处理](#数据处理)
  * 1.1.1 [数据预处理](#数据预处理)
  * 1.1.2 [modtran模拟](#modtran模拟)
  * 1.1.3 [卫星数据读取](#卫星数据读取)
- 1.2 [甲烷点源监测全流程](#甲烷点源监测)
  * 1.2.1 [甲烷浓度反演](#浓度反演)
  * 1.2.2 [甲烷烟羽筛选](#烟羽识别)
  * 1.2.3 [甲烷排放量计算](#排放估算)
- 1.3 [数据可视化](#数据可视化)

2.[后续跟进内容](#后续跟进内容)


## 版块介绍
接下来将对项目的各个组成部分进行介绍 

### 数据处理
#### 数据预处理
数据预处理主要包括数据的读取， 数据的读取主要是读取高光谱数据，获得所需波段的传感器radiance数据，
将其传到匹配滤波算法中进行处理。



#### Modtran模拟

#### 卫星数据处理
针对卫星数据的处理，主要是读取卫星数据，获取卫星数据的辐亮度数据，将其传到匹配滤波算法中进行处理。\
对于AHSI传感器，利用AHSI_data.py文件进行数据读取，使用方法如下：
```
from MF import AHSI_data
# main functions:

get_raster_array(filepath):
    # 读取栅格数据
    # filepath: 文件路径
    # return: 栅格数据
    
rad_calibration(dataset, cal_file="GF5B_AHSI_RadCal_SWIR.raw"):**
    # 辐亮度校正
    # dataset: 数据集
    # cal_file: 校正文件
    # return: 校正后的数据集

export_array_to_tiff(result, filepath, output_folder)
    # 将结果导出为tiff文件
    # result: 甲烷浓度增强数组
    # filepath: 文件路径
    # output_folder: 校正后的数据集
    # return: none
    
image_coordinate(image_path):
    # 基于同名rpb文件进行影像校正,会在同路径生成一个新的校正后的影像
    # image_path: 影像路径
    # return: none

```
### 匹配滤波算法
在MF文件夹中存储有关匹配滤波算法的python文件

用法
```
from MF import matched_filter
# main functions:

open_unit_absorption_spectrum(filepath, min_wavelength, max_wavelength):
    # 打开吸收光谱文件
    # filepath: 文件路径
    # min_wavelength: 最小波长
    # max_wavelength: 最大波长
    # return: 波长，吸收光谱  
    
matched_filter(data_array, unit_absorption_spectrum, is_iterate=False, is_albedo=False, is_filter=False):
    # 匹配滤波函数
    # data_array: 数据数组
    # unit_absorption_spectrum: 单位吸收光谱
    # is_iterate: 是否迭代
    # is_albedo: 是否考虑反射率
    # is_filter: 是否进行l1滤波
    # return: 甲烷浓度增强结果 
```


### 数据可视化


## 后续跟进内容
1. 完善匹配滤波算法
2. 完善数据处理流程






























