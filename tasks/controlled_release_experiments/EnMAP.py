import numpy as np
from matplotlib import pyplot as plt

import time
import math
import os

from methane_retrieval_algorithms.matchedfilter import matched_filter as mf
from methane_retrieval_algorithms.columnwise_matchedfilter import (
    columnwise_matched_filter as cmf,
)
from methane_retrieval_algorithms.ml_matchedfilter import ml_matched_filter as mlmf
from methane_retrieval_algorithms.columnwise_ml_matchedfilter import (
    columnwise_ml_matched_filter as cmlmf,
)
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


def EnMAP_test(filepath, output_folder):
    _, radiance_cube = sd.EnMAP_data.get_enmap_bands_array(filepath, 2150, 2500)

    # sza, _ = sd.EnMAP_data.get_SZA_altitude(filepath)
    # altitude = 0
    # if altitude > 5:
    #     altitude = 5
    sza = 25
    altitude = 0
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "EnMAP", 0, 50000, 2150, 2500, sza, altitude
    )

    # 原始匹配滤波算法结果测试
    mf_enhancement = mf(radiance_cube, uas, True, True, True)
    cmf_enhancement = cmf(radiance_cube, uas, True, True, True, 5)
    # 多层匹配滤波算法结果测试
    # methane_enhancement_mlmf = mfs.ml_matched_filter_new(radiance_cube,uas, True)

    # 输出结果到tiff文件
    filename = os.path.basename(filepath)

    sd.EnMAP_data.export_enmap_array_to_tiff(
        mf_enhancement, filepath, output_folder, filename.replace(".TIF", "_mf.tif")
    )
    sd.EnMAP_data.export_enmap_array_to_tiff(
        cmf_enhancement, filepath, output_folder, filename.replace(".TIF", "_cmf.tif")
    )

    return mf_enhancement, cmf_enhancement


filepath = "/home/emeric/Documents/stanford/EnMAP/enmap1.tif"


output_folder = "/home/emeric/Documents/stanford/EnMAP/"
start_time = time.time()
result = EnMAP_test(filepath, output_folder)
end_time = time.time()
print("Time cost: ", end_time - start_time)
