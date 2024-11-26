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


def EnMAP_test():
    filepath = r"J:\stanford\EnMAP\ENMAP01-____L1B-DT0000005368_20221116T184046Z_004_V010501_20241017T040758Z\ENMAP01-____L1B-DT0000005368_20221116T184046Z_004_V010501_20241017T040758Z-SPECTRAL_IMAGE_SWIR.TIF"
    _, radiance_cube = sd.EnMAP_data.get_enmap_bands_array(filepath, 2150, 2500)
    sza, _ = sd.EnMAP_data.get_SZA_altitude(filepath)
    altitude = 0
    if altitude > 5:
        altitude = 5
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "EnMAP", 0, 50000, 2150, 2500, sza, altitude
    )

    # 原始匹配滤波算法结果测试
    mf_enhancement = mf(radiance_cube, uas, True, True, True)
    cmf_enhancement = cmf(radiance_cube, uas, True, True, True, 5)
    # 多层匹配滤波算法结果测试
    # methane_enhancement_mlmf = mfs.ml_matched_filter_new(radiance_cube,uas, True)

    output_folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    # vnir =  r"I:\stanford_campaign\EnMAP\dims_op_oc_oc-en_701862725_1\ENMAP.HSI.L1B\ENMAP01-____L1B-DT0000005368_20221116T184050Z_005_V010501_20241017T042058Z\ENMAP01-____L1B-DT0000005368_20221116T184050Z_005_V010501_20241017T042058Z-SPECTRAL_IMAGE_VNIR.TIF"

    sd.EnMAP_data.export_enmap_array_to_tiff(
        mf_enhancement, filepath, output_folder, filename.replace(".TIF", "_mf.tif")
    )
    sd.EnMAP_data.export_enmap_array_to_tiff(
        cmf_enhancement, filepath, output_folder, filename.replace(".TIF", "_cmf.tif")
    )

    return mf_enhancement, cmf_enhancement


start_time = time.time()
result = EnMAP_test()
end_time = time.time()
print("Time cost: ", end_time - start_time)
