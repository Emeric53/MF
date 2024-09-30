import sys
import numpy as np

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")

import MatchedFilter.mf_methods as mf

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
uas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mf.matched_filter(
    data_cube=a,
    unit_absorption_spectrum=uas,
    albedoadjust=False,
    iterate=False,
    sparsity=False,
)
