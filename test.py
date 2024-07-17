from MatchedFilter.matched_filter import matched_filter
from tools.needed_function import read_tiff
import matplotlib.pyplot as plt
import numpy as np
filepath = r"I:\simulated_images_nonoise\wetland_q_1000_u_2_stability_D.tif"
result = read_tiff(filepath)
average = np.mean(result,axis=(1,2))
plt.plot(average)
plt.show()
