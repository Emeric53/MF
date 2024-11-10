import re
import numpy as np

a = np.ones((10, 10, 10))
b = a[:, :, 1:2]
print(b.shape)
