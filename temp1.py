import re
import numpy as np

prisma = r"C:\Users\RS\VSCode\matchedfiltermethod\src\data\satellite_channels\AHSI_channels.npz"
data = np.load(prisma)
print(data["fwhms"])
