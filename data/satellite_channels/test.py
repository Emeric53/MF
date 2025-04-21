import numpy as np
import matplotlib.pyplot as plt

# Load data
a = np.load(
    "/home/emeric/Documents/GitHub/MF/data/satellite_channels/AHSI_channels.npz"
)
print("Data file variables:", a.files)
print("Central wavelengths (nm):\n", a["central_wvls"])
print("FWHMs (nm):\n", a["fwhms"])

central_wvls = a["central_wvls"]
fwhms = a["fwhms"]

# Set target wavelength range
min_wavelength = 2150
max_wavelength = 2500

# Filter data within the target wavelength range
filtered_indices = np.where(
    (central_wvls >= min_wavelength) & (central_wvls <= max_wavelength)
)
filtered_wvls = central_wvls[filtered_indices]
filtered_fwhms = fwhms[filtered_indices]

# Plot FWHM distribution
plt.figure(figsize=(10, 6))  # Set figure size for better visual appearance
plt.plot(
    filtered_wvls, filtered_fwhms, marker="o", linestyle="-", markersize=5
)  # Use markers and lines for clarity
plt.xlabel("Central Wavelength (nm)", fontsize=12)  # Clear labels with larger font size
plt.ylabel("FWHM (nm)", fontsize=12)
plt.title(
    "AHSI Channel FWHM Distribution (Wavelength Range: 2150-2500nm)",
    fontsize=14,
    fontweight="bold",
)  # Specific and prominent title
x_range = max_wavelength - min_wavelength  # 计算原始 x 轴范围
x_extension = x_range * 0.05  # 例如，扩展 5% 的范围 (您可以调整这个百分比)
extended_min_wavelength = min_wavelength - x_extension  # 新的 x 轴最小值
extended_max_wavelength = max_wavelength + x_extension  # 新的 x 轴最大值
plt.xlim(extended_min_wavelength, extended_max_wavelength)  # 设置扩展后的 x 轴范围

plt.grid(True, linestyle="--", alpha=0.7)  # Add grid lines for better readability
plt.xticks(fontsize=10)  # Adjust tick label font size
plt.yticks(fontsize=10)
plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()
# plt.savefig("/home/emeric/Desktop/AHSI.png")  # Save the plot as an image file
