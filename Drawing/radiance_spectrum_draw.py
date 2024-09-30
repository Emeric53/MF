import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
wavelength = np.linspace(1000, 2500, 1500)
mod_data = np.sin(wavelength / 100) + 2
gf5b_data = mod_data * 0.9
zy1f_data = mod_data * 0.8
prisma_data = mod_data * 0.7
enmap_data = mod_data * 0.6

# 创建一个画布，并在画布上添加自定义网格布局的子图
fig = plt.figure(figsize=(12, 8))

# 顶部大图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
ax1.plot(wavelength, mod_data, label="Mod, FWHM ~ 0.2nm", color="gray")
ax1.plot(wavelength, gf5b_data, label="GF5B obs, FWHM ~ 8nm", color="blue")
ax1.plot(wavelength, zy1f_data, label="ZY1F obs, FWHM ~ 16nm", color="orange")
ax1.plot(wavelength, prisma_data, label="PRISMA obs, FWHM ~ 11nm", color="green")
ax1.plot(wavelength, enmap_data, label="EnMAP obs, FWHM ~ 10nm", color="red")
ax1.set_ylabel("Radiance (μW cm⁻² sr⁻¹ nm⁻¹)")
ax1.legend()
ax1.grid(True)

# 左下角小图
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=1)
ax2.plot(wavelength, mod_data, color="gray")
ax2.plot(wavelength, gf5b_data, color="blue")
ax2.plot(wavelength, zy1f_data, color="orange")
ax2.plot(wavelength, prisma_data, color="green")
ax2.plot(wavelength, enmap_data, color="red")
ax2.set_xlim([1600, 1900])
ax2.set_ylim([0, 1.6])
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Radiance (μW cm⁻² sr⁻¹ nm⁻¹)")
ax2.grid(True)

# 右下角小图
ax3 = plt.subplot2grid((3, 3), (2, 2), colspan=1)
ax3.plot(wavelength, mod_data, color="gray")
ax3.plot(wavelength, gf5b_data, color="blue")
ax3.plot(wavelength, zy1f_data, color="orange")
ax3.plot(wavelength, prisma_data, color="green")
ax3.plot(wavelength, enmap_data, color="red")
ax3.set_xlim([2100, 2500])
# ax3.set_ylim([0, 0.6])
ax3.set_xlabel("Wavelength (nm)")
ax3.grid(True)

# 调整布局
plt.tight_layout()
print("The radiance spectrum is drawn successfully.")
# plt.show()
