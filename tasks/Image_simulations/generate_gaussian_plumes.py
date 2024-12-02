import numpy as np
import matplotlib.pyplot as plt
import os


def new_calc_sigmas(CATEGORY, x1):
    x = np.abs(x1)

    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))

    if CATEGORY == "A":
        a = 213
        b = 0.894
    elif CATEGORY == "B":
        a = 184.5
        b = 0.894
    elif CATEGORY == "C":
        a = 150
        b = 0.894
    elif CATEGORY == "D":
        a = 130
        b = 0.894
    elif CATEGORY == "E":
        a = 104
        b = 0.894
    elif CATEGORY == "F":
        a = 23.3
        b = 0.625
    elif CATEGORY == "G":
        a = 77.6
        b = 0.813
    else:
        sys.exit()

    sig_y = a * ((x / 1000) ** b)

    return sig_y


def gaussian_fuc(Q, u, windir, x, y, xs, ys, STABILITY):
    # 将风速分解为正北方向和正东方向的分量
    ux = u * np.cos((windir - 180.0) * np.pi / 180.0)
    uy = u * np.sin((windir - 180.0) * np.pi / 180.0)

    # 创建相对于排放源位置的坐标网格
    x1, y1 = np.meshgrid(x - xs, y - ys)

    # 计算顺风距离和横风距离
    dot_product = ux * x1 + uy * y1
    magnitudes = np.sqrt(ux**2 + uy**2) * np.sqrt(x1**2 + y1**2)
    subtended = np.arccos(dot_product / (magnitudes + 1e-15))  # 防止除以零
    hypotenuse = np.sqrt(x1**2 + y1**2)

    downwind = np.cos(subtended) * hypotenuse
    crosswind = np.sin(subtended) * hypotenuse

    # 只考虑顺风方向的点
    ind = downwind > 0
    C = np.zeros_like(x1)

    # 计算扩散参数
    sig_y = new_calc_sigmas(STABILITY, downwind[ind])

    # 使用高斯烟羽模型公式计算浓度分布
    C[ind] = (
        Q
        / (np.sqrt(2 * np.pi) * u * sig_y)
        * np.exp(-(crosswind[ind] ** 2) / (2 * sig_y**2))
    )
    # 参数设置
    molar_mass_CH4 = 16.04  # 甲烷的摩尔质量，g/mol
    molar_volume_STP = 0.0224  # 摩尔体积，m^3/mol at STP
    # 将 g/m^2 转换为 ppm
    C_ppmm = C / (molar_mass_CH4 / molar_volume_STP) * 1e6
    return C, C_ppmm


def plot_gaussian_plume(file_path, x, y):
    concentration = np.load(file_path)
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, concentration.T, levels=50, cmap="viridis")
    plt.colorbar(label="Concentration (g/m^3)")
    plt.xlabel("Downwind Distance (m)")
    plt.ylabel("Crosswind Distance (m)")
    plt.title(f"Gaussian Plume - {os.path.basename(file_path)}")
    plt.show()


if __name__ == "__main__":
    # 示例参数
    Q = 1000  # 排放率，单位 g/s
    windir = 180  # 风向，单位度
    x = np.linspace(-1470, 1500, 100)  # 下风向距离，单位 m
    y = np.linspace(-1470, 1500, 100)  # 横风向距离，单位 m
    xs, ys = 0, 0  # 排放源位置
    for stability in ["A", "B", "C", "D", "E", "F", "G"]:
        for windspeed in [2, 4, 6, 8, 10]:
            _, concentration = gaussian_fuc(
                Q, windspeed, windir, x, y, xs, ys, stability
            )
            file_path = f"data/simulated_plumes/gaussianplume_1000_{windspeed}_stability_{stability}.npy"
            plot_gaussian_plume(file_path, x, y)
            # np.save(
            #     f"data\simulated_plumes\\gaussianplume_1000_{windspeed}_stability_{stability}.npy",
            #     concentration,
            # )
# import numpy as np
# import sys
# import matplotlib.pyplot as plt
# import os


# def new_calc_sigmas(CATEGORY, x1):
#     x = np.abs(x1)
#     x = np.maximum(x, 0.1)  # 避免 x 为零或负值

#     # 根据 Pasquill-Gifford 稳定度类别计算参数
#     if CATEGORY == "A":
#         a_y, b_y = 213, 0.894
#     elif CATEGORY == "B":
#         a_y, b_y = 156, 0.894
#     elif CATEGORY == "C":
#         a_y, b_y = 104, 0.894
#     elif CATEGORY == "D":
#         a_y, b_y = 68, 0.894
#     elif CATEGORY == "E":
#         a_y, b_y = 50.5, 0.894
#     elif CATEGORY == "F":
#         a_y, b_y = 34, 0.894
#     else:
#         print("无效的稳定度类别")
#         sys.exit()

#     x_km = x / 1000.0  # 将距离转换为 km

#     sig_y = a_y * (x_km**b_y)
#     sig_y = np.maximum(sig_y, 0.1)

#     return sig_y


# def gaussian_fuc(Q, u, windir, x, y, xs, ys, STABILITY):
#     # 将风向转换为数学坐标系角度
#     theta = np.deg2rad(270 - windir)
#     # 调整风向
#     x1, y1 = np.meshgrid(x - xs, y - ys, indexing="ij")
#     x_prime = x1 * np.cos(theta) + y1 * np.sin(theta)
#     y_prime = -x1 * np.sin(theta) + y1 * np.cos(theta)

#     # 只考虑下风向
#     x_prime = np.maximum(x_prime, 0.1)

#     # 计算 σ_y
#     sig_y = new_calc_sigmas(STABILITY, x_prime)

#     # 计算浓度，积分了 z 方向
#     C = (Q / (np.sqrt(2 * np.pi) * u * sig_y)) * np.exp(-(y_prime**2) / (2 * sig_y**2))

#     return C


# def plot_gaussian_plume(concentration, x, y, file_path):
#     X, Y = np.meshgrid(x, y, indexing="ij")
#     plt.figure(figsize=(10, 8))
#     plt.pcolormesh(X, Y, concentration, shading="auto", cmap="viridis")
#     plt.colorbar(label="浓度 (g/s·m²)")
#     plt.xlabel("X 距离 (m)")
#     plt.ylabel("Y 距离 (m)")
#     plt.title(f"高斯烟羽 - {os.path.basename(file_path)}")
#     plt.axis("equal")
#     plt.show()


# if __name__ == "__main__":
#     # 示例参数
#     Q = 1000  # 排放率，单位 g/s
#     windir = 180  # 风向，单位度
#     x = np.arange(-1500, 1500, 30)  # 下风向距离，单位 m，分辨率为 30 m
#     y = np.arange(-1500, 1500, 30)  # 横风向距离，单位 m，分辨率为 30 m
#     xs, ys = 0, 0  # 排放源位置

#     for stability in ["A", "B", "C", "D", "E", "F"]:
#         for windspeed in [2, 4, 6, 8, 10]:
#             concentration = gaussian_fuc(Q, windspeed, windir, x, y, xs, ys, stability)

#             # 保存结果
#             os.makedirs("data/simulated_plumes_new", exist_ok=True)
#             file_path = f"data/simulated_plumes_new/gaussianplume_{Q}_{windspeed}_stability_{stability}.npy"
#             np.save(file_path, concentration)

#             # 绘制结果
#             plot_gaussian_plume(concentration, x, y, file_path)
