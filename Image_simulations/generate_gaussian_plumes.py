import numpy as np
import sys
import matplotlib.pyplot as plt

def new_calc_sigmas(CATEGORY, x1):
    x = np.abs(x1)

    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))

    if CATEGORY == 'A':
        a = 213
        b = 0.894
    elif CATEGORY == 'B':
        a = 184.5
        b = 0.894
    elif CATEGORY == 'C':
        a = 150
        b = 0.894
    elif CATEGORY == 'D':
        a = 130
        b = 0.894
    elif CATEGORY == 'E':
        a = 104
        b = 0.894
    elif CATEGORY == 'F':
        a = 23.3
        b = 0.625
    elif CATEGORY == 'G':
        a = 77.6
        b = 0.813
    else:
        sys.exit()

    sig_y = a * ((x / 1000) ** b)

    return sig_y

def gaussian_fuc(Q, u, windir, x, y, xs, ys, STABILITY):
    # 将风速分解为正北方向和正东方向的分量
    ux = u * np.cos((windir - 180.) * np.pi / 180.)
    uy = u * np.sin((windir - 180.) * np.pi / 180.)

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
    C[ind] = Q / (np.sqrt(2 * np.pi) * u * sig_y) * np.exp(-crosswind[ind]**2 / (2 * sig_y**2))
    # 参数设置
    molar_mass_CH4 = 16.04  # 甲烷的摩尔质量，g/mol
    molar_volume_STP = 0.0224  # 摩尔体积，m^3/mol at STP
    # 将 g/m^2 转换为 ppm
    C_ppmm = C /(molar_mass_CH4/molar_volume_STP) * 1e6
    return C,C_ppmm


if __name__ == "__main__":
    # 示例参数
    Q = 1000  # 排放率，单位 g/s
    windir = 180  # 风向，单位度
    x = np.linspace(-1470, 1500, 100)  # 下风向距离，单位 m
    y = np.linspace(-1470, 1500, 100)  # 横风向距离，单位 m
    xs, ys = 0, 0  # 排放源位置
    for stability in ['A','B','C','D','E','F','G']:
        for windspeed in [2,4,6,8,10]:
                _,concentration = gaussian_fuc(Q, windspeed, windir, x, y, xs, ys, stability)
                np.save(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\simulated_plumes\\gaussianplume_1000_{windspeed}_stability_{stability}.npy",concentration)
    
    