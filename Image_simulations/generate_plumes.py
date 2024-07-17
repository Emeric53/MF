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
    u1 = u
    x1, y1 = np.meshgrid(x - xs, y - ys)  # 创建二维网格

    # 象矢的风向是从正北方向开始，沿顺时针方向计算风向的方向，
    # 通过以下两行可将风速分解到正北和正东方向
    ux = u1 * np.sin((windir - 180.) * np.pi / 180.)
    uy = u1 * np.cos((windir - 180.) * np.pi / 180.)

    # 为了计算顺风距离和横风距离，需要计算风向与点与源之间连线的夹角，
    # 这里采用向量积的内积和模的计算，向量是用点除以模长来计算，减掉向量的夹角的cos值
    dot_product = ux * x1 + uy * y1
    magnitudes = u1 * np.sqrt(x1 ** 2. + y1 ** 2.)

    subtended = np.arccos(dot_product / (magnitudes + 1e-15))  # 夹角
    hypotenuse = np.sqrt(x1 ** 2. + y1 ** 2.)

    downwind = np.cos(subtended) * hypotenuse  # 顺风距离
    crosswind = np.sin(subtended) * hypotenuse  # 横风距离

    # 展平成一维数组
    downwind_flat = downwind.ravel()
    crosswind_flat = crosswind.ravel()
    ind = downwind_flat > 0
    sig_y = new_calc_sigmas(STABILITY, downwind_flat)

    # 计算浓度并重塑回二维数组
    C_flat = np.zeros_like(downwind_flat)
    C_flat[ind] = Q / (np.sqrt(2. * np.pi) * u1 * sig_y[ind]) * np.exp(-crosswind_flat[ind] ** 2. / (2. * sig_y[ind] ** 2.))
    C = C_flat.reshape(downwind.shape)
    # 参数设置
    molar_mass_CH4 = 16.04  # 甲烷的摩尔质量，g/mol
    molar_volume_STP = 0.0224  # 摩尔体积，m^3/mol at STP
    # 将 g/m^2 转换为 ppm
    C_ppm = C * 0.716 * 1e6
    return C, C_ppm

# 示例参数
Q = 100  # 排放率，单位 g/s
u = 5  # 风速，单位 m/s
windir = 270  # 风向，单位度
x = np.linspace(-3000, 3000, 200)  # 下风向距离，单位 m
y = np.linspace(-600, 600, 200)  # 横风向距离，单位 m
xs, ys = 0, 0  # 排放源位置
STABILITY = 'E'  # 稳定度分类

# 计算浓度分布
concentration,_ = gaussian_fuc(Q, u, windir, x, y, xs, ys, STABILITY)
print(np.max(concentration))

# 绘制结果
plt.figure(figsize=(10, 6))
plt.contourf(x, y, concentration.T, levels=100, cmap='jet')
plt.colorbar(label='Concentration (ppm)')
plt.xlabel('x (metres)')
plt.ylabel('y (metres)')
plt.title('ASC = {}  wind speed = {:.2f} m/s'.format(STABILITY, u))
plt.show()