import numpy as np
import matplotlib.pyplot as plt
import hapi
from pylab import *

# HITRAN 数据库设置
hapi.db_begin("C:\\Users\\RS\\Documents\\Hitran")  # 设置数据文件夹的路径

# 加载分子数据
hapi.fetch('CO2', 2, 1, 4000, 6666)  # 二氧化碳, 分子标签2，同位素标签1，波数范围6200-7200 cm^-1
hapi.fetch('CH4', 6, 1, 4000, 6666)  # 甲烷, 分子标签6，同位素标签1，波数范围6200-7200 cm^-1
hapi.fetch('H2O', 1, 1, 4000, 6666)  # 水蒸气, 分子标签1，同位素标签1，波数范围6200-7200 cm^-1

x,y = hapi.getStickXY('CH4')

plot(x, y); plt.show()


# 设置条件
T = 296  # 温度, K
P = 1    # 压力, atm
L = 1000000    # 光程长度, cm

nu1, coef_CO2 = hapi.absorptionCoefficient_Voigt(SourceTables='CO2', OmegaStep = 0.01,Environment={'T': T, 'p': P})
nu2, coef_CH4 = hapi.absorptionCoefficient_Voigt(SourceTables='CH4',OmegaStep = 0.01, Environment={'T': T, 'p': P})
nu3, coef_H2O = hapi.absorptionCoefficient_Voigt(SourceTables='H2O', OmegaStep = 0.01,Environment={'T': T, 'p': P})
subplot(2,2,1); plot(nu1,coef_CO2); title('CO2 k(w): p=1 atm, T=296K')
subplot(2,2,2); plot(nu2,coef_CH4); title('CH4 k(w): p=1 atm, T=296K')
subplot(2,2,3); plot(nu3,coef_H2O); title('H2O k(w): p=1 atm, T=296K')
show()

nu,transm = hapi.transmittanceSpectrum(nu1,coef1,Environment={'l':1000.})
# print(coef_CH4)
#
# nu,trans = transmittanceSpectrum(nu2, coef_CH4)
# # 计算透射率
# trans_CO2 = np.exp(-coef_CO2 * L)
# trans_CH4 = np.exp(-coef_CH4 * L)
# trans_H2O = np.exp(-coef_H2O * L)
# print(trans_CH4)
# # 绘图
# plt.figure(figsize=(10, 6))
# plt.plot(nu, trans, label='CO2')
# plt.xlabel('Wavenumber (cm$^{-1}$)')
# plt.ylabel('Transmission')
# plt.title('High-Resolution Transmission Spectra')
# plt.legend()
# plt.grid(True)
# plt.show()
