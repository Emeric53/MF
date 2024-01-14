"""绘制甲烷的透射光谱图像"""

import pandas as pd
import matplotlib.pyplot as plt


fulltrans = pd.read_csv("H:\\mathced-filter processing\\fulltrans.csv")


df = fulltrans[['FREQ(CM-1)', 'CH4 TRANS']]
# 将波数转换为wavelength（nm）
df['wavelength（nm）'] = 1 / (df['FREQ(CM-1)']) * 10**7
print(df.values)
# 选择wavelength范围为2000nm-2500nm的数据
filtered_data = df[(df['wavelength（nm）'] >= 2100) & (df['wavelength（nm）'] <= 2500)]

# 使用移动平均进行平滑处理
window_size = 10  # 调整窗口大小以控制平滑程度
filtered_data['平滑透射率'] = filtered_data['CH4 TRANS'].rolling(window=window_size).mean()

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['wavelength（nm）'], filtered_data['平滑透射率'])
plt.xlabel('Wavelength(nm)')
plt.ylabel('Transmittance')
plt.title('Methane transmittance over SWIR')
plt.grid(True)
plt.show()
