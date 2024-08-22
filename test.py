import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

all_result = np.load("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\resultdict.npz")



# 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum = np.nanmean(data_cube, axis=(1,2))
    target_spectrum = background_spectrum*unit_absorption_spectrum

    # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
    radiancediff_with_back =data_cube - background_spectrum[:, None,None]
    covariance = np.zeros((bands, bands))
    for row in range(rows): 
        for col in range(cols):
            covariance += np.outer(radiancediff_with_back[:, row, col], radiancediff_with_back[:, row, col])
    covariance = covariance/(rows*cols)
    covariance_inverse = np.linalg.inv(covariance)

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (radiancediff_with_back[:,row,col].T @ covariance_inverse @ target_spectrum)
            denominator = (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[row,col] = numerator/denominator
# test1 = all_result["non_result"].flatten()
# from matplotlib import pyplot as plt
# plt.hist(test1,bins=100,color='g')
# # 计算方差和标准差
# mean = np.mean(test1)
# std_dev = np.std(test1)

# # 在图中显示方差和标准差
# plt.title(f'Histogram\nVariance: {mean:.2f}, Standard Deviation: {std_dev:.2f}')

# # 显示图表
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# # plt.hist(test2,bins=50,color='r')
# plt.savefig("distribution.png")


# matrix_size = 100
# num_pixels = 200
# enhance_value = 1  # 设定增强的强度值
# plume = np.zeros((matrix_size, matrix_size))
# np.random.seed(42)  # 设置随机种子以保证结果可重复
# indices = np.random.choice(matrix_size * matrix_size, num_pixels, replace=False)
# all_indices = np.arange(plume.size)
# unenhanced_indices = np.setdiff1d(all_indices, indices)
# row_indices, col_indices = np.unravel_index(indices, (matrix_size, matrix_size))
# non_row_indices,non_col_indices = np.unravel_index(unenhanced_indices, (matrix_size, matrix_size))
# np.put(plume, indices, enhance_value)

# print(plume[row_indices,col_indices])
# print(plume[non_row_indices,non_col_indices])



x = []
y = []
albedo = []
# 生成示例数据
enhancements = np.arange(0,10000,10)
result = np.load("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\resultdict.npz")
for i in range(1000):
    for a in range(200):
        x.append(enhancements[i])
    y.extend(result["result"][i])
    albedo.extend(result["albedo"][i])

x = np.array(x)
y = np.array(y)
albedo = np.array(albedo)
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# 创建绘图
fig, ax = plt.subplots(figsize=(12, 10))

# 绘制散点图
scatter = ax.scatter(x, y, c=albedo, cmap='viridis', alpha=0.5)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Albedo factor')

# 线性回归拟合
model = LinearRegression().fit(x.reshape(-1, 1), y)
y_pred = model.predict(x.reshape(-1, 1))

# 绘制回归线
sns.lineplot(x=x, y=y_pred, ax=ax, color='red', label='Linear regression')
sns.lineplot(x=enhancements,y=enhancements,  color='green', label='1:1 line')
# 添加统计信息
r2 = model.score(x.reshape(-1, 1), y)
bias = np.mean(y_pred - y)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# 添加统计信息文本
ax.text(0.05, 0.95, f'$y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}$', transform=ax.transAxes, color='red', fontsize=12, verticalalignment='top')
ax.text(0.05, 0.90, f'$R^2 = {r2:.3f}$', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.05, 0.85, f'$BIAS = {bias:.3f}$', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.05, 0.80, f'$RMSE = {rmse:.3f}$', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.05, 0.75, f'$MAE = {mae:.3f}$', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# 设置标签和标题
ax.set_xlabel('True CH$_4$ enhancement (ppb)')
ax.set_ylabel('Retrieved CH$_4$ enhancement (ppb)')
ax.set_title('Scatter plot with Linear Regression and Statistics')

plt.legend()
plt.savefig("test.png")


ax, fig = plt.subplots(figsize=(12, 10))




