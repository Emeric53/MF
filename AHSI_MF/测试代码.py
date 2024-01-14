import numpy as np
# import math
# e = math.e
# # originallist=[1*math.pow(e,-1.2), 1*math.pow(e,-1.3), 1*math.pow(e,-1.4),
# #               1*math.pow(e,-1.5), 1*math.pow(e,-1.6)]
# # for i in range(len(originallist)):
# #     originallist[i]= math.log(originallist[i],e)
# # x = np.array(originallist)
# # y = np.array([1,5])
# # print(x)
# # slope = np.polyfit(y, x, 1)
# # print(slope)
# # z = np.array([10,20,30,40,50])
# # slope = np.polyfit(y, x, 1)
# # print(slope)
# print(math.log(0.99,math.e)/20)
import numpy as np

# 假设您有一个名为data的三维数组，维度为(波段, 行, 列)
# 这里使用随机数据作为示例
data = np.random.rand(4, 3, 5)  # 生成一个4x3x5的随机数组作为示例
print(data)
# 沿着波段维度计算均值
mean_along_band = np.mean(data, axis=1)  # 在波段维度上计算均值，得到一个3x5的数组
print(mean_along_band)
# 获取每一行的均值
mean_along_band_per_row = np.mean(mean_along_band, axis=1)  # 沿着列维度计算均值，得到每一行的均值

# 打印结果
print("每一行的均值为:", mean_along_band_per_row)