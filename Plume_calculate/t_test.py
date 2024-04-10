import scipy.stats as stats

# 假设的样本数据
sample1 = [4,5,6,7,8,9]
sample2 = [3,4,5,6,7,8]

# 进行t检验
t_statistic, p_value = stats.ttest_ind(sample1, sample2)
print(p_value)

