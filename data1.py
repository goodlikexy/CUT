import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv("/home/hz/projects/CUTS/UNN-main/CUTS/data/train/data/C1_csv/C1_all_1.csv")

# 查看相关性矩阵
correlation_matrix = data.corr()

# 查看时间序列特征
# 可以使用自相关和互相关分析