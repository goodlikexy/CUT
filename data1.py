import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/test_b/data/merged_output_normalized_1.csv')

# 获取维度
print(f"文件维度: {df.shape}")
# shape[0]是行数，shape[1]是列数