import numpy as np
import pandas as pd

# 加载 .npz 文件
npz_file_path = 'outputs/cuts_example_2025_0224_132514_092603.npz'  # 替换为您的文件路径
data = np.load(npz_file_path)
print(data)

# 假设您想查看名为 'normal_segments' 的数组
normal_segments = data['pred_cm']  # 替换为您实际的数组名称

# 将数据转换为 DataFrame
df_normal_segments = pd.DataFrame(normal_segments.reshape(-1, normal_segments.shape[1]))

# 打印 DataFrame 的前几行
print(df_normal_segments.head()) 