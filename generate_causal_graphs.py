import numpy as np
from causal_graph_utils import CausalGraphUtils

# 创建 CausalGraphUtils 实例
graph_utils = CausalGraphUtils()

# 加载矩阵文件
coef_data = np.load("/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/test_d/data_processed/discovered_graph_coef.npz")
thresholded_data = np.load("/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/test_d/data_processed/thresholded_graph.npz")

# 获取矩阵
coef_matrix = coef_data['coef_matrix']
thresholded_matrix = thresholded_data['thresholded_matrix']

# 将矩阵转换为下三角矩阵
lower_triangular_coef = graph_utils.make_lower_triangular(coef_matrix)
lower_triangular_thresholded = graph_utils.make_lower_triangular(thresholded_matrix)

# 提取相关边的位置
edges = np.argwhere(lower_triangular_thresholded > 0)

# 创建一个新的矩阵来存储权重，确保是浮点数类型
weighted_edges = np.zeros_like(lower_triangular_thresholded, dtype=float)

# 从 coef 矩阵中提取权重
for edge in edges:
    i, j = edge
    # 确保从 coef 矩阵中提取的值是有效的
    weight = lower_triangular_coef[i, j]
    
    # 打印调试信息
    print(f"Edge ({i}, {j}) - Coef Weight: {weight}")

    if weight != 0:  # 只在权重不为零时赋值
        weighted_edges[i, j] = weight

# 打印 weighted_edges 矩阵以检查权重
print("Weighted Edges Matrix:")
print(weighted_edges)

# 生成因果图
graph_utils.generate_causal_graph(
    causal_matrix=weighted_edges,
    filename="weighted_graph.png",
    output_dir="/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/test_d/data_processed/",
    threshold=0.1
)